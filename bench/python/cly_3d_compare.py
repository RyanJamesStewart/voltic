#!/usr/bin/env python3
"""CLY-3D cross-solver comparison harness.

Reads `cly3d_data.csv` (written by `cargo run --release --bin cly_3d`) and
times + checks the accuracy of:
  * py_lets_be_rational            — Jäckel LBR scalar reference (Python loop)
  * py_vollib_vectorized           — numpy-vectorized wrapper around the same
  * volfi 0.1.8 iv_call            — wol-fi's batch IV (calls only)

Output: `/tmp/cly3d_results.json` merging Wren K's voltic rows with these
three Python rows, plus a "notes" block for any deviation. Metrics match
the Rust harness (ns/option = median of 7 timed passes after warmup,
`taskset -c 0`).

Voltic rows: this script reads them from
`/tmp/cly3d_voltic_partial.json` (produced by the Rust harness — see
`bench/cly_3d.rs`), or accepts CLI flags `--voltic-fast` / `--voltic-ctx`
each of which expects a "ns_per_option max_abs_err nan cat" tuple as four
floats / ints.

Important — volfi shape contract (per Wren K's v1.0.1 deviation note):
every option on CLY-3D has a unique (K, T), so the precomputed-context
API (`iv_call(F, K, disc, T, c)`) has the same per-row shape as a scalar
IV call. volfi's own `bench_vollib.py` uses `iv_call` for this reason; we
match it.

Usage:
    taskset -c 0 \
        /home/alfreddataba/work/voltic/.venv/bin/python \
        bench/python/cly_3d_compare.py cly3d_data.csv \
        --voltic-fast-ns 100.0 --voltic-fast-err 1e-12 --voltic-fast-nan 0 --voltic-fast-cat 0 \
        --voltic-ctx-ns  120.0 --voltic-ctx-err  1e-12 --voltic-ctx-nan  0 --voltic-ctx-cat  0 \
        --out /tmp/cly3d_results.json
"""
import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
import time

PASSES = 7

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--passes", type=int, default=PASSES)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--out", default="/tmp/cly3d_results.json")
ap.add_argument("--per-row-out", default="/tmp/cly3d_per_row_python.csv")
# Voltic results (passed in by the orchestrator after running the Rust bin).
ap.add_argument("--voltic-fast-ns", type=float)
ap.add_argument("--voltic-fast-err", type=float)
ap.add_argument("--voltic-fast-nan", type=int)
ap.add_argument("--voltic-fast-cat", type=int)
ap.add_argument("--voltic-ctx-ns", type=float)
ap.add_argument("--voltic-ctx-err", type=float)
ap.add_argument("--voltic-ctx-nan", type=int)
ap.add_argument("--voltic-ctx-cat", type=int)
args = ap.parse_args()

# --- load the dataset ------------------------------------------------------
S, K, T, R, P, SIG, FLAG = [], [], [], [], [], [], []
with open(args.csv) as f:
    for row in csv.DictReader(f):
        S.append(float(row["spot"])); K.append(float(row["strike"]))
        T.append(float(row["tte"])); R.append(float(row["rate"]))
        P.append(float(row["price"])); SIG.append(float(row["sigma_true"]))
        FLAG.append(row["kind"].strip())
if args.limit:
    S, K, T, R, P, SIG, FLAG = (x[:args.limit] for x in (S, K, T, R, P, SIG, FLAG))
N = len(S)
print(f"dataset: {N} options from {args.csv}")

CAT_THRESHOLD = 1e-3


def accuracy(solved):
    worst = 0.0
    nan = 0
    cat = 0
    for i in range(N):
        v = solved[i]
        if v is None or (isinstance(v, float) and (math.isnan(v) or not math.isfinite(v))):
            nan += 1
            continue
        e = abs(v - SIG[i])
        if e > worst:
            worst = e
        if e >= CAT_THRESHOLD:
            cat += 1
    return worst, nan, cat


def timed(fn):
    fn()  # warmup
    samples = []
    for _ in range(args.passes):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    med = statistics.median(samples)
    return med, med / N * 1e9


rows = []
notes = []


# Always include voltic rows if supplied (from the Rust harness).
if args.voltic_fast_ns is not None:
    rows.append({
        "solver": "voltic implied_vol_fast",
        "lang": "Rust SIMD",
        "ns_per_option": args.voltic_fast_ns,
        "wall_s": args.voltic_fast_ns * N / 1e9,
        "max_abs_err": args.voltic_fast_err,
        "nan": args.voltic_fast_nan,
        "cat_ge_1e-3": args.voltic_fast_cat,
    })
if args.voltic_ctx_ns is not None:
    rows.append({
        "solver": "voltic implied_vol_with_context_batch",
        "lang": "Rust SIMD",
        "ns_per_option": args.voltic_ctx_ns,
        "wall_s": args.voltic_ctx_ns * N / 1e9,
        "max_abs_err": args.voltic_ctx_err,
        "nan": args.voltic_ctx_nan,
        "cat_ge_1e-3": args.voltic_ctx_cat,
    })


# --- py_lets_be_rational (scalar LBR reference) ----------------------------
solved_lbr = [None] * N
try:
    from py_lets_be_rational import (
        implied_volatility_from_a_transformed_rational_guess as lbr_iv,
    )

    def run_lbr():
        for i in range(N):
            try:
                # signature: (price, F, K, T, q)  with q = +1 for call, -1 for put
                F = S[i] * math.exp(R[i] * T[i])
                # LBR works in undiscounted (forward) prices; voltic's
                # `bs_price` returns the spot-valued (discounted) price for
                # discount = exp(-r·T). Convert: c_undisc = c_disc / DF.
                disc = math.exp(-R[i] * T[i])
                c_undisc = P[i] / disc
                q = 1.0 if FLAG[i] == "c" else -1.0
                solved_lbr[i] = lbr_iv(c_undisc, F, K[i], T[i], q)
            except Exception:
                solved_lbr[i] = float("nan")

    wall_s, ns_per = timed(run_lbr)
    worst, nan, cat = accuracy(solved_lbr)
    rows.append({
        "solver": "py_lets_be_rational scalar",
        "lang": "Python loop over C++",
        "ns_per_option": ns_per,
        "wall_s": wall_s,
        "max_abs_err": worst,
        "nan": nan,
        "cat_ge_1e-3": cat,
    })
    print(f"py_lets_be_rational : {ns_per:>12.1f} ns/option   max err {worst:.3e}  NaN {nan}  cat {cat}")
except ImportError as e:
    notes.append(f"py_lets_be_rational missing: {e!s}")
    print(f"py_lets_be_rational not installed: {e}")


# --- py_vollib_vectorized --------------------------------------------------
solved_pvv = [None] * N
try:
    import numpy as np
    from py_vollib_vectorized import vectorized_implied_volatility as pvv_iv

    Sa = np.asarray(S, dtype=float)
    Ka = np.asarray(K, dtype=float)
    Ta = np.asarray(T, dtype=float)
    Ra = np.asarray(R, dtype=float)
    Pa = np.asarray(P, dtype=float)
    Fa = np.asarray(FLAG)
    out_box = [None]

    def run_pvv():
        out_box[0] = pvv_iv(Pa, Sa, Ka, Ta, Ra, Fa, return_as="numpy")

    wall_s, ns_per = timed(run_pvv)
    res = list(out_box[0])
    solved_pvv = res
    worst, nan, cat = accuracy(res)
    rows.append({
        "solver": "py_vollib_vectorized",
        "lang": "numpy-vectorized LBR",
        "ns_per_option": ns_per,
        "wall_s": wall_s,
        "max_abs_err": worst,
        "nan": nan,
        "cat_ge_1e-3": cat,
    })
    print(f"py_vollib_vectorized: {ns_per:>12.1f} ns/option   max err {worst:.3e}  NaN {nan}  cat {cat}")
except ImportError as e:
    notes.append(f"py_vollib_vectorized missing: {e!s}")
    print(f"py_vollib_vectorized not installed: {e}")
except Exception as e:
    notes.append(f"py_vollib_vectorized runtime: {type(e).__name__}: {e!s}")
    print(f"py_vollib_vectorized runtime fail: {type(e).__name__}: {e!s:.150}")


# --- volfi 0.1.8 iv_call ---------------------------------------------------
# Per Wren K's deviation note: every CLY-3D option has a unique (K, T),
# so iv_call (volfi's per-row API) is the apples-to-apples shape —
# matching volfi's own bench_vollib.py. We pass spot-valued call price.
solved_volfi = [None] * N
try:
    import volfi

    # volfi signature (from oracle_mpmath.py): iv_call(F, K, disc, T, c)
    # All calls on CLY-3D (S=100 < K).
    F_arr = [S[i] * math.exp(R[i] * T[i]) for i in range(N)]
    DISC_arr = [math.exp(-R[i] * T[i]) for i in range(N)]
    K_arr = list(K)
    T_arr = list(T)
    P_arr = list(P)
    out_box = [None]

    def run_volfi():
        out_box[0] = volfi.iv_call(F_arr, K_arr, DISC_arr, T_arr, P_arr)

    wall_s, ns_per = timed(run_volfi)
    res = list(out_box[0])
    solved_volfi = res
    worst, nan, cat = accuracy(res)
    rows.append({
        "solver": "volfi 0.1.8 iv_call",
        "lang": "Rust (wol-fi) batch",
        "ns_per_option": ns_per,
        "wall_s": wall_s,
        "max_abs_err": worst,
        "nan": nan,
        "cat_ge_1e-3": cat,
    })
    print(f"volfi iv_call       : {ns_per:>12.1f} ns/option   max err {worst:.3e}  NaN {nan}  cat {cat}")
except ImportError as e:
    notes.append(f"volfi missing: {e!s}")
    print(f"volfi not installed: {e}")
except Exception as e:
    notes.append(f"volfi runtime: {type(e).__name__}: {e!s}")
    print(f"volfi runtime fail: {type(e).__name__}: {e!s:.150}")


# --- host metadata ---------------------------------------------------------
def cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "unknown"


# --- per-row error CSV (so Wren F can compute banded stats) ----------------
with open(args.per_row_out, "w") as f:
    f.write("spot,strike,tte,sigma_true,sigma_lbr,sigma_pvv,sigma_volfi\n")
    for i in range(N):
        f.write(f"{S[i]:.17e},{K[i]:.17e},{T[i]:.17e},{SIG[i]:.17e},")
        sl = solved_lbr[i] if solved_lbr[i] is not None else float("nan")
        sp = solved_pvv[i] if solved_pvv[i] is not None else float("nan")
        sv = solved_volfi[i] if solved_volfi[i] is not None else float("nan")
        f.write(f"{sl:.17e},{sp:.17e},{sv:.17e}\n")
print(f"per-row Python solver CSV written: {args.per_row_out}")


# --- write merged JSON -----------------------------------------------------
result = {
    "grid_spec": {
        "source": "ThiopheneIV (arXiv:2605.22427) §A.1 'Dataset grids' — cross-checked with FlashIV (arXiv:2605.29102) §4.2 Table 3; underlying paper is Cui, Liu, Yao 2021 (J. Futures Markets, paywalled)",
        "n_points": N,
        "S": 100.0,
        "r": 0.03,
        "K_range": [105.0, 800.0],
        "T_range": [0.01, 2.0],
        "sigma_range": [0.01, 0.99],
        "discretization": "K = linspace(105, 800, 40), T = linspace(0.01, 2, 40), sigma = linspace(0.01, 0.99, 40); 40^3 = 64000 raw, filtered to call price > 1e-20 -> 51,321 cases (matches FlashIV Table 3 and ThiopheneIV Table 3 cell count exactly)",
        "kinds": "all calls (S < K everywhere on grid; ITM cases not present)",
    },
    "host": {
        "cpu": cpu_name(),
        "pin": "taskset -c 0",
        "passes": args.passes,
    },
    "results": rows,
    "notes": notes,
}
with open(args.out, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nwrote {args.out}")
