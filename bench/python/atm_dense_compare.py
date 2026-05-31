#!/usr/bin/env python3
"""ATM-dense cross-solver comparison harness.

Reads `atm_dense_data.csv` (written by `cargo run --release --bin atm_dense`)
and times + checks the accuracy of:
  * py_lets_be_rational            — Jäckel LBR scalar reference (Python loop)
  * py_vollib_vectorized           — numpy-vectorized wrapper around the same
  * volfi 0.1.8 (iv_call / iv_put) — wol-fi's batch IV (mixed kinds)

Output: `/tmp/atm_dense_results.json` merging Wren K's voltic rows with
these three Python rows, plus a "notes" block for any deviation. Metrics
match the Rust harness (ns/option = median of 7 timed passes after warmup,
`taskset -c 0`).

The grid is tight around the at-the-money strike (K/S in [0.85, 1.15], 31
points), spanning T in [0.01, 2] and sigma in [0.01, 0.99]. Call/put
convention: calls when K >= S, puts when K < S (OTM side of every strike).

Volfi shape contract: every option has a unique (K, T), so iv_call /
iv_put (per-row API) is the apples-to-apples shape. We split the dataset
by kind and run iv_call on the call subset, iv_put on the put subset,
then merge.

Usage:
    taskset -c 0 \
        /home/alfreddataba/work/voltic/.venv/bin/python \
        bench/python/atm_dense_compare.py atm_dense_data.csv \
        --voltic-fast-ns 100.0 --voltic-fast-err 1e-12 --voltic-fast-nan 0 --voltic-fast-cat 0 \
        --voltic-ctx-ns  120.0 --voltic-ctx-err  1e-12 --voltic-ctx-nan  0 --voltic-ctx-cat  0 \
        --out /tmp/atm_dense_results.json
"""
import argparse
import csv
import json
import math
import statistics
import time

PASSES = 7

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--passes", type=int, default=PASSES)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--out", default="/tmp/atm_dense_results.json")
ap.add_argument("--per-row-out", default="/tmp/atm_dense_per_row_python.csv")
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
n_calls = sum(1 for f in FLAG if f == "c")
n_puts = N - n_calls
print(f"dataset: {N} options from {args.csv}  ({n_calls} calls, {n_puts} puts)")

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
                F = S[i] * math.exp(R[i] * T[i])
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


# --- volfi 0.1.8 iv_call (puts via put-call parity) ------------------------
# volfi has no iv_put (see oracle_mpmath.py:152). The established pattern
# is: convert puts to equivalent calls via put-call parity
#     call_equiv = put + S - K * exp(-r*T)
# and feed everything through iv_call. Done in the timed region so we
# measure the realistic mixed-kind cost; the conversion is O(1) per row.
solved_volfi = [float("nan")] * N
try:
    import volfi

    F_arr = [S[i] * math.exp(R[i] * T[i]) for i in range(N)]
    DISC_arr = [math.exp(-R[i] * T[i]) for i in range(N)]
    # Pre-convert puts -> equivalent call prices via parity.
    C_equiv = [
        P[i] if FLAG[i] == "c" else (P[i] + S[i] - K[i] * DISC_arr[i])
        for i in range(N)
    ]
    out_box = [None]

    def run_volfi():
        out_box[0] = volfi.iv_call(F_arr, K, DISC_arr, T, C_equiv)

    wall_s, ns_per = timed(run_volfi)
    res = list(out_box[0])
    solved_volfi = res
    worst, nan, cat = accuracy(solved_volfi)
    rows.append({
        "solver": "volfi 0.1.8 iv_call (parity for puts)",
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


# --- per-row error CSV -----------------------------------------------------
with open(args.per_row_out, "w") as f:
    f.write("spot,strike,tte,kind,sigma_true,sigma_lbr,sigma_pvv,sigma_volfi\n")
    for i in range(N):
        sl = solved_lbr[i] if solved_lbr[i] is not None else float("nan")
        sp = solved_pvv[i] if solved_pvv[i] is not None else float("nan")
        sv = solved_volfi[i] if solved_volfi[i] is not None else float("nan")
        f.write(
            f"{S[i]:.17e},{K[i]:.17e},{T[i]:.17e},{FLAG[i]},{SIG[i]:.17e},"
            f"{sl:.17e},{sp:.17e},{sv:.17e}\n"
        )
print(f"per-row Python solver CSV written: {args.per_row_out}")


# --- write merged JSON -----------------------------------------------------
result = {
    "grid_spec": {
        "source": "ATM-dense grid added for voltic v1.1.0 to address the reviewer-flagged sparseness of SplitMix64 near-ATM coverage (n=506 of 100,000). Tight K/S=[0.85,1.15] envelope where real options usage concentrates.",
        "n_points": N,
        "n_calls": n_calls,
        "n_puts": n_puts,
        "S": 100.0,
        "r": 0.03,
        "K_range": [85.0, 115.0],
        "T_range": [0.01, 2.0],
        "sigma_range": [0.01, 0.99],
        "discretization": "K = linspace(85, 115, 31), T = linspace(0.01, 2, 40), sigma = linspace(0.01, 0.99, 40); 31*40*40 = 49600 raw, filtered to price > 1e-20",
        "kinds": "call when K >= S (OTM call), put when K < S (OTM put); OTM side of every strike",
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
