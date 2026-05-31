#!/usr/bin/env python3
"""Independent 200-bit mpmath oracle for voltic / py_lets_be_rational / volfi.

Measures distance-from-the-f64-BS-inversion-physical-floor rather than
distance-from-each-other. That converts the v1.0.0 cross-check headline
("matches LBR to 1e-15") into "at f64 floor, independently verified against
200-bit mpmath".

INPUTS:
  - The canonical dataset CSV (the SplitMix64-seeded 1M rows that the Rust
    harness writes): the price column there is the **f64-rounded** BS price
    that voltic / LBR / volfi all see.
  - The voltic IV CSV from `cargo run --release --bin rational_iv`, same
    row order. (We reuse rather than re-run.)

OUTPUTS:
  - bench/python/oracle_results.csv (one row per subsampled option)
  - stdout: per-band summary stats + the two derived numbers
      max(voltic_err / f64_floor)  -- 1.0 means voltic is at the floor
      max(voltic_err - lbr_err)    -- positive means voltic gaps to LBR

ALGORITHM:
  For each row:
    1. Promote (S, K, T, r, σ_true) to mpf at 200 bits of precision.
    2. Compute the 200-bit BS price using mp.erfc-based normal CDF.
    3. sigma_truth_mpmath := invert the **mpmath-computed price** via
       mp.findroot warm-seeded at sigma_true. Used as a sanity check on
       the oracle itself; must agree with sigma_true to << 1e-40.
    4. sigma_truth_from_f64_price := invert the **f64 price the dataset
       carries** (the one the f64 solvers see). This is the inversion's
       physical floor: no f64 solver can do better than this.
    5. f64_floor := |sigma_truth_from_f64_price - sigma_true|
    6. Three solver errors:
         voltic_err := |sigma_voltic - sigma_truth_from_f64_price|
         lbr_err    := |sigma_lbr    - sigma_truth_from_f64_price|
         volfi_err  := |sigma_volfi  - sigma_truth_from_f64_price|

BANDING (per Wren I):
  q = N(-d2)  ≈ risk-neutral OTM-call prob; |k|/√T uses log-moneyness.
    deep_otm: q < 0.01
    deep_itm: q > 0.99
    near_atm: |k|/√T < 5e-3
    other:    everything else

Usage:
  taskset -c 0 .venv/bin/python bench/python/oracle_mpmath.py \\
      --data /tmp/voltic_data.csv \\
      --voltic /tmp/voltic_rational_iv.csv \\
      --out bench/python/oracle_results.csv \\
      [--n 100000] [--seed 0]
"""
import argparse
import csv
import math
import os
import random
import sys
import time
from collections import defaultdict

try:
    import mpmath as mp
except ImportError:
    sys.exit("install mpmath: pip install mpmath")

try:
    import py_lets_be_rational as lbr
except ImportError:
    sys.exit("install py_lets_be_rational: pip install py_lets_be_rational")

try:
    import volfi
except ImportError:
    sys.exit("install volfi (cd .../volfi/bindings/python && pip install -e .)")

# --- 200-bit working precision -------------------------------------------
mp.mp.prec = 200  # ~60 decimal digits
ONE_HALF = mp.mpf("0.5")
SQRT2 = mp.sqrt(2)


def bs_price_mp(S, K, T, r, sigma, kind):
    """200-bit Black-Scholes price.

    N(x) = 0.5 * erfc(-x / sqrt(2)) — equivalent to mp.ncdf but mp.erfc is
    measurably faster at high precision.
    """
    s = sigma * mp.sqrt(T)
    d1 = (mp.log(S / K) + (r + ONE_HALF * sigma * sigma) * T) / s
    d2 = d1 - s
    Ke_rT = K * mp.exp(-r * T)
    if kind == "c":
        return S * (ONE_HALF * mp.erfc(-d1 / SQRT2)) - Ke_rT * (ONE_HALF * mp.erfc(-d2 / SQRT2))
    else:
        # P = K*e^{-rT}*N(-d2) - S*N(-d1)
        return Ke_rT * (ONE_HALF * mp.erfc(d2 / SQRT2)) - S * (ONE_HALF * mp.erfc(d1 / SQRT2))


def invert_mp(S, K, T, r, target_price_mp, kind, warm_seed):
    """200-bit IV: find σ s.t. BS(σ) = target_price_mp, warm-seeded at sigma_true."""
    def f(sigma):
        # findroot may probe negative; clamp via fabs so the function is well-defined.
        return bs_price_mp(S, K, T, r, mp.fabs(sigma), kind) - target_price_mp
    try:
        root = mp.findroot(f, warm_seed, tol=mp.mpf("1e-60"), maxsteps=100)
        return mp.fabs(root)
    except (ValueError, ZeroDivisionError) as e:
        return mp.mpf("nan")


def band_of(S, K, T, r, sigma_true, kind):
    """OTM-probability banding under each option's own kind.

    For a call: prob-call-finishes-OTM = N(-d2).
    For a put : prob-put-finishes-OTM  = N( d2).
    deep_otm = q < 0.01  (deeply unlikely to pay off)
    deep_itm = q > 0.99  (almost certain to pay off)
    near_atm = |log(S/K)| / sqrt(T) < 5e-3
    """
    s = sigma_true * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma_true * sigma_true) * T) / s
    d2 = d1 - s
    if kind == "c":
        q = 0.5 * math.erfc(d2 / math.sqrt(2.0))  # N(-d2)
    else:
        q = 0.5 * math.erfc(-d2 / math.sqrt(2.0))  # N(d2)
    if q < 0.01:
        return "deep_otm"
    if q > 0.99:
        return "deep_itm"
    log_m = math.log(S / K)
    if abs(log_m) / math.sqrt(T) < 5e-3:
        return "near_atm"
    return "other"


def lbr_iv(S, K, T, r, price, kind):
    """Black-box LBR call — undiscounted forward convention (same as cross_validate.py)."""
    F = S * math.exp(r * T)
    fwd_price = price * math.exp(r * T)
    theta = 1.0 if kind == "c" else -1.0
    try:
        v = lbr.implied_volatility_from_a_transformed_rational_guess(fwd_price, F, K, T, theta)
    except Exception:
        return float("nan")
    if not math.isfinite(v):
        return float("nan")
    return v


def volfi_iv_one(S, K, T, r, price, kind):
    """Black-box volfi.iv_call. Volfi has no iv_put; use put-call parity:
       call_equiv = put + S - K*exp(-rT) gives an equivalent call price."""
    disc = math.exp(-r * T)
    F = S * math.exp(r * T)
    if kind == "c":
        c = price
    else:
        c = price + S - K * disc
    # volfi.iv_call(f, k, d, t, p) — f=forward, d=discount, p=spot-valued call price
    try:
        out = volfi.iv_call([F], [K], [disc], [T], [c])
        v = float(out[0])
    except Exception:
        return float("nan")
    if not math.isfinite(v):
        return float("nan")
    return v


def percentile(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    if p >= 1.0:
        return sorted_vals[-1]
    idx = int(p * len(sorted_vals))
    if idx >= len(sorted_vals):
        idx = len(sorted_vals) - 1
    return sorted_vals[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/voltic_data.csv",
                    help="canonical dataset CSV (spot,strike,tte,rate,price,sigma_true,kind)")
    ap.add_argument("--voltic", default="/tmp/voltic_rational_iv.csv",
                    help="voltic IV CSV from `cargo run --bin rational_iv`")
    ap.add_argument("--out", default="bench/python/oracle_results.csv",
                    help="output CSV")
    ap.add_argument("--n", type=int, default=100_000,
                    help="subsample size (default 100k)")
    ap.add_argument("--seed", type=int, default=0,
                    help="subsample RNG seed for reproducibility")
    args = ap.parse_args()

    # --- load ----------------------------------------------------------------
    rows = []
    with open(args.data) as f:
        rd = csv.reader(f)
        next(rd)
        for r in rd:
            rows.append((float(r[0]), float(r[1]), float(r[2]), float(r[3]),
                         float(r[4]), float(r[5]), r[6]))
    print(f"loaded {len(rows)} rows from {args.data}", file=sys.stderr)

    voltic_iv_all = []
    with open(args.voltic) as f:
        rd = csv.reader(f)
        next(rd)
        for r in rd:
            voltic_iv_all.append(float(r[1]))
    assert len(rows) == len(voltic_iv_all), "voltic IV row count mismatch"
    print(f"loaded {len(voltic_iv_all)} voltic IVs from {args.voltic}", file=sys.stderr)

    # --- subsample (deterministic) -------------------------------------------
    n_total = len(rows)
    n_take = min(args.n, n_total)
    rng = random.Random(args.seed)
    idxs = rng.sample(range(n_total), n_take) if n_take < n_total else list(range(n_total))
    idxs.sort()
    print(f"subsampling {n_take} rows (seed={args.seed})", file=sys.stderr)

    # --- the run -------------------------------------------------------------
    t0 = time.time()
    out_rows = []
    band_buckets = defaultdict(lambda: {
        "f64_floor": [], "voltic_err": [], "lbr_err": [], "volfi_err": [],
        "voltic_over_floor": [], "voltic_minus_lbr": [],
    })

    max_self_consistency = mp.mpf(0)  # max |sigma_truth_mpmath - sigma_true|
    n_lbr_nan = 0
    n_volfi_nan = 0
    n_voltic_nan = 0

    log_every = max(1, n_take // 20)
    for prog, i in enumerate(idxs):
        S, K, T, r, price, sigma_true, kind = rows[i]
        sigma_voltic = voltic_iv_all[i]

        # promote
        Smp = mp.mpf(repr(S))
        Kmp = mp.mpf(repr(K))
        Tmp = mp.mpf(repr(T))
        rmp = mp.mpf(repr(r))
        sigma_true_mp = mp.mpf(repr(sigma_true))
        price_f64_mp = mp.mpf(repr(price))

        # 200-bit BS price at sigma_true
        price_mp = bs_price_mp(Smp, Kmp, Tmp, rmp, sigma_true_mp, kind)

        # 200-bit inversion of the mpmath price → must recover sigma_true
        sigma_truth_mpmath = invert_mp(Smp, Kmp, Tmp, rmp, price_mp, kind, sigma_true_mp)
        sc = mp.fabs(sigma_truth_mpmath - sigma_true_mp)
        if sc > max_self_consistency:
            max_self_consistency = sc

        # 200-bit inversion of the f64 price → the floor
        sigma_truth_floor = invert_mp(Smp, Kmp, Tmp, rmp, price_f64_mp, kind, sigma_true_mp)
        f64_floor = float(mp.fabs(sigma_truth_floor - sigma_true_mp))

        # f64 solvers
        sigma_lbr = lbr_iv(S, K, T, r, price, kind)
        sigma_volfi = volfi_iv_one(S, K, T, r, price, kind)

        truth = float(sigma_truth_floor)
        voltic_err = abs(sigma_voltic - truth) if math.isfinite(sigma_voltic) else float("nan")
        lbr_err = abs(sigma_lbr - truth) if math.isfinite(sigma_lbr) else float("nan")
        volfi_err = abs(sigma_volfi - truth) if math.isfinite(sigma_volfi) else float("nan")

        if not math.isfinite(sigma_voltic):
            n_voltic_nan += 1
        if not math.isfinite(sigma_lbr):
            n_lbr_nan += 1
        if not math.isfinite(sigma_volfi):
            n_volfi_nan += 1

        b = band_of(S, K, T, r, sigma_true, kind)

        out_rows.append((S, K, T, r, price, kind, sigma_true,
                         float(sigma_truth_mpmath), f64_floor,
                         sigma_voltic, sigma_lbr, sigma_volfi,
                         voltic_err, lbr_err, volfi_err, b))

        bk = band_buckets[b]
        bk["f64_floor"].append(f64_floor)
        if math.isfinite(voltic_err):
            bk["voltic_err"].append(voltic_err)
            if f64_floor > 0:
                bk["voltic_over_floor"].append(voltic_err / f64_floor)
            if math.isfinite(lbr_err):
                bk["voltic_minus_lbr"].append(voltic_err - lbr_err)
        if math.isfinite(lbr_err):
            bk["lbr_err"].append(lbr_err)
        if math.isfinite(volfi_err):
            bk["volfi_err"].append(volfi_err)

        if (prog + 1) % log_every == 0:
            dt = time.time() - t0
            rate_ = (prog + 1) / dt
            print(f"  {prog+1}/{n_take}  {dt:.1f}s  {rate_:.1f} opt/s "
                  f"(self-consistency max={float(max_self_consistency):.2e})",
                  file=sys.stderr)

    wall = time.time() - t0
    print(f"\nwall time: {wall:.1f}s  ({wall/n_take*1e3:.3f} ms/opt)", file=sys.stderr)

    # --- write CSV -----------------------------------------------------------
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["S", "K", "T", "r", "c", "kind", "sigma_true",
                    "sigma_truth_mpmath", "f64_floor",
                    "sigma_voltic", "sigma_lbr", "sigma_volfi",
                    "voltic_err", "lbr_err", "volfi_err", "band"])
        for row in out_rows:
            w.writerow([f"{row[0]:.17e}", f"{row[1]:.17e}", f"{row[2]:.17e}",
                        f"{row[3]:.17e}", f"{row[4]:.17e}", row[5],
                        f"{row[6]:.17e}", f"{row[7]:.17e}", f"{row[8]:.6e}",
                        f"{row[9]:.17e}", f"{row[10]:.17e}", f"{row[11]:.17e}",
                        f"{row[12]:.6e}", f"{row[13]:.6e}", f"{row[14]:.6e}",
                        row[15]])
    print(f"wrote {len(out_rows)} rows to {args.out}", file=sys.stderr)

    # --- oracle self-consistency check ---------------------------------------
    print()
    print(f"=== Oracle self-consistency check ===")
    print(f"max |sigma_truth_mpmath - sigma_true| = {float(max_self_consistency):.3e}")
    if max_self_consistency > mp.mpf("1e-40"):
        print("WARNING: oracle self-consistency check FAILED (> 1e-40).")
        print("The mpmath BS/invert pair did not recover sigma_true. Oracle suspect.")
    else:
        print("OK: oracle self-consistency holds at < 1e-40.")

    # --- per-band summary ----------------------------------------------------
    print()
    print(f"=== Per-band summary (n={n_take}) ===")
    print(f"voltic NaNs={n_voltic_nan}  lbr NaNs={n_lbr_nan}  volfi NaNs={n_volfi_nan}")
    print()

    def stats_line(name, vals):
        if not vals:
            return f"  {name:14s}: no data"
        vs = sorted(vals)
        n = len(vs)
        return (f"  {name:14s}: n={n:>7}  median={percentile(vs, 0.5):.3e}  "
                f"p90={percentile(vs, 0.90):.3e}  p99={percentile(vs, 0.99):.3e}  "
                f"max={vs[-1]:.3e}")

    for b in ("deep_otm", "near_atm", "deep_itm", "other"):
        bk = band_buckets.get(b)
        if not bk:
            print(f"[{b}] no rows")
            continue
        print(f"[{b}] n={len(bk['f64_floor'])}")
        print(stats_line("f64_floor", bk["f64_floor"]))
        print(stats_line("voltic_err", bk["voltic_err"]))
        print(stats_line("lbr_err", bk["lbr_err"]))
        print(stats_line("volfi_err", bk["volfi_err"]))
        print()

    # --- the two derived numbers --------------------------------------------
    all_over = []
    all_minus = []
    for bk in band_buckets.values():
        all_over.extend(bk["voltic_over_floor"])
        all_minus.extend(bk["voltic_minus_lbr"])
    if all_over:
        print(f"max(voltic_err / f64_floor)  = {max(all_over):.3e}  "
              f"(1.0 = at floor)")
    if all_minus:
        print(f"max(voltic_err - lbr_err)    = {max(all_minus):.3e}  "
              f"(positive = voltic gaps to LBR)")


if __name__ == "__main__":
    main()
