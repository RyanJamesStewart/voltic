#!/usr/bin/env python3
"""Python-side benchmark harness — the 4 comparison rows.

Reads the *same* dataset the Rust harness uses (the CSV written by
`cargo run --release --bin bench -- --csv data.csv`), then times and checks the
accuracy of:
  * py_vollib                — the reference (wraps Jäckel's LetsBeRational C++);
                               pure-Python loop, one option at a time
  * py_vollib_vectorized     — the numpy-vectorized variant of the same
  * QuantLib (Python binding)— QuantLib's `blackFormulaImpliedStdDevChambers`
                               / `BlackScholesCalculator` IV
  * (jackal's numbers come from the Rust harness; this script does not call it)

Methodology, matching the Rust side:
  * same workload (the CSV);
  * an explicit warmup pass, discarded;
  * single-threaded (pin with `taskset -c 0 python bench.py …`);
  * median of N timed passes;
  * accuracy reported alongside throughput: max |solved σ − σ_true| over the
    rows each tool solved (and how many it could not), stratified by moneyness
    band (deep OTM S/K < 0.7, near ATM 0.95–1.05, deep ITM S/K > 1.3).

Usage:
    taskset -c 0 python bench/python/bench.py data.csv [--passes 7] [--limit N]

The tools (and pinned versions) are in `bench/python/requirements.txt`.
"""
import csv
import math
import statistics
import sys
import time

PASSES = 7
LIMIT = None
args = sys.argv[1:]
if not args:
    sys.exit("usage: python bench/python/bench.py <data.csv> [--passes N] [--limit M]")
csv_path = args[0]
i = 1
while i < len(args):
    if args[i] == "--passes":
        PASSES = int(args[i + 1]); i += 2
    elif args[i] == "--limit":
        LIMIT = int(args[i + 1]); i += 2
    else:
        i += 1

# --- load the dataset ------------------------------------------------------
S, K, T, R, P, SIG, FLAG = [], [], [], [], [], [], []
with open(csv_path) as f:
    for row in csv.DictReader(f):
        S.append(float(row["spot"])); K.append(float(row["strike"]))
        T.append(float(row["tte"])); R.append(float(row["rate"]))
        P.append(float(row["price"])); SIG.append(float(row["sigma_true"]))
        FLAG.append(row["kind"].strip())
if LIMIT:
    S, K, T, R, P, SIG, FLAG = (x[:LIMIT] for x in (S, K, T, R, P, SIG, FLAG))
N = len(S)
print(f"dataset: {N} options from {csv_path}")


def band(s, k):
    m = s / k
    if m < 0.7: return "deep OTM (S/K<0.7)"
    if 0.95 <= m <= 1.05: return "near ATM (0.95-1.05)"
    if m > 1.3: return "deep ITM (S/K>1.3)"
    return "other"


def report_accuracy(name, solved):
    """`solved[i]` = the tool's σ for option i, or NaN if it couldn't solve it."""
    overall_worst, overall_nan, bands = 0.0, 0, {}
    for i in range(N):
        b = band(S[i], K[i])
        bd = bands.setdefault(b, [0.0, 0, 0])  # [worst, nan, count]
        bd[2] += 1
        v = solved[i]
        if v is None or (isinstance(v, float) and math.isnan(v)):
            overall_nan += 1; bd[1] += 1; continue
        e = abs(v - SIG[i])
        overall_worst = max(overall_worst, e)
        bd[0] = max(bd[0], e)
    print(f"  {name}: max |σ−σ_true| overall = {overall_worst:.3e}  ({overall_nan} of {N} unsolved)")
    for b in ("deep OTM (S/K<0.7)", "near ATM (0.95-1.05)", "deep ITM (S/K>1.3)"):
        if b in bands:
            w, nn, c = bands[b]
            print(f"     {b:<24}: max abs err {w:.3e}  ({nn} of {c} unsolved)")


def time_pass(fn):
    fn()  # warmup, discarded
    samples = []
    for _ in range(PASSES):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    med = statistics.median(samples)
    return med / N * 1e9, 1e9 / (med / N * 1e9)  # ns/option, options/sec


# --- py_vollib (the reference; pure-Python loop) ---------------------------
try:
    from py_vollib.black_scholes.implied_volatility import implied_volatility as pv_iv
    last = [None] * N

    def run_pv():
        for i in range(N):
            try:
                last[i] = pv_iv(P[i], S[i], K[i], T[i], R[i], FLAG[i])
            except Exception:
                last[i] = float("nan")
    nspo, ops = time_pass(run_pv)
    print(f"\npy_vollib                : {nspo:>12.1f} ns/option   {ops:>12.3e} options/sec")
    report_accuracy("py_vollib (reference)", last)
except ImportError:
    print("\npy_vollib not installed — skipping (pip install py_vollib)")

# --- py_vollib_vectorized --------------------------------------------------
try:
    import numpy as np
    from py_vollib_vectorized import vectorized_implied_volatility as pvv_iv
    Sa, Ka, Ta, Ra, Pa = (np.asarray(x, dtype=float) for x in (S, K, T, R, P))
    Fa = np.asarray(FLAG)
    out = [None]

    def run_pvv():
        out[0] = pvv_iv(Pa, Sa, Ka, Ta, Ra, Fa, return_as="numpy")
    nspo, ops = time_pass(run_pvv)
    res = out[0]
    print(f"\npy_vollib_vectorized     : {nspo:>12.1f} ns/option   {ops:>12.3e} options/sec")
    report_accuracy("py_vollib_vectorized", list(res))
except ImportError:
    print("\npy_vollib_vectorized not installed — skipping (pip install py_vollib_vectorized)")

# --- QuantLib --------------------------------------------------------------
try:
    import QuantLib as ql

    def ql_iv(price, s, k, t, r, flag):
        # Standard QuantLib IV: build the BS process + a European option, ask
        # for the implied vol given the (undiscounted-forward) Black formula.
        # Use the European-option `impliedVolatility` solver — the apples-to-
        # apples one for a single contract.
        day_count = ql.Actual365Fixed()
        cal = ql.NullCalendar()
        today = ql.Date(15, 5, 2026)
        ql.Settings.instance().evaluationDate = today
        expiry = today + int(round(t * 365))
        opt_type = ql.Option.Call if flag == "c" else ql.Option.Put
        payoff = ql.PlainVanillaPayoff(opt_type, k)
        exercise = ql.EuropeanExercise(expiry)
        option = ql.EuropeanOption(payoff, exercise)
        u = ql.SimpleQuote(s)
        rate_q = ql.SimpleQuote(r)
        sigma_q = ql.SimpleQuote(0.20)
        ts_r = ql.FlatForward(today, ql.QuoteHandle(rate_q), day_count)
        ts_q = ql.FlatForward(today, ql.QuoteHandle(ql.SimpleQuote(0.0)), day_count)
        vol_ts = ql.BlackConstantVol(today, cal, ql.QuoteHandle(sigma_q), day_count)
        process = ql.BlackScholesMertonProcess(
            ql.QuoteHandle(u), ql.YieldTermStructureHandle(ts_q),
            ql.YieldTermStructureHandle(ts_r), ql.BlackVolTermStructureHandle(vol_ts))
        try:
            return option.impliedVolatility(price, process, 1e-10, 200, 1e-4, 5.0)
        except Exception:
            return float("nan")

    last_ql = [None] * N

    def run_ql():
        for i in range(N):
            last_ql[i] = ql_iv(P[i], S[i], K[i], T[i], R[i], FLAG[i])
    nspo, ops = time_pass(run_ql)
    print(f"\nQuantLib (Python binding): {nspo:>12.1f} ns/option   {ops:>12.3e} options/sec")
    report_accuracy("QuantLib", last_ql)
except ImportError:
    print("\nQuantLib not installed — skipping (pip install QuantLib)")

print("\n(jackal's rows come from `cargo run --release --bin bench`; this harness covers the 4 comparison tools only.)")
