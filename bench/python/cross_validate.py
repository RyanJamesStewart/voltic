#!/usr/bin/env python3
"""Cross-validate voltic::implied_vol_rational vs py_lets_be_rational.

Reads:
  - The dataset CSV that the Rust harness writes (the canonical 1M-option
    dataset, seeded by SplitMix64(0x5EEDBEEFCAFEF00D)).
  - The voltic-IV CSV that `cargo run --release --bin rational_iv` writes
    (one IV per row, same order as the dataset).

Runs py_lets_be_rational on each input as the oracle. Compares the two IVs
per input, reports:
  - max |voltic_iv - py_lbr_iv|
  - distribution (percentiles, histogram)
  - per-band stratification (deep OTM / near ATM / deep ITM)
  - count of disagreements (where py_lbr returned a non-NaN and voltic NaN'd,
    or vice versa)

CLEAN-ROOM DISCIPLINE: py_lets_be_rational is called as a black box via its
public API only. Its source has NOT been read by the implementer.

Usage:
  taskset -c 0 .venv/bin/python bench/python/cross_validate.py \\
      /tmp/voltic_data.csv /tmp/voltic_rational_iv.csv
"""
import csv
import math
import sys
import time

try:
    import py_lets_be_rational as lbr
except ImportError:
    sys.exit("install py_lets_be_rational: pip install py_lets_be_rational")

if len(sys.argv) < 3:
    sys.exit("usage: cross_validate.py <dataset.csv> <voltic_iv.csv>")

dataset_path = sys.argv[1]
voltic_iv_path = sys.argv[2]

# Read dataset.
rows = []
with open(dataset_path) as f:
    rd = csv.reader(f)
    next(rd)  # header
    for r in rd:
        rows.append({
            'S': float(r[0]),
            'K': float(r[1]),
            'T': float(r[2]),
            'r': float(r[3]),
            'price': float(r[4]),
            'sigma_true': float(r[5]),
            'kind': r[6],
        })
print(f"loaded {len(rows)} options from {dataset_path}", file=sys.stderr)

# Read voltic IVs.
voltic_iv = []
with open(voltic_iv_path) as f:
    rd = csv.reader(f)
    next(rd)  # header
    for r in rd:
        voltic_iv.append(float(r[1]))
print(f"loaded {len(voltic_iv)} voltic IVs from {voltic_iv_path}", file=sys.stderr)

assert len(rows) == len(voltic_iv), "row count mismatch"

# Run py_lbr on each input.
# Its IV function: implied_volatility_from_a_transformed_rational_guess(
#     price_undiscounted, F, K, T, theta)
# where price_undiscounted = forward-valued price = price·exp(rT)
# and theta = +1 for call, -1 for put.
py_iv = []
t0 = time.time()
n_lbr_fail = 0
for r in rows:
    F = r['S'] * math.exp(r['r'] * r['T'])
    fwd_price = r['price'] * math.exp(r['r'] * r['T'])
    theta = 1.0 if r['kind'] == 'c' else -1.0
    try:
        iv = lbr.implied_volatility_from_a_transformed_rational_guess(
            fwd_price, F, r['K'], r['T'], theta
        )
    except Exception:
        iv = float('nan')
    if not math.isfinite(iv):
        n_lbr_fail += 1
    py_iv.append(iv)
elapsed = time.time() - t0
print(f"py_lbr: {len(rows)} options in {elapsed:.2f}s ({elapsed/len(rows)*1e6:.1f} us/opt)", file=sys.stderr)
print(f"py_lbr failures (NaN): {n_lbr_fail}", file=sys.stderr)

# Compare.
def band(s, k):
    m = s / k
    if m < 0.7:
        return 'deep_otm'
    elif 0.95 <= m <= 1.05:
        return 'near_atm'
    elif m > 1.3:
        return 'deep_itm'
    return 'other'

stats = {'all': [], 'deep_otm': [], 'near_atm': [], 'deep_itm': [], 'other': []}
both_solved = 0
voltic_only = 0
lbr_only = 0
neither = 0
disagreements = []  # (idx, voltic_iv, py_iv, abs_diff, sigma_true)

for i, (r, v, p) in enumerate(zip(rows, voltic_iv, py_iv)):
    v_ok = math.isfinite(v)
    p_ok = math.isfinite(p)
    if v_ok and p_ok:
        both_solved += 1
        diff = abs(v - p)
        b = band(r['S'], r['K'])
        stats['all'].append(diff)
        stats[b].append(diff)
        if diff > 1e-10:
            disagreements.append((i, v, p, diff, r['sigma_true']))
    elif v_ok and not p_ok:
        voltic_only += 1
    elif p_ok and not v_ok:
        lbr_only += 1
    else:
        neither += 1

print()
print("=== Cross-validation results ===")
print(f"Total options:       {len(rows):>10}")
print(f"Both solved:         {both_solved:>10}")
print(f"voltic-only solved:  {voltic_only:>10}")
print(f"py_lbr-only solved:  {lbr_only:>10}  ← gaps in voltic to investigate")
print(f"Neither solved:      {neither:>10}")
print()
print("Agreement (|voltic - py_lbr|) by band:")
for b in ['all', 'deep_otm', 'near_atm', 'deep_itm', 'other']:
    diffs = sorted(stats[b])
    if not diffs:
        print(f"  {b:10s}: no data")
        continue
    n = len(diffs)
    print(f"  {b:10s}: n={n:>7} max={diffs[-1]:.3e}  p99={diffs[int(0.99*n)]:.3e}  "
          f"p90={diffs[int(0.90*n)]:.3e}  median={diffs[n//2]:.3e}")

# Show worst disagreements for diagnosis.
if disagreements:
    disagreements.sort(key=lambda x: x[3], reverse=True)
    print()
    print(f"Worst {min(10, len(disagreements))} disagreements (>1e-10):")
    for i, v, p, d, st in disagreements[:10]:
        r = rows[i]
        print(f"  idx={i} S={r['S']:.2f} K={r['K']:.2f} T={r['T']:.4f} "
              f"r={r['r']:.4f} σ_true={st:.4f} kind={r['kind']}: "
              f"voltic={v:.6e} py_lbr={p:.6e} |Δ|={d:.2e}")
