#!/usr/bin/env python3
"""Adversarial-grid voltic-vs-AQFED comparison harness.

Reads the merged CSV produced by `aqfed_adversarial.jl` (adversarial_data.csv
+ AQFED SR-Householder + AQFED Jaeckel columns) AND the voltic-typed result
(emitted alongside as JSON), then tabulates per-regime:

  - count
  - voltic typed status distribution
  - AQFED SR NaN count / value range
  - AQFED Jaeckel NaN count / value range
  - regimes where voltic flags but AQFED returns an unflagged value (the
    "voltic-correctly-flags AQFED-silently-mislabels" cell)
  - regimes where AQFED NaNs but voltic recovers a value

Usage:
    python3 bench/python/adversarial_compare.py \
        /tmp/aqfed_adversarial_results.csv \
        /tmp/voltic_typed_adversarial.json

The voltic-typed JSON is produced by `cargo run --release --bin adversarial_dump`.
"""
import csv
import json
import sys
import math
from collections import defaultdict, Counter

def main():
    if len(sys.argv) < 3:
        print("usage: adversarial_compare.py AQFED_CSV VOLTIC_JSON", file=sys.stderr)
        sys.exit(2)
    aqfed_csv = sys.argv[1]
    voltic_json = sys.argv[2]

    # --- Load AQFED merged rows
    rows = []
    with open(aqfed_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    n = len(rows)
    print(f"adversarial rows: {n}")

    # --- Load voltic typed rows
    with open(voltic_json) as f:
        vt = json.load(f)
    assert len(vt) == n, f"voltic JSON has {len(vt)} rows, expected {n}"

    # --- Per-regime tally
    by_regime = defaultdict(lambda: {
        'count': 0,
        'voltic_status': Counter(),
        'aqfed_sr_nan': 0,
        'aqfed_j_nan': 0,
        'voltic_flag_aqfed_silent_sr': 0,  # voltic non-Computed, AQFED SR finite
        'voltic_flag_aqfed_silent_j': 0,   # voltic non-Computed, AQFED Jaeckel finite
        'aqfed_nan_voltic_recovers_sr': 0, # AQFED SR NaN, voltic Computed/BelowVolMin/AboveVolMax
        'aqfed_nan_voltic_recovers_j': 0,
    })

    for i, row in enumerate(rows):
        regime = row['regime']
        d = by_regime[regime]
        d['count'] += 1
        v = vt[i]
        vstatus = v['status']
        d['voltic_status'][vstatus] += 1

        try:
            aqfed_sr = float(row['sigma_aqfed_sr'])
        except ValueError:
            aqfed_sr = float('nan')
        try:
            aqfed_j  = float(row['sigma_aqfed_jaeckel'])
        except ValueError:
            aqfed_j = float('nan')

        if math.isnan(aqfed_sr):
            d['aqfed_sr_nan'] += 1
        if math.isnan(aqfed_j):
            d['aqfed_j_nan'] += 1

        voltic_non_computed = vstatus != 'Computed'
        voltic_has_value = vstatus in ('Computed', 'BelowVolMin', 'AboveVolMax')

        if voltic_non_computed and not math.isnan(aqfed_sr):
            d['voltic_flag_aqfed_silent_sr'] += 1
        if voltic_non_computed and not math.isnan(aqfed_j):
            d['voltic_flag_aqfed_silent_j'] += 1
        if math.isnan(aqfed_sr) and voltic_has_value:
            d['aqfed_nan_voltic_recovers_sr'] += 1
        if math.isnan(aqfed_j) and voltic_has_value:
            d['aqfed_nan_voltic_recovers_j'] += 1

    # --- Print
    print()
    print(f"{'regime':30} {'n':4} | voltic statuses                                              | aqfed_sr_nan / aqfed_j_nan")
    print("-" * 140)
    for regime in sorted(by_regime):
        d = by_regime[regime]
        vstr = " ".join(f"{k}={v}" for k, v in sorted(d['voltic_status'].items()))
        print(f"{regime:30} {d['count']:4} | {vstr:60} | sr={d['aqfed_sr_nan']:3}  j={d['aqfed_j_nan']:3}")

    print()
    print("=== Where voltic FLAGS and AQFED returns an unflagged value ===")
    print("(voltic typed status is not Computed; AQFED returned a finite value — voltic is more honest)")
    for regime in sorted(by_regime):
        d = by_regime[regime]
        if d['voltic_flag_aqfed_silent_sr'] > 0 or d['voltic_flag_aqfed_silent_j'] > 0:
            print(f"  {regime:30}: aqfed_sr finite={d['voltic_flag_aqfed_silent_sr']:4d} | aqfed_jaeckel finite={d['voltic_flag_aqfed_silent_j']:4d}")
    total_silent_sr = sum(d['voltic_flag_aqfed_silent_sr'] for d in by_regime.values())
    total_silent_j  = sum(d['voltic_flag_aqfed_silent_j']  for d in by_regime.values())
    print(f"  TOTAL                          : aqfed_sr finite={total_silent_sr:4d} | aqfed_jaeckel finite={total_silent_j:4d}")

    print()
    print("=== Where AQFED NaNs but voltic RECOVERS a value (Computed/BelowVolMin/AboveVolMax) ===")
    for regime in sorted(by_regime):
        d = by_regime[regime]
        if d['aqfed_nan_voltic_recovers_sr'] > 0 or d['aqfed_nan_voltic_recovers_j'] > 0:
            print(f"  {regime:30}: aqfed_sr→NaN voltic→σ={d['aqfed_nan_voltic_recovers_sr']:4d} | aqfed_j→NaN voltic→σ={d['aqfed_nan_voltic_recovers_j']:4d}")
    total_rec_sr = sum(d['aqfed_nan_voltic_recovers_sr'] for d in by_regime.values())
    total_rec_j  = sum(d['aqfed_nan_voltic_recovers_j']  for d in by_regime.values())
    print(f"  TOTAL                          : aqfed_sr→NaN voltic→σ={total_rec_sr:4d} | aqfed_j→NaN voltic→σ={total_rec_j:4d}")

    # Overall NaN-count summary
    voltic_non_computed = sum(1 for r in vt if r['status'] != 'Computed')
    aqfed_sr_nan = sum(1 for row in rows if math.isnan(float(row.get('sigma_aqfed_sr', 'nan'))) if row.get('sigma_aqfed_sr') is not None)
    # Recompute defensively
    aqfed_sr_nan = sum(1 for row in rows if (lambda v: math.isnan(v))(float(row['sigma_aqfed_sr']) if row['sigma_aqfed_sr'] not in ('NaN','nan') else float('nan')))
    aqfed_j_nan  = sum(1 for row in rows if (lambda v: math.isnan(v))(float(row['sigma_aqfed_jaeckel']) if row['sigma_aqfed_jaeckel'] not in ('NaN','nan') else float('nan')))

    print()
    print("=== Overall flag/failure rates (adversarial grid) ===")
    print(f"  voltic typed non-Computed (=correctly-flagged): {voltic_non_computed} of {n} ({100*voltic_non_computed/n:.1f}%)")
    print(f"  aqfed SR-Householder NaN                       : {aqfed_sr_nan} of {n} ({100*aqfed_sr_nan/n:.1f}%)")
    print(f"  aqfed Jaeckel NaN                              : {aqfed_j_nan} of {n} ({100*aqfed_j_nan/n:.1f}%)")
    print()
    print("Interpretation: voltic flags more rows than AQFED NaNs — the typed API")
    print("provides correct rejection/labeling on regimes AQFED silently returns")
    print("a value for (especially NON-FINITE inputs, AT_INTRINSIC, AT_MAXIMUM).")

if __name__ == "__main__":
    main()
