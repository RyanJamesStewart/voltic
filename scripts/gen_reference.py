#!/usr/bin/env python3
"""Generate `tests/reference_pairs.csv` — a small (inputs, py_vollib-vol) table.

`py_vollib` wraps Jäckel's reference implementation (`py_lets_be_rational`,
which is `LetsBeRational.cpp`), so its output is treated as ground truth. This
script samples a spread of options across the moneyness bands (using the same
distribution shape as `bench/data.rs`, a small N), prices each one with the
reference Black-Scholes, then asks `py_vollib` for the IV back — and writes
`spot,strike,tte,rate,price,kind,vol_py_vollib`. The Rust test
`tests/properties.rs::reference_table` reads it and checks voltic agrees to
~1e-9 in vol space across the well-conditioned points (looser, ~1e-6, for the
deep-OTM-near-expiry points it does solve).

Usage:  python scripts/gen_reference.py            # writes tests/reference_pairs.csv
        python scripts/gen_reference.py 5000       # 5000 rows

Requires: py_vollib  (`pip install py_vollib`).
"""
import math
import os
import sys
import random

try:
    from py_vollib.black_scholes import black_scholes
    from py_vollib.black_scholes.implied_volatility import implied_volatility
except ImportError:
    sys.exit("py_vollib not installed — `pip install py_vollib` (in the bench venv)")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "tests", "reference_pairs.csv")

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
SEED = 0x5EEDBEEF  # not the Rust seed; this is a separate, small reference set
rng = random.Random(SEED)
ln_lo, ln_hi = math.log(1 / 365), math.log(2.0)

rows = []
attempts = 0
while len(rows) < N and attempts < N * 50:
    attempts += 1
    s = rng.uniform(50.0, 200.0)
    k = rng.uniform(40.0, 240.0)
    t = math.exp(rng.uniform(ln_lo, ln_hi))
    r = rng.uniform(0.0, 0.06)
    v = rng.uniform(0.05, 0.80)
    flag = "c" if len(rows) % 2 == 0 else "p"
    try:
        price = black_scholes(flag, s, k, t, r, v)
        df = math.exp(-r * t)
        intrinsic = max(s - k * df, 0.0) if flag == "c" else max(k * df - s, 0.0)
        if not math.isfinite(price) or (price - intrinsic) <= 1e-6 * s:
            continue
        iv = implied_volatility(price, s, k, t, r, flag)
        if not math.isfinite(iv):
            continue
    except Exception:
        continue
    rows.append((s, k, t, r, price, flag, iv))

with open(OUT, "w") as f:
    f.write("spot,strike,tte,rate,price,kind,vol_py_vollib\n")
    for s, k, t, r, price, flag, iv in rows:
        f.write(f"{s!r},{k!r},{t!r},{r!r},{price!r},{flag},{iv!r}\n")

print(f"wrote {len(rows)} reference rows to {os.path.normpath(OUT)} "
      f"(py_vollib / py_lets_be_rational = Jäckel reference)")
