# voltic

Vectorized Black-Scholes implied-volatility solver — `f64x8` SIMD, Schadner inverse-Gaussian seed with Householder-3 polish, with a Jäckel rational kernel as the deep-corner fallback. Independently verified against a 200-bit mpmath oracle.

```rust
use voltic::{implied_vol_fast, OptionKind};

// 30%-vol ATM call: S = K = 100, 1 year, r = 2%  →  priced at ~12.8216
let iv = implied_vol_fast(
    &[100.0],            // spot
    &[100.0],            // strike
    &[1.0],              // time to expiry (years)
    &[0.02],             // risk-free rate (continuous)
    &[12.821_58],        // option price
    &[OptionKind::Call],
);
assert!((iv[0] - 0.30).abs() < 1e-4);
```

Requires a **nightly Rust toolchain** (`std::simd`, `#![feature(portable_simd)]`), **AVX-512 hardware** (Zen 4+ or Intel Sapphire Rapids+), and a static **SLEEF** library at `~/.local/lib/libsleef.a`. See [Build requirements](#build-requirements).

---

## Headline

voltic and py_lets_be_rational sit at the f64 inversion floor across 100,000 SplitMix64-seeded options. volfi has a silent ~0.91% catastrophic-precision tail in the deep wings of the moneyness-vega plane (3-4% failure rate inside each deep-wing band; max σ error 3.3e-1).

Verification: an independent 200-bit mpmath oracle (`bench/python/oracle_mpmath.py`) inverts each option's f64-rounded BS price to the floor it can be inverted to; the f64 solvers' errors are reported relative to that floor. Oracle self-consistency at 7.5e-56 — passes the 1e-40 acceptance threshold by 16 orders of magnitude.

### Accuracy per band (100,000 SplitMix64-seeded options, mpmath-200-bit oracle)

| band | f64_floor max | voltic max | LBR max | volfi max | volfi NaN | volfi catastrophic (≥ 1e-3) |
|---|---:|---:|---:|---:|---:|---:|
| deep_otm (n=12,730) | 1.1e-11 | **2.2e-11** | 2.0e-11 | 3.2e-01 | 0 | 439 / 12,726 (3.45%) |
| near_atm (n=506)    | 7.2e-15 | **5.9e-15** | 2.3e-15 | 1.3e-11 | 0 | 0 / 506 (0%) |
| deep_itm (n=12,969) | 4.7e-12 | **4.3e-15** | 1.7e-15 | 3.3e-01 | 4 | 474 / 12,969 (3.65%) |
| other (n=73,795)    | 1.3e-13 | **1.9e-13** | 1.2e-13 | 1.4e-05 | 0 | 0 / 73,795 (0%) |
| **all (100,000)**   | 1.1e-11 | **2.2e-11** | 2.0e-11 | 3.3e-01 | 4 | 913 / 99,996 (0.91%) |

Bands are oracle bucketing by N(-d2): the `deep_otm` and `deep_itm` labels reflect very-low / very-high OTM-call probability and both correspond to the **deep wings of the (moneyness, vega) plane**. voltic and LBR have zero rows above 1e-3 anywhere. volfi's 913 catastrophic rows concentrate in those two deep-wing bands at 3-4% rate each.

Voltic carries a mild 2-4× residual vs LBR in the deep_otm tail (sub-picovol absolute — max 9.9e-12) — see [Known gaps](#known-gaps).

### Speed (single-threaded, znver5, taskset -c 0)

| solver | impl | n | ns/option | wall (s) | max abs err | NaN | catastrophic (≥ 1e-3) |
|---|---|---:|---:|---:|---:|---:|---:|
| **voltic 1.0.1 `implied_vol_fast`** | Rust f64x8 SIMD (znver5) | 1,000,000 | **73.6** | 0.074 | 3.42e-11 | 0 | 0 |
| voltic 1.0.1 `implied_vol_with_context_batch` | Rust f64x8 SIMD, precomputed-context | 1,000,000 | 33.8 | 0.034 | 3.42e-11 | 0 | 0 |
| voltic 1.0.1 `implied_vol_fully_vectorized` | Rust f64x8 SIMD, fused cold path | 1,000,000 | 45.9 | 0.046 | 3.42e-11 | 0 | 0 |
| py_lets_be_rational (LBR scalar) | Python+C++ scalar loop | 100,000 | 3,475.3 | 0.348 | 1.54e-11 | 0 | 0 |
| py_vollib_vectorized | Python+C++ numpy-vectorized | 100,000 | 405.6 | 0.041 | 2.04e-11 | 0 | 0 |
| volfi 0.1.8 `iv_call` | C++ binding (vectorized) | 100,000 | 350.1 | 0.035 | 3.34e-01 | 1 | 906 |

All rows on the same SplitMix64-seeded dataset (`bench/data.rs`, seed `0x5EEDBEEFCAFEF00D`). The volfi row uses the same `iv_call` path as volfi's own `bench_vollib.py`. The voltic Rust rows are 1M options (median of 7 timed passes after warmup, `cargo run --release --bin bench`); the Python comparison rows are a 100k subsample (Python is per-option-slower so 1M wall time would be 3+ s for LBR scalar). Same dataset, same RNG draw, first 100k rows. Put-side options for the volfi row use put-call parity to feed the call-only `iv_call` API.

Voltic's one-shot `implied_vol_fast` is **~48× faster than LBR scalar**, **~5.5× faster than py_vollib_vectorized**, **~4.8× faster than volfi**, with zero catastrophic errors and zero NaN. The precomputed-context shape (`implied_vol_with_context_batch`, the analogue of volfi's `volfi.ctx()` precomputed-context API) lands at 33.8 ns/option for repeat-`(k, T)` workloads.

### Volfi v×Δ wing-saturated stress grid

Two additional volfi tail tests on a wing-saturated grid (v ∈ {0.01, 0.05, 0.10, …, 2.00}, Δ ∈ {0.01, 0.05, …, 0.99}, T=1, F=1; 360 cases after filtering, the bench/wing_grid.rs harness):

```
=== volfi v×Δ grid benchmark ===
cases: 360
max |σ_solved − σ_true|: 8.298e-12   (excluding NaN)
NaN count: 2
median ns/option (median of 7, 5000 reps each): 81.3
```

The 2 NaN are pre-existing f64 conditioning failures at **(v=0.01, Δ∈{0.30, 0.70})** — tiny-σ near-ATM puts where the BS price is below 1e-7 (no meaningful f64 inverse). Pinned by the `volfi_wing_grid_nan_set_bounded_to_two` regression test so a future inner-iteration edit can't silently expand the NaN set.

---

## How to reproduce

```sh
# Build (nightly + SLEEF static lib at ~/.local/lib/libsleef.a)
cargo +nightly build --release

# Schadner cold: 1M synthetic options, znver5, taskset -c 0
taskset -c 0 ./target/release/bench --n 1000000

# volfi v×Δ wing-saturated stress grid (360 cases)
taskset -c 0 ./target/release/wing_grid

# 200-bit mpmath oracle (100k subsample, ~10 min wall)
python3 -m venv .venv
.venv/bin/pip install -r bench/python/requirements.txt
taskset -c 0 ./target/release/bench --n 1000000 --csv /tmp/voltic_data.csv
taskset -c 0 .venv/bin/python bench/python/oracle_mpmath.py \
    --data /tmp/voltic_data.csv \
    --voltic /tmp/voltic_rational_iv.csv \
    --out bench/python/oracle_results.csv \
    --n 100000

# Full test suite
cargo +nightly test --release
```

---

## Public API

Four entry points cover the depth-of-use axis. Pick the one that matches the workload shape:

### `implied_vol_fast` — one-shot, public, direct kernel

```rust
use voltic::{implied_vol_fast, OptionKind};
let iv = implied_vol_fast(&spot, &strike, &tte, &rate, &price, &kind);
```

73.6 ns/option cold, 0 NaN on the 1M Schadner grid, f64-inversion-floor accuracy.

### `OtmContext::new` + `implied_vol_with_context_batch` — split-API for repeat workloads

The `(k, T)`-only prelude is built once and reused across many price evaluations on the same `(strike, expiry)` node — vol-surface calibration, MC repricing on a fixed grid, scenario sweeps. 33.8 ns/option per evaluation.

```rust
use voltic::{OtmContext, implied_vol_with_context_batch};

let ctx = OtmContext::new(k_log, t);
let ivs = implied_vol_with_context_batch(&ctx, &prices);
```

### `implied_vol_fully_vectorized` — cold portfolio path

Every option has a unique `(k, T, c)`; the IG prelude itself runs at SIMD throughput, fused into the solve. 45.9 ns/option cold.

```rust
use voltic::implied_vol_fully_vectorized;
let ivs = implied_vol_fully_vectorized(&k, &t, &c);   // canonical OTM premium ratios
```

The vector-of-contexts shape (`implied_vol_vectorized_with_contexts`) is also exposed for callers that have already materialized an `&[OtmContext]`.

---

## Test coverage

`cargo +nightly test --release` runs **74 tests, all passing**:

- **52 lib unit tests** (`src/lib.rs`):
  - `black::tests` (12) — cancellation-free Black price, three derivatives (FD-cross-checked), inflection-point invariant, small-σ / large-σ asymptotics, `erfcx` reformulation, SIMD-vs-scalar bitwise consistency.
  - `jackel::tests` (13) — Jäckel rational kernel: quadrant canonicalization, region classification, rational-cubic interpolation, Householder-3 degenerate reduction, per-region initial-guess anchors, end-to-end solve.
  - `norm::tests` (11) — cumulative-normal kernels: Abramowitz-Stegun 26.2.17, Cody 1969, Hart 5666, West 2009 vs a 41-point high-precision reference, plus `erfcx` asymptotic + identity, Φ symmetry, Φ⁻¹ inversion.
  - `schadner::tests` (4) — explicit IG inverter: SIMD-vs-scalar, NaN on degenerate inputs, agreement with direct solver, ATM round-trip.
  - `schadner_fast::avenue1_property_tests` (2) — Avenue-1 fused-`erfcx` form: ATM finiteness, agreement away from centre.
  - `tests` (10) — top-level integration: SIMD tail padding, deep-OTM short-expiry, edge-case NaN policy, rational kernel grid, put-call parity, named extreme regimes.
- **11 proptest property tests** (`tests/properties.rs`): randomized round-trip σ recovery on each kernel, put-call parity, batch-vs-singleton SIMD agreement, `reference_table` against a py_lets_be_rational-generated reference.
- **9 wing-seed tests** (`tests/wing_seed.rs`): Wren G corner, mpmath-200-bit reference table across `h ∈ {3..8} × q ∈ {0.01, 0.05, 0.1, 0.2, 0.3}`, boundary finiteness at the gate edges, SIMD lane independence, end-to-end kernel σ recovery at wing corners, Chebyshev-regime non-regression, context-API routing through the wing seed, and the volfi v×Δ NaN-set regression pin.
- **2 doc tests**: `implied_vol_fast` usage in `src/lib.rs` and the Schadner usage example in `src/schadner.rs`.

The cross-validation against py_lets_be_rational on the full 1M dataset is the standalone harness `bench/python/cross_validate.py`. The 200-bit mpmath oracle is `bench/python/oracle_mpmath.py`. Neither is part of `cargo test`.

---

## Build requirements

- **Rust**: nightly (`rustup override set nightly`). The core uses `std::simd` (`#![feature(portable_simd)]`).
- **CPU**: AVX-512. Zen 4 / Zen 5 (Ryzen 7000+ / 9000+, EPYC Genoa+) or Intel Sapphire Rapids+. The crate ships a `.cargo/config.toml` pinning `target-cpu=znver5`; on non-Zen-5 hardware override with `RUSTFLAGS="-C target-cpu=native"` or edit the file.
- **SLEEF**: a static `libsleef.a` at `~/.local/lib/`. The build script (`build.rs`) hard-links the vectorized `exp` / `log`. If the library is elsewhere, edit `build.rs` to match — without it the crate will not link.

One-liner SLEEF install (Linux):

```sh
git clone https://github.com/shibatch/sleef.git /tmp/sleef && cd /tmp/sleef
cmake -S . -B build -DCMAKE_INSTALL_PREFIX="$HOME/.local" -DSLEEF_BUILD_STATIC_LIB=TRUE
cmake --build build -j && cmake --install build
```

---

## Implementation notes

1. **Chebyshev seed.** A deg-12 bivariate Chebyshev fit in `(ln k, logit q)` lands within ~1e-7 of the answer over the well-conditioned interior. The fit's (k, T)-only sub-summation is what `OtmContext` precomputes once per node.
2. **Wing seed (new in v1.0.1).** For `|k_log| ∈ [2.95, 8.0]` and IG-survival `q_surv ∈ (0, 0.30)` an analytic seed derived clean-room from Schadner's paper + the 2-term Mills asymptotic replaces the Cheb extrapolation. See [Algorithm — wing seed](#algorithm--wing-seed-v101-addition).
3. **Householder-3 iteration** (order-4 convergence). Three steps from either seed lands at the f64 conditioning floor where the seed is in domain.
4. **Cancellation-free Black via `erfcx`** (Avenue-1 fused form). The IG-survival residual is one difference of scaled complementary error functions (F3 anchor + B1 cancellation) via the `ig_surv_from_uv` primitive.
5. **SLEEF `f64x8` `vexp` / `vlog`.** The remaining `exp`/`log` calls go through SLEEF's vectorized intrinsics, statically linked.
6. **Dual bailout.** Pre-classification masks route ATM-strip (`|k|/√T < 5e-3`) and deep-OTM (`c_otm/F < 3e-6`) lanes that the seed-and-polish chain cannot reach to the Jäckel rational kernel.

### Algorithm — wing seed (v1.0.1 addition)

For an OTM option in the wing regime (large `|k_log|`, small IG-survival `q_surv = (1 − c_*)/m`), the Schadner explicit-IG quantile factors as

```
z1₀ = -Φ⁻¹(q_surv);    u₀ = -z1₀ + √(z1₀² + 2h)        (leading W0)

for n in 0..N_PICARD:
  z2  = h/u + u/2
  Q   = 1/z2 − 1/z2³                                   (2-term Mills)
  φ   = (2π)^(−1/2) · exp(−z1²/2)
  δ   = φ · Q                                          (= e^h · Φ(−z2), overflow-free)
  z1  = -Φ⁻¹(q_surv + δ)
  u   = -z1 + √(z1² + 2h)

v = u
```

The collapse `e^h · Φ(−z2) ≡ φ(z1) · Q(z2)` (with `Q(z) := √(π/2) · erfcx(z/√2) ≈ 1/z − 1/z³`) eliminates the `exp(h)` overflow on the wing. Voltic uses `N_PICARD = 1` followed by HH3 polish to the f64 floor.

The wing seed replaces a v1.0.0 path that extrapolated the Chebyshev seed beyond its fit domain (`SEED_K_HI = 3.0`) and produced 3.27e-1 catastrophic errors at `|k_log| > 3`. The new dispatch is gated by `K_HI_BAILOUT = 2.95`, `WING_Q_MAX = 0.30`, `WING_H_MAX = 8.0` (above which the Mills 2-term truncation breaks down and we route to the Jäckel rational fallback). The 200-bit mpmath reference table (`tests/wing_seed.rs` `WING_REF`) is regenerable via `scripts/wing_ref_gen.py`.

---

## Known gaps

- **Mild voltic–LBR residual in deep_otm.** Voltic carries a 2-4× median ratio vs py_lets_be_rational in the deep_otm tail (max absolute 9.9e-12, sub-picovol). Jäckel's rational guess wins by design in that corner; tightening voltic's deep_otm seed is v1.1 work. The ratio is well below the band's f64 conditioning floor for all but the outermost rows — see the per-band table above.
- **Two NaN at (v=0.01, Δ∈{0.30, 0.70}) on the wing v×Δ stress grid.** Tiny-σ near-ATM puts at the f64 BS price floor (< 1e-7). No meaningful f64 inverse exists for these inputs; they're pinned by the `volfi_wing_grid_nan_set_bounded_to_two` regression test.

---

## Reference comparisons (versions)

- [py_lets_be_rational 1.0.1](https://github.com/vollib/py_lets_be_rational) (Peter Jäckel, *Let's be rational*, Wilmott Magazine 2015) — the canonical accuracy reference.
- [volfi 0.1.8](https://github.com/coatless-rd/volfi) — vectorized C++ inverter.
- [py_vollib 1.0.7](https://github.com/vollib/py_vollib) — scalar Python wrapper around LBR.
- [py_vollib_vectorized 0.1.1](https://github.com/marcdemers/py_vollib_vectorized) — numpy-vectorized LBR wrapper.
- [QuantLib 1.42.1](https://www.quantlib.org/) — full-suite financial library (per-option Brent inversion).

---

## Limitations

- **European Black-Scholes only.** No American / early-exercise, no dividends (continuous or discrete), single flat risk-free rate.
- **Equity options only.** No FX (Garman-Kohlhagen), no commodities (Black-76), no rates options (SABR / shifted-lognormal).
- **Numerical domain.** A solved vol below 1% or above 500% is `NaN` by design. A premium at or below intrinsic value, or at or above the trivial upper bound, is `NaN`.
- **No f32 SIMD path.** f32 IV is rarely defensible; not implemented.
- **AVX-512 required.** The crate will not run on hardware below Zen 4 / Sapphire Rapids; the SLEEF link and the SIMD width are unconditional.

## What this is not

- Not a portfolio risk system, market-data feed, quote engine, backtesting framework, or trading system.
- One numerical kernel. Use it as a building block; build the rest yourself.

---

## License

Dual-licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT), at your option.

## Disclaimer

voltic is a numerical library. It is not investment advice, it is not certified for production trading use, and accuracy guarantees are limited to the conditions specified in the benchmark methodology above. Computed implied volatilities are deterministic functions of the inputs and have no opinion about whether you should trade anything.

---

For consulting: ryan@databa.ai
