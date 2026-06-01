# voltic

Vectorized Black-Scholes implied-volatility solver: `f64x8` SIMD, Schadner inverse-Gaussian seed with Householder-3 polish, with a Jäckel rational kernel as the deep-corner fallback. Independently verified against a 200-bit mpmath oracle.

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

voltic sits at the f64 inversion floor across 100,000 SplitMix64-seeded options, tied with py_lets_be_rational on accuracy and 48 times faster at f64x8 SIMD throughput. Verification is an independent 200-bit mpmath oracle (`bench/python/oracle_mpmath.py`) that inverts each option's f64-rounded BS price to the floor it can be inverted to; the f64 solvers' errors are reported relative to that floor. Oracle self-consistency at 7.5e-56 passes the 1e-40 acceptance threshold by 16 orders of magnitude.[^1]

The accuracy table below also surfaces a ~0.91% catastrophic-precision tail in volfi at the deep wings of the moneyness-vega plane (3-4% failure rate inside each deep-wing band; max σ error 3.3e-1).

### Accuracy per band (100,000 SplitMix64-seeded options, mpmath-200-bit oracle)

| band | f64_floor max | voltic max | LBR max | volfi max | volfi NaN | volfi catastrophic (≥ 1e-3) |
|---|---:|---:|---:|---:|---:|---:|
| deep_otm (n=12,730) | 1.1e-11 | **2.2e-11** | 2.0e-11 | 3.2e-01 | 0 | 439 / 12,726 (3.45%) |
| near_atm (n=506)    | 7.2e-15 | **5.9e-15** | 2.3e-15 | 1.3e-11 | 0 | 0 / 506 (0%) |
| deep_itm (n=12,969) | 4.7e-12 | **4.3e-15** | 1.7e-15 | 3.3e-01 | 4 | 474 / 12,969 (3.65%) |
| other (n=73,795)    | 1.3e-13 | **1.9e-13** | 1.2e-13 | 1.4e-05 | 0 | 0 / 73,795 (0%) |
| **all (100,000)**   | 1.1e-11 | **2.2e-11** | 2.0e-11 | 3.3e-01 | 4 | 913 / 99,996 (0.91%) |

Bands are oracle bucketing by N(-d2): the `deep_otm` and `deep_itm` labels reflect very-low / very-high OTM-call probability and both correspond to the **deep wings of the (moneyness, vega) plane**. voltic and LBR have zero rows above 1e-3 anywhere. volfi's 913 catastrophic rows concentrate in those two deep-wing bands at 3-4% rate each.

Voltic carries a sub-picovol residual vs LBR concentrated in the |h|≥4 deep-wing tail (n=230, max 9.9e-12). See [Accuracy: known gaps](#accuracy-known-gaps).

The volfi finding is reproduced via volfi's own `otm_context` API and volfi-self-priced inputs; the wing-saturated v×Δ stress grid (360 cases) reproduces it on a third dataset. Full methodology in [`research/volfi-verification-methodology.md`](research/volfi-verification-methodology.md).

---

### Speed (single-threaded, znver5, taskset -c 0)

| solver | impl | n | ns/option | wall (s) | max abs err | NaN |
|---|---|---:|---:|---:|---:|---:|
| **voltic 1.1.0 `implied_vol_fast`** | Rust f64x8 SIMD (znver5) | 1,000,000 | **73.6** | 0.074 | 3.42e-11 | 0 |
| py_lets_be_rational (LBR scalar) | Python+C++ scalar loop | 100,000 | 3,475.3 | 0.348 | 1.54e-11 | 0 |
| py_vollib_vectorized | Python+C++ numpy-vectorized | 100,000 | 405.6 | 0.041 | 2.04e-11 | 0 |

All rows on the same SplitMix64-seeded dataset (`bench/data.rs`, seed `0x5EEDBEEFCAFEF00D`). The voltic Rust rows are 1M options (median of 7 timed passes after warmup, `cargo run --release --bin bench`); the Python comparison rows are a 100k subsample (Python is per-option-slower so 1M wall time would be 3+ s for LBR scalar). Same dataset, same RNG draw, first 100k rows.[^2]

Voltic's one-shot `implied_vol_fast` is about 48 times faster than LBR scalar and about 5.5 times faster than py_vollib_vectorized, with zero catastrophic errors and zero NaN.

### Benchmarks: CLY-3D (Cui-Liu-Yao 2021 standard grid)

The CLY-3D grid (51,321 deep-OTM-weighted points; defined in Cui, Liu, Yao 2021 and used as the standard benchmark in the May 2026 FlashIV and ThiopheneIV preprints) provides a second independent dataset alongside SplitMix64. Grid: `S=100`, `r=0.03`, `K ∈ linspace(105, 800, 40)`, `T ∈ linspace(0.01, 2, 40)`, `σ ∈ linspace(0.01, 0.99, 40)`, filtered to call price > 1e-20 (matches FlashIV Table 3 and ThiopheneIV Table 3 cell count exactly).

| solver | impl | ns/option | wall (s) | max abs err | NaN |
|---|---|---:|---:|---:|---:|
| **voltic 1.1.0 `implied_vol_fast`** | Rust f64x8 SIMD (znver5) | **89** | 0.0046 | 1.539e-09 | 13 |
| py_lets_be_rational (scalar) | Python+C++ scalar loop | 3,268 | 0.168 | 1.539e-09 | 0 |
| py_vollib_vectorized | Python+C++ numpy-vectorized | 372 | 0.019 | 1.539e-09 | 0 |

voltic, LBR, and py_vollib_vectorized all sit at 1.539e-9 max abs σ error (the f64 reverse-Black floor at the deep-OTM near-expiry corner). voltic is 36.7 times faster than LBR scalar and 4.2 times faster than py_vollib_vectorized.

Voltic's 13 NaN are rows where `σ_true = VOL_MIN = 0.01` exactly, excluded by the open-interval domain. See [Accuracy: known gaps](#accuracy-known-gaps).

### Benchmarks: ATM-dense (near-at-the-money grid)

Real options markets are densest at the money. The SplitMix64 dataset (1M synthetic) has only n=506 near-ATM rows by the `|k|/√T < 5e-3` definition (0.05%). To verify accuracy at the volume regime, voltic v1.1.0 adds an ATM-dense grid: `S=100`, `r=0.03`, `K ∈ linspace(85, 115, 31)`, `T ∈ linspace(0.01, 2, 40)`, `σ ∈ linspace(0.01, 0.99, 40)`, filtered to call price > 1e-20 (n=48,831 after filtering, OTM side of every strike).

| solver | ns/option | max abs err | NaN |
|---|---:|---:|---:|
| **voltic 1.1.0 `implied_vol_fast`** | **67.8** | 3.338e-03 | 288 |
| py_lets_be_rational (scalar) | 5,315 | 3.338e-03 | 0 |
| py_vollib_vectorized | 498 | 3.338e-03 | 0 |

voltic and LBR sit at the same max error (3.338e-3, governed by 2 deep-wing cases shared by all three solvers); voltic is 78 times faster than LBR scalar on the ATM regime. voltic's 288 NaN are rows where `σ_true = VOL_MIN = 0.01` exactly, excluded by voltic's open-interval domain. See [Accuracy: known gaps](#accuracy-known-gaps).

---

## How to reproduce

```sh
# Build (nightly + SLEEF static lib at ~/.local/lib/libsleef.a)
cargo +nightly build --release

# Schadner cold: 1M synthetic options, znver5, taskset -c 0
taskset -c 0 ./target/release/bench --n 1000000

# Wing v×Δ stress grid (360 cases)
taskset -c 0 ./target/release/wing_grid

# CLY-3D standard grid (Cui-Liu-Yao 2021; 51,321 cases)
cargo +nightly run --release --bin cly_3d
python3 bench/python/cly_3d_compare.py

# ATM-dense grid (near-at-the-money, 48,831 cases)
cargo +nightly run --release --bin atm_dense
python3 bench/python/atm_dense_compare.py

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

### `implied_vol_fast`: one-shot, public, direct kernel

```rust
use voltic::{implied_vol_fast, OptionKind};
let iv = implied_vol_fast(&spot, &strike, &tte, &rate, &price, &kind);
```

73.6 ns/option cold, 0 NaN on the 1M Schadner grid, f64-inversion-floor accuracy.

### `OtmContext::new` + `implied_vol_with_context_batch`: throughput-prioritized split API

The `(k, T)`-only prelude is built once and reused across many price evaluations on the same `(strike, expiry)` node, e.g. vol-surface calibration, MC repricing on a fixed grid, scenario sweeps. About 34 to 40 ns/option per evaluation. This path skips the rational-fallback step `implied_vol_fast` uses, so on unfiltered grids it produces catastrophic errors (e.g. ~20% of rows on CLY-3D, max σ error 0.85); it is only correct on pre-filtered interior input. Use `implied_vol_fast` for accuracy-critical paths.

```rust
use voltic::{OtmContext, implied_vol_with_context_batch};

let ctx = OtmContext::new(k_log, t);
let ivs = implied_vol_with_context_batch(&ctx, &prices);
```

### `implied_vol_fully_vectorized`: cold portfolio path

Every option has a unique `(k, T, c)`; the IG prelude itself runs at SIMD throughput, fused into the solve. 45.9 ns/option cold.

```rust
use voltic::implied_vol_fully_vectorized;
let ivs = implied_vol_fully_vectorized(&k, &t, &c);   // canonical OTM premium ratios
```

The vector-of-contexts shape (`implied_vol_vectorized_with_contexts`) is also exposed for callers that have already materialized an `&[OtmContext]`.

### `implied_vol_typed` (v1.2): boundary-status API

Returns an `ImpliedVolResult { value, status }` per option. The status arm distinguishes seven outcomes the bare-NaN API conflates: `Computed`, `BelowVolMin { computed }`, `AboveVolMax { computed }`, `BelowIntrinsic`, `AboveMaximum`, `NonFinite`, `FailedToConverge`. The legacy `implied_vol_fast` / `implied_vol` map all non-`Computed` to a single NaN; the typed surface keeps the regime and (for `BelowVolMin` / `AboveVolMax`) the sigma the iteration actually found.

```rust
use voltic::{implied_vol_typed_batch, ImpliedVolStatus, OptionKind};

let results = implied_vol_typed_batch(&spot, &strike, &tte, &rate, &price, &kind);
for r in &results {
    match r.status {
        ImpliedVolStatus::Computed => { /* r.value is the IV */ }
        ImpliedVolStatus::BelowVolMin { computed } => { /* sub-VOL_MIN root */ }
        ImpliedVolStatus::AboveVolMax { computed } => { /* super-VOL_MAX root */ }
        ImpliedVolStatus::BelowIntrinsic
        | ImpliedVolStatus::AboveMaximum
        | ImpliedVolStatus::NonFinite
        | ImpliedVolStatus::FailedToConverge => { /* domain-rejection */ }
    }
}
```

The typed path runs a Householder-3 update (FlashIV eq. 6, AQFED.jl parity) on a wide internal bracket `[1e-8, 50.0]`, with a three-term price-residual floor and a sigma-resolution-aware classification gate. Computed sigma is guaranteed within a 1e-6 absolute sigma-resolution budget of the f64-stored price's true sigma; 200-bit mpmath verification on stratified CLY-3D and ATM-dense samples confirms worst deviation 3.41e-14 and 5.57e-8 respectively.

The legacy f64-returning entry points (`implied_vol_fast`, `OtmContext::*`, `implied_vol_fully_vectorized`) are byte-identical with v1.1.0; use the typed API only when you need the regime structure.


---

## Test coverage

`cargo +nightly test --release` runs **97 tests, all passing**:

- **52 lib unit tests** (`src/lib.rs`):
  - `black::tests` (12): cancellation-free Black price, three derivatives (FD-cross-checked), inflection-point invariant, small-σ / large-σ asymptotics, `erfcx` reformulation, SIMD-vs-scalar bitwise consistency.
  - `jackel::tests` (13): Jäckel rational kernel: quadrant canonicalization, region classification, rational-cubic interpolation, Householder-3 degenerate reduction, per-region initial-guess anchors, end-to-end solve.
  - `norm::tests` (11): cumulative-normal kernels: Abramowitz-Stegun 26.2.17, Cody 1969, Hart 5666, West 2009 vs a 41-point high-precision reference, plus `erfcx` asymptotic + identity, Φ symmetry, Φ⁻¹ inversion.
  - `schadner::tests` (4): explicit IG inverter: SIMD-vs-scalar, NaN on degenerate inputs, agreement with direct solver, ATM round-trip.
  - `schadner_fast::avenue1_property_tests` (2): Avenue-1 fused-`erfcx` form: ATM finiteness, agreement away from centre.
  - `tests` (10): top-level integration: SIMD tail padding, deep-OTM short-expiry, edge-case NaN policy, rational kernel grid, put-call parity, named extreme regimes.
- **11 proptest property tests** (`tests/properties.rs`): randomized round-trip σ recovery on each kernel, put-call parity, batch-vs-singleton SIMD agreement, `reference_table` against a py_lets_be_rational-generated reference.
- **9 wing-seed tests** (`tests/wing_seed.rs`): Wren G corner, mpmath-200-bit reference table across `h ∈ {3..8} × q ∈ {0.01, 0.05, 0.1, 0.2, 0.3}`, boundary finiteness at the gate edges, SIMD lane independence, end-to-end kernel σ recovery at wing corners, Chebyshev-regime non-regression, context-API routing through the wing seed, and the volfi v×Δ NaN-set regression pin.
- **18 typed-result tests** (`tests/typed_result.rs`): typed-API status discrimination across the seven `ImpliedVolStatus` arms, sigma-resolution-aware classification, internal-bracket reachability for sub-VOL_MIN and super-VOL_MAX roots, accept-floor gate, vega-conditioning gate, and adversarial-grid expected-status agreement.
- **5 backward-compat tests** (`tests/backward_compat.rs`): every `implied_vol` / `implied_vol_fast` NaN row maps to a non-Computed typed status, every Computed typed status round-trips through the legacy f64 API as a finite value, and the v1.1.0 NaN counts (CLY-3D 13, ATM-dense 288, wing v x Delta 2, Schadner cold 0) reproduce bitwise.
- **2 doc tests**: `implied_vol_fast` usage in `src/lib.rs` and the Schadner usage example in `src/schadner.rs`.

The cross-validation against py_lets_be_rational on the full 1M dataset is the standalone harness `bench/python/cross_validate.py`. The 200-bit mpmath oracle is `bench/python/oracle_mpmath.py`. Neither is part of `cargo test`.

---

## Build requirements

- **Rust**: nightly (`rustup override set nightly`). The core uses `std::simd` (`#![feature(portable_simd)]`).
- **CPU**: AVX-512. Zen 4 / Zen 5 (Ryzen 7000+ / 9000+, EPYC Genoa+) or Intel Sapphire Rapids+. The crate ships a `.cargo/config.toml` pinning `target-cpu=znver5`; on non-Zen-5 hardware override with `RUSTFLAGS="-C target-cpu=native"` or edit the file.
- **SLEEF**: a static `libsleef.a` at `~/.local/lib/`. The build script (`build.rs`) hard-links the vectorized `exp` / `log`. If the library is elsewhere, edit `build.rs` to match; without it the crate will not link.

One-liner SLEEF install (Linux):

```sh
git clone https://github.com/shibatch/sleef.git /tmp/sleef && cd /tmp/sleef
cmake -S . -B build -DCMAKE_INSTALL_PREFIX="$HOME/.local" -DSLEEF_BUILD_STATIC_LIB=TRUE
cmake --build build -j && cmake --install build
```

---

## Implementation notes

1. **Chebyshev seed.** A deg-12 bivariate Chebyshev fit in `(ln k, logit q)` lands within ~1e-7 of the answer over the well-conditioned interior. The fit's (k, T)-only sub-summation is what `OtmContext` precomputes once per node.
2. **Wing seed.** For `|k_log| ∈ [2.95, 8.0]` and IG-survival `q_surv ∈ (0, 0.30)` an analytic seed derived clean-room from Schadner's paper plus the 2-term Mills asymptotic replaces the Cheb extrapolation. See [Algorithm: wing seed](#algorithm-wing-seed).
3. **Householder-3 iteration** (order-4 convergence). Three steps from either seed lands at the f64 conditioning floor where the seed is in domain.
4. **Cancellation-free Black via `erfcx`** (Avenue-1 fused form). The IG-survival residual is one difference of scaled complementary error functions (F3 anchor + B1 cancellation) via the `ig_surv_from_uv` primitive.
5. **SLEEF `f64x8` `vexp` / `vlog`.** The remaining `exp`/`log` calls go through SLEEF's vectorized intrinsics, statically linked.
6. **Dual bailout.** Pre-classification masks route ATM-strip (`|k|/√T < 5e-3`) and deep-OTM (`c_otm/F < 3e-6`) lanes that the seed-and-polish chain cannot reach to the Jäckel rational kernel.

### Algorithm: wing seed

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

## Accuracy: known gaps

The v1.0.1 README described a "mild 2-4x gap to LBR in the deep_otm tail." That framing was too coarse. The actual picture is sharper and worth stating precisely: voltic is at LBR parity across most of deep_otm, and the headline gap lives in a thin n=230 tail.

**Empirical finding (SplitMix64 dataset, deep_otm band, n=12,730)**

On 44.6% of deep_otm rows, voltic and LBR tie within 1.5x of the f64 BS-inversion floor (i.e. neither solver has room to do better given input price round-off). On another 17.8% of rows voltic genuinely outperforms LBR by a factor of 2 or more. On the remaining 33.6% voltic genuinely underperforms LBR by 2x or more. Median (voltic_err - lbr_err) is +8.42e-15, mean is +9.66e-14, and the Pearson correlation between voltic_err and lbr_err across all 12,730 rows is 0.746.

**Stratification by |h| = |ln(F/K)| / σ_total**

| \|h\| band | n | voltic_max | LBR_max | f64_floor_max |
|---|---|---|---|---|
| [2, 3) | 6,253 | 8.96e-13 | 6.94e-13 | 8.12e-13 |
| [3, 4) | 6,242 | 1.18e-11 | 1.66e-11 | 1.05e-11 |
| [4, 10) | 230 | 2.19e-11 | 2.01e-11 | 1.15e-11 |

In the |h| ∈ [3, 4) middle of deep_otm, voltic's worst row beats LBR's worst row. The v1.0.1 headline 2.19e-11 vs 2.01e-11 lives entirely in the n=230 |h|≥4 tail, which is 1.8% of deep_otm rows.

Voltic and LBR are within roughly 2x of the physical f64 BS-inversion floor in this regime; the floor is a property of input price round-off, not of the solver. The residual gap is not closable by structural rewrite — see [FlashIV equivalence finding](#flashiv-equivalence-finding).

Other known gaps:

- **Two NaN at (v=0.01, Δ∈{0.30, 0.70}) on the wing v×Δ stress grid.** Tiny-σ near-ATM puts at the f64 BS price floor (< 1e-7). No meaningful f64 inverse exists for these inputs; they are pinned by the `volfi_wing_grid_nan_set_bounded_to_two` regression test.
- **13 NaN on CLY-3D `implied_vol_fast`.** Rows where `σ_true = VOL_MIN = 0.01` exactly, deliberately excluded by voltic's open-interval domain contract. See [Domain contract](#domain-contract).
- **288 NaN on ATM-dense `implied_vol_fast`.** Same case: rows where `σ_true = VOL_MIN = 0.01` exactly, excluded by the open-interval domain contract. The accurate solvers all share a 3.338e-3 max abs err governed by two deep-wing cases on this grid that none of them resolves.

---

## FlashIV equivalence finding

Le Floc'h and Healy's FlashIV (arxiv 2605.29102 §3.2 Eq. 4) log-price residual decomposition is algebraically equivalent at f64 precision to voltic's existing cancellation-free `b_normalized` evaluator (`src/black.rs:60-79`); both factor through the same `erfcx`-difference structure. The 2.19e-11 |h|≥4 ceiling is f64-conditioning-floor bound, not algebra bound. Full proof + side-by-side run in [`research/flashiv-equivalence-finding.md`](research/flashiv-equivalence-finding.md).

---

## Domain contract

Voltic solves for σ on the open interval `σ ∈ (VOL_MIN, VOL_MAX)` where `VOL_MIN = 0.01` and `VOL_MAX = 4.0`. Bench NaN counts on synthetic grids that land exactly on `σ = VOL_MIN` (13 on CLY-3D, 288 on ATM-dense) are honest out-of-domain rejection. The v1.2 `implied_vol_typed` API surfaces `BelowVolMin { computed }` with the sigma the iteration actually found for callers who want the boundary value.

---

## Reference comparisons (versions)

- [py_lets_be_rational 1.0.1](https://github.com/vollib/py_lets_be_rational) (Peter Jäckel, *Let's be rational*, Wilmott Magazine 2015): the canonical accuracy reference.
- [volfi 0.1.8](https://github.com/coatless-rd/volfi): vectorized C++ inverter.
- [py_vollib 1.0.7](https://github.com/vollib/py_vollib): scalar Python wrapper around LBR.
- [py_vollib_vectorized 0.1.1](https://github.com/marcdemers/py_vollib_vectorized): numpy-vectorized LBR wrapper.
- [QuantLib 1.42.1](https://www.quantlib.org/): full-suite financial library (per-option Brent inversion).

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

## Notes on the benchmark numbers

[^1]: The 200-bit mpmath oracle (`bench/python/oracle_mpmath.py`) computes Black-Scholes prices via `mp.mpf` arithmetic and `mp.erfc` for the normal CDF. Voltic's residual evaluator uses an `erfcx`-based cancellation-free formulation in f64. The two code paths are structurally independent, so the `f64_floor` column measures the f64 BS-inversion conditioning limit, not a shared-implementation correlated error.

[^2]: Benchmark numbers are AMD Ryzen 9 9950X (znver5) with `taskset -c 0` and `target-cpu=znver5`. SLEEF vexp/vlog inlining and FMA contraction differ across AVX-512 microarchitectures; Sapphire Rapids or Genoa may show different per-row ULPs at the f64 floor while keeping the same accuracy class.

---

## License

Dual-licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT), at your option.

## Disclaimer

voltic is a numerical library. It is not investment advice, it is not certified for production trading use, and accuracy guarantees are limited to the conditions specified in the benchmark methodology above. Computed implied volatilities are deterministic functions of the inputs and have no opinion about whether you should trade anything.

---

For consulting: ryan@databa.ai
