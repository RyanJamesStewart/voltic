# voltic

voltic is a fast, vectorized Black-Scholes implied volatility solver. One operation, four entry points (single-shot, split-context, batched-context, fully vectorized), all `f64x8` SIMD, all converging to the f64 conditioning floor of the inversion problem with zero `NaN` across a 1,000,000-option synthetic Schadner grid.

v1.0 introduces an `OtmContext` split API for repeat-`(k, T)` workloads (vol-surface calibration, MC repricing on a fixed grid), a SIMD-batched context build, a fused fully-vectorized cold path, a Householder-3 inner iteration (order-4 convergence), a deg-12 Chebyshev seed in `(ln k, logit q)`, a fused cancellation-free Black via `erfcx` (analytic F3 + B1), SLEEF `f64x8` vectorized `exp`/`log`, and an AVX-512 znver5 build target. The net is that voltic now beats volfi v0.1.8 at every API depth on identical hardware and dataset.

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

## Benchmark

Hardware: **AMD Ryzen 9 9950X** (Zen 5, native AVX-512: `avx512{f,dq,ifma,cd,bw,vl,bf16,vbmi,vbmi2,vnni,bitalg,vpopcntdq}`), base 4.3 GHz / boost ≈ 5.75 GHz. Single-threaded, `taskset -c 0`, target-cpu pinned to `znver5` via `.cargo/config.toml`. Workload: one persisted synthetic dataset of **1,000,000 options** (`bench/data.rs`, seed `0x5EEDBEEFCAFEF00D`; see [Reproducibility](#reproducibility)). Rows: median of 5 timed passes after one discarded warmup, `cargo run --release --bin bench`.

### voltic vs volfi v0.1.8 — same hardware, same grid, every API depth

| API path | voltic ns/option | volfi ns/option | speedup | max abs σ error |
|---|---:|---:|---:|---:|
| split context, repeat workload (SIMD batched) — `implied_vol_with_context_batch` | **34.0** | 46.6 (`implied_variance_otm`) | 1.37× | 3.42e-11 |
| cold vector-context (SIMD) — `implied_vol_vectorized_with_contexts` | **40.7** | — | — | 3.42e-11 |
| cold fully vectorized (SIMD prelude + solve) — `implied_vol_fully_vectorized` | **45.5** | — | — | 3.42e-11 |
| one-shot public API (direct kernel) — `implied_vol_fast` | **73.8** | 358.2 (`implied_volatility_call`) | 4.85× | 3.42e-11 |

volfi additionally produced **7,488 outliers ≥ 1e-2** max abs σ error on the deep-OTM corner of the same 1M grid; voltic returned **0 NaN and 0 outliers**. voltic wins all four API depths on speed and coverage. The accuracy floor (3.42e-11) is the dataset's intrinsic f64 Black-Scholes inversion floor — both solvers hit it on this grid.

### Accuracy + coverage, stratified by moneyness

Max `|solved σ − σ_true|` for the voltic kernels over the 1,000,000-option dataset (`σ_true` is the volatility that produced each price; the dataset is consistent by construction):

| band | definition | direct (`implied_vol`) | explicit (`implied_vol_explicit`) | rational (`implied_vol_rational`) | fast (`implied_vol_fast`) | options solved |
|---|---|---:|---:|---:|---:|---:|
| deep OTM | S/K < 0.7 | 1.79e-11 | 2.99e-11 | 3.42e-11 | 3.42e-11 | 225,065 / 225,065 |
| near ATM | 0.95 ≤ S/K ≤ 1.05 | 3.63e-12 | 3.22e-12 | 4.57e-12 | 1.17e-11 | 119,384 / 119,384 |
| deep ITM | S/K > 1.3 | 1.00e-11 | 1.02e-11 | 1.25e-11 | 1.25e-11 | 198,152 / 198,152 |

Zero `NaN` across all kernels on all bands. The worst case (~1e-11 deep OTM, ~5e-12 ATM strip) is the dataset's intrinsic f64 BS-inversion floor; voltic does not, and cannot, do better. Nothing here is a claim of sub-conditioning-floor precision.

---

## Public API

Four entry points cover the depth-of-use axis. Pick the one that matches the workload shape:

### `implied_vol_fast` — one-shot, public, direct kernel

The default; takes the same six slices as the v1 `implied_vol` and dispatches through the v2 Chebyshev seed + Householder-3 inner iteration with a dual bailout to the rational kernel on the deep-OTM / ATM corners (zero `NaN`).

```rust
use voltic::{implied_vol_fast, OptionKind};
let iv = implied_vol_fast(&spot, &strike, &tte, &rate, &price, &kind);
```

### `OtmContext::new` + `implied_vol_with_context` — split API for repeat workloads

The `(k, T)`-only prelude is built once and reused across many price evaluations on the same `(strike, expiry)` node — vol-surface calibration, MC repricing on a fixed grid, scenario sweeps.

```rust
use voltic::{OtmContext, implied_vol_with_context, canonical_c_from_price};

let ctx = OtmContext::new(k_log, t);      // build the (k, T) prelude once
for &price in prices {
    let c = canonical_c_from_price(&ctx, spot, price, is_call);
    let iv = implied_vol_with_context(&ctx, c);
    // ...
}
```

### `implied_vol_with_context_batch` — one context × many prices, SIMD per 8

The repeat-workload shape, vectorized: one `OtmContext` plus a price slice, solved 8-wide. Fastest shape voltic offers (34 ns/option).

```rust
use voltic::{OtmContext, implied_vol_with_context_batch};

let ctx = OtmContext::new(k_log, t);
let ivs = implied_vol_with_context_batch(&ctx, &prices);   // Vec<f64>, len == prices.len()
```

### `implied_vol_fully_vectorized` — cold portfolio path

Every option has a unique `(k, T, c)`; the IG prelude itself runs at SIMD throughput, fused into the solve. Recommended for portfolios (45.5 ns/option cold).

```rust
use voltic::implied_vol_fully_vectorized;
let ivs = implied_vol_fully_vectorized(&k, &t, &c);   // canonical OTM premium ratios
```

The vector-of-contexts shape (`implied_vol_vectorized_with_contexts`) is also exposed for callers that have already materialized an `&[OtmContext]`.

---

## Build requirements

- **Rust**: nightly (`rustup override set nightly`). The core uses `std::simd` (`#![feature(portable_simd)]`).
- **CPU**: AVX-512. Zen 4 / Zen 5 (Ryzen 7000+ / 9000+, EPYC Genoa+) or Intel Sapphire Rapids+. The crate ships a `.cargo/config.toml` that pins `target-cpu=znver5`; on non-Zen-5 hardware override with `RUSTFLAGS="-C target-cpu=native"` or edit the file.
- **SLEEF**: a static `libsleef.a` at `~/.local/lib/`. The build script (`build.rs`) hard-links the vectorized `exp` / `log`. If the library is elsewhere, edit `build.rs` to match — without it the crate will not link.

One-liner SLEEF install (Linux):

```sh
git clone https://github.com/shibatch/sleef.git /tmp/sleef && cd /tmp/sleef
cmake -S . -B build -DCMAKE_INSTALL_PREFIX="$HOME/.local" -DSLEEF_BUILD_STATIC_LIB=TRUE
cmake --build build -j && cmake --install build
```

Then:

```sh
rustup override set nightly      # in this directory
cargo +nightly build --release --bin bench
cargo +nightly test --release    # 65/65 should pass
taskset -c 0 ./target/release/bench --n 1000000
```

---

## Reference comparisons

The Python comparison harness (`bench/python/bench.py`) runs the reference implementations on the same dataset the Rust harness writes (`--csv`). voltic's rows come from `cargo run --release --bin bench`; the others come from the Python harness, single-threaded, `taskset -c 0`, median of N timed passes after one discarded warmup. The Python venv is pinned in `bench/python/requirements.txt`.

Workload sizes differ by what each tool can complete in a reasonable wall-clock: voltic and volfi numbers are on the full 1,000,000-option dataset; the Python and QuantLib rows are on the first 100,000 options of the same dataset (a held-out slice) because per-option scalar Python and QuantLib are 50–700× slower than voltic and would not finish on 1M within the bench window. The dataset is consistent by construction in both cases (the 100k slice has the same distribution by parity of indices).

| Solver | Throughput | Max abs σ error | Unsolved / outliers |
|---|---:|---:|---:|
| voltic 1.0 `implied_vol_with_context_batch` (1M) | **34.0 ns/option** | 3.42e-11 | 0 |
| voltic 1.0 `implied_vol_fully_vectorized` (1M) | **45.3 ns/option** | 3.42e-11 | 0 |
| voltic 1.0 `implied_vol_fast` one-shot (1M) | **74.3 ns/option** | 3.42e-11 | 0 |
| volfi v0.1.8 `implied_variance_otm` (1M) | 46.6 ns/option | 3.42e-11 | — |
| volfi v0.1.8 `implied_volatility_call` (1M) | 358.2 ns/option | ≥1e-2 on 7,488 | 7,488 deep-OTM outliers |
| py_vollib_vectorized 0.1.1 (100k) | 408.1 ns/option | 2.04e-11 | 0 |
| py_vollib 1.0.7 (scalar, 100k) | 4,498.5 ns/option | 2.04e-11 | 0 |
| QuantLib 1.42.1 (Python binding, 100k) | 31,515.5 ns/option | 3.73e-01 | 2,164 unsolved |

Notes on the QuantLib row. The harness drives `EuropeanOption::impliedVolatility(price, process, accuracy=1e-10, maxIterations=200, minVol=1e-4, maxVol=5.0)` — the standard per-option binding. The 2,164 unsolved entries are options where the Brent solver hit a bracket failure on the deep-OTM / short-expiry corner; the 3.73e-01 max-error row is the residual on the rows it *did* return, several of which converged to a non-root because the bracket excluded the true σ. This is the per-option binding's behaviour out of the box, not an indictment of QuantLib's internal Black machinery; a hand-tuned QuantLib bench would route through `blackFormulaImpliedStdDevChambers` and likely close most of the accuracy gap. We report the standard binding because that is what a typical caller writes.

py_vollib and py_vollib_vectorized both wrap Peter Jäckel's `LetsBeRational` C++ — the same reference voltic's rational kernel is cross-validated against — so their accuracy floor (2.04e-11) is the same f64 conditioning floor voltic hits (3.42e-11 in the rational kernel, 1.79e-11 in the direct kernel). The throughput delta is the gap a vectorized Rust SIMD kernel opens against a Cython-wrapped scalar C++ inverter.

### Accuracy cross-validation: voltic vs `py_lets_be_rational` (the Jäckel oracle)

`bench/python/cross_validate.py` runs voltic's `implied_vol_rational` kernel on the dataset, then runs Jäckel's `py_lets_be_rational` (the canonical reference impl) on the same inputs as a black-box oracle, and reports the per-option |voltic − py_lbr| disagreement. Run on the first 100,000-option slice:

```
100,000 options, both solved 100,000 (0 voltic-only, 0 py_lbr-only, 0 neither):
  band        n        max         p99         p90         median
  all         100,000  1.58e-11    1.01e-12    1.54e-14    3.33e-16
  deep_otm     22,502  1.58e-11    1.87e-12    5.47e-14    2.78e-16
  near_atm     11,914  9.55e-13    2.25e-14    2.67e-15    4.44e-16
  deep_itm     19,800  8.62e-12    1.14e-12    3.63e-14    2.78e-16
```

Median disagreement is at the f64 round-off floor (~3e-16); p99 is in the low picoseconds-of-vol range; max is the dataset's intrinsic f64 BS-inversion conditioning floor. voltic's rational kernel and Jäckel's reference disagree by less than the inversion problem's own conditioning permits.

### Coverage on the 1,000,000-option Schadner grid

| Solver | NaN | Outliers ≥ 1e-2 max σ err |
|---|---:|---:|
| voltic 1.0 (all four kernels) | 0 | 0 |
| volfi v0.1.8 `implied_variance_otm` | — | 0 |
| volfi v0.1.8 `implied_volatility_call` | — | 7,488 (deep-OTM corner) |
| py_vollib (100k slice) | 0 | 0 |
| py_vollib_vectorized (100k slice) | 0 | 0 |
| QuantLib (100k slice, default per-option binding) | 2,164 | many — see note above |

---

## Algorithm

The fast kernel is the order-of-the-day winner on this grid:

1. **Chebyshev seed.** A deg-12 bivariate Chebyshev fit in `(ln k, logit q)` lands within ~1e-7 of the answer over the well-conditioned interior. The fit's (k, T)-only sub-summation is what `OtmContext` precomputes once per node.
2. **Householder-3 iteration** (order-4 convergence). Three steps from the Chebyshev seed lands at the f64 conditioning floor everywhere the seed is in domain. Replaces v1's four-Halley step count for the same accuracy at lower latency.
3. **Cancellation-free Black via `erfcx`** (the Avenue-1 fused form). The IG-survival residual is computed analytically as a single difference of scaled complementary error functions (F3 anchor + B1 cancellation) via the `ig_surv_from_uv` primitive — one fewer `exp` than the naïve form, and no centre-cancellation loss.
4. **L(x) bug fix.** The Halley/Householder derivative chain had a stale `L(x)` term; corrected to match the analytic Black price derivative.
5. **SLEEF `f64x8` `vexp` / `vlog`.** The remaining `exp` / `log` calls go through SLEEF's vectorized intrinsics instead of scalar libc, statically linked.
6. **Dual bailout.** Two pre-classification masks route the small corner that the Chebyshev seed cannot reach to the Jäckel rational kernel (cf. [Jäckel 2015](https://www.jaeckel.org/LetsBeRational.pdf)): `|k|/√T < 5e-3` (ATM-ceiling of the seed's scaled-probit arm) and `c_otm / F < 3e-6` (deep-OTM where Halley/Householder cannot drive σ-error below 1e-7 within three steps).

The fallback rational kernel is voltic's clean-room SIMD implementation of Peter Jäckel, *Let's be rational* (Wilmott Magazine, January 2015) — the canonical machine-precision IV inverter. The fast kernel implements Schadner, *"An Explicit Solution to Black-Scholes Implied Volatility"* (arXiv:2604.24480, 2026) reformulated through the OtmContext split. See `src/schadner_fast.rs`, `src/otm_context.rs`, `src/jackel.rs`, `src/black.rs`.

---

## Tests and quality

`cargo +nightly test --release` runs **65 tests, all passing**, enumerated from `cargo +nightly test --release -- --list`:

- **52 lib unit tests** (`src/lib.rs`), grouped by module:
  - `black::tests` (12) — the cancellation-free Black price: `b_at_forward_matches_erf_form`, `b_bounded_by_b_max`, `b_double_prime_matches_fd_of_bprime`, `b_double_prime_zero_at_sigma_c`, `b_large_sigma_matches_asymptotic_3_4`, `b_prime_matches_finite_difference`, `b_small_sigma_matches_asymptotic_3_3`, `b_triple_prime_matches_fd_of_bdoubleprime`, `erfcx_form_agrees_with_naive_in_centre`, `erfcx_form_extends_to_deep_otm`, `scalar_simd_consistency`, `sigma_c_is_inflection_point`. Covers the analytic price, its first three derivatives (via finite-difference cross-check), the inflection-point invariant, the small-σ / large-σ asymptotic expansions, the `erfcx` reformulation against the naïve form across the (k, σ) plane, and SIMD-vs-scalar bitwise consistency.
  - `jackel::tests` (13) — the rational kernel: `canonicalize_handles_all_four_quadrants`, `classify_region_assigns_correct_index`, `dg_r_helpers_compile`, `dg_rational_cubic_matches_endpoints`, `householder3_reduces_to_newton_when_higher_derivs_zero`, `initial_guess_centre_close_to_truth`, `initial_guess_centre_left_matches_anchors`, `initial_guess_centre_right_matches_anchors`, `region_boundaries_are_ordered`, `region_boundaries_sigma_c_matches_black`, `solve_middle_converges_to_truth`, `solve_rational_end_to_end`, `unified_initial_guess_across_all_regions`. Covers Jäckel's quadrant canonicalization, region classification, rational-cubic interpolation, the Householder-3 step's reduction to Newton in the degenerate case, the initial-guess anchors per region, and the end-to-end rational solve.
  - `norm::tests` (11) — the cumulative-normal kernels: `as_matches_reference_coarsely`, `cody_matches_reference`, `erfcx_at_zero_is_one`, `erfcx_large_argument_asymptotic`, `erfcx_matches_identity_in_centre`, `hart_matches_reference`, `pdf_is_derivative_of_cdf`, `phi_inv_is_inverse_of_phi`, `phi_inv_recovers_reference`, `symmetry`, `west_matches_reference`. Covers Abramowitz-Stegun 26.2.17, Cody 1969, Hart 5666, and West 2009 against a 41-point high-precision reference, plus `erfcx`'s asymptotic and identity properties, Φ symmetry, and Φ⁻¹ inversion.
  - `schadner::tests` (4) — the explicit IG inverter: `batch_equals_singletons`, `edge_cases_return_nan`, `matches_direct_solver_on_grid`, `round_trip_atm_call`. SIMD-vs-scalar consistency, NaN on degenerate inputs, agreement with the direct solver on the synthetic grid, ATM round-trip.
  - `schadner_fast::avenue1_property_tests` (2) — the Avenue-1 fused-`erfcx` form: `avenue1_finite_at_atm`, `avenue1_matches_naive_away_from_atm`. Finiteness through the ATM strip and agreement with the naïve form away from the centre.
  - `tests` (10) — top-level integration: `batch_with_padding_tail`, `deep_otm_short_expiry_is_handled_or_nan`, `edge_cases_return_nan_not_garbage`, `implied_vol_rational_handles_grid`, `implied_vol_rational_handles_x_zero_atm`, `implied_vol_rational_recovers_known_vol_atm`, `put_call_parity_on_solved_vols`, `round_trip_atm_call`, `round_trip_grid`, `zero_rate_and_high_vol`. Covers SIMD tail padding, deep-OTM short-expiry behaviour, edge-case NaN policy, the rational kernel on a synthetic grid, put-call parity on solved vols, and named extreme regimes.
- **11 proptest property tests** (`tests/properties.rs`): `batch_equals_singletons`, `explicit_agrees_with_direct`, `put_call_parity`, `rational_agrees_with_direct`, `rational_batch_equals_singletons`, `rational_put_call_parity`, `rational_round_trip_high_precision`, `reference_table`, `reference_table_explicit`, `reference_table_rational`, `round_trip_recovers_sigma`. Randomized round-trip recovery of σ on each kernel, put-call parity on solved vols, batch-vs-singleton SIMD agreement, and the `reference_table` properties that pin direct / explicit / rational against a `py_lets_be_rational`-generated reference table.
- **2 doc tests**: the `implied_vol_fast` usage example in `src/lib.rs` and the Schadner usage example in `src/schadner.rs`.

The cross-validation against `py_lets_be_rational` on the full 1M-option dataset is the standalone harness `bench/python/cross_validate.py`, not part of `cargo test`. See the [Reference comparisons](#reference-comparisons) section for the numbers.

`cargo +nightly run --release --bin bench -- --n 1000000` additionally runs the full benchmark: throughput per ns/option on every voltic API depth, accuracy stratified by moneyness band (deep OTM / near ATM / deep ITM), the A5 split-context bench, the A5.1 SIMD-prelude + fused vectorized cold path, and the cumulative-normal kernel frontier. Pass `--csv data.csv` to dump the dataset for the Python comparison harness (`bench/python/bench.py`).

---

## Reproducibility

```sh
RUSTFLAGS="-C target-cpu=znver5" taskset -c 0 \
    cargo +nightly run --release --bin bench -- --n 1000000

# The criterion harness for the rigorous headline ns/option:
RUSTFLAGS="-C target-cpu=znver5" taskset -c 0 cargo +nightly bench
```

**Dataset.** Synthetic options drawn (independently per option) from: spot ~ Uniform[$50, $200], strike ~ Uniform[$40, $240], time-to-expiry ~ exp(Uniform[ln(1/365), ln(2)]) (log-uniform, one day to two years), risk-free rate ~ Uniform[0, 6%], volatility ~ Uniform[5%, 80%], call/put alternating by index parity. Each price is computed by Black-Scholes from those parameters and the option is kept only if its premium exceeds intrinsic by more than `1e-6 · spot`. RNG is SplitMix64 seeded with `0x5EEDBEEFCAFEF00D` (`bench/data.rs`). Byte-identical across re-runs.

---

## Limitations

voltic solves one numerical kernel under one model; the edges are deliberate.

- **European Black-Scholes only.** No American / early-exercise, no dividends (continuous or discrete), single flat risk-free rate (no term structure).
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

For consulting, custom work, or commercial integration: ryan@databa.ai
