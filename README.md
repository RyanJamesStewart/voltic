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

Voltic carries a mild 2-4× residual vs LBR in the deep_otm tail (sub-picovol absolute — max 9.9e-12) — see [Accuracy: known gap and v1.1 roadmap](#accuracy-known-gap-and-v11-roadmap).

---

## Verification methodology

Claims of the form "library X has a numerical bug" are usually wrong: nine times out of ten the user fed X the wrong inputs through a misconfigured adapter. Before publishing the volfi finding we ran five independent checks designed to catch exactly that class of mistake. All five agreed.

### 1. Hand-coded direct repro outside the oracle adapter

We pulled the top 10 worst volfi rows from the oracle output and called `volfi.iv_call(F, K, disc, T, c)` directly in a standalone script, with no adapter layer of any kind. All 10 rows reproduced ULP-identical to the oracle run. Maximum disagreement across the 10 rows was 0.0 floats.

### 2. Put-call parity via two independent paths

For each put-side row we computed σ two ways. Path A: our oracle's parity transform `c = p + S - K*disc`, then `volfi.iv_call(c)`. Path B: volfi's own `bs_call` from their published `bench_vollib.py` evaluated at `sigma_true` to produce a call price, then `volfi.iv_call` on that price. Path A and Path B agreed to better than 1e-13 absolute σ on every row tested, and both produced the same catastrophic answer. The error does not depend on which side of parity the row originated on.

### 3. Alternate volfi entry point

We re-ran the worst rows through `volfi.iv_otm` instead of `volfi.iv_call`. The error reproduced on both paths. On Row 10 (true σ ≈ 0.786), `iv_otm` returned 0.4903 and `iv_call` returned 0.4961; both are off by roughly 0.29-0.30 absolute. The failure is not confined to a single volfi entry point.

### 4. volfi-self-priced, volfi-self-inverted

To remove our BS-pricer from the loop entirely, we used volfi's own `bs_call` formula to construct the call price `c` at `sigma_true`, then fed that price back into `volfi.iv_call`. The returned σ was the same wrong value. The failure mode is independent of which Black-Scholes formula produced the input price.

### 5. Bench pattern equivalence to volfi's own `bench_vollib.py`

The call shape used in our benchmark, `volfi.iv_call(F, K, disc, T, c)`, is the same pattern volfi's own published benchmark `bench_vollib.py` uses. The prices we feed volfi agree with what volfi's own `bs_call` produces to within 3.55e-15 absolute, which is the f64 ULP at the relevant magnitudes.

The failure mode is intrinsic to volfi at this regime; 913 of 99,996 rows show err > 1e-3, with floor-ratio (volfi_err / f64_BS_inversion_floor) median 1.34e+11, max 3.04e+15.

---

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

### Benchmarks: CLY-3D (Cui-Liu-Yao 2021 standard grid)

The CLY-3D grid (51,321 deep-OTM-weighted points; defined in Cui, Liu, Yao 2021 and used as the standard benchmark in the May 2026 FlashIV and ThiopheneIV preprints) provides a second independent dataset alongside SplitMix64. Grid: `S=100`, `r=0.03`, `K ∈ linspace(105, 800, 40)`, `T ∈ linspace(0.01, 2, 40)`, `σ ∈ linspace(0.01, 0.99, 40)`, filtered to call price > 1e-20 (matches FlashIV Table 3 and ThiopheneIV Table 3 cell count exactly).

| solver | impl | ns/option | wall (s) | max abs err | NaN | catastrophic (≥ 1e-3) |
|---|---|---:|---:|---:|---:|---:|
| **voltic 1.0.2 `implied_vol_fast`** | Rust f64x8 SIMD (znver5) | **89** | 0.0046 | 1.539e-09 | 13 | 0 |
| voltic 1.0.2 `implied_vol_with_context_batch` (cold)† | Rust f64x8 SIMD | 40 | 0.0021 | 8.50e-01 | 0 | 9,977 |
| py_lets_be_rational (scalar) | Python+C++ scalar loop | 3,268 | 0.168 | 1.539e-09 | 0 | 0 |
| py_vollib_vectorized | Python+C++ numpy-vectorized | 372 | 0.019 | 1.539e-09 | 0 | 0 |
| volfi 0.1.8 `iv_call` | C++ binding (vectorized) | 418 | 0.021 | 2.385 | 4,836 | 5,488 |

voltic ≈ LBR ≈ py_vollib_vectorized at 1.539e-9 max abs σ error (all three sit at the f64 reverse-Black floor at the deep-OTM near-expiry corner) while voltic is **36.7× faster than LBR scalar** and **4.2× faster than py_vollib_vectorized**.

The volfi catastrophic-tail reproduces on CLY-3D: 4,836 NaN + 5,488 catastrophic out of 51,321 cases, concentrated in the K/S > 2 band (42,290 cases). This is the same defect class as the v1.0.1 SplitMix64 finding, observed independently on the CLY-3D grid.

Voltic's 13 NaN are by-design rejection in the f64-double-underflow regime where `ln(c) < -708`. The Bachelier-microscopic branch FlashIV §3 defines handles this regime; queued for v1.1. See [Accuracy: known gap and v1.1 roadmap](#accuracy-known-gap-and-v11-roadmap).

† `implied_vol_with_context_batch` trades accuracy for raw throughput by skipping the rational-fallback path; this is the documented split-context API contract. Use `implied_vol_fast` for accuracy-critical paths.

---

## How to reproduce

```sh
# Build (nightly + SLEEF static lib at ~/.local/lib/libsleef.a)
cargo +nightly build --release

# Schadner cold: 1M synthetic options, znver5, taskset -c 0
taskset -c 0 ./target/release/bench --n 1000000

# volfi v×Δ wing-saturated stress grid (360 cases)
taskset -c 0 ./target/release/wing_grid

# CLY-3D standard grid (Cui-Liu-Yao 2021; 51,321 cases)
cargo +nightly run --release --bin cly_3d
python3 bench/python/cly_3d_compare.py

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

## Accuracy: known gap and v1.1 roadmap

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

**Root cause**

At |h| ≈ 4.3, the cancellation-free residual evaluator `b_normalized` in `src/black.rs` computes a difference of two `erfcx` values that agree to roughly 1.5 significant digits, then multiplies the difference by ½·exp(-9). The f64 rounding of that inner subtraction is the catastrophic step: the surviving bits of the residual are dominated by round-off rather than by the analytic signal. LBR sidesteps this by switching to an explicit 17-term asymptotic series once |h| crosses its threshold, so its high-|h| path never executes the cancellation.

**v1.1 roadmap**

Le Floc'h and Healy's "FlashIV" preprint (arxiv 2605.29102, May 27 2026) publishes a log-price residual decomposition `c = ½ [erfcx((-h - t)/√2) − e^(−x) · erfcx((-h + t)/√2)]` that structurally avoids the cancellation: the two `erfcx` terms enter as a scaled combination rather than as a raw subtraction at near-equal magnitudes. Voltic v1.1 will implement this evaluator in SIMD f64x8 alongside the existing path, gated on |h|. Expected outcome: voltic matches or beats LBR on the |h|≥4 tail at unchanged throughput. The residual rewrite costs under 1 ns per option amortized across the lane, well below the householder iteration cost that dominates the inner loop.

Both voltic and LBR are within roughly 2x of the physical f64 BS-inversion floor in this regime. The engineering target is at-LBR-parity, not the floor itself; the floor is a property of input price round-off, not of the solver.

Other known gaps:

- **Two NaN at (v=0.01, Δ∈{0.30, 0.70}) on the wing v×Δ stress grid.** Tiny-σ near-ATM puts at the f64 BS price floor (< 1e-7). No meaningful f64 inverse exists for these inputs; they're pinned by the `volfi_wing_grid_nan_set_bounded_to_two` regression test.
- **13 NaN on CLY-3D `implied_vol_fast`.** Deep-OTM near-expiry double-underflow regime where `ln(c) < -708`. By-design rejection; the Bachelier-microscopic branch FlashIV §3 defines handles this regime, queued for v1.1.

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
