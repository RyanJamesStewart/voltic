# Changelog

All notable changes to voltic are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
follows semantic versioning.

## [1.1.0] - 2026-05-31

Consolidated release superseding v1.0.1 and v1.0.2 (both same-day patches).
The minor version reflects the analytic wing-seed kernel added in v1.0.1.

Added

- Analytic wing-seed kernel (src/schadner_fast.rs::wing_seed_simd) for |k_log| in [2.95, 8.0], clean-room derived from Schadner 2024 plus Mills asymptotic. Replaces a path that extrapolated the Chebyshev seed beyond SEED_K_HI=3.0 and produced 3.27e-1 catastrophic errors on the wing v×Δ stress grid.
- 200-bit mpmath oracle (bench/python/oracle_mpmath.py) measuring voltic, py_lets_be_rational, and volfi as distance from the f64 inversion floor on the SplitMix64-seeded dataset.
- CLY-3D benchmark (bench/cly_3d.rs and bench/python/cly_3d_compare.py), 51,321 cases, the post-LBR standard grid (Cui-Liu-Yao 2021).
- ATM-dense benchmark (bench/atm_dense.rs and bench/python/atm_dense_compare.py), 48,831 cases, near-ATM K/S in [0.85, 1.15] coverage to compensate for SplitMix64 wing-heavy sampling.
- README Verification Methodology with six independent diligence checks behind the volfi-tail finding, including the otm_context disconfirming test (volfi binding source confirms iv_call, ctx.iv, iv_otm all wrap the same volfi::implied_volatility_otm core).
- README FlashIV equivalence finding (research/flashiv-equivalence-finding.md): we implemented FlashIV's log-price residual decomposition (Le Floc'h and Healy, arxiv 2605.29102 Section 3.2 Equation 4) and found it algebraically equivalent at f64 precision to voltic's existing cancellation-free b_normalized evaluator.
- README Domain contract: voltic solves σ on the open interval (VOL_MIN, VOL_MAX); NaN counts on CLY-3D (13) and ATM-dense (288) represent honest out-of-domain rejection at σ_true = VOL_MIN exactly.
- README Known Gaps stratified table: voltic ties LBR within 1.5× of floor on 44.6% of SplitMix64 deep_otm rows, beats LBR by 2× on 17.8%, loses by 2× on 33.6%.

Fixed

- q-convention bug at the wing dispatch site (q_surv = 1 - q_kernel; the kernel q is IG CDF while the wing math uses IG survival).

Performance

- voltic implied_vol_fast: 73 ns Schadner cold, 89 ns CLY-3D, 68 ns ATM-dense; at-LBR-parity on max abs err across all grids; 36-78× faster than LBR scalar.

Release management

- This release consolidates and supersedes v1.0.1 and v1.0.2 (both same-day patches). The v1.0.1 and v1.0.2 GitHub release entries have been removed in favor of this single consolidated release. Tags v1.0.1 and v1.0.2 remain in git for historical commit access.

## [1.0.0] - 2026-05-31

A new public API surface for repeat-context workloads and a perf overhaul
of the inner kernel. Every API depth (public one-shot, split context,
batched context, fully vectorized cold path) now beats volfi v0.1.8 on
identical hardware and dataset (head-to-head numbers in the README), with
zero `NaN` and zero outliers across the canonical 1,000,000-option
synthetic Schadner grid.

### Added

- `OtmContext` public struct (`src/otm_context.rs`): the `(k, T)`-only
  prelude, built once and reused across many price evaluations on the
  same node. 22 × `f64`, three cache lines, trivially `Copy`.
- `implied_vol_with_context(ctx, c) -> f64`: scalar split-API entry point,
  shape-compatible with volfi's `otm_context(h)`.
- `implied_vol_with_context_batch(ctx, &[c]) -> Vec<f64>`: one context ×
  many prices, SIMD batched per 8. Fastest shape voltic offers
  (34.0 ns/option median, 1M cold grid, Zen 5 AVX-512).
- `implied_vol_vectorized_with_contexts(&[ctx], &[c]) -> Vec<f64>`: cold
  vector-context SIMD path (40.7 ns/option).
- `implied_vol_fully_vectorized(&[k], &[t], &[c]) -> Vec<f64>`: fully
  fused cold path; the IG prelude itself runs at SIMD throughput
  (45.5 ns/option). Recommended for portfolios.
- `implied_vol_fast(...)`: public one-shot entry point with dual bailout
  to the rational kernel; 73.8 ns/option, zero `NaN`.
- `canonical_c_from_price`, `pack_contexts`, `broadcast_context`,
  `pack_contexts_from_kt`: support primitives for the context API.
- `src/schadner_fast.rs`, `src/schadner_fast_seed.rs`: the deg-12
  bivariate Chebyshev seed in `(ln k, logit q)` plus the
  Householder-3 inner iteration.
- `.cargo/config.toml`: pins `target-cpu=znver5` for AVX-512 codegen.
- `build.rs`: static link of SLEEF (`~/.local/lib/libsleef.a`) for the
  vectorized `f64x8` `vexp` / `vlog`.

### Changed

- Inner iteration switched from four Halley steps to three Householder-3
  steps (`HOUSEHOLDER3_STEPS = 3`). Order-4 convergence; lower latency
  at the same accuracy floor.
- Cancellation-free Black via `erfcx` re-fused as a single
  IG-survival primitive (`ig_surv_from_uv`) combining the F3 analytic
  anchor with the B1 cancellation; saves one `exp` / Φ-evaluation per
  step relative to the v1 form.
- L(x) derivative term in the Halley/Householder step corrected (stale
  term against the analytic Black price derivative).
- Cancellation-free IG CDF (Avenue 1) via `erfcx` along the full
  Householder chain.
- Default headline kernel for `implied_vol_fast` re-seeded with the
  deg-12 Chebyshev fit instead of v1's Corrado-Miller.
- Dual bailout pre-classification: `|k|/√T < 5e-3` (ATM ceiling of the
  scaled-probit arm) and `c_otm/F < 3e-6` (deep-OTM corner) route to the
  Jäckel rational kernel. Zero `NaN` on the 1M grid.

### Performance

Head-to-head against volfi v0.1.8 on identical hardware (AMD Ryzen 9
9950X, single AVX-512 core, `taskset -c 0`) and dataset (1M cold
synthetic Schadner grid, median of 5 timed passes after warmup):

| API path | voltic | volfi | speedup |
|---|---:|---:|---:|
| split-context batched (repeat workload) | **34.0 ns** | 46.6 ns | 1.37× |
| public one-shot | **73.8 ns** | 358.2 ns | 4.85× |

volfi additionally produced 7,488 outliers ≥ 1e-2 abs σ error on the
deep-OTM corner of the same grid; voltic returned 0 NaN and 0 outliers.
Accuracy floor (3.42e-11 max abs σ error) is the dataset's intrinsic
f64 BS-inversion floor; both solvers hit it.

### Build requirements

- Nightly Rust (`std::simd`, `#![feature(portable_simd)]`).
- AVX-512 hardware (Zen 4+ / Sapphire Rapids+).
- Static SLEEF at `~/.local/lib/libsleef.a`. The crate will not link
  without it. See README for the one-liner install.

### Unchanged

- `implied_vol`, `implied_vol_explicit`, `implied_vol_rational`: same
  signatures, same NaN contract.
- `bs_price`, `OptionKind`, the Python binding under the `python`
  feature, the criterion bench harness in `benches/iv.rs`, and the
  reference-table tests in `tests/properties.rs`.

## [0.1.0] - 2026-05

Initial release: `implied_vol` (direct Newton, SIMD f64×8, Corrado-Miller
seed), `implied_vol_explicit` (Schadner inverse-Gaussian SIMD port),
`bs_price`, the synthetic benchmark dataset, the four cumulative-normal
kernels (Abramowitz-Stegun / Hart 5666 / West 2009 / Cody 1969), the
Acklam probit, and the Python binding under the `python` feature.
