# Changelog

All notable changes to voltic are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
follows semantic versioning.

## [1.0.0] — 2026-05-31

A new public API surface for repeat-context workloads and a perf overhaul
of the inner kernel. Every API depth — public one-shot, split context,
batched context, fully vectorized cold path — now beats volfi v0.1.8 on
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

## [0.1.0] — 2026-05

Initial release: `implied_vol` (direct Newton, SIMD f64×8, Corrado-Miller
seed), `implied_vol_explicit` (Schadner inverse-Gaussian SIMD port),
`bs_price`, the synthetic benchmark dataset, the four cumulative-normal
kernels (Abramowitz-Stegun / Hart 5666 / West 2009 / Cody 1969), the
Acklam probit, and the Python binding under the `python` feature.
