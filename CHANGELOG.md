# Changelog

All notable changes to voltic are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
follows semantic versioning.

## [1.0.2] — 2026-05-31

Additive release: a second independent benchmark (CLY-3D), an explicit
verification methodology section documenting the v1.0.1 volfi finding,
and a sharper diagnostic of the deep_otm LBR-parity story with a v1.1
roadmap. **No change to the IV kernel.**

### Added

- **CLY-3D benchmark** (`bench/cly_3d.rs` + `bench/python/cly_3d_compare.py`).
  51,321 cases on the post-LBR standard grid (Cui, Liu, Yao 2021;
  same grid the May 2026 FlashIV and ThiopheneIV preprints use as
  benchmark). Grid: `S=100`, `r=0.03`, `K ∈ linspace(105, 800, 40)`,
  `T ∈ linspace(0.01, 2, 40)`, `σ ∈ linspace(0.01, 0.99, 40)`, filtered
  to call price > 1e-20. Cell count matches FlashIV Table 3 and
  ThiopheneIV Table 3 exactly.
- **README "Verification methodology"** section. Five independent
  diligence checks behind the v1.0.1 volfi finding: hand-coded direct
  repro outside the oracle adapter, put-call parity via two independent
  paths, alternate volfi entry point (`iv_otm` vs `iv_call`),
  volfi-self-priced volfi-self-inverted, bench pattern equivalence to
  volfi's own `bench_vollib.py`. All five agree; the failure mode is
  intrinsic to volfi at the deep-wing regime.

### Refined

- **Known gaps → "Accuracy: known gap and v1.1 roadmap".** New
  empirical disclosure based on per-row analysis: on 44.6% of deep_otm
  rows voltic and LBR tie within 1.5x of the f64 BS-inversion floor;
  on 17.8% voltic beats LBR by ≥2x; on 33.6% voltic loses to LBR by
  ≥2x. The v1.0.1 headline 2.19e-11 vs 2.01e-11 lives in the |h|≥4
  tail (n=230, 1.8% of deep_otm). Root cause: an `erfcx` cancellation
  at |h|≈4.3 in `b_normalized` (`src/black.rs`). The v1.1 roadmap
  cites the FlashIV (arxiv 2605.29102, May 27 2026) log-price residual
  decomposition as the structural fix; expected to land voltic at or
  above LBR parity on the |h|≥4 tail at unchanged throughput.

### Performance

CLY-3D (51,321 cases, znver5, taskset -c 0, median of 7 passes):

| solver | ns/option | max abs err | NaN | catastrophic (≥ 1e-3) |
|---|---:|---:|---:|---:|
| voltic 1.0.2 `implied_vol_fast` | **89** | 1.539e-09 | 13 | 0 |
| voltic 1.0.2 `implied_vol_with_context_batch` (cold)† | 40 | 8.50e-01 | 0 | 9,977 |
| py_lets_be_rational (scalar) | 3,268 | 1.539e-09 | 0 | 0 |
| py_vollib_vectorized | 372 | 1.539e-09 | 0 | 0 |
| volfi 0.1.8 `iv_call` | 418 | 2.385 | 4,836 | 5,488 |

voltic ≈ LBR ≈ py_vollib_vectorized at 1.539e-9 max abs σ error (all
three sit at the f64 reverse-Black floor at deep-OTM near-expiry).
voltic is 36.7× faster than LBR scalar and 4.2× faster than
py_vollib_vectorized.

The volfi catastrophic tail reproduces on CLY-3D: 4,836 NaN + 5,488
catastrophic out of 51,321, concentrated in the K/S > 2 band (42,290
cases). Same defect class as the v1.0.1 SplitMix64 finding, observed
independently on the CLY-3D grid.

Voltic's 13 NaN are by-design rejection in the f64-double-underflow
regime where `ln(c) < -708`. The Bachelier-microscopic branch FlashIV §3
defines handles this regime; queued for v1.1.

† `implied_vol_with_context_batch` trades accuracy for raw throughput
by skipping the rational-fallback path; documented split-context API
contract. Use `implied_vol_fast` for accuracy-critical paths.

### Unchanged

- No algorithmic change to the IV kernel. `implied_vol_fast`,
  `implied_vol_with_context_batch`, `implied_vol_fully_vectorized`,
  and all internal kernels are bit-identical to v1.0.1.

## [1.0.1] — 2026-05-31

Analytic wing-seed for the deep-wing regime, plus an independent 200-bit
mpmath oracle that reframes accuracy claims around the f64 inversion floor.

### Added

- **Analytic wing-seed** for `|k_log| ∈ [2.95, 8.0]`, derived clean-room
  from Schadner's IG-quantile and the 2-term Mills asymptotic. Replaces a
  v1.0.0 path that extrapolated the Chebyshev seed beyond its fit domain
  (`SEED_K_HI = 3.0`) and produced 3.27e-1 catastrophic errors on a
  wing-saturated stress grid. Dispatch gates: `K_HI_BAILOUT = 2.95`,
  `WING_Q_MAX = 0.30`, `WING_H_MAX = 8.0`. The leading term is W0
  followed by `N_PICARD = 1` then HH3 polish; the collapse
  `e^h · Φ(−z2) ≡ φ(z1) · Q(z2)` (with `Q(z) := √(π/2) · erfcx(z/√2)
  ≈ 1/z − 1/z³`) eliminates the `exp(h)` overflow on the wing.
  See `src/schadner_fast.rs::wing_seed_simd`.
- **200-bit mpmath oracle** `bench/python/oracle_mpmath.py` measuring
  voltic, py_lets_be_rational, and volfi as distance-from-the-f64-
  inversion-floor on the SplitMix64-seeded dataset. Oracle self-
  consistency at 7.5e-56 (passes 1e-40 acceptance by 16 orders). Reveals
  that voltic and LBR sit at the floor while volfi has a silent ~0.91%
  catastrophic-precision tail in the deep wings of the moneyness-vega
  plane (3-4% rate per deep-wing band, max σ error 3.3e-1).
- **`bench/wing_grid.rs`** — volfi-style v×Δ wing-saturated stress
  harness, 360 cases after filtering. Measures throughput and the NaN set
  at the conditioning edge of the inversion problem (81 ns/option, 2
  pre-existing NaN — see Known issues).
- **`tests/wing_seed.rs`** (9 tests) — Wren G corner, mpmath-200-bit
  reference table at `h ∈ {3..8} × q ∈ {0.01..0.30}`, boundary
  finiteness, SIMD lane independence, end-to-end kernel σ recovery at
  wing corners, Chebyshev-regime non-regression, context-API routing
  through the wing seed, and the `volfi_wing_grid_nan_set_bounded_to_two`
  regression pin.
- **`scripts/wing_ref_gen.py`** — regenerates `WING_REF` in
  `tests/wing_seed.rs` from mpmath at 200 bits. Not wired to CI; present
  for reproducibility.

### Fixed

- **q-convention bug at the wing dispatch site.** The IG kernel's `q` is
  the IG CDF; the wing analytic uses IG survival. Fix:
  `q_surv = 1 − q_kernel` at the dispatch boundary. Caught during
  integration; verifier-confirmed.

### Performance

Schadner cold benchmark (1M synthetic options, znver5, taskset -c 0,
median of 7 timed passes after warmup):

- voltic `implied_vol_fast` one-shot: 73.6 ns / 3.42e-11 max abs σ
  error / 0 NaN. Unchanged from v1.0.0 outside the wing regime.

Volfi v×Δ wing-saturated stress grid (360 cases, median of 7):

- voltic `implied_vol_fast`: 81.3 ns / 8.30e-12 / 2 NaN.
- Pre-wing v1.0.0 result on the same grid: 3.27e-1 catastrophic. Net
  accuracy win of ~11 orders of magnitude.

Head-to-head against the LBR/volfi/py_vollib_vectorized reference set on
a 100k SplitMix64-seeded subsample (same dataset, znver5, taskset -c 0):

| solver | ns/option | max abs err | NaN | cat (≥ 1e-3) |
|---|---:|---:|---:|---:|
| voltic 1.0.1 `implied_vol_fast` | 73.6 | 3.42e-11 | 0 | 0 |
| py_lets_be_rational (scalar) | 3,475 | 1.54e-11 | 0 | 0 |
| py_vollib_vectorized | 406 | 2.04e-11 | 0 | 0 |
| volfi 0.1.8 `iv_call` | 350 | 3.34e-01 | 1 | 906 |

### Known issues

- voltic carries a mild 2-4× residual to py_lets_be_rational in the
  `deep_otm` band (max absolute 9.9e-12 — sub-picovol). Jäckel's
  rational guess wins by design in that corner; tightening voltic's
  deep_otm seed is v1.1 work.
- Two NaN at `(v=0.01, Δ∈{0.30, 0.70})` on the wing v×Δ stress grid:
  tiny-σ near-ATM puts at the f64 BS price floor (< 1e-7), no
  meaningful f64 inverse. Pinned by
  `tests/wing_seed.rs::volfi_wing_grid_nan_set_bounded_to_two`.

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
