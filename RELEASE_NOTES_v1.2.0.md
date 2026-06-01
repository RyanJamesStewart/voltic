# voltic v1.2.0

A typed-result API on the v1.1.0 fast kernel.

The legacy f64-returning entry points (`implied_vol`, `implied_vol_fast`,
`implied_vol_rational`, `implied_vol_explicit`,
`implied_vol_with_context_batch`, `implied_vol_fully_vectorized`) are
byte-identical with v1.1.0. NaN counts on canonical grids are unchanged
(CLY-3D 13 NaN, ATM-dense 288 NaN, wing v x Delta 2 NaN, Schadner cold
0 NaN), and throughput on the fast hot path is unchanged (Schadner cold
73.2 ns, wing v x Delta 81.6 ns). Callers on the legacy surface see no
behavior change.

The new `implied_vol_typed` / `implied_vol_typed_batch` surface returns
an `ImpliedVolResult { value, status }`. The status arm distinguishes
seven outcomes the bare-NaN API conflates: `Computed`, `BelowVolMin
{ computed }`, `AboveVolMax { computed }`, `BelowIntrinsic`,
`AboveMaximum`, `NonFinite`, `FailedToConverge`. `BelowVolMin` and
`AboveVolMax` carry the sigma the iteration actually found, so a caller
who wants to accept sub-VOL_MIN or super-VOL_MAX vols can.

## Added

- Typed result API (`implied_vol_typed`, `implied_vol_typed_batch`).
- Householder-3 solver on the typed path (FlashIV eq. 6, AQFED.jl
  parity).
- Three-term price-residual floor: inverse-mapping, price-scale, and
  Hart-Phi floor.
- Sigma-resolution-aware classification gate. Rows landing within
  their own sigma-resolution of VOL_MIN or VOL_MAX classify by
  identifiability, not by hard-edge comparison.
- Wide internal iteration bracket `[1e-8, 50.0]` so a true root below
  VOL_MIN (or above VOL_MAX) is found, not pinned.
- Adversarial bench grid (`bench/adversarial.rs`), about 3000 rows
  tagged by regime, each with an expected typed-status assertion.
- Typed verification tools: `bench/spot_check.rs`,
  `bench/verify_301.rs`, `bench/full_cly3d_scan.rs`.
- PyO3 binding `voltic.implied_vol_typed`.

## Verified

- 97/97 lib + integration + doc tests pass under
  `cargo +nightly test --release`.
- 200-bit mpmath truth on stratified samples confirms Computed sigma
  deviation inside the documented 1e-6 sigma-resolution budget:
  worst 3.41e-14 on CLY-3D (51,321 rows), worst 5.57e-8 on ATM-dense
  (48,831 rows).
- HH3 algorithm cross-checked against AQFED.jl
  `src/black/iv_solver_householder.jl` and FlashIV eq. 6 (Le Floc'h
  and Healy, arxiv 2605.29102 sec. 3.1).
- Independent verifier on 27 shifted-Computed sample rows: 0 contract
  violations against the mpmath truth.

## Compatibility

- MSRV unchanged (Rust 1.94 stable for non-SIMD callers; nightly
  required for the `std::simd` path, same as v1.1.0).
- No public API removals. No behavior change on the legacy f64
  surface.
- Python wheel (`voltic` on PyPI) ships the new
  `implied_vol_typed` function alongside the existing exports.

## CI

- Pinned the nightly toolchain in `.github/workflows/ci.yml` to
  `nightly-2026-05-12` so rustfmt and clippy rules stop drifting
  between runs.
- Crate-level `#![allow]` attributes added with one-line reasons:
  `improper_ctypes` on `src/norm.rs` (Sleef SIMD FFI vector types
  are not Rust-FFI-safe by spec); `clippy::excessive_precision` on
  `src/schadner_fast.rs` (Chebyshev seed coefficients in the
  include!()-d seed file carry beyond-f64 digits as published);
  `clippy::absurd_extreme_comparisons`, `clippy::assign_op_pattern`,
  `clippy::manual_clamp`, `clippy::manual_range_contains`, and
  `unused_imports`/`unused_variables`/`dead_code` on
  `src/schadner_fast.rs` (protected hot path per v1.2 byte-identity
  rule); `clippy::manual_clamp`, `clippy::manual_memcpy`,
  `clippy::manual_div_ceil`, and `non_snake_case` on
  `src/otm_context.rs` (NaN-clamp semantics, manual SIMD copy paths,
  and SIMD mask type naming); `clippy::excessive_precision` on
  `tests/wing_seed.rs` (published mpmath-200-bit reference table).
- Bench files passed clippy after small mechanical hygiene fixes
  (iterator-style loops, `writeln!`, `copy_from_slice`, range
  contains, type aliases).
- `cargo clippy --all-targets -- -D warnings` runs clean.

More info: ryan@databa.ai
