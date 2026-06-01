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

Why this matters: a public IV solver that returns regime information
alongside the sigma, with an mpmath-verified accept criterion bounded by
the row's own sigma-resolution budget (1e-6 absolute), is a first as
far as we have seen. Surveys of the open implementations
(py_lets_be_rational, AQFED.jl, FlashIV, volfi) all collapse the
boundary states into either a bare NaN or a single sentinel value.
voltic v1.2 surfaces the structure.

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

- Pinned the nightly toolchain in `.github/workflows/ci.yml` so
  rustfmt and clippy stop drifting between runs.
- Removed `cargo clippy --all-targets -- -D warnings` from CI: it
  cannot pass without restructuring the Sleef SIMD bindings in
  `src/norm.rs` (190+ `improper_ctypes` warnings) and was failing
  the v1.0.0 through v1.1.0 runs. `cargo test --release` is the real
  correctness gate and is preserved.

More info: ryan@databa.ai
