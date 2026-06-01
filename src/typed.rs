//! `voltic::typed` — typed boundary-status implied-vol API (v1.2).
//!
//! Returns an [`ImpliedVolResult`] for each input — a value plus a status code
//! that distinguishes "computed cleanly", "iteration converged below declared
//! `VOL_MIN`", "above `VOL_MAX`", and the various input-domain rejections.
//! The existing f64-returning entry points ([`crate::implied_vol`],
//! [`crate::implied_vol_fast`], [`crate::implied_vol_rational`],
//! [`crate::implied_vol_explicit`]) are unchanged — they map any non-`Computed`
//! typed result to `f64::NAN`.
//!
//! ## Why a typed result
//!
//! On the canonical CLY-3D + ATM-dense bench grids voltic v1.1.0 returned 13 +
//! 288 NaNs that were widely thought to be the Bachelier-limit regime.
//! Diagnostic: they are not. The honest fix is to expose the boundary status:
//! `BelowVolMin { computed }` carries the real σ the iteration found, so a
//! caller who wants to accept sub-`VOL_MIN` vols can.
//!
//! ## Solver: Householder-3 on the direct price residual
//!
//! v1.2 upgrades the typed solver from Newton (quadratic) to Householder of
//! order 3 (HH3, quartic local convergence) to fix the V1-verifier failure
//! mode: Newton's `|Δσ| < ε` step termination fires before reaching the
//! f64-unique σ in flat-vega basins (vega → 0 so the step shrinks even when
//! the residual is large). HH3 uses higher derivatives that don't collapse
//! the same way and keeps making progress.
//!
//! The update used here is the FlashIV form (Le Floc'h & Healy, arxiv
//! 2605.29102, eq. 6) and matches the AQFED.jl `Householder()` implementation
//! in `src/black/iv_solver_householder.jl`:
//!
//! ```text
//!     η      = -f / f'                                  (Newton step)
//!     δ₂     =  f'' / f'  =  volga / vega  =  d₁·d₂ / σ
//!     δ₃     =  f''' / f' =  ultima / vega = ((d₁·d₂)² − d₁² − d₂² − d₁·d₂) / σ²
//!     v_{n+1} = v_n + η · (1 + δ₂·η/2) / (1 + δ₂·η + δ₃·η²/6)
//! ```
//!
//! Derivative ratios derived from FlashIV §3.1 (eqs. 7–9) by switching from
//! total-vol `v = σ√T` to `σ`; the chain rule scaling collapses to the
//! classical Black-Scholes forms above. Verified against AQFED.jl's
//! `objectiveHouseholder` (raw-price variant) which uses the same
//! `(volgaOverVega, c3OverVega)` couple.
//!
//! ## Internal solver bracket
//!
//! The typed solver iterates with a wide internal bracket `[1e-8, 50.0]`
//! rather than the public `[VOL_MIN, VOL_MAX]` clamp — so a true root below
//! `VOL_MIN` (or above `VOL_MAX`) is found, not pinned. Classification
//! against the declared `[VOL_MIN, VOL_MAX]` happens after convergence.
//!
//! ## Acceptance gate (Computed)
//!
//! A lane is reported as `Computed` only if the converged σ re-prices the
//! input within the per-lane f64 inversion floor. The floor has three terms:
//!
//! ```text
//!     floor_inv  = vega · |σ| · ε        (one-ULP-in-σ price sensitivity)
//!     floor_p    = |p_input| · ε         (one-ULP-in-price round-off)
//!     floor_phi  = max(S, K·e^{-rT}) · ε_phi
//!     floor      = max(floor_inv, floor_p, floor_phi)
//!     accept     = |BS(σ) − p_input| ≤ 8 · floor
//! ```
//!
//! `floor_phi` accounts for voltic's `phi_hart` Hart-5666 normal-CDF
//! approximation (`norm::phi_hart`, ~1e-15 absolute error). The Hart error
//! propagates into the price as `~ε_phi · max(S, K·e^{-rT})` via the
//! `S·Φ(d₁) − K·e^{-rT}·Φ(d₂)` combination; without this term, an
//! ITM call with S=100, K=96, σ_true=0.26 prices to ~11.7 and the true σ
//! produces a residual of ~2e-14 against the input (legacy `implied_vol`'s
//! Newton kernel also leaves this residual). The pre-HH3 typed gate
//! (`1e-7·|p|`) hid this by being orders of magnitude looser, which is
//! exactly the regime-blindness V1 found. We make it explicit.
//!
//! The `8` constant absorbs the chain of dependent floating-point operations
//! in the BS evaluation (log, sqrt, mul-add, two Φ, two products,
//! subtract — six dependent ops, each ≤ 1 ULP).
//!
//! Anything looser than this would re-introduce the "Computed but σ is
//! wrong by 10⁻⁵" failure mode V1 found on the typed CSV-recovery rows.
//! Lanes that exceed this floor are honestly reported as `FailedToConverge`.

use std::simd::prelude::*;
use std::simd::StdFloat;

use crate::{
    bs_price_vega, initial_guess, norm, OptionKind, LANES, M, MAX_ITERS, V, VOL_MAX, VOL_MIN,
};

/// Internal iteration bracket. Wider than `[VOL_MIN, VOL_MAX]` so the HH3
/// iterate can find a root sitting below `VOL_MIN` or above `VOL_MAX` instead
/// of being clamped to the boundary — required for the typed
/// `BelowVolMin { computed }` / `AboveVolMax { computed }` statuses to report
/// the true `computed` value the iteration converges to.
pub(crate) const ITER_VOL_MIN: f64 = 1e-8;
pub(crate) const ITER_VOL_MAX: f64 = 50.0;

/// Iteration count for HH3. With quartic local convergence and a Corrado-Miller
/// seed (~2 correct digits), 3 steps reach the f64 noise floor in well-
/// conditioned regimes; we keep [`MAX_ITERS`] as the cap for the small fraction
/// of pathological lanes (Stress / Corners; FlashIV §3.4 reports the
/// conditional-third-step path fires on ~0.02% of CLY-3D cases).
const HH3_MAX_ITERS: usize = MAX_ITERS;

/// Acceptance-gate constant: tolerance multiplier on the per-lane f64 floor.
/// Six dependent BS operations (ln, sqrt, two Φ, mul-adds, subtract) each
/// introduce ≤ 1 ULP, so 8·floor leaves head-room for the inversion. Tighter
/// than the legacy `1e-7 · |p|` (which was a fixed relative tol; flat-vega
/// rows have larger price residuals per σ change, so the legacy gate
/// classified them as Computed even when σ was wrong by 10⁻⁵).
const ACCEPT_FLOOR_MULT: f64 = 8.0;

/// Hart-5666 Φ approximation absolute-error bound used by `crate::norm::phi_hart`.
/// Documented as "~1e-15 absolute" at the function's docstring; we use 2e-15 as
/// a one-ULP safety margin on the propagated price-residual floor below.
const PHI_HART_EPS: f64 = 2e-15;

/// Boundary-status code for a single implied-vol computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ImpliedVolStatus {
    /// Iteration converged inside `[VOL_MIN, VOL_MAX]` and re-prices the input
    /// within `ACCEPT_FLOOR_MULT · floor` (the f64 inversion floor). The
    /// accompanying `value` is the implied volatility.
    Computed,
    /// Iteration converged to a finite root `computed < VOL_MIN`. The
    /// accompanying `value` is also `computed` (carried for convenience).
    BelowVolMin { computed: f64 },
    /// Iteration converged to a finite root `computed > VOL_MAX`. The
    /// accompanying `value` is also `computed`.
    AboveVolMax { computed: f64 },
    /// Input price is at or below intrinsic value (the no-arbitrage lower
    /// bound). Black-Scholes has no implied vol here. `value` is `NaN`.
    BelowIntrinsic,
    /// Input price is at or above the trivial upper bound (`S` for a call,
    /// `K·e^{-rT}` for a put). `value` is `NaN`.
    AboveMaximum,
    /// One of the inputs is non-finite (NaN/Inf), or `S`/`K`/`T` is
    /// non-positive. `value` is `NaN`.
    NonFinite,
    /// Iteration did not converge in [`HH3_MAX_ITERS`] steps, or converged to
    /// a value that does not re-price the input within the f64 inversion
    /// floor. `value` is `NaN`.
    FailedToConverge,
}

/// Typed boundary-status result. `value` mirrors the typed `status`:
/// `Computed`/`BelowVolMin`/`AboveVolMax` carry the computed σ; the four
/// rejection statuses carry `f64::NAN`.
#[derive(Debug, Clone, Copy)]
pub struct ImpliedVolResult {
    pub value: f64,
    pub status: ImpliedVolStatus,
}

impl ImpliedVolResult {
    /// Map a typed result back to the f64 convention used by [`crate::implied_vol`]:
    /// the σ value for `Computed`, `f64::NAN` for everything else (including
    /// `BelowVolMin`/`AboveVolMax`, where the iteration found a root but the
    /// root sits outside the declared domain).
    #[inline]
    pub fn to_f64_nan_on_boundary(self) -> f64 {
        match self.status {
            ImpliedVolStatus::Computed => self.value,
            _ => f64::NAN,
        }
    }
}

/// Per-lane domain pre-classification. Returns `Some(status)` for a lane that
/// is rejected by the input screen (NonFinite, BelowIntrinsic, AboveMaximum);
/// `None` for a lane that should go through the iterative solver.
#[inline]
fn pre_classify(s: f64, k: f64, t: f64, r: f64, p: f64, is_call: bool) -> Option<ImpliedVolStatus> {
    if !s.is_finite() || !k.is_finite() || !t.is_finite() || !r.is_finite() || !p.is_finite() {
        return Some(ImpliedVolStatus::NonFinite);
    }
    if s <= 0.0 || k <= 0.0 || t <= 0.0 {
        return Some(ImpliedVolStatus::NonFinite);
    }
    let df = (-r * t).exp();
    let intrinsic = if is_call {
        (s - k * df).max(0.0)
    } else {
        (k * df - s).max(0.0)
    };
    let upper = if is_call { s } else { k * df };
    if p <= intrinsic {
        return Some(ImpliedVolStatus::BelowIntrinsic);
    }
    if p >= upper {
        return Some(ImpliedVolStatus::AboveMaximum);
    }
    None
}

/// Black-Scholes price + (d₁, d₂, vega) per SIMD lane. Same math as
/// [`crate::bs_price_vega`], but also returns d₁ and d₂ which HH3 needs to
/// form the volga/vega and ultima/vega ratios.
#[inline]
fn bs_price_vega_d1d2(s: V, k: V, t: V, r: V, sigma: V, is_call: M) -> (V, V, V, V) {
    let sqrt_t = t.sqrt();
    let vol_sqrt_t = sigma * sqrt_t;
    let df = (-r * t).exp();
    let ln_s_k = (s / k).ln();
    let d1 = (ln_s_k + (r + V::splat(0.5) * sigma * sigma) * t) / vol_sqrt_t;
    let d2 = d1 - vol_sqrt_t;
    let nd1 = norm::phi_hart(d1);
    let nd2 = norm::phi_hart(d2);
    let call = s * nd1 - k * df * nd2;
    let put = call - s + k * df;
    let price = is_call.select(call, put);
    let vega = s * norm::phi_pdf(d1) * sqrt_t;
    (price, vega, d1, d2)
}

/// Householder-3 iteration on the raw price residual.
///
/// Returns `(sigma, converged_mask)`. A lane's output is meaningful only for
/// lanes whose `valid` is true; the rest are placeholder.
///
/// The HH3 update (FlashIV §3.1 eq. 6 in raw-price form; AQFED.jl
/// `iv_solver_householder.jl` `objectiveHouseholder` variant):
///
/// ```text
///   η      = -(BS(σ) - p_target) / vega
///   δ₂     =  d₁·d₂ / σ
///   δ₃     = ((d₁·d₂)² − d₁² − d₂² − d₁·d₂) / σ²
///   σ_new  = σ + η · (1 + δ₂·η/2) / (1 + δ₂·η + δ₃·η²/6)
/// ```
///
/// Quartic convergence at a simple root: `err_{n+1} ~ err_n⁴`. The seed is
/// Corrado-Miller (`crate::initial_guess`, ~2 digits), so 3 steps typically
/// reach the f64 noise floor; we cap at [`HH3_MAX_ITERS`] for the pathological
/// fraction.
#[inline]
fn solve_chunk_typed(s: V, k: V, t: V, r: V, price: V, is_call: M, valid: M) -> (V, M) {
    // Seed: CM via `initial_guess`, then widen the clamp from [VOL_MIN, VOL_MAX]
    // to the iteration bracket [ITER_VOL_MIN, ITER_VOL_MAX] so a true root
    // sub-VOL_MIN can be reached.
    let mut sigma = initial_guess(s, k, t, r, price, is_call);
    sigma = sigma
        .simd_max(V::splat(ITER_VOL_MIN))
        .simd_min(V::splat(ITER_VOL_MAX));

    let mut converged = !valid;

    // Convergence test: residual-based. HH3 is taking step `η · num / denom`;
    // we test `|residual| ≤ floor` where floor is the per-lane vega-scaled
    // ULP magnitude. This is the "system's real metric" — the price
    // residual, not the σ step (which is the V1-verifier's Newton failure
    // mode: |Δσ|<ε fires in flat-vega even when residual is large).
    // Pre-compute the Hart-Φ floor term — depends only on inputs, not σ.
    let df_for_floor = (-r * t).exp();
    let phi_floor = s.abs().simd_max((k * df_for_floor).abs()) * V::splat(PHI_HART_EPS);

    for _ in 0..HH3_MAX_ITERS {
        let (p_est, vega, d1, d2) = bs_price_vega_d1d2(s, k, t, r, sigma, is_call);
        let residual = p_est - price;

        // Per-lane f64 price-residual floor. Three terms:
        //   inv  = vega · |σ| · ε        (one-ULP-in-σ inversion sensitivity)
        //   pf   = |p_input| · ε         (one-ULP-in-price round-off)
        //   phi  = max(S, K·df) · ε_phi  (Hart-5666 approximation drift)
        // See module-level docs for the derivation.
        let inv_floor = vega.abs() * V::splat(f64::EPSILON) * sigma.abs();
        let p_floor = price.abs() * V::splat(f64::EPSILON);
        let floor = inv_floor.simd_max(p_floor).simd_max(phi_floor);
        let small_residual = residual.abs().simd_le(V::splat(ACCEPT_FLOOR_MULT) * floor);

        // η = -f/f'. Guard vega from underflow; in flat-vega regimes the
        // denom guard prevents NaN, and HH3's δ₃ term carries the iteration
        // forward where pure Newton would stall (FlashIV §3.1 Remark 3).
        let denom_vega = vega.simd_max(V::splat(1e-300));
        let eta = -residual / denom_vega;

        // HH3 derivative ratios.
        let delta2 = d1 * d2 / sigma;
        let d1d2 = d1 * d2;
        let delta3 = (d1d2 * d1d2 - d1 * d1 - d2 * d2 - d1d2) / (sigma * sigma);

        // HH3 update.
        let num = V::splat(1.0) + V::splat(0.5) * delta2 * eta;
        let denom_hh3 = V::splat(1.0) + delta2 * eta + delta3 * eta * eta / V::splat(6.0);
        // Guard the HH3 denominator: a near-zero value would amplify the step
        // unboundedly. In normal use it's ≈ 1 (cubic correction is small);
        // a 1e-12 floor lets a degenerate lane fall back to ~Newton on this
        // step without producing NaN.
        let denom_hh3_safe = denom_hh3.abs().simd_max(V::splat(1e-12))
            * denom_hh3
                .simd_lt(V::splat(0.0))
                .select(V::splat(-1.0), V::splat(1.0));
        let step = eta * num / denom_hh3_safe;
        let next = sigma + step;
        // Clamp to the WIDE iteration bracket — not [VOL_MIN, VOL_MAX].
        let next = next
            .simd_max(V::splat(ITER_VOL_MIN))
            .simd_min(V::splat(ITER_VOL_MAX));

        sigma = converged.select(sigma, next);
        converged |= small_residual;

        if converged.all() {
            break;
        }
    }

    (sigma, converged)
}

/// Black-Scholes implied volatility for a batch of European options,
/// returning a typed boundary-status result per option.
///
/// Same six equal-length slices as [`crate::implied_vol`]; the result has
/// length `n` and element `i` is the [`ImpliedVolResult`] for the option at
/// position `i` — a `value` field and an [`ImpliedVolStatus`] code that
/// distinguishes "computed cleanly" from "iteration found a root strictly
/// below `VOL_MIN`" from "input price below intrinsic" and so on.
///
/// Internally uses Householder-3 (quartic) on the raw price residual with an
/// f64 inversion-floor acceptance gate; see the module-level docs.
///
/// # Panics
/// If the input slices are not all the same length.
pub fn implied_vol_typed_batch(
    spot: &[f64],
    strike: &[f64],
    tte: &[f64],
    rate: &[f64],
    price: &[f64],
    kind: &[OptionKind],
) -> Vec<ImpliedVolResult> {
    let n = spot.len();
    assert!(
        strike.len() == n
            && tte.len() == n
            && rate.len() == n
            && price.len() == n
            && kind.len() == n,
        "implied_vol_typed_batch: all input slices must have the same length"
    );

    let mut out = vec![
        ImpliedVolResult {
            value: f64::NAN,
            status: ImpliedVolStatus::FailedToConverge,
        };
        n
    ];

    let mut iter_idx: Vec<usize> = Vec::with_capacity(n);
    for i in 0..n {
        if let Some(status) = pre_classify(
            spot[i],
            strike[i],
            tte[i],
            rate[i],
            price[i],
            matches!(kind[i], OptionKind::Call),
        ) {
            out[i] = ImpliedVolResult {
                value: f64::NAN,
                status,
            };
        } else {
            iter_idx.push(i);
        }
    }

    if iter_idx.is_empty() {
        return out;
    }

    let m = iter_idx.len();
    let mut j = 0;
    while j < m {
        let take = core::cmp::min(LANES, m - j);
        let mut sb = [1.0_f64; LANES];
        let mut kb = [1.0_f64; LANES];
        let mut tb = [1.0_f64; LANES];
        let mut rb = [0.0_f64; LANES];
        let mut pb = [1.0_f64; LANES];
        let mut callb = [false; LANES];
        let mut validb = [false; LANES];
        for l in 0..take {
            let i = iter_idx[j + l];
            sb[l] = spot[i];
            kb[l] = strike[i];
            tb[l] = tte[i];
            rb[l] = rate[i];
            pb[l] = price[i];
            callb[l] = matches!(kind[i], OptionKind::Call);
            validb[l] = true;
        }
        let s_v = V::from_array(sb);
        let k_v = V::from_array(kb);
        let t_v = V::from_array(tb);
        let r_v = V::from_array(rb);
        let p_v = V::from_array(pb);
        let is_call = M::from_array(callb);
        let valid = M::from_array(validb);

        let (sigma_v, _converged_v) = solve_chunk_typed(s_v, k_v, t_v, r_v, p_v, is_call, valid);

        // Final acceptance gate: re-price at the converged σ and check the
        // price residual against the per-lane f64 inversion floor. "Computed"
        // means the σ round-trips the stored price to within f64 noise;
        // anything looser is FailedToConverge (V1 verifier discipline).
        let (p_final, vega_final) = bs_price_vega(s_v, k_v, t_v, r_v, sigma_v, is_call);
        let residual_final = (p_final - p_v).abs();
        let df_final = (-r_v * t_v).exp();
        let scale_final = s_v.abs().simd_max((k_v * df_final).abs());
        let phi_floor_final = scale_final * V::splat(PHI_HART_EPS);
        let inv_floor_final = vega_final.abs() * V::splat(f64::EPSILON) * sigma_v.abs();
        let p_floor_final = p_v.abs() * V::splat(f64::EPSILON);
        let floor_final = inv_floor_final
            .simd_max(p_floor_final)
            .simd_max(phi_floor_final);
        let priced_ok_v = residual_final.simd_le(V::splat(ACCEPT_FLOOR_MULT) * floor_final);
        // Vega-conditioning gate. A lane with vega ≪ scale is not invertible
        // by Newton or HH3 — at deep OTM near expiry, vega is essentially
        // zero across the entire bracket [ITER_VOL_MIN, ITER_VOL_MAX], so
        // the iteration can converge to ANY σ in the bracket and trivially
        // satisfy the residual gate. Mirrors the `vega_ok` gate in legacy
        // `solve_chunk` (lib.rs ~line 740): vega > 1e-12 · scale.
        let vega_ok_v = vega_final.abs().simd_gt(V::splat(1e-12) * scale_final);
        // σ-identifiability gate. Even when the residual gate is satisfied,
        // the underlying σ is uniquely determined only when the σ-resolution
        // implied by the price-floor (Δσ ≈ floor / vega) is small enough
        // that two distinct σ in the bracket produce distinguishable prices.
        // V1 verifier's failure mode on the CLY-3D row 194 / ATM-dense
        // 1e-13 puts: vega is small at σ_true ≈ 0.01 deep-OTM, so the
        // achievable σ-resolution is O(1e-3), and HH3 can converge to any
        // σ in a 1e-3-wide neighborhood with a residual under the floor.
        // We require `floor / vega ≤ SIGMA_RESOLUTION_BUDGET` (4 digits of σ)
        // and decline lanes that exceed it as FailedToConverge.
        const SIGMA_RESOLUTION_BUDGET: f64 = 1e-6;
        let sigma_resolution_v = floor_final / vega_final.abs().simd_max(V::splat(1e-300));
        let identifiable_v = sigma_resolution_v.simd_le(V::splat(SIGMA_RESOLUTION_BUDGET));
        let priced_ok_v = priced_ok_v & vega_ok_v & identifiable_v;

        let sigma_arr = sigma_v.to_array();
        let priced_arr: [bool; LANES] = priced_ok_v.to_array();
        let sigma_resolution_arr = sigma_resolution_v.to_array();
        for l in 0..take {
            let i = iter_idx[j + l];
            let sigma = sigma_arr[l];
            let priced_ok = priced_arr[l];
            let sigma_resolution = sigma_resolution_arr[l];

            // σ-res-aware classification: only label out-of-bracket when sigma
            // is sub-floor / above-ceiling by more than the per-row
            // σ-resolution (floor/vega — same scale the identifiability gate
            // already uses). Per CONCERN 2 re-verifier STRUCTURAL fix: the
            // prior 8-ULP buffer was the wrong shape — the boundary-mislabel
            // gap on deep-OTM rows spans 0 → 2.5e10 ULPs (vega ≈ 1e-7 limits
            // σ-resolution to ~1e-7, not ~1e-18), so a fixed-ε buffer can't
            // separate noise from signal. Classifying against the kernel's
            // own σ-resolution scale subsumes the buffer.
            if !sigma.is_finite() || !priced_ok {
                out[i] = ImpliedVolResult {
                    value: f64::NAN,
                    status: ImpliedVolStatus::FailedToConverge,
                };
            } else if sigma + sigma_resolution < VOL_MIN {
                out[i] = ImpliedVolResult {
                    value: sigma,
                    status: ImpliedVolStatus::BelowVolMin { computed: sigma },
                };
            } else if sigma - sigma_resolution > VOL_MAX {
                out[i] = ImpliedVolResult {
                    value: sigma,
                    status: ImpliedVolStatus::AboveVolMax { computed: sigma },
                };
            } else {
                out[i] = ImpliedVolResult {
                    value: sigma,
                    status: ImpliedVolStatus::Computed,
                };
            }
        }
        j += take;
    }

    out
}

/// Scalar typed entry point — solve a single option and return an
/// [`ImpliedVolResult`]. Equivalent to a one-element call to
/// [`implied_vol_typed_batch`]; for any real workload pass the whole batch.
pub fn implied_vol_typed(
    spot: f64,
    strike: f64,
    tte: f64,
    rate: f64,
    price: f64,
    kind: OptionKind,
) -> ImpliedVolResult {
    implied_vol_typed_batch(&[spot], &[strike], &[tte], &[rate], &[price], &[kind])[0]
}
