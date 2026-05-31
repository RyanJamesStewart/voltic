//! `voltic::schadner` — the explicit (closed-form) implied-volatility solver.
//!
//! Schadner, *"An Explicit Solution to Black-Scholes Implied Volatility"*
//! (arXiv:2604.24480, 2026), observes that the Black-Scholes call price is the
//! survival function of an inverse Gaussian, so the implied vol is, in closed
//! form,
//!
//! ```text
//!   σ(K, C) = (2/√T) · [ ℱ_IG⁻¹( (1 − c)/m ; μ = 2|k|, λ = 1 ) ]^(−1/2)
//! ```
//!
//! with `c` the spot-normalized call price, `k = ln(K/F)` the forward
//! log-moneyness, `m = 1` for `K > F` and `m = K/F` for `K < F`, and at the
//! forward (`k = 0`) it collapses to the pure probit
//! `σ = (2/√T)·Φ⁻¹((c+1)/2)`.
//!
//! "Explicit" relocates the work rather than removing it: the only
//! non-elementary operation is the inverse Gaussian quantile `ℱ_IG⁻¹`, which
//! has no elementary closed form and is itself a root of the IG CDF. The IG
//! CDF *does* have a closed form in Φ, so this module evaluates `ℱ_IG⁻¹` the
//! same way `voltic`'s direct solver evaluates its inverse: a lane-packed
//! Newton with masked per-lane convergence, here on the inverse Gaussian CDF
//! (two [`norm::phi_hart`] calls per step) instead of the Black-Scholes price.
//!
//! A finding from wiring it up: the explicit formula does *not* free you from
//! a good initial guess. A naive probit start overshoots the IG-quantile root
//! by orders of magnitude for an out-of-the-money option (the probit assumes
//! the forward), and the IG density is flat in the tail just as the BS price
//! is flat in σ there, so the relocated Newton has the same pathology. This
//! module therefore seeds the IG-quantile Newton with the *same*
//! Corrado-Miller rational guess [`crate::implied_vol`] uses, mapped through
//! `x = 4/(σ√T)²`. That makes the benchmark a clean isolation: identical
//! guess, identical convergence machinery; the only variable is whether the
//! residual is the Black-Scholes price (direct) or the inverse Gaussian CDF
//! (explicit). `bench/` answers which is faster on identical hardware.
//!
//! Faithfulness. Schadner posted reference demo code (`wol-fi/direct_vola`,
//! unlicensed, cited not vendored); his `iv_fig` uses the same probit
//! at-the-forward collapse, the same Acklam `ndtri` (two Halley steps, matched
//! in [`norm::phi_inv`]), and iterates a vol-proportional variable rather than
//! the IG quantile `x` directly, exactly as here. He disclaims the demo as not
//! the implementation behind the paper's speed figure, so the paper's
//! 3.4×-vs-Jäckel is from unreleased code and not reproducible; this module is
//! a faithful, *favorably-optimized* SIMD port (his demo seeds only the probit
//! and runs a fixed bracket + 30 safeguarded-Halley iterations; this gives the
//! method voltic's Corrado-Miller seed and a lean masked Newton), so the
//! benchmark's gap to the direct solver is a charitable upper bound on the
//! method, not a critique of his code.
//!
//! API parity with [`crate::implied_vol`]: same six equal-length slices, same
//! `NaN` discipline (a lane that fails the domain screen, does not converge,
//! leaves `[VOL_MIN, VOL_MAX]`, or does not re-price to the input is `NaN`).
//!
//! ```
//! use voltic::{implied_vol_explicit, OptionKind};
//! let iv = implied_vol_explicit(&[100.0], &[100.0], &[1.0], &[0.02], &[12.821_58], &[OptionKind::Call]);
//! assert!((iv[0] - 0.30).abs() < 1e-4);
//! ```

use std::simd::prelude::*;
use std::simd::StdFloat;

use crate::norm;
use crate::{bs_price_vega, initial_guess, screen, OptionKind};
use crate::{LANES, M, MAX_ITERS, V, VOL_MAX, VOL_MIN};

/// Below this `|k| = |ln(K/F)|` the option is at the forward and the formula
/// collapses to the probit; the inverse-Gaussian mean `μ = 2|k| → 0` is
/// degenerate there, so the probit arm (computed unconditionally as the IG
/// initial guess anyway) is selected instead.
const K_ZERO_EPS: f64 = 1e-7;
// The Newton iterate is the total vol `v = σ√T`, not the IG quantile `x`
// itself: `x = 4/v²` is violently nonlinear (a tail overshoot in `x` lands at a
// clamp and oscillates), whereas `v` lives in the same tame, bounded bracket
// the direct solver clamps `σ` into. Same clamp-don't-reflect discipline; and
// it makes the benchmark an honest isolation (both Newtons iterate in vol
// space; only the residual differs: BS price vs IG CDF).

/// Inverse-Gaussian CDF `F_IG(x; μ, λ=1)` and density `f_IG` (the Newton
/// slope), lane-packed. From Schadner's survival representation,
/// `1 − F_IG(x;μ,1) = Φ(−√x/μ + 1/√x) − e^{2/μ}·Φ(−√x/μ − 1/√x)`, and
/// `f_IG(x;μ,1) = (2π x³)^(−1/2)·exp(−(x−μ)²/(2 μ² x))`. The caller caps `μ`
/// (it is `2/|k|`, so `μ → ∞` at the forward, where the probit arm is taken).
#[inline]
fn ig_cdf_pdf(x: V, mu: V) -> (V, V) {
    let sqrt_x = x.sqrt();
    let inv_sqrt_x = V::splat(1.0) / sqrt_x;
    let a = sqrt_x / mu; // √x / μ

    // Survival: Φ(−a + 1/√x) − e^{2/μ}·Φ(−a − 1/√x).
    let term1 = norm::phi_hart(-a + inv_sqrt_x);
    let term2 = norm::phi_hart(-a - inv_sqrt_x);
    let surv = term1 - norm::vexp(V::splat(2.0) / mu) * term2;
    let cdf = V::splat(1.0) - surv;

    // Density: (2π x³)^(−1/2) · exp(−(x−μ)²/(2 μ² x)).
    let two_pi = V::splat(2.0 * core::f64::consts::PI);
    let inv_norm = (two_pi * x * x * x).sqrt();
    let dx = x - mu;
    let expo = norm::vexp(-(dx * dx) / (V::splat(2.0) * mu * mu * x));
    let pdf = expo / inv_norm;
    (cdf, pdf)
}

/// Solve one lane-packed batch by Schadner's explicit formula. Mirrors
/// `crate::solve_chunk`: invalid lanes return `NaN`, and within the valid
/// lanes a lane that does not converge, leaves `[VOL_MIN, VOL_MAX]`, or does
/// not re-price to the input premium is also `NaN`.
#[inline]
fn solve_chunk_explicit(s: V, k: V, t: V, r: V, price: V, is_call: M, valid: M) -> V {
    let nan = V::splat(f64::NAN);
    let sqrt_t = t.sqrt();
    let df = norm::vexp(-r * t); // e^{−rT}
    let fwd = s / df; // forward F = S·e^{rT}
    let k_log = norm::vlog(k / fwd); // k = ln(K/F)
    let ak = k_log.abs();
    let ek = norm::vexp(k_log); // e^k = K/F

    // m = 1 for K > F (k > 0); m = K/F = e^k for K < F.
    let m = k_log.simd_gt(V::splat(0.0)).select(V::splat(1.0), ek);

    // Convert both kinds to the call-equivalent spot-normalized price via
    // put-call parity (C − P = S − K·e^{−rT}, i.e. c = p + 1 − e^k in
    // spot-normalized terms); then a single call formula serves both. With
    // c so defined, (1 − c)/m equals Schadner's put argument (e^k − p)/m.
    let xn = price / s;
    let c = is_call.select(xn, xn + V::splat(1.0) - ek);

    // Probit arm: exact at the forward (k = 0), where the formula collapses to
    // σ = (2/√T)·Φ⁻¹((c+1)/2).
    let v_probit = V::splat(2.0) * norm::phi_inv((c + V::splat(1.0)) * V::splat(0.5));
    let sigma_atm = v_probit / sqrt_t;

    // IG arm. Target probability q = (1 − c)/m, mean μ = 2/|k| (large near the
    // forward, where the probit arm wins and these lanes are masked off; capped
    // so an extreme-moneyness lane cannot overflow e^{2/μ}). Solve
    // F_IG(x; μ, 1) = q for x > 0 by masked Newton, then σ = 2/(√T·√x).
    let q = (V::splat(1.0) - c) / m;
    let mu = (V::splat(2.0) / ak.simd_max(V::splat(1e-12))).simd_min(V::splat(1e12));

    // v₀ from voltic's own Corrado-Miller rational guess (moneyness-aware, the
    // same start the direct solver uses): σ₀ → v₀ = σ₀√T.
    let sigma0 = initial_guess(s, k, t, r, price, is_call);
    let v_lo = V::splat(VOL_MIN) * sqrt_t;
    let v_hi = V::splat(VOL_MAX) * sqrt_t;
    let mut v = (sigma0 * sqrt_t).simd_max(v_lo).simd_min(v_hi);

    // A lane is excluded from the IG iteration if it is invalid, at the
    // forward (probit arm wins), or its target probability is out of (0, 1).
    let near_fwd = ak.simd_lt(V::splat(K_ZERO_EPS));
    let q_ok = q.simd_gt(V::splat(0.0)) & q.simd_lt(V::splat(1.0));
    let mut converged = !valid | near_fwd | !q_ok;

    // Step tolerance in vol units (matches the direct solver's scale) carried
    // back to v via √T; plus an absolute residual on the probability.
    let tol_v = V::splat(1e-12) * sqrt_t;
    let res_tol = V::splat(1e-13);
    for _ in 0..MAX_ITERS {
        // x = 4/v²; Newton on G(v) = F_IG(4/v²; μ, 1) − q, with
        // G'(v) = f_IG(x)·dx/dv = f_IG(x)·(−8/v³), so the update is
        // v ← v + (F_IG(x) − q)·v³ / (8·f_IG(x)).
        let x = V::splat(4.0) / (v * v);
        let (cdf, pdf) = ig_cdf_pdf(x, mu);
        let residual = cdf - q;
        // Guard a vanishing density (the IG tail) so a flat lane cannot divide
        // by ~0; the residual test below retires it instead.
        let denom = (V::splat(8.0) * pdf).simd_max(V::splat(1e-300));
        let next = (v + residual * v * v * v / denom)
            .simd_max(v_lo)
            .simd_min(v_hi);

        let small_step = (next - v).abs().simd_le(tol_v);
        let small_residual = residual.abs().simd_le(res_tol);
        v = converged.select(v, next);
        converged |= small_step | small_residual;
        if converged.all() {
            break;
        }
    }
    let sigma_ig = v / sqrt_t;

    // Select the arm: probit at the forward, IG quantile otherwise.
    let sigma = near_fwd.select(sigma_atm, sigma_ig);

    // Acceptance, the same honesty gate as the direct solver: σ inside the
    // open bracket, finite, and the option re-prices to the input premium to
    // a relative tolerance with an absolute floor scaled to the underlying.
    let (p_final, _vega) = bs_price_vega(s, k, t, r, sigma, is_call);
    let scale = price.abs().simd_max(s.abs()).simd_max(k.abs());
    let priced_ok = (p_final - price)
        .abs()
        .simd_le(V::splat(1e-7) * price.abs() + V::splat(1e-10) * scale);
    let inside =
        sigma.simd_gt(V::splat(VOL_MIN * 1.0000001)) & sigma.simd_lt(V::splat(VOL_MAX * 0.9999999));
    let finite = sigma.is_finite();
    let accept = valid & converged & priced_ok & inside & finite;
    accept.select(sigma, nan)
}

/// Black-Scholes implied volatility for a batch of European options via
/// Schadner's explicit inverse-Gaussian formula (arXiv:2604.24480).
///
/// Drop-in for [`crate::implied_vol`]: same six equal-length slices, same
/// return shape, same `NaN` contract (element `i` is `NaN` if that input has
/// no Black-Scholes implied vol in `[VOL_MIN, VOL_MAX]`). The two differ only
/// in *how* the inverse is taken; on a clean round-trip dataset they agree to
/// the conditioning floor (see `tests/properties.rs`).
///
/// # Panics
/// If the input slices are not all the same length.
pub fn implied_vol_explicit(
    spot: &[f64],
    strike: &[f64],
    tte: &[f64],
    rate: &[f64],
    price: &[f64],
    kind: &[OptionKind],
) -> Vec<f64> {
    let n = spot.len();
    assert!(
        strike.len() == n
            && tte.len() == n
            && rate.len() == n
            && price.len() == n
            && kind.len() == n,
        "implied_vol_explicit: all input slices must have the same length"
    );
    let mut out = vec![0.0_f64; n];

    let mut i = 0;
    while i < n {
        let take = core::cmp::min(LANES, n - i);
        let mut sb = [1.0_f64; LANES];
        let mut kb = [1.0_f64; LANES];
        let mut tb = [1.0_f64; LANES];
        let mut rb = [0.0_f64; LANES];
        let mut pb = [1.0_f64; LANES];
        let mut callb = [false; LANES];
        let mut realb = [false; LANES];
        for j in 0..take {
            sb[j] = spot[i + j];
            kb[j] = strike[i + j];
            tb[j] = tte[i + j];
            rb[j] = rate[i + j];
            pb[j] = price[i + j];
            callb[j] = matches!(kind[i + j], OptionKind::Call);
            realb[j] = true;
        }
        let s = V::from_array(sb);
        let k = V::from_array(kb);
        let t = V::from_array(tb);
        let r = V::from_array(rb);
        let p = V::from_array(pb);
        let is_call = M::from_array(callb);
        let real = M::from_array(realb);

        let valid = real & screen(s, k, t, r, p, is_call);
        let res = solve_chunk_explicit(s, k, t, r, p, is_call, valid);
        out[i..i + take].copy_from_slice(&res.to_array()[..take]);
        i += take;
    }
    out
}

/// Single-option convenience (calls [`implied_vol_explicit`] with one-element
/// slices). For real workloads pass the whole batch.
pub fn implied_vol_explicit_one(
    spot: f64,
    strike: f64,
    tte: f64,
    rate: f64,
    price: f64,
    kind: OptionKind,
) -> f64 {
    implied_vol_explicit(&[spot], &[strike], &[tte], &[rate], &[price], &[kind])[0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bs_price;

    #[test]
    fn round_trip_atm_call() {
        let (s, k, t, r) = ([100.0], [100.0], [1.0], [0.02]);
        let kind = [OptionKind::Call];
        let price = bs_price(&s, &k, &t, &r, &[0.30], &kind);
        let iv = implied_vol_explicit(&s, &k, &t, &r, &price, &kind);
        assert!(
            (iv[0] - 0.30).abs() < 1e-9,
            "iv={} price={}",
            iv[0],
            price[0]
        );
    }

    #[test]
    fn matches_direct_solver_on_grid() {
        // The explicit formula and the direct Newton solver must agree to the
        // round-trip conditioning floor across a well-conditioned grid.
        let mut s = Vec::new();
        let mut k = Vec::new();
        let mut t = Vec::new();
        let mut r = Vec::new();
        let mut sig = Vec::new();
        let mut kind = Vec::new();
        for &spot in &[80.0_f64, 100.0, 130.0] {
            for &strike in &[70.0_f64, 90.0, 100.0, 110.0, 140.0] {
                for &tte in &[0.05_f64, 0.25, 1.0, 2.0] {
                    for &rate in &[0.0_f64, 0.03, 0.06] {
                        for &v in &[0.08_f64, 0.2, 0.5, 0.8] {
                            for &kd in &[OptionKind::Call, OptionKind::Put] {
                                s.push(spot);
                                k.push(strike);
                                t.push(tte);
                                r.push(rate);
                                sig.push(v);
                                kind.push(kd);
                            }
                        }
                    }
                }
            }
        }
        let price = bs_price(&s, &k, &t, &r, &sig, &kind);
        let iv = implied_vol_explicit(&s, &k, &t, &r, &price, &kind);
        let mut worst = 0.0_f64;
        let mut solved = 0usize;
        for idx in 0..s.len() {
            let df = (-r[idx] * t[idx]).exp();
            let intrinsic = match kind[idx] {
                OptionKind::Call => (s[idx] - k[idx] * df).max(0.0),
                OptionKind::Put => (k[idx] * df - s[idx]).max(0.0),
            };
            if price[idx] - intrinsic <= 1e-8 * (s[idx] + k[idx]) {
                continue; // not well-posed at f64 precision
            }
            assert!(
                !iv[idx].is_nan(),
                "explicit NaN'd well-posed S={} K={} T={} r={} v={} {:?}",
                s[idx],
                k[idx],
                t[idx],
                r[idx],
                sig[idx],
                kind[idx]
            );
            solved += 1;
            worst = worst.max((iv[idx] - sig[idx]).abs());
        }
        assert!(solved > 200, "only {solved} well-posed points");
        assert!(worst < 1e-6, "explicit worst abs vol error = {worst:e}");
    }

    #[test]
    fn batch_equals_singletons() {
        let n = 23;
        let s: Vec<f64> = (0..n).map(|_| 100.0).collect();
        let k: Vec<f64> = (0..n).map(|i| 70.0 + (i as f64) * 3.0).collect();
        let t: Vec<f64> = (0..n).map(|_| 0.6).collect();
        let r: Vec<f64> = (0..n).map(|_| 0.02).collect();
        let v: Vec<f64> = (0..n).map(|i| 0.1 + 0.02 * (i as f64)).collect();
        let kind: Vec<OptionKind> = (0..n)
            .map(|i| {
                if i % 2 == 0 {
                    OptionKind::Call
                } else {
                    OptionKind::Put
                }
            })
            .collect();
        let p = bs_price(&s, &k, &t, &r, &v, &kind);
        let batched = implied_vol_explicit(&s, &k, &t, &r, &p, &kind);
        for idx in 0..n {
            let one = implied_vol_explicit(
                &[s[idx]],
                &[k[idx]],
                &[t[idx]],
                &[r[idx]],
                &[p[idx]],
                &[kind[idx]],
            );
            assert_eq!(
                batched[idx].to_bits(),
                one[0].to_bits(),
                "lane {idx}: batched {} vs single {}",
                batched[idx],
                one[0]
            );
        }
    }

    #[test]
    fn edge_cases_return_nan() {
        let s = [100.0, 100.0, -1.0, 100.0];
        let k = [100.0, 100.0, 100.0, 100.0];
        let t = [1.0, 0.0, 1.0, 1.0];
        let r = [0.0, 0.0, 0.0, 0.0];
        let p = [0.001, 5.0, 5.0, 200.0];
        let kind = [OptionKind::Call; 4];
        let iv = implied_vol_explicit(&s, &k, &t, &r, &p, &kind);
        for (idx, v) in iv.iter().enumerate() {
            assert!(v.is_nan(), "lane {idx} should be NaN, got {v}");
        }
    }
}
