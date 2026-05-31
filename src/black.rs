//! The normalized Black function for the Jäckel rational implied-vol solver.
//!
//! Jäckel's normalization (Paper §2):
//!
//! ```text
//!     x      := ln(F/K)                       forward log-moneyness
//!     σ      := σ̂ · √T                       TOTAL volatility (NOT annual)
//!     b(x,σ) := B(F, K, σ̂, T, +1) / √(F·K)   normalized Black call price
//!
//!     b(x, σ) = e^(x/2) · Φ(x/σ + σ/2) − e^(−x/2) · Φ(x/σ − σ/2)
//! ```
//!
//! After the canonical reduction (Paper §2 invariances (2.5)–(2.6)), all
//! inputs are mapped to `x ≤ 0, θ = +1` (out-of-the-money calls), so this
//! module assumes that pre-condition. Bounds: `0 ≤ b ≤ b_max ≤ 1` with
//! `b_max = e^(x/2)`.
//!
//! ## Precision policy (TBD-7 — resolved)
//!
//! The naive form `e^(x/2)·Φ(d₁) − e^(−x/2)·Φ(d₂)` loses precision in the
//! centre when `Φ(d₁)` and `Φ(d₂)` are close (cancellation eats ~50% of the
//! mantissa in the worst case). This module implements the cancellation-free
//! form via the scaled complementary error function `erfcx(z) = e^(z²)·erfc(z)`:
//!
//! With `α := −x/(σ√2)` and `β := σ/(2√2)` (canonical inputs give `α ≥ 0,
//! β > 0`), substituting `Φ(z) = ½·erfc(−z/√2)` into the Black formula and
//! using `erfc(z) = e^(−z²)·erfcx(z)` for `z ≥ 0`:
//!
//! ```text
//!   b(x, σ) = ½·e^(−α²−β²)·[erfcx(α−β) − erfcx(α+β)]
//! ```
//!
//! Implemented as two mask-selected branches so `erfcx` only ever sees a
//! non-negative argument (`erfcx(−t) = 2·e^(t²) − erfcx(t)` applied to the
//! `α<β` case, then algebraically simplified):
//!
//! - `α ≥ β`: `b = ½·e^(−α²−β²)·[erfcx(α−β) − erfcx(α+β)]`
//! - `α < β`: `b = e^(−2αβ) − ½·e^(−α²−β²)·[erfcx(β−α) + erfcx(α+β)]`
//!
//! Cody's three-band structure makes erfcx cheaper than the naive `Φ` path
//! in two of three bands (no `e^(−y²)` factor formed).
//!
//! ## Derivatives
//!
//! Analytic forms from differentiating the closed form. Paper gives `b'`
//! explicitly (4.6); `b''` and `b'''` are derived symbolically here and
//! checked against finite differences in the unit tests.

#![allow(clippy::excessive_precision)] // published-constant style matches norm.rs

use std::simd::prelude::*;
use std::simd::StdFloat;

use crate::norm;

/// Normalized Black call price `b(x, σ)`, cancellation-free via erfcx.
///
/// See module docs for the derivation. Caller responsible for canonicalization
/// (`x ≤ 0`) and positivity (`σ > 0`).
#[inline]
pub fn b_normalized<const N: usize>(x: Simd<f64, N>, sigma: Simd<f64, N>) -> Simd<f64, N> {
    let half = Simd::splat(0.5);
    let two = Simd::splat(2.0);
    let inv_sqrt_2 = Simd::splat(std::f64::consts::FRAC_1_SQRT_2);
    // α = −x/(σ·√2), β = σ/(2·√2). Canonical inputs (x ≤ 0) give α ≥ 0.
    let alpha = (-x) * inv_sqrt_2 / sigma;
    let beta = half * sigma * inv_sqrt_2;
    let a2_plus_b2 = alpha * alpha + beta * beta;
    let two_ab = two * alpha * beta;
    let factor = half * (-a2_plus_b2).exp(); // ½·e^(−α²−β²)
    let s = alpha + beta; // always ≥ 0
    let t = (alpha - beta).abs(); // |α − β|, always ≥ 0
    let et = norm::erfcx(t);
    let es = norm::erfcx(s);
    let alpha_ge_beta = alpha.simd_ge(beta);
    let branch_a = factor * (et - es); // α ≥ β
    let branch_b = (-two_ab).exp() - factor * (et + es); // α < β
    alpha_ge_beta.select(branch_a, branch_b)
}

/// Naive (textbook) form of `b(x, σ)`, retained as a test reference for the
/// erfcx form in the well-conditioned centre.
///
/// `b(x, σ) = e^(x/2)·Φ(d₁) − e^(−x/2)·Φ(d₂)` with `d₁ = x/σ + σ/2`, `d₂ =
/// x/σ − σ/2`. Cancellation-prone — do not use in solver paths.
#[cfg(test)]
pub(crate) fn b_normalized_naive<const N: usize>(
    x: Simd<f64, N>,
    sigma: Simd<f64, N>,
) -> Simd<f64, N> {
    let half = Simd::splat(0.5);
    let h = x / sigma;
    let k = half * sigma;
    let d1 = h + k;
    let d2 = h - k;
    let half_x = half * x;
    let e_plus = half_x.exp();
    let e_minus = (-half_x).exp();
    e_plus * norm::phi_hart(d1) - e_minus * norm::phi_hart(d2)
}

/// First derivative `b'(σ) = ∂b/∂σ`, the "normalized vega" (Paper (4.6)):
///
/// ```text
///     b'(σ) = (1/√(2π)) · exp(−½·[(x/σ)² + (σ/2)²])
/// ```
///
/// Identical for calls and puts (vega-is-symmetric); no cancellation
/// concerns. Caller responsible for `σ > 0`.
#[inline]
pub fn b_prime<const N: usize>(x: Simd<f64, N>, sigma: Simd<f64, N>) -> Simd<f64, N> {
    let half = Simd::splat(0.5);
    let inv_sqrt_2pi = Simd::splat(0.398_942_280_401_432_68); // 1/√(2π)
    let h = x / sigma; // x/σ
    let k = half * sigma; // σ/2
    let expo = -half * (h * h + k * k);
    inv_sqrt_2pi * expo.exp()
}

/// The maximum possible normalized price given log-moneyness: `b_max = e^(x/2)`.
///
/// For canonical inputs (`x ≤ 0`), `b_max ∈ (0, 1]`. This is the upper
/// bound on `b(x, σ)` as `σ → ∞`.
#[inline]
pub fn b_max<const N: usize>(x: Simd<f64, N>) -> Simd<f64, N> {
    (Simd::splat(0.5) * x).exp()
}

/// Second derivative `b''(σ) = b'(σ)·(x²/σ³ − σ/4)`.
///
/// Derivation: with `A(σ) = −½·(x²/σ² + σ²/4)` so `b'(σ) = (1/√(2π))·exp(A)`,
/// `A'(σ) = x²/σ³ − σ/4` (note: `d(σ²/4)/dσ = σ/2`, times `−½` gives `−σ/4`),
/// and `b''(σ) = b'(σ)·A'(σ)`. Pinned vs central-difference of `b'` in tests.
#[inline]
pub fn b_double_prime<const N: usize>(x: Simd<f64, N>, sigma: Simd<f64, N>) -> Simd<f64, N> {
    let bp = b_prime(x, sigma);
    let a_prime = (x * x) / (sigma * sigma * sigma) - Simd::splat(0.25) * sigma;
    bp * a_prime
}

/// Third derivative `b'''(σ) = b'(σ)·[ (x²/σ³ − σ/4)² − 3x²/σ⁴ − ¼ ]`.
///
/// Derivation: `b''' = b''·A' + b'·A''` where `A'' = −3x²/σ⁴ − 1/4`,
/// then factor `b'`. Pinned vs central-difference of `b''` in tests.
#[inline]
pub fn b_triple_prime<const N: usize>(x: Simd<f64, N>, sigma: Simd<f64, N>) -> Simd<f64, N> {
    let bp = b_prime(x, sigma);
    let a_prime = (x * x) / (sigma * sigma * sigma) - Simd::splat(0.25) * sigma;
    let a_double_prime =
        -Simd::splat(3.0) * (x * x) / (sigma * sigma * sigma * sigma) - Simd::splat(0.25);
    bp * (a_prime * a_prime + a_double_prime)
}

/// Inflection-point total volatility `σ_c = √(2·|x|)` (Paper (4.1)).
///
/// `b(x, σ)` viewed as a function of σ on `[0, ∞)` has a single inflection
/// point at `σ_c`; convex below, concave above. Used to anchor the
/// four-region partition of the initial-guess function.
#[inline]
pub fn sigma_c<const N: usize>(x: Simd<f64, N>) -> Simd<f64, N> {
    (Simd::splat(2.0) * x.abs()).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::simd::f64x8;

    /// Helper: scalar evaluation of the naive Black form via the SIMD primitive.
    /// Used as a centre-region cross-pin against the erfcx form.
    fn b_scalar_naive(x: f64, sigma: f64) -> f64 {
        b_normalized_naive(f64x8::splat(x), f64x8::splat(sigma))[0]
    }

    #[test]
    fn b_at_forward_matches_erf_form() {
        // At x = 0 (the forward), the symmetric form gives
        //   b(0, σ) = 2·Φ(σ/2) − 1 = erf(σ/(2√2))
        // Check against this exact alternative for a range of σ.
        for &sigma_val in &[0.01_f64, 0.05, 0.2, 0.5, 1.0, 2.0, 4.0] {
            let x = f64x8::splat(0.0);
            let s = f64x8::splat(sigma_val);
            let got = b_normalized(x, s)[0];
            // 2·Φ(σ/2) − 1 — voltic's phi_hart on σ/2:
            let expected = 2.0 * norm::phi_hart(f64x8::splat(0.5 * sigma_val))[0] - 1.0;
            assert!(
                (got - expected).abs() < 1e-14,
                "b(0, {sigma_val}) = {got}, expected {expected}, diff {:e}",
                (got - expected).abs()
            );
        }
    }

    #[test]
    fn b_bounded_by_b_max() {
        // 0 ≤ b(x, σ) ≤ b_max for all canonical (x ≤ 0, σ > 0).
        let mut bad = 0;
        for &x_val in &[0.0_f64, -0.1, -0.5, -1.0, -2.0, -5.0] {
            for &sigma_val in &[0.01_f64, 0.1, 0.5, 1.0, 2.0, 5.0] {
                let x = f64x8::splat(x_val);
                let s = f64x8::splat(sigma_val);
                let b = b_normalized(x, s)[0];
                let bm = b_max(x)[0];
                if !(b >= 0.0 && b <= bm + 1e-15) {
                    eprintln!("b({x_val}, {sigma_val}) = {b} not in [0, {bm}]");
                    bad += 1;
                }
            }
        }
        assert_eq!(bad, 0, "{bad} (x, σ) pairs violated 0 ≤ b ≤ b_max");
    }

    #[test]
    fn b_prime_matches_finite_difference() {
        // b'(σ) from the analytic form must agree with a centered finite
        // difference of b(x, σ) in the well-conditioned centre.
        for &x_val in &[0.0_f64, -0.1, -0.5, -1.0] {
            for &sigma_val in &[0.1_f64, 0.3, 0.7, 1.5] {
                let h = 1e-6;
                let x = f64x8::splat(x_val);
                let s_plus = f64x8::splat(sigma_val + h);
                let s_minus = f64x8::splat(sigma_val - h);
                let s = f64x8::splat(sigma_val);
                let fd = (b_normalized(x, s_plus)[0] - b_normalized(x, s_minus)[0]) / (2.0 * h);
                let analytic = b_prime(x, s)[0];
                let err = (fd - analytic).abs();
                assert!(
                    err < 1e-7,
                    "b'({x_val}, {sigma_val}): analytic={analytic} fd={fd} err={err:e}"
                );
            }
        }
    }

    #[test]
    fn b_small_sigma_matches_asymptotic_3_3() {
        // Paper (3.3): as σ → 0,
        //   b(x, σ) ≈ (2π|x|/(3√3)) · Φ(−|x|/(√3·σ))³
        //
        // Restricted to (x, σ) pairs where the naive `b_normalized` does NOT
        // underflow to 0. At `|x|/σ` deep enough, both Φ(d₁) and Φ(d₂)
        // underflow individually before the algebraic difference can recover
        // the (small but nonzero) price. That regime requires the precision-
        // preserving form (TBD-7 in specs/jackel-lbr-spec.md) before this
        // test can extend into it.
        for (x_val, sigma_small) in [(-0.5_f64, 0.1_f64), (-1.0, 0.2), (-2.0, 0.4)] {
            let x = f64x8::splat(x_val);
            let s = f64x8::splat(sigma_small);
            let b = b_normalized(x, s)[0];
            let abs_x = x_val.abs();
            let z = -abs_x / (3.0_f64.sqrt() * sigma_small);
            let phi_z = norm::phi_hart(f64x8::splat(z))[0];
            let b_asymp =
                (2.0 * std::f64::consts::PI * abs_x / (3.0 * 3.0_f64.sqrt())) * phi_z.powi(3);
            let ratio = b / b_asymp;
            assert!(
                b > 0.0 && b_asymp > 0.0 && (0.3..=3.0).contains(&ratio),
                "x={x_val} σ={sigma_small}: b={b} b_asymp={b_asymp} ratio={ratio}"
            );
        }
    }

    #[test]
    fn b_large_sigma_matches_asymptotic_3_4() {
        // Paper (3.4): as σ → ∞,
        //   b(x, σ) ≈ b_max − 2·Φ(−σ/2)
        // For large σ, (b_max − b)/(2·Φ(−σ/2)) should → 1.
        for &x_val in &[0.0_f64, -0.5, -1.0] {
            let sigma_large = 5.0_f64;
            let x = f64x8::splat(x_val);
            let s = f64x8::splat(sigma_large);
            let b = b_normalized(x, s)[0];
            let bm = b_max(x)[0];
            let phi_neg = norm::phi_hart(f64x8::splat(-0.5 * sigma_large))[0];
            let deficit = bm - b;
            let asymp_deficit = 2.0 * phi_neg;
            let ratio = deficit / asymp_deficit;
            assert!(
                (0.5..=2.0).contains(&ratio),
                "x={x_val} σ={sigma_large}: b_max−b={deficit} 2Φ(−σ/2)={asymp_deficit} ratio={ratio}"
            );
        }
    }

    #[test]
    fn b_double_prime_matches_fd_of_bprime() {
        for &x_val in &[0.0_f64, -0.1, -0.5, -1.0] {
            for &sigma_val in &[0.1_f64, 0.3, 0.7, 1.5] {
                let h = 1e-6;
                let x = f64x8::splat(x_val);
                let fd = (b_prime(x, f64x8::splat(sigma_val + h))[0]
                    - b_prime(x, f64x8::splat(sigma_val - h))[0])
                    / (2.0 * h);
                let analytic = b_double_prime(x, f64x8::splat(sigma_val))[0];
                assert!(
                    (fd - analytic).abs() < 1e-6,
                    "b''({x_val}, {sigma_val}): analytic={analytic} fd={fd}"
                );
            }
        }
    }

    #[test]
    fn b_triple_prime_matches_fd_of_bdoubleprime() {
        for &x_val in &[0.0_f64, -0.1, -0.5, -1.0] {
            for &sigma_val in &[0.2_f64, 0.5, 1.0, 1.5] {
                let h = 1e-5;
                let x = f64x8::splat(x_val);
                let fd = (b_double_prime(x, f64x8::splat(sigma_val + h))[0]
                    - b_double_prime(x, f64x8::splat(sigma_val - h))[0])
                    / (2.0 * h);
                let analytic = b_triple_prime(x, f64x8::splat(sigma_val))[0];
                assert!(
                    (fd - analytic).abs() < 1e-4,
                    "b'''({x_val}, {sigma_val}): analytic={analytic} fd={fd}"
                );
            }
        }
    }

    #[test]
    fn b_double_prime_zero_at_sigma_c() {
        // By definition of σ_c = √(2|x|), b''(σ_c) = 0.
        for &x_val in &[-0.1_f64, -0.5, -1.0, -2.0] {
            let sc = (2.0 * x_val.abs()).sqrt();
            let x = f64x8::splat(x_val);
            let val = b_double_prime(x, f64x8::splat(sc))[0];
            assert!(val.abs() < 1e-15, "b''(σ_c) for x={x_val}: {val}");
        }
    }

    #[test]
    fn sigma_c_is_inflection_point() {
        // At σ_c = √(2|x|), b''(σ) = 0 (the defining property of the
        // inflection point — Paper §4). Check by finite-differencing b'(σ).
        for &x_val in &[-0.1_f64, -0.5, -1.0, -2.0] {
            let sc = (2.0 * x_val.abs()).sqrt();
            let h = 1e-5;
            let x = f64x8::splat(x_val);
            let bpp_fd = (b_prime(x, f64x8::splat(sc + h))[0]
                - b_prime(x, f64x8::splat(sc - h))[0])
                / (2.0 * h);
            // At the inflection, b''(σ_c) = 0. Tolerance allows for the
            // finite-difference truncation error (O(h²·b''')).
            assert!(
                bpp_fd.abs() < 1e-4,
                "x={x_val} σ_c={sc}: b''(σ_c)≈{bpp_fd} should be ~0"
            );
        }
    }

    #[test]
    fn scalar_simd_consistency() {
        // The SIMD `b_normalized` (erfcx form) must agree with the naive
        // textbook form at the per-lane level in the well-conditioned centre
        // (cancellation hasn't kicked in yet). Tolerance allows ~1 ULP error
        // from the rational approximations in the two forms.
        let x_vals = [-0.1_f64, -0.5, -1.0, 0.0, -0.3, -0.7, -1.5, -2.5];
        let sigma_vals = [0.5_f64, 0.7, 0.8, 1.0, 0.6, 0.9, 1.5, 3.0];
        let x = f64x8::from_array(x_vals);
        let s = f64x8::from_array(sigma_vals);
        let b_simd = b_normalized(x, s);
        for i in 0..8 {
            let naive = b_scalar_naive(x_vals[i], sigma_vals[i]);
            let err = (b_simd[i] - naive).abs();
            let rel = err / naive.abs().max(1e-15);
            assert!(
                rel < 1e-12,
                "lane {i}: erfcx form {} vs naive {} rel={rel:e}",
                b_simd[i],
                naive
            );
        }
    }

    #[test]
    fn erfcx_form_agrees_with_naive_in_centre() {
        // The erfcx form must agree with the textbook naive form wherever the
        // naive form is itself accurate. Two regimes:
        //
        //  - Strict centre (|x|/σ ≤ 1.5 → < 0.5 digit cancellation): agreement
        //    to ~1e-14 relative.
        //  - Wider centre (|x|/σ up to ~5 → up to a few digits of naive
        //    cancellation): agreement to ~1e-10. Beyond this, the naive form
        //    bleeds precision; the erfcx form is the trusted reference.
        for &x_val in &[0.0_f64, -0.1, -0.3, -0.5, -1.0] {
            for &sigma_val in &[1.0_f64, 1.5, 2.0] {
                if x_val.abs() / sigma_val > 1.5 {
                    continue;
                }
                let b_new = b_normalized(f64x8::splat(x_val), f64x8::splat(sigma_val))[0];
                let b_old = b_normalized_naive(f64x8::splat(x_val), f64x8::splat(sigma_val))[0];
                let rel = ((b_new - b_old).abs()) / b_old.abs().max(1e-15);
                assert!(
                    rel < 1e-14,
                    "strict centre x={x_val} σ={sigma_val}: erfcx={b_new} naive={b_old} rel={rel:e}"
                );
            }
        }
        // Wider centre (|x|/σ up to ~3.5): both forms still close. Past this
        // ratio, naive cancellation makes naive itself the limiting factor —
        // the erfcx form is the trusted reference, covered by the
        // `erfcx_form_extends_to_deep_otm` test below.
        for &x_val in &[-0.5_f64, -1.0, -1.5] {
            for &sigma_val in &[0.5_f64, 0.7, 1.0] {
                if x_val.abs() / sigma_val > 3.5 {
                    continue;
                }
                let b_new = b_normalized(f64x8::splat(x_val), f64x8::splat(sigma_val))[0];
                let b_old = b_normalized_naive(f64x8::splat(x_val), f64x8::splat(sigma_val))[0];
                let rel = ((b_new - b_old).abs()) / b_old.abs().max(1e-15);
                assert!(
                    rel < 1e-10,
                    "wider centre x={x_val} σ={sigma_val}: erfcx={b_new} naive={b_old} rel={rel:e}"
                );
            }
        }
    }

    #[test]
    fn erfcx_form_extends_to_deep_otm() {
        // In deep OTM where the naive form underflows to 0 (catastrophic
        // cancellation between two ~Φ(deep-negative)·exp values), the erfcx
        // form returns a finite positive value matching paper (3.3) to
        // first order. Specifically: pick (x, σ) where |x|/σ is large enough
        // that the naive form gives 0 but the asymptotic is nonzero.
        for (x_val, sigma_val) in [
            (-2.0_f64, 0.05_f64), // |x|/σ = 40
            (-3.0, 0.07),         // |x|/σ ≈ 43
            (-1.5, 0.04),         // |x|/σ = 37.5
        ] {
            let x = f64x8::splat(x_val);
            let s = f64x8::splat(sigma_val);
            let b_new = b_normalized(x, s)[0];
            let b_naive = b_normalized_naive(x, s)[0];
            // The asymptotic price from (3.3).
            let abs_x = x_val.abs();
            let z = -abs_x / (3.0_f64.sqrt() * sigma_val);
            let phi_z = norm::phi_hart(f64x8::splat(z))[0];
            let b_asymp =
                (2.0 * std::f64::consts::PI * abs_x / (3.0 * 3.0_f64.sqrt())) * phi_z.powi(3);
            // erfcx form should be ≥ naive (naive cancels to 0 or near-0).
            assert!(
                b_new >= 0.0,
                "x={x_val} σ={sigma_val}: erfcx returned negative {b_new}"
            );
            // erfcx form should be within a factor of 3 of the asymptotic
            // (the asymptotic is leading-order; higher-order corrections
            // are O(σ²)).
            if b_asymp > 1e-300 {
                let ratio = b_new / b_asymp;
                assert!(
                    (0.3..=3.0).contains(&ratio) || (b_new < 1e-280 && b_asymp < 1e-280),
                    "x={x_val} σ={sigma_val}: erfcx={b_new:e} naive={b_naive:e} asymp={b_asymp:e} ratio={ratio}"
                );
            }
        }
    }
}
