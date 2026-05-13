//! Cumulative-normal kernels Φ(x) = P[N(0,1) ≤ x] and the density φ(x).
//!
//! Φ is evaluated twice per Newton iteration (for d1 and d2), so it is the
//! inner-inner loop of the solver. Every variant here is **branch-free** — a
//! region split is a `mask.select(...)`, never an `if`, so a lane-packed batch
//! never pays for a divergent branch when one lane is in the tail.
//!
//! Three rational approximations are implemented so the accuracy/throughput
//! frontier can be measured (`cargo run --release --bin bench`, then
//! `scripts/plot_phi.py`):
//!
//! | kernel        | provenance                          | abs error | cost |
//! |---------------|-------------------------------------|-----------|------|
//! | [`phi_as`]    | Abramowitz & Stegun 26.2.17 (1964)  | ~7.5e-8   | low  |
//! | [`phi_hart`]  | Hart, *Computer Approximations* 5666 (1968) | ~1e-15 | mid |
//! | [`phi_west`]  | West, *Wilmott* 2009 (Hart 5666 + tail arm) | ~1e-15 | mid |
//! | [`phi_cody`]  | Cody, rational-Chebyshev erfc, *Math.Comp.* 1969 | ~1e-18 | high |
//!
//! The solver uses [`phi_hart`]. Measured on a Zen 5 AVX-512 core: ~8e-9 max
//! *relative* error over a 41-point grid x ∈ [−8, 8] — far below the ~1e-6
//! conditioning floor of implied vol — at ~4.3 ns/call, the fastest of the
//! three accurate kernels. West (a near-clone that adds a continued-fraction
//! tail arm) measures slightly slower for no accuracy gain on this hardware;
//! Cody buys ~50× better relative error at ~1.6× the cost — accuracy the IV
//! problem cannot use; AS's ~1e-2 relative error in the deep wing is coarser
//! than the floor. (See the README "cumulative normal kernel choice" and
//! `scripts/plot_phi.py` for the frontier.)

#![allow(clippy::excessive_precision)] // published kernel coefficients are written as published

use std::simd::prelude::*;
use std::simd::StdFloat;

/// 1/√(2π).
const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_68; // 1/√(2π)
/// √(2π).
const SQRT_2PI: f64 = 2.506_628_274_631_000_5; // √(2π)
/// 1/√2.
const INV_SQRT_2: f64 = std::f64::consts::FRAC_1_SQRT_2; // 1/√2

/// Standard-normal density φ(x) = (1/√(2π)) e^{-x²/2}.
#[inline]
pub fn phi_pdf<const N: usize>(x: Simd<f64, N>) -> Simd<f64, N> {
    Simd::splat(INV_SQRT_2PI) * (Simd::splat(-0.5) * x * x).exp()
}

// ---------------------------------------------------------------------------
// Abramowitz & Stegun 26.2.17 — the fast/coarse anchor (~7.5e-8).
// ---------------------------------------------------------------------------
//
// Φ(x) ≈ 1 - φ(x)·(b1 t + b2 t² + … + b5 t⁵), t = 1/(1 + p|x|), mirrored for
// x < 0. Five mul-adds + one reciprocal + one exp. Branch-free already (only a
// sign select). Too coarse for the IV conditioning floor — included only to
// anchor the cheap end of the frontier graph.
#[allow(clippy::unreadable_literal, clippy::excessive_precision)]
mod ab_steg {
    pub const P: f64 = 0.2316419;
    pub const B: [f64; 5] = [
        0.319381530,
        -0.356563782,
        1.781477937,
        -1.821255978,
        1.330274429,
    ];
}

/// Abramowitz & Stegun 26.2.17 cumulative normal. ~7.5e-8 absolute error.
#[inline]
pub fn phi_as<const N: usize>(x: Simd<f64, N>) -> Simd<f64, N> {
    let z = x.abs();
    let t = Simd::splat(1.0) / (Simd::splat(1.0) + Simd::splat(ab_steg::P) * z);
    let b = &ab_steg::B;
    // Horner in t, ascending b1..b5: ((((b5 t + b4) t + b3) t + b2) t + b1) t
    let poly = ((((Simd::splat(b[4]) * t + Simd::splat(b[3])) * t + Simd::splat(b[2])) * t
        + Simd::splat(b[1]))
        * t
        + Simd::splat(b[0]))
        * t;
    let tail = phi_pdf(z) * poly; // ≈ 1 - Φ(z) for z ≥ 0
    let neg = x.simd_lt(Simd::splat(0.0));
    let res = neg.select(tail, Simd::splat(1.0) - tail);
    res.simd_max(Simd::splat(0.0)).simd_min(Simd::splat(1.0))
}

// ---------------------------------------------------------------------------
// Hart 5666 — the classic rational (~1e-15). The arm West's algorithm builds on.
// ---------------------------------------------------------------------------
//
// 1 - Φ(z) ≈ e^{-z²/2} · Np(z) / Dp(z) for z ≥ 0, with Np degree 6 and Dp
// degree 7 (Hart, *Computer Approximations*, algorithm 5666). Mirrored for
// z < 0. Branch-free (sign select only). For |x| > 37 the result underflows to
// 0/1; we clamp.
#[allow(clippy::unreadable_literal, clippy::excessive_precision)]
mod hart {
    /// Numerator, evaluated Horner-style high→low (degree 6).
    pub const NP: [f64; 7] = [
        3.526_249_659_989_109e-2,
        0.700_383_064_443_688,
        6.373_962_203_531_65,
        33.912_866_078_383,
        112.079_291_497_871,
        221.213_596_169_931,
        220.206_867_912_376,
    ];
    /// Denominator, Horner-style high→low (degree 7).
    pub const DP: [f64; 8] = [
        8.838_834_764_831_84e-2,
        1.755_667_163_182_64,
        16.064_177_579_207,
        86.780_732_202_946_1,
        296.564_248_779_674,
        637.333_633_378_831,
        793.826_512_519_948,
        440.413_735_824_752,
    ];
}

#[inline]
fn horner6<const N: usize>(c: &[f64; 7], z: Simd<f64, N>) -> Simd<f64, N> {
    (((((Simd::splat(c[0]) * z + Simd::splat(c[1])) * z + Simd::splat(c[2])) * z
        + Simd::splat(c[3]))
        * z
        + Simd::splat(c[4]))
        * z
        + Simd::splat(c[5]))
        * z
        + Simd::splat(c[6])
}

#[inline]
fn horner7<const N: usize>(c: &[f64; 8], z: Simd<f64, N>) -> Simd<f64, N> {
    ((((((Simd::splat(c[0]) * z + Simd::splat(c[1])) * z + Simd::splat(c[2])) * z
        + Simd::splat(c[3]))
        * z
        + Simd::splat(c[4]))
        * z
        + Simd::splat(c[5]))
        * z
        + Simd::splat(c[6]))
        * z
        + Simd::splat(c[7])
}

/// Hart 5666 cumulative normal. Branch-free; ~1e-15 absolute error.
#[inline]
pub fn phi_hart<const N: usize>(x: Simd<f64, N>) -> Simd<f64, N> {
    let z = x.abs();
    let expo = (Simd::splat(-0.5) * z * z).exp();
    let tail = expo * horner6(&hart::NP, z) / horner7(&hart::DP, z); // ≈ 1 - Φ(z)
    let neg = x.simd_lt(Simd::splat(0.0));
    let res = neg.select(tail, Simd::splat(1.0) - tail);
    res.simd_max(Simd::splat(0.0)).simd_min(Simd::splat(1.0))
}

// ---------------------------------------------------------------------------
// West 2009 — Hart 5666's rational arm + a continued-fraction tail (~1e-15).
// ---------------------------------------------------------------------------
//
// Graeme West, "Better approximations to cumulative normal functions", Wilmott
// Magazine, 2005/2009. For |x| < √50 it uses Hart 5666's rational; for
// √50 ≤ |x| < 37 a 5-deep continued fraction (so the tail stays accurate where
// the rational would lose digits); |x| ≥ 37 → 0/1. All three regimes evaluated
// branch-free and `select`ed. This is the solver's kernel.
const WEST_RATIONAL_CUTOFF: f64 = 7.071_067_811_865_475; // √50 ≈ 7.0711
const WEST_ZERO_CUTOFF: f64 = 37.0;

/// West 2009 cumulative normal. Branch-free; ~1e-15 absolute error.
#[inline]
pub fn phi_west<const N: usize>(x: Simd<f64, N>) -> Simd<f64, N> {
    let z = x.abs();
    let expo = (Simd::splat(-0.5) * z * z).exp();

    // Rational arm (Hart 5666).
    let tail_rational = expo * horner6(&hart::NP, z) / horner7(&hart::DP, z);

    // Continued-fraction arm: 1-Φ(z) ≈ φ(z) / (z + 1/(z + 2/(z + 3/(z + 4/(z + 0.65)))))
    // built from the inside out — each step is a select-free reciprocal.
    let cf = {
        let mut b = z + Simd::splat(0.65);
        b = z + Simd::splat(4.0) / b;
        b = z + Simd::splat(3.0) / b;
        b = z + Simd::splat(2.0) / b;
        b = z + Simd::splat(1.0) / b;
        expo / b / Simd::splat(SQRT_2PI)
    };

    let use_rational = z.simd_lt(Simd::splat(WEST_RATIONAL_CUTOFF));
    let is_huge = z.simd_ge(Simd::splat(WEST_ZERO_CUTOFF));
    // For huge z the tail probability is 0.
    let tail = is_huge.select(Simd::splat(0.0), use_rational.select(tail_rational, cf));

    let neg = x.simd_lt(Simd::splat(0.0));
    let res = neg.select(tail, Simd::splat(1.0) - tail);
    res.simd_max(Simd::splat(0.0)).simd_min(Simd::splat(1.0))
}

// ---------------------------------------------------------------------------
// Cody 1969 — rational-Chebyshev erfc, three magnitude bands (~1e-18).
// ---------------------------------------------------------------------------
//
// W. J. Cody, "Rational Chebyshev approximation for the error function",
// Math. Comp. 23 (1969). erfc(y), y ≥ 0:
//   y ≤ 0.5      : erf via A/B (deg 4),  erfc = 1 - erf
//   0.5 < y ≤ 4  : erfc via C/D (deg 8) · e^{-y²}
//   y > 4        : erfc via P/Q asymptotic (deg 5) · e^{-y²}/y
// All three evaluated branch-free; `select` on the band. Φ(x)=½·erfc(-x/√2).
#[allow(clippy::unreadable_literal, clippy::excessive_precision)]
mod cody {
    pub const A: [f64; 5] = [
        3.16112374387056560e0,
        1.13864154151050156e2,
        3.77485237685302021e2,
        3.20937758913846947e3,
        1.85777706184603153e-1,
    ];
    pub const B: [f64; 4] = [
        2.36012909523441209e1,
        2.44024637934444173e2,
        1.28261652607737228e3,
        2.84423683343917062e3,
    ];
    pub const C: [f64; 9] = [
        5.64188496988670089e-1,
        8.88314979438837594e0,
        6.61191906371416295e1,
        2.98635138197400131e2,
        8.81952221241769090e2,
        1.71204761263407058e3,
        2.05107837782607147e3,
        1.23033935479799725e3,
        2.15311535474403846e-8,
    ];
    pub const D: [f64; 8] = [
        1.57449261107098347e1,
        1.17693950891312499e2,
        5.37181101862009858e2,
        1.62138957456669019e3,
        3.29079923573345963e3,
        4.36261909014324716e3,
        3.43936767414372164e3,
        1.23033935480374942e3,
    ];
    pub const P: [f64; 6] = [
        3.05326634961232344e-1,
        3.60344899949804439e-1,
        1.25781726111229246e-1,
        1.60837851487422766e-2,
        6.58749161529837803e-4,
        1.63153871373020978e-2,
    ];
    pub const Q: [f64; 5] = [
        2.56852019228982242e0,
        1.87295284992346047e0,
        5.27905102951428412e-1,
        6.05183413124413191e-2,
        2.33520497626869185e-3,
    ];
    /// 1/√π.
    pub const SQRPI: f64 = 5.64189583547756287e-1;
}

#[inline]
fn cody_erfc<const N: usize>(y_in: Simd<f64, N>) -> Simd<f64, N> {
    // erfc(y) for y ≥ 0.
    let y = y_in.abs();
    let ysq = y * y;

    // Band 1: erf via A/B then 1 - erf.
    let erfc_b1 = {
        let mut xnum = Simd::splat(cody::A[4]) * ysq;
        let mut xden = ysq;
        let mut i = 0;
        while i < 3 {
            xnum = (xnum + Simd::splat(cody::A[i])) * ysq;
            xden = (xden + Simd::splat(cody::B[i])) * ysq;
            i += 1;
        }
        let erf = y * (xnum + Simd::splat(cody::A[3])) / (xden + Simd::splat(cody::B[3]));
        Simd::splat(1.0) - erf
    };

    // Band 2: C/D · e^{-y²}.
    let erfc_b2 = {
        let mut xnum = Simd::splat(cody::C[8]) * y;
        let mut xden = y;
        let mut i = 0;
        while i < 7 {
            xnum = (xnum + Simd::splat(cody::C[i])) * y;
            xden = (xden + Simd::splat(cody::D[i])) * y;
            i += 1;
        }
        let r = (xnum + Simd::splat(cody::C[7])) / (xden + Simd::splat(cody::D[7]));
        (Simd::splat(-1.0) * ysq).exp() * r
    };

    // Band 3: asymptotic.
    let erfc_b3 = {
        let zinv = Simd::splat(1.0) / ysq;
        let mut xnum = Simd::splat(cody::P[5]) * zinv;
        let mut xden = zinv;
        let mut i = 0;
        while i < 4 {
            xnum = (xnum + Simd::splat(cody::P[i])) * zinv;
            xden = (xden + Simd::splat(cody::Q[i])) * zinv;
            i += 1;
        }
        let r = zinv * (xnum + Simd::splat(cody::P[4])) / (xden + Simd::splat(cody::Q[4]));
        let r = (Simd::splat(cody::SQRPI) - r) / y;
        (Simd::splat(-1.0) * ysq).exp() * r
    };

    let b1 = y.simd_le(Simd::splat(0.5));
    let b3 = y.simd_gt(Simd::splat(4.0));
    b1.select(erfc_b1, b3.select(erfc_b3, erfc_b2))
}

/// Cody 1969 cumulative normal. Branch-free; ~1e-18 absolute error.
#[inline]
pub fn phi_cody<const N: usize>(x: Simd<f64, N>) -> Simd<f64, N> {
    let t = x * Simd::splat(INV_SQRT_2); // Φ(x) = ½ erfc(-t)
    let half_erfc = Simd::splat(0.5) * cody_erfc(t.abs());
    let neg = x.simd_lt(Simd::splat(0.0));
    let res = neg.select(half_erfc, Simd::splat(1.0) - half_erfc);
    res.simd_max(Simd::splat(0.0)).simd_min(Simd::splat(1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() < tol,
            "left={a} right={b} |Δ|={} tol={tol}",
            (a - b).abs()
        );
    }

    // Reference values from a high-precision source (mpmath ncdf).
    const REF: &[(f64, f64)] = &[
        (-5.0, 2.866_515_719_235_352_1e-7),
        (-3.0, 1.349_898_031_630_094_5e-3),
        (-1.96, 2.499_789_514_822_046e-2),
        (-1.0, 1.586_552_539_314_570_5e-1),
        (-0.5, 3.085_375_387_259_868_9e-1),
        (0.0, 5.0e-1),
        (0.25, 5.987_063_256_829_237e-1),
        (1.0, 8.413_447_460_685_429e-1),
        (1.96, 9.750_021_048_517_795e-1),
        (2.0, 9.772_498_680_518_208e-1),
        (3.0, 9.986_501_019_683_699e-1),
        (5.0, 9.999_997_133_484_281e-1),
        (8.0, 9.999_999_999_999_993e-1),
    ];

    #[test]
    fn west_matches_reference() {
        for &(x, y) in REF {
            let got = phi_west(f64x8::splat(x))[0];
            approx_eq(got, y, 1e-12);
        }
    }

    #[test]
    fn hart_matches_reference() {
        for &(x, y) in REF {
            let got = phi_hart(f64x8::splat(x))[0];
            approx_eq(got, y, 1e-12);
        }
    }

    #[test]
    fn cody_matches_reference() {
        for &(x, y) in REF {
            let got = phi_cody(f64x8::splat(x))[0];
            approx_eq(got, y, 1e-14);
        }
    }

    #[test]
    fn as_matches_reference_coarsely() {
        for &(x, y) in REF {
            if x.abs() > 6.0 {
                continue; // AS is for the bulk, not the deep tail
            }
            let got = phi_as(f64x8::splat(x))[0];
            approx_eq(got, y, 1e-7);
        }
    }

    #[test]
    fn symmetry() {
        for v in [0.1_f64, 0.5, 1.0, 1.7, 2.5, 3.3, 4.1, 6.0] {
            let p = f64x8::splat(v);
            let m = f64x8::splat(-v);
            approx_eq(phi_west(p)[0] + phi_west(m)[0], 1.0, 1e-12);
            approx_eq(phi_hart(p)[0] + phi_hart(m)[0], 1.0, 1e-12);
            approx_eq(phi_cody(p)[0] + phi_cody(m)[0], 1.0, 1e-15);
        }
    }

    #[test]
    fn pdf_is_derivative_of_cdf() {
        let h = 1e-6;
        for v in [-1.5_f64, -0.3, 0.0, 0.7, 2.0] {
            let num =
                (phi_west(f64x8::splat(v + h))[0] - phi_west(f64x8::splat(v - h))[0]) / (2.0 * h);
            approx_eq(num, phi_pdf(f64x8::splat(v))[0], 1e-7);
        }
    }
}
