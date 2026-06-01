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
#![allow(improper_ctypes)] // Sleef SIMD FFI: vector types are not Rust-FFI-safe by spec

use std::simd::prelude::*;
use std::simd::StdFloat;

// ---------------------------------------------------------------------------
// SLEEF AVX-512 vectorized exp/log bindings.
// `Sleef_expd8_u10avx512f` / `Sleef_logd8_u10avx512f` operate on a single
// `__m512d` (8 × f64). `std::simd::Simd<f64, 8>` on AVX-512 x86_64 has the
// same layout/ABI as `__m512d`, so we transmute at the boundary.
// ---------------------------------------------------------------------------
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use core::arch::x86_64::__m512d;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
extern "C" {
    fn Sleef_expd8_u10avx512f(a: __m512d) -> __m512d;
    fn Sleef_logd8_u10avx512f(a: __m512d) -> __m512d;
}

/// Vectorized exp on `Simd<f64, 8>` via SLEEF (u10 ≈ 1 ulp).
#[inline(always)]
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub fn vexp_f64x8(x: Simd<f64, 8>) -> Simd<f64, 8> {
    unsafe {
        let v: __m512d = core::mem::transmute(x);
        let r = Sleef_expd8_u10avx512f(v);
        core::mem::transmute::<__m512d, Simd<f64, 8>>(r)
    }
}

/// Vectorized ln on `Simd<f64, 8>` via SLEEF (u10 ≈ 1 ulp).
#[inline(always)]
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub fn vlog_f64x8(x: Simd<f64, 8>) -> Simd<f64, 8> {
    unsafe {
        let v: __m512d = core::mem::transmute(x);
        let r = Sleef_logd8_u10avx512f(v);
        core::mem::transmute::<__m512d, Simd<f64, 8>>(r)
    }
}

/// Fallback for non-AVX-512 builds: delegate to the per-lane libc exp.
#[inline(always)]
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
pub fn vexp_f64x8(x: Simd<f64, 8>) -> Simd<f64, 8> {
    x.exp()
}

/// Fallback for non-AVX-512 builds: delegate to the per-lane libc ln.
#[inline(always)]
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
pub fn vlog_f64x8(x: Simd<f64, 8>) -> Simd<f64, 8> {
    x.ln()
}

/// Generic SIMD exp: routes f64x8 on AVX-512 to SLEEF, otherwise falls back
/// to `.exp()` (std::simd per-lane libc dispatch). The N==8 branch is a
/// compile-time const fold, so non-8 monomorphizations have zero runtime
/// dispatch cost.
#[inline(always)]
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub fn vexp<const N: usize>(x: Simd<f64, N>) -> Simd<f64, N> {
    if N == 8 {
        unsafe {
            let v: __m512d = core::mem::transmute_copy(&x);
            let r = Sleef_expd8_u10avx512f(v);
            let out: Simd<f64, 8> = core::mem::transmute(r);
            core::mem::transmute_copy::<Simd<f64, 8>, Simd<f64, N>>(&out)
        }
    } else {
        x.exp()
    }
}

#[inline(always)]
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub fn vlog<const N: usize>(x: Simd<f64, N>) -> Simd<f64, N> {
    if N == 8 {
        unsafe {
            let v: __m512d = core::mem::transmute_copy(&x);
            let r = Sleef_logd8_u10avx512f(v);
            let out: Simd<f64, 8> = core::mem::transmute(r);
            core::mem::transmute_copy::<Simd<f64, 8>, Simd<f64, N>>(&out)
        }
    } else {
        x.ln()
    }
}

#[inline(always)]
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
pub fn vexp<const N: usize>(x: Simd<f64, N>) -> Simd<f64, N> {
    x.exp()
}

#[inline(always)]
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
pub fn vlog<const N: usize>(x: Simd<f64, N>) -> Simd<f64, N> {
    x.ln()
}

/// 1/√(2π).
const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_68; // 1/√(2π)
/// √(2π).
const SQRT_2PI: f64 = 2.506_628_274_631_000_5; // √(2π)
/// 1/√2.
const INV_SQRT_2: f64 = std::f64::consts::FRAC_1_SQRT_2; // 1/√2

/// Standard-normal density φ(x) = (1/√(2π)) e^{-x²/2}.
#[inline]
pub fn phi_pdf<const N: usize>(x: Simd<f64, N>) -> Simd<f64, N> {
    Simd::splat(INV_SQRT_2PI) * vexp(Simd::splat(-0.5) * x * x)
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
    let expo = vexp(Simd::splat(-0.5) * z * z);
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
    let expo = vexp(Simd::splat(-0.5) * z * z);

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
        vexp(Simd::splat(-1.0) * ysq) * r
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
        vexp(Simd::splat(-1.0) * ysq) * r
    };

    let b1 = y.simd_le(Simd::splat(0.5));
    let b3 = y.simd_gt(Simd::splat(4.0));
    b1.select(erfc_b1, b3.select(erfc_b3, erfc_b2))
}

/// Internal: compute both the merged B2/B3 erfcx value AND the raw B1 rational
/// `(1 − erf(y))` WITHOUT yet applying the `exp(y²)` factor.
///
/// Returns `(b23_value, b1_raw)` where:
/// - `b23_value` = the B2/B3 merged erfcx selection (correct for `y > 0.5`,
///   meaningless on B1 lanes but cheap to compute branch-free).
/// - `b1_raw` = `1 − erf(y)` from the Band-1 rational (no exp applied).
///
/// The public `erfcx(y)` wrapper combines them with an `exp(y²)` and the
/// `y ≤ 0.5` mask. The fused `ig_surv_from_uv` instead uses `b1_raw`
/// directly, because the surrounding `e^(−u²)·e^(u²)·R1 = R1` cancellation
/// removes the exp entirely on B1 lanes (Research D fusion B1).
#[inline]
fn erfcx_split_b1exp<const N: usize>(
    y: Simd<f64, N>,
    ysq: Simd<f64, N>,
) -> (Simd<f64, N>, Simd<f64, N>) {
    // Band 1 raw: (1 − erf(y)) without the e^(y²) factor.
    let b1_raw = {
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

    let erfcx_b2 = {
        let mut xnum = Simd::splat(cody::C[8]) * y;
        let mut xden = y;
        let mut i = 0;
        while i < 7 {
            xnum = (xnum + Simd::splat(cody::C[i])) * y;
            xden = (xden + Simd::splat(cody::D[i])) * y;
            i += 1;
        }
        (xnum + Simd::splat(cody::C[7])) / (xden + Simd::splat(cody::D[7]))
    };

    let erfcx_b3 = {
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
        (Simd::splat(cody::SQRPI) - r) / y
    };

    let b3 = y.simd_gt(Simd::splat(4.0));
    let b23 = b3.select(erfcx_b3, erfcx_b2);
    (b23, b1_raw)
}

/// Scaled complementary error function `erfcx(y) = e^(y²)·erfc(y)`, for `y ≥ 0`.
///
/// The Jäckel rational solver's precision-preserving normalized Black price
/// (`black::b_normalized`) needs `erfcx` because the cancellation in the
/// direct form `Φ(d₁) − Φ(d₂)` kills mantissa bits in the centre. Cody's
/// rational structure gives `erfcx` essentially for free in two of three
/// bands — the `e^(−y²)` factor that `erfc` carries simply isn't formed.
///
/// Branch-free over Cody's three bands. Caller responsible for `y ≥ 0`;
/// negative arguments must be handled via `erfcx(−y) = 2·e^(y²) − erfcx(y)`.
#[inline]
pub fn erfcx<const N: usize>(y_in: Simd<f64, N>) -> Simd<f64, N> {
    // erfcx(y) = e^(y²)·erfc(y). For Cody's three bands:
    //   Band 1 (y ≤ 0.5): erfc = 1 − erf,  so erfcx = e^(y²)·(1 − erf). Needs exp.
    //   Band 2 (0.5 < y ≤ 4): Cody's form is (C/D)·e^(−y²), so erfcx = C/D directly. No exp.
    //   Band 3 (y > 4):       Cody's form is asymp·e^(−y²)/y, so erfcx = asymp/y. No exp.
    let y = y_in.abs();
    let ysq = y * y;
    let (b23, b1_raw) = erfcx_split_b1exp(y, ysq);
    let erfcx_b1 = vexp(ysq) * b1_raw;
    let b1 = y.simd_le(Simd::splat(0.5));
    b1.select(erfcx_b1, b23)
}

/// Fused IG survival `S(x; μ) = ½·e^(−u²)·[erfcx(u) − erfcx(v)]` in one shot.
///
/// Three analytic fusions (Research D, 2026-05-31) collapse dead `exp` work:
/// - **F3**: on `u < 0` lanes the reflection `erfcx(−|u|) = 2·e^(u²) − erfcx(|u|)`
///   has its `e^(u²)` cancel analytically with the outer `½·e^(−u²)`.
/// - **B1**: on `|u| ≤ 0.5` lanes the Cody Band-1 form
///   `erfcx(|u|) = e^(u²)·(1 − erf(|u|))` has its internal `e^(u²)` cancel
///   with the outer `e^(−u²)` — we use the raw `(1 − erf)` directly.
/// - **Reciprocal**: folding the reflection into the bracket means only
///   `e^(−u²)` is needed (one exp per lane, period — no second `e^(+u²)`).
///
/// Caller guarantees `v ≥ 0`. `u` may be signed.
#[inline]
pub fn ig_surv_from_uv<const N: usize>(u: Simd<f64, N>, v: Simd<f64, N>) -> Simd<f64, N> {
    let abs_u = u.abs();
    let u2 = u * u;

    // Split erfcx(|u|) into B2/B3 value and B1 raw (1 − erf), no exp on B1.
    let (e_u_b23, r1_u) = erfcx_split_b1exp(abs_u, u2);
    let is_b1_u = abs_u.simd_le(Simd::splat(0.5));

    // v ≥ 0 always (caller contract); full erfcx path.
    let erfcx_v = erfcx(v);

    // One exp per lane, period.
    let em_u2 = vexp(-u2);

    // B1 fusion: e^(−u²)·erfcx(|u|) = e^(−u²)·e^(u²)·R1 = R1  on B1 lanes.
    //            e^(−u²)·erfcx(|u|) = e^(−u²)·B23           on B2/B3 lanes.
    let term_u = is_b1_u.select(r1_u, em_u2 * e_u_b23);
    let term_v = em_u2 * erfcx_v;

    let half = Simd::splat(0.5);
    // Positive-u branch: S = ½·e^(−u²)·[erfcx(u) − erfcx(v)] = ½·(term_u − term_v).
    let surv_pos = half * (term_u - term_v);
    // Negative-u branch: erfcx(u) = 2·e^(u²) − erfcx(|u|); the ½·e^(−u²)·2·e^(u²)
    // collapses analytically to 1, giving S = 1 − ½·(term_u + term_v).
    let surv_neg = Simd::splat(1.0) - half * (term_u + term_v);

    u.is_sign_negative().select(surv_neg, surv_pos)
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

// ---------------------------------------------------------------------------
// Acklam 2003 — inverse cumulative normal Φ⁻¹(p), the probit (~1e-9), refined
// to full f64 with two Halley steps against the Hart Φ above.
// ---------------------------------------------------------------------------
//
// Peter Acklam, "An algorithm for computing the inverse normal cumulative
// distribution function" (2003): a single low-order rational in the central
// region `p ∈ [low, 1−low]` and a `√(−2 ln·)` rational in each tail, mirrored
// by sign. Branch-free here: all three arms are evaluated and `select`ed on
// the region mask, so a lane-packed batch never pays for one lane being in a
// tail. The bare rational is ~1.15e-9 relative; two Halley correction steps
// (using [`phi_hart`] for the residual and [`phi_pdf`] for the slope) take it
// to ~1e-15, which is what the Schadner explicit solver needs for its
// at-the-forward branch and its initial guess. Two steps (not one) mirror the
// reference `ndtri` in Schadner's demo (`wol-fi/direct_vola`).
#[allow(clippy::unreadable_literal, clippy::excessive_precision)]
mod acklam {
    pub const LOW: f64 = 0.02425; // central-region boundary (HIGH = 1 − LOW)
    pub const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    pub const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    pub const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    pub const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
}

/// Inverse standard-normal CDF (probit) Φ⁻¹(p) for `p ∈ (0, 1)`. Branch-free;
/// Acklam's rational refined by two Halley steps to ~1e-15. Outside `(0, 1)`
/// the inputs are clamped to a tiny interior margin, so a caller that has
/// already screened its probabilities never sees `±∞`.
#[inline]
pub fn phi_inv<const N: usize>(p_in: Simd<f64, N>) -> Simd<f64, N> {
    // Clamp into the open interval so ln() of a tail arm is always finite even
    // for a lane the caller will end up rejecting anyway.
    let tiny = Simd::splat(1e-300);
    let p = p_in
        .simd_max(tiny)
        .simd_min(Simd::splat(1.0) - Simd::splat(f64::EPSILON));

    let a = &acklam::A;
    let b = &acklam::B;
    let c = &acklam::C;
    let d = &acklam::D;

    // Central arm: q = p − ½, r = q²,  x = poly_a(r)·q / poly_b(r).
    let qc = p - Simd::splat(0.5);
    let rc = qc * qc;
    let num_c = ((((Simd::splat(a[0]) * rc + Simd::splat(a[1])) * rc + Simd::splat(a[2])) * rc
        + Simd::splat(a[3]))
        * rc
        + Simd::splat(a[4]))
        * rc
        + Simd::splat(a[5]);
    let den_c = ((((Simd::splat(b[0]) * rc + Simd::splat(b[1])) * rc + Simd::splat(b[2])) * rc
        + Simd::splat(b[3]))
        * rc
        + Simd::splat(b[4]))
        * rc
        + Simd::splat(1.0);
    let x_central = num_c * qc / den_c;

    // Tail arm, written for the lower tail; the upper tail is the negation with
    // `1 − p` in place of `p`. Evaluate one shared rational on
    // `q = √(−2 ln(min(p, 1−p)))` and flip its sign for the upper tail.
    let lower = p.simd_lt(Simd::splat(acklam::LOW));
    let p_tail = lower.select(p, Simd::splat(1.0) - p);
    let qt = (Simd::splat(-2.0) * vlog(p_tail)).sqrt();
    let num_t = (((((Simd::splat(c[0]) * qt + Simd::splat(c[1])) * qt + Simd::splat(c[2])) * qt
        + Simd::splat(c[3]))
        * qt
        + Simd::splat(c[4]))
        * qt
        + Simd::splat(c[5]))
        * Simd::splat(1.0);
    let den_t = (((Simd::splat(d[0]) * qt + Simd::splat(d[1])) * qt + Simd::splat(d[2])) * qt
        + Simd::splat(d[3]))
        * qt
        + Simd::splat(1.0);
    // Acklam's tail rational is already negative for a small probability; the
    // upper tail is its negation (with `1 − p` fed through the shared arm).
    let raw = num_t / den_t;
    let x_tail = lower.select(raw, -raw);

    let in_central = p.simd_ge(Simd::splat(acklam::LOW))
        & p.simd_le(Simd::splat(1.0) - Simd::splat(acklam::LOW));
    let mut x = in_central.select(x_central, x_tail);

    // Two Halley steps on F(x) = Φ(x) − p, F'(x) = φ(x): with e = Φ(x) − p and
    // u = e/φ(x),  x ← x − u / (1 + x·u/2). Lifts the ~1e-9 rational to ~1e-15.
    // Two iterations (not one) to match the reference `ndtri` in Schadner's
    // demo (`wol-fi/direct_vola`), so the explicit solver's probit arm is
    // faithful to the method it is benchmarked against.
    for _ in 0..2 {
        let e = phi_hart(x) - p;
        let u = e / phi_pdf(x);
        x = x - u / (Simd::splat(1.0) + Simd::splat(0.5) * x * u);
    }
    x
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
    fn phi_inv_recovers_reference() {
        // Φ⁻¹(Φ(x)) == x to ~1e-12 over the reference grid (the tails too).
        for &(x, p) in REF {
            if !(1e-12..=1.0 - 1e-12).contains(&p) {
                continue; // outside the probit's representable interior
            }
            let got = phi_inv(f64x8::splat(p))[0];
            approx_eq(got, x, 1e-10);
        }
    }

    #[test]
    fn phi_inv_is_inverse_of_phi() {
        // Round-trip the other way: Φ(Φ⁻¹(p)) == p across the central + tail
        // regions, including right at Acklam's region boundary.
        for p in [
            1e-9_f64,
            1e-4,
            0.02425,
            0.05,
            0.2,
            0.5,
            0.7,
            0.97575,
            0.9999,
            1.0 - 1e-9,
        ] {
            let x = phi_inv(f64x8::splat(p))[0];
            approx_eq(phi_hart(f64x8::splat(x))[0], p, 1e-12);
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
    fn erfcx_matches_identity_in_centre() {
        // erfcx(y) = e^(y²)·erfc(y) — check against this identity for moderate
        // y (where e^(y²)·erfc(y) doesn't underflow/overflow).
        for y in [0.0_f64, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0] {
            let got = erfcx(f64x8::splat(y))[0];
            // erfc via the existing path (cody_erfc) then scale.
            let erfc_y = cody_erfc(f64x8::splat(y))[0];
            let expected = (y * y).exp() * erfc_y;
            assert!(
                (got - expected).abs() < 1e-13 * expected.max(1e-12),
                "erfcx({y}): got {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn erfcx_large_argument_asymptotic() {
        // For large y: erfcx(y) ≈ 1/(y·√π).
        let one_over_sqrt_pi = 0.564_189_583_547_756_3_f64;
        for y in [5.0_f64, 8.0, 12.0, 20.0] {
            let got = erfcx(f64x8::splat(y))[0];
            let asymp = one_over_sqrt_pi / y;
            // First-order error is O(1/y²), so relative err ≈ 1/(2y²).
            let rel = (got - asymp).abs() / asymp;
            assert!(
                rel < 0.5 / (y * y),
                "erfcx({y}): {got} vs asymp {asymp} rel={rel}"
            );
        }
    }

    #[test]
    fn erfcx_at_zero_is_one() {
        // erfcx(0) = e^0 · erfc(0) = 1·1 = 1.
        let got = erfcx(f64x8::splat(0.0))[0];
        assert!((got - 1.0).abs() < 1e-15, "erfcx(0) = {got}, expected 1");
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
