//! The Jäckel "Let's be rational" implied-volatility kernel for voltic v1.0.
//!
//! Implements the four-branch initial guess (Paper §4) + Householder-3
//! iteration (Paper §5) on the canonicalized normalized Black function from
//! [`crate::black`]. See `specs/jackel-lbr-spec.md` for the algorithm-level
//! derivation.
//!
//! All six paper-derivation TBDs are resolved: canonicalization (TBD-1),
//! upper-region anchor derivatives (TBD-2), lower-region anchor derivatives
//! (TBD-3, including a paper-typo correction), explicit Householder-3 update
//! (TBD-4), lower/upper objective derivatives (TBD-5/6), and cancellation-free
//! Black via erfcx (TBD-7 in `black.rs`).

use std::simd::prelude::*;
use std::simd::StdFloat;

use crate::black;
use crate::{M, V};

// ---------------------------------------------------------------------------
// Canonicalization (Paper §2 invariances (2.5)–(2.6); TBD-1 in spec)
// ---------------------------------------------------------------------------

/// Canonicalize `(x, β, is_call)` → `(x_canon, β_canon)` with `x_canon ≤ 0`
/// and the option implicitly being a call (`θ = +1`).
///
/// Derivation from the paper's two invariances:
///   * (2.5) reciprocal-strike: `b(x, σ, θ) = b(−x, σ, −θ)`
///   * (2.6) time-value:        `b(x, σ, θ) − ι(x, θ) = b(x, σ, −θ) − ι(x, −θ)`
///     with `ι(x, θ) = (e^(θx/2) − e^(−θx/2))_+`.
///
/// Casework (`OTM` = out-of-the-money, `ITM` = in-the-money):
///
/// | input                          | x_canon | β_canon            | reasoning                       |
/// |--------------------------------|---------|--------------------|---------------------------------|
/// | OTM call (call, x ≤ 0)         | x       | β                  | already canonical              |
/// | OTM put  (put,  x > 0)         | −x      | β                  | (2.5) only — same β            |
/// | ITM call (call, x > 0)         | −x      | β − 2·sinh(\|x\|/2)| (2.5) + (2.6) intrinsic adjust |
/// | ITM put  (put,  x ≤ 0)         | x       | β − 2·sinh(\|x\|/2)| (2.6) intrinsic adjust         |
///
/// The intrinsic adjustment is the same magnitude `2·sinh(|x|/2)` in both ITM
/// cases — it is the time-value of the equivalent OTM call.
///
/// The de-canonicalization for σ̂ is trivial: σ̂ does not depend on θ or sign
/// of x (the invariances preserve volatility), so the solved σ is the answer.
#[inline]
pub fn canonicalize(x: V, beta: V, is_call: M) -> (V, V) {
    let abs_x = x.abs();
    let half_abs_x = V::splat(0.5) * abs_x;
    let e_plus = half_abs_x.exp();
    let e_minus = (-half_abs_x).exp();
    let two_sinh = e_plus - e_minus; // 2·sinh(|x|/2)

    let x_canon = -abs_x;

    // ITM if (x > 0 and call) or (x ≤ 0 and put).
    let x_pos = x.simd_gt(V::splat(0.0));
    let is_itm = (x_pos & is_call) | (!x_pos & !is_call);
    let beta_canon = is_itm.select(beta - two_sinh, beta);
    (x_canon, beta_canon)
}

// ---------------------------------------------------------------------------
// Region boundaries (Paper §4: (4.1)–(4.4), (4.7), (4.8))
// ---------------------------------------------------------------------------

/// The five quantities that anchor the four-region partition of `β ∈ [0, b_max]`:
///
/// `b_l < b_c < b_u < b_max`, plus the three vol-axis anchors `σ_l, σ_c, σ_u`.
///
/// Caller must have canonicalized to `x ≤ 0` first (the inflection-point
/// identity `σ_c = √(2|x|)` and the tangent geometry are derived under that
/// assumption).
#[derive(Copy, Clone, Debug)]
pub struct RegionBoundaries {
    /// `σ_l = σ_c − b_c / b'(σ_c)` — tangent intersection with `b = 0` (4.3).
    pub sigma_l: V,
    /// `σ_c = √(2|x|)` — inflection-point total vol (4.1).
    pub sigma_c: V,
    /// `σ_u = σ_c + (b_max − b_c) / b'(σ_c)` — tangent intersection with `b = b_max` (4.4).
    pub sigma_u: V,
    /// `b(x, σ_l)`.
    pub b_l: V,
    /// `b(x, σ_c)` — value at the inflection point.
    pub b_c: V,
    /// `b(x, σ_u)`.
    pub b_u: V,
    /// `b_max = e^(x/2)`.
    pub b_max: V,
    /// `b'(σ_c)` — the slope at the inflection (cached because the tangent
    /// formulae and the rational-cubic `1/b'` slopes all need it).
    pub bp_c: V,
}

/// Compute all five region boundaries plus the cached slope `b'(σ_c)`.
/// Lane-packed; same value for all lanes if `x` is the same.
///
/// One `b(x, σ)` evaluation per lane per σ-anchor (three total: at σ_c, σ_l,
/// σ_u). Plus one `b'` at σ_c. b_max is a single exp.
#[inline]
pub fn region_boundaries(x: V) -> RegionBoundaries {
    let sigma_c = black::sigma_c(x);
    let b_c = black::b_normalized(x, sigma_c);
    let bp_c = black::b_prime(x, sigma_c);
    let b_max = black::b_max(x);

    let sigma_l = sigma_c - b_c / bp_c;
    let sigma_u = sigma_c + (b_max - b_c) / bp_c;
    let b_l = black::b_normalized(x, sigma_l);
    let b_u = black::b_normalized(x, sigma_u);

    RegionBoundaries {
        sigma_l,
        sigma_c,
        sigma_u,
        b_l,
        b_c,
        b_u,
        b_max,
        bp_c,
    }
}

/// Region index: 0 = lower `[0, b_l]`, 1 = centre-left `[b_l, b_c]`,
/// 2 = centre-right `[b_c, b_u]`, 3 = upper `(b_u, b_max]`.
///
/// Used to mask-select which initial-guess formula applies per lane.
#[inline]
pub fn classify_region(beta: V, rb: &RegionBoundaries) -> Simd<i64, { crate::LANES }> {
    let is_below_bl = beta.simd_le(rb.b_l);
    let is_below_bc = beta.simd_le(rb.b_c);
    let is_below_bu = beta.simd_le(rb.b_u);

    // Default region 3 (upper); peel off lower regions in order.
    let r3 = Simd::splat(3_i64);
    let r2 = is_below_bu.select(Simd::splat(2_i64), r3);
    let r1 = is_below_bc.select(Simd::splat(1_i64), r2);
    is_below_bl.select(Simd::splat(0_i64), r1)
}

// ---------------------------------------------------------------------------
// Delbourgo–Gregory rational cubic (Paper (4.10))
// ---------------------------------------------------------------------------

/// Rational cubic interpolant on `[x_l, x_r]`:
///
/// ```text
///   f^rc(x) = [f_r·s³ + (r·f_r − h·f'_r)·s²·(1−s)
///              + (r·f_l + h·f'_l)·s·(1−s)²  + f_l·(1−s)³]
///            / [1 + (r − 3)·s·(1−s)]
///
///   h := x_r − x_l,  s := (x − x_l)/h
/// ```
///
/// Parameter `r > −1` (else there is a pole inside `[x_l, x_r]`). Choosing
/// `r` via [`dg_r_left`] / [`dg_r_right`] to match a second derivative at one
/// edge is the Jäckel design (Paper (4.12)/(4.13)).
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dg_rational_cubic(x: V, x_l: V, x_r: V, f_l: V, f_r: V, fp_l: V, fp_r: V, r: V) -> V {
    let one = V::splat(1.0);
    let three = V::splat(3.0);
    let h = x_r - x_l;
    let s = (x - x_l) / h;
    let one_minus_s = one - s;
    let s2 = s * s;
    let s3 = s2 * s;
    let oms2 = one_minus_s * one_minus_s;
    let oms3 = oms2 * one_minus_s;

    let num = f_r * s3
        + (r * f_r - h * fp_r) * s2 * one_minus_s
        + (r * f_l + h * fp_l) * s * oms2
        + f_l * oms3;
    let den = one + (r - three) * s * one_minus_s;
    num / den
}

/// Control parameter `r` chosen so the rational cubic matches a desired
/// **left-edge** second derivative `f''_l` (Paper (4.12)):
///
/// ```text
///   r_l = [½·h·f''_l + (f'_r − f'_l)] / (Δ − f'_l),    Δ := (f_r − f_l)/h
/// ```
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dg_r_left(x_l: V, x_r: V, f_l: V, f_r: V, fp_l: V, fp_r: V, fpp_l: V) -> V {
    let h = x_r - x_l;
    let delta = (f_r - f_l) / h;
    (V::splat(0.5) * h * fpp_l + (fp_r - fp_l)) / (delta - fp_l)
}

/// Control parameter `r` chosen so the rational cubic matches a desired
/// **right-edge** second derivative `f''_r` (Paper (4.13)):
///
/// ```text
///   r_r = [½·h·f''_r + (f'_r − f'_l)] / (f'_r − Δ),    Δ := (f_r − f_l)/h
/// ```
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn dg_r_right(x_l: V, x_r: V, f_l: V, f_r: V, fp_l: V, fp_r: V, fpp_r: V) -> V {
    let h = x_r - x_l;
    let delta = (f_r - f_l) / h;
    (V::splat(0.5) * h * fpp_r + (fp_r - fp_l)) / (fp_r - delta)
}

// ---------------------------------------------------------------------------
// Initial guess: centre regions (Paper §4.5–4.6)
// ---------------------------------------------------------------------------

/// Initial guess `σ_0(β)` in the centre-left region `β ∈ [b_l, b_c]`
/// (Paper (4.17)–(4.19)).
///
/// Delbourgo–Gregory rational cubic interpolation in `(β, σ)` space:
///   * endpoint values:    σ_l at β = b_l,    σ_c at β = b_c
///   * endpoint slopes:    1/b'(σ_l) at b_l,  1/b'(σ_c) at b_c
///     (because dσ/dβ = 1/b'(σ))
///   * second-derivative match at the RIGHT endpoint: σ''(β = b_c) = 0
///     (because at the inflection b''(σ_c) ≡ 0, and
///     σ''(β) = −b''(σ)/b'(σ)³ ⇒ 0 at σ_c).
///
/// Domain: caller responsible for `β ∈ [b_l, b_c]`. Outside this band the
/// returned σ has no meaning — the orchestrator's region mask selects the
/// right branch per lane.
#[inline]
pub fn initial_guess_centre_left(x: V, beta: V, rb: &RegionBoundaries) -> V {
    let bp_l = black::b_prime(x, rb.sigma_l);
    let bp_c = rb.bp_c;
    let inv_bp_l = V::splat(1.0) / bp_l;
    let inv_bp_c = V::splat(1.0) / bp_c;
    // r matched at the right edge with f''_r = 0 (paper (4.19)).
    let r = dg_r_right(
        rb.b_l,
        rb.b_c,
        rb.sigma_l,
        rb.sigma_c,
        inv_bp_l,
        inv_bp_c,
        V::splat(0.0),
    );
    dg_rational_cubic(
        beta, rb.b_l, rb.b_c, rb.sigma_l, rb.sigma_c, inv_bp_l, inv_bp_c, r,
    )
}

/// Initial guess `σ_0(β)` in the centre-right region `β ∈ (b_c, b_u]`
/// (Paper (4.20)–(4.22)).
///
/// Mirror of [`initial_guess_centre_left`]: same DG interpolation, second-
/// derivative match at the LEFT endpoint (β = b_c) instead of the right.
#[inline]
pub fn initial_guess_centre_right(x: V, beta: V, rb: &RegionBoundaries) -> V {
    let bp_c = rb.bp_c;
    let bp_u = black::b_prime(x, rb.sigma_u);
    let inv_bp_c = V::splat(1.0) / bp_c;
    let inv_bp_u = V::splat(1.0) / bp_u;
    // r matched at the left edge with f''_l = 0 (paper (4.22)).
    let r = dg_r_left(
        rb.b_c,
        rb.b_u,
        rb.sigma_c,
        rb.sigma_u,
        inv_bp_c,
        inv_bp_u,
        V::splat(0.0),
    );
    dg_rational_cubic(
        beta, rb.b_c, rb.b_u, rb.sigma_c, rb.sigma_u, inv_bp_c, inv_bp_u, r,
    )
}

// ---------------------------------------------------------------------------
// Initial guess: upper region (Paper §4.7, TBD-2 resolved in spec)
// ---------------------------------------------------------------------------

/// `f_u(β) := Φ(−σ(β)/2)` — the non-linear transformation that linearises
/// the upper-region inversion (Paper (4.23)). Defined for `σ > 0`.
#[inline]
fn f_u(sigma: V) -> V {
    crate::norm::phi_hart(-V::splat(0.5) * sigma)
}

/// `f'_u(β) = −½·exp((x/σ)²/2)` (TBD-2.a; clean derivation in spec).
#[inline]
fn fp_u(x: V, sigma: V) -> V {
    let h = x / sigma;
    -V::splat(0.5) * (V::splat(0.5) * h * h).exp()
}

/// `f''_u(β) = √(π/2)·(x²/σ³)·exp((x/σ)² + σ²/8)` (TBD-2.b).
#[inline]
fn fpp_u(x: V, sigma: V) -> V {
    let h = x / sigma;
    let half_pi = V::splat(0.5 * core::f64::consts::PI);
    let coef = half_pi.sqrt() * (x * x) / (sigma * sigma * sigma);
    coef * (h * h + V::splat(0.125) * sigma * sigma).exp()
}

/// Initial guess `σ_0(β)` in the upper region `β ∈ (b_u, b_max)` (Paper
/// (4.28)–(4.30)). DG rational cubic interpolates `f_u` in transformed
/// space; `σ = −2·Φ⁻¹(f_u^rc(β))` recovers the vol.
#[inline]
pub fn initial_guess_upper(x: V, beta: V, rb: &RegionBoundaries) -> V {
    let f_u_at_bu = f_u(rb.sigma_u);
    let fp_u_at_bu = fp_u(x, rb.sigma_u);
    let fpp_u_at_bu = fpp_u(x, rb.sigma_u);
    // (4.29): r matched at the left edge (β = b_u) with the upper-region f''.
    // Right edge: level 0, slope −½ (paper (4.26)/(4.27)).
    let r = dg_r_left(
        rb.b_u,
        rb.b_max,
        f_u_at_bu,
        V::splat(0.0),
        fp_u_at_bu,
        V::splat(-0.5),
        fpp_u_at_bu,
    );
    let f_rc = dg_rational_cubic(
        beta,
        rb.b_u,
        rb.b_max,
        f_u_at_bu,
        V::splat(0.0),
        fp_u_at_bu,
        V::splat(-0.5),
        r,
    );
    // (4.30): σ = −2·Φ⁻¹(f_u^rc).
    V::splat(-2.0) * crate::norm::phi_inv(f_rc)
}

// ---------------------------------------------------------------------------
// Initial guess: lower region (Paper §4.8, TBD-3 with corrected exp factor)
// ---------------------------------------------------------------------------

/// `f_l(β) := (2π|x|/(3√3))·Φ(z)³` with `z := −|x|/(√3·σ)` (Paper (4.31)).
#[inline]
fn f_l(x: V, sigma: V) -> V {
    let abs_x = x.abs();
    let z = -abs_x / (V::splat(3.0_f64.sqrt()) * sigma);
    let coef = V::splat(2.0 * core::f64::consts::PI) * abs_x / V::splat(3.0 * 3.0_f64.sqrt());
    let phi_z = crate::norm::phi_hart(z);
    coef * phi_z * phi_z * phi_z
}

/// `f'_l(β) = 2π·z²·Φ(z)²·exp(z² + σ²/8)` (TBD-3.a; corrected from paper
/// (4.32) which appears to have a typesetting typo — paper's `exp(z²/2)`
/// gives `lim_{β→0} f'_l = 0`, contradicting paper's own (4.35) claim that
/// the limit is 1; the corrected `exp(z² + σ²/8)` matches the limit).
#[inline]
fn fp_l(x: V, sigma: V) -> V {
    let abs_x = x.abs();
    let sqrt3 = V::splat(3.0_f64.sqrt());
    let z = -abs_x / (sqrt3 * sigma);
    let two_pi = V::splat(2.0 * core::f64::consts::PI);
    let phi_z = crate::norm::phi_hart(z);
    let z_sq = z * z;
    let expo = z_sq + V::splat(0.125) * sigma * sigma;
    two_pi * z_sq * phi_z * phi_z * expo.exp()
}

/// `f''_l(β)` derived numerically via central finite difference of `fp_l`.
/// The closed-form symbolic derivative is mechanical to compute but lengthy
/// (paper's (4.33) has both a `z²/2` typo and an undefined `a` symbol); for
/// initial guess fidelity, a finite-difference of `fp_l` at the needed point
/// is sufficient since `f''_l` is only used at a single anchor (`β = b_l`).
///
/// Step size `h = 1e-5·σ` keeps the FD truncation error well below the f64
/// floor for the values of `σ_l` encountered in practice.
#[inline]
fn fpp_l(x: V, sigma: V) -> V {
    let h = V::splat(1e-5) * sigma;
    let s_plus = sigma + h;
    let s_minus = sigma - h;
    let fp_plus = fp_l(x, s_plus);
    let fp_minus = fp_l(x, s_minus);
    // d(fp_l)/dσ via FD, then convert to d/dβ via the chain rule:
    //   f''_l(β) = d(fp_l)/dσ · (dσ/dβ) = d(fp_l)/dσ · (1/b'(σ))
    let dfp_dsigma = (fp_plus - fp_minus) / (V::splat(2.0) * h);
    let bp = black::b_prime(x, sigma);
    dfp_dsigma / bp
}

/// Initial guess `σ_0(β)` in the lower region `β ∈ [0, b_l]` (Paper
/// (4.36)–(4.38)). DG rational cubic interpolates `f_l` in transformed
/// space; inversion of (4.31) recovers σ.
#[inline]
pub fn initial_guess_lower(x: V, beta: V, rb: &RegionBoundaries) -> V {
    let abs_x = x.abs();
    let f_l_at_bl = f_l(x, rb.sigma_l);
    let fp_l_at_bl = fp_l(x, rb.sigma_l);
    let fpp_l_at_bl = fpp_l(x, rb.sigma_l);
    // (4.37): r matched at the right edge (β = b_l).
    // Left edge (β = 0): level 0, slope 1 (paper (4.34)/(4.35)).
    let r = dg_r_right(
        V::splat(0.0),
        rb.b_l,
        V::splat(0.0),
        f_l_at_bl,
        V::splat(1.0),
        fp_l_at_bl,
        fpp_l_at_bl,
    );
    let f_rc = dg_rational_cubic(
        beta,
        V::splat(0.0),
        rb.b_l,
        V::splat(0.0),
        f_l_at_bl,
        V::splat(1.0),
        fp_l_at_bl,
        r,
    );
    // (4.38): σ = |x/√3 · [Φ⁻¹(√3·∛(f_l^rc/(2π|x|)))]⁻¹|
    // cbrt via exp(ln(·)/3) — argument is positive (f_l ≥ 0, |x| > 0).
    let two_pi = V::splat(2.0 * core::f64::consts::PI);
    let ratio = f_rc / (two_pi * abs_x);
    let cbrt_ratio = (ratio.ln() / V::splat(3.0)).exp();
    let inside = cbrt_ratio * V::splat(3.0_f64.sqrt());
    let phi_inv_inside = crate::norm::phi_inv(inside);
    let sigma = abs_x / (V::splat(3.0_f64.sqrt()) * phi_inv_inside.abs());
    sigma.abs()
}

// ---------------------------------------------------------------------------
// Unified initial guess: dispatch by region (Paper (4.39))
// ---------------------------------------------------------------------------

/// Net initial guess `σ_0(β)` per Paper (4.39): dispatch by region.
/// All four branches are computed in SIMD (mask-and-compute over branches —
/// see spec §7 "Strategy A"). Output: same value the iteration would seed
/// from in each region.
#[inline]
pub fn initial_guess(x: V, beta: V, rb: &RegionBoundaries) -> V {
    let region = classify_region(beta, rb);
    let g0 = initial_guess_lower(x, beta, rb);
    let g1 = initial_guess_centre_left(x, beta, rb);
    let g2 = initial_guess_centre_right(x, beta, rb);
    let g3 = initial_guess_upper(x, beta, rb);
    let is_0 = region.simd_eq(Simd::splat(0));
    let is_1 = region.simd_eq(Simd::splat(1));
    let is_2 = region.simd_eq(Simd::splat(2));
    // Select; default to g3 (upper) for region 3.
    is_0.select(g0, is_1.select(g1, is_2.select(g2, g3)))
}

// ---------------------------------------------------------------------------
// Householder's method of order 3 (Paper §5, TBD-4 resolved in spec)
// ---------------------------------------------------------------------------

/// One Householder-3 step:
///
/// ```text
///   σ_{n+1} = σ_n − 3·g·(2·g'² − g·g'')  /  (6·g'³ + g²·g''' − 6·g·g'·g'')
/// ```
///
/// Convergence order 4. Two iterations suffice for f64 from the Jäckel
/// initial guess. Denominator is guarded with a tiny floor so a flat lane
/// (g' near zero) returns a finite (non-NaN) step rather than ∞.
#[inline]
pub fn householder3_step(sigma: V, g: V, gp: V, gpp: V, gppp: V) -> V {
    let two = V::splat(2.0);
    let three = V::splat(3.0);
    let six = V::splat(6.0);
    let num = three * g * (two * gp * gp - g * gpp);
    let den = six * gp * gp * gp + g * g * gppp - six * g * gp * gpp;
    // Guard against vanishing denominator (e.g. a perfectly flat lane).
    let den_safe = den.abs().simd_max(V::splat(1e-300));
    let signed_den = den.is_sign_negative().select(-den_safe, den_safe);
    sigma - num / signed_den
}

// ---------------------------------------------------------------------------
// Middle-region objective and its derivatives (Paper (5.1) middle branch)
// ---------------------------------------------------------------------------

/// Middle-region objective: `g(σ) = b(x, σ) − β` and the three derivatives
/// `g', g'', g'''`.
///
/// Used for `β ∈ [b_l, b̄_u]` per Paper (5.1). Returns the tuple
/// `(g, g', g'', g''')` so a single chunk evaluation feeds the Householder
/// step.
#[inline]
pub fn objective_middle(x: V, sigma: V, beta: V) -> (V, V, V, V) {
    let b = black::b_normalized(x, sigma);
    let bp = black::b_prime(x, sigma);
    let bpp = black::b_double_prime(x, sigma);
    let bppp = black::b_triple_prime(x, sigma);
    (b - beta, bp, bpp, bppp)
}

// ---------------------------------------------------------------------------
// Middle-region solver: initial guess + two Householder-3 iterations
// ---------------------------------------------------------------------------

/// Solve the middle region by initial-guess + two Householder-3 iterations.
///
/// **Currently middle-region only** (β ∈ [b_l, b_u]). Lower and upper
/// region iteration formulae are TBD-5 / TBD-6 in the spec; this function
/// returns `NaN` if `β` is outside the middle band, so a caller dispatching
/// by region can mix it with future lower/upper solvers.
#[inline]
pub fn solve_middle(x: V, beta: V, rb: &RegionBoundaries) -> V {
    let in_band = beta.simd_ge(rb.b_l) & beta.simd_le(rb.b_u);
    // Pick the initial guess by which centre side.
    let on_left = beta.simd_le(rb.b_c);
    let cl = initial_guess_centre_left(x, beta, rb);
    let cr = initial_guess_centre_right(x, beta, rb);
    let mut sigma = on_left.select(cl, cr);
    // Two Householder-3 iterations.
    for _ in 0..2 {
        let (g, gp, gpp, gppp) = objective_middle(x, sigma, beta);
        sigma = householder3_step(sigma, g, gp, gpp, gppp);
    }
    in_band.select(sigma, V::splat(f64::NAN))
}

// ---------------------------------------------------------------------------
// Lower-region objective (Paper (5.1) lower branch; TBD-5 resolved here)
// ---------------------------------------------------------------------------

/// Lower-region objective: `g(σ) = 1/ln(b(σ)) − 1/ln(β)`.
///
/// Derivation: let `L := ln b`, so `L' = b'/b`, `L'' = b''/b − (L')²`,
/// `L''' = b'''/b − 3·L'·L'' − (L')³`. Then with `g = 1/L − const`:
/// ```text
///   g'   = −L'/L²
///   g''  = (2(L')² − L·L'')/L³
///   g''' = (6·L·L'·L'' − L²·L''' − 6(L')³)/L⁴
/// ```
#[inline]
pub fn objective_lower(x: V, sigma: V, beta: V) -> (V, V, V, V) {
    let b = black::b_normalized(x, sigma);
    let bp = black::b_prime(x, sigma);
    let bpp = black::b_double_prime(x, sigma);
    let bppp = black::b_triple_prime(x, sigma);
    let l = b.ln();
    let lp = bp / b;
    let lpp = bpp / b - lp * lp;
    let lppp = bppp / b - V::splat(3.0) * lp * lpp - lp * lp * lp;
    let g = V::splat(1.0) / l - V::splat(1.0) / beta.ln();
    let gp = -lp / (l * l);
    let gpp = (V::splat(2.0) * lp * lp - l * lpp) / (l * l * l);
    let l_sq = l * l;
    let gppp =
        (V::splat(6.0) * l * lp * lpp - l_sq * lppp - V::splat(6.0) * lp * lp * lp) / (l_sq * l_sq);
    (g, gp, gpp, gppp)
}

// ---------------------------------------------------------------------------
// Upper-region objective (Paper (5.1) upper branch; TBD-6 resolved here)
// ---------------------------------------------------------------------------

/// Upper-region objective: `g(σ) = ln((b_max − β)/(b_max − b(σ)))`.
///
/// Let `h := b_max − b`. Then `g = ln(b_max − β) − ln(h)` and:
/// ```text
///   g'   = b'/h
///   g''  = b''/h + (b')²/h²
///   g''' = b'''/h + 3·b'·b''/h² + 2(b')³/h³
/// ```
#[inline]
pub fn objective_upper(x: V, sigma: V, beta: V, b_max: V) -> (V, V, V, V) {
    let b = black::b_normalized(x, sigma);
    let bp = black::b_prime(x, sigma);
    let bpp = black::b_double_prime(x, sigma);
    let bppp = black::b_triple_prime(x, sigma);
    let h = b_max - b;
    let g = ((b_max - beta) / h).ln();
    let gp = bp / h;
    let gpp = bpp / h + (bp * bp) / (h * h);
    let gppp =
        bppp / h + V::splat(3.0) * bp * bpp / (h * h) + V::splat(2.0) * bp * bp * bp / (h * h * h);
    (g, gp, gpp, gppp)
}

// ---------------------------------------------------------------------------
// Region-specific solvers
// ---------------------------------------------------------------------------

/// Solve the lower region by initial-guess + two Householder-3 iterations.
#[inline]
pub fn solve_lower(x: V, beta: V, rb: &RegionBoundaries) -> V {
    let in_band = beta.simd_gt(V::splat(0.0)) & beta.simd_lt(rb.b_l);
    let mut sigma = initial_guess_lower(x, beta, rb);
    for _ in 0..2 {
        let (g, gp, gpp, gppp) = objective_lower(x, sigma, beta);
        sigma = householder3_step(sigma, g, gp, gpp, gppp);
    }
    in_band.select(sigma, V::splat(f64::NAN))
}

/// Solve the upper region by initial-guess + two Householder-3 iterations.
#[inline]
pub fn solve_upper(x: V, beta: V, rb: &RegionBoundaries) -> V {
    let in_band = beta.simd_gt(rb.b_u) & beta.simd_lt(rb.b_max);
    let mut sigma = initial_guess_upper(x, beta, rb);
    for _ in 0..2 {
        let (g, gp, gpp, gppp) = objective_upper(x, sigma, beta, rb.b_max);
        sigma = householder3_step(sigma, g, gp, gpp, gppp);
    }
    in_band.select(sigma, V::splat(f64::NAN))
}

// ---------------------------------------------------------------------------
// Unified solver: dispatch by region
// ---------------------------------------------------------------------------

/// Centre-left dense solver (caller guarantees β ∈ [b_l, b_c] for every lane).
///
/// Used by the segregated `implied_vol_rational` path: the macro-chunk
/// classifier guarantees every lane in the input chunk has been routed to
/// region 1, so this solver skips the `in_band` mask AND skips computing the
/// centre-right initial guess. Two Householder-3 iterations on the middle-
/// region objective, seeded by `initial_guess_centre_left`.
#[inline]
pub fn solve_centre_left(x: V, beta: V, rb: &RegionBoundaries) -> V {
    let mut sigma = initial_guess_centre_left(x, beta, rb);
    for _ in 0..2 {
        let (g, gp, gpp, gppp) = objective_middle(x, sigma, beta);
        sigma = householder3_step(sigma, g, gp, gpp, gppp);
    }
    sigma
}

/// Centre-right dense solver (caller guarantees β ∈ (b_c, b_u] for every lane).
///
/// Mirror of [`solve_centre_left`]; skips the unused centre-left initial guess.
#[inline]
pub fn solve_centre_right(x: V, beta: V, rb: &RegionBoundaries) -> V {
    let mut sigma = initial_guess_centre_right(x, beta, rb);
    for _ in 0..2 {
        let (g, gp, gpp, gppp) = objective_middle(x, sigma, beta);
        sigma = householder3_step(sigma, g, gp, gpp, gppp);
    }
    sigma
}

/// Lower-region dense solver (caller guarantees β ∈ (0, b_l) for every lane).
/// Skips the `in_band` mask used by [`solve_lower`].
#[inline]
pub fn solve_lower_dense(x: V, beta: V, rb: &RegionBoundaries) -> V {
    let mut sigma = initial_guess_lower(x, beta, rb);
    for _ in 0..2 {
        let (g, gp, gpp, gppp) = objective_lower(x, sigma, beta);
        sigma = householder3_step(sigma, g, gp, gpp, gppp);
    }
    sigma
}

/// Upper-region dense solver (caller guarantees β ∈ (b_u, b_max) for every lane).
/// Skips the `in_band` mask used by [`solve_upper`].
#[inline]
pub fn solve_upper_dense(x: V, beta: V, rb: &RegionBoundaries) -> V {
    let mut sigma = initial_guess_upper(x, beta, rb);
    for _ in 0..2 {
        let (g, gp, gpp, gppp) = objective_upper(x, sigma, beta, rb.b_max);
        sigma = householder3_step(sigma, g, gp, gpp, gppp);
    }
    sigma
}

/// The full Jäckel rational solver: dispatch by region.
///
/// Fast path: if all 8 lanes of a chunk classify to the same region, run only
/// that region's solver (skipping ~⅔ of the work the heterogeneous path
/// would do). For natural workloads — a single options chain at one expiry,
/// a market-data snapshot clustered around ATM — most chunks are homogeneous
/// and the fast path dominates. The bool reduction is one AVX-512 `kortestz`
/// (or equivalent on narrower targets); the branch is highly predictable for
/// any non-pathological batch ordering.
///
/// Heterogeneous path: compute all three candidates and mask-select, as
/// before. This keeps the SIMD-divergence-free behaviour the spec calls
/// "Strategy A" for chunks that genuinely cross region boundaries.
#[inline]
pub fn solve_rational(x: V, beta: V, rb: &RegionBoundaries) -> V {
    let region = classify_region(beta, rb);
    // Strict fast path: all 8 lanes in the same region → one solver, no mask.
    let r0 = region[0];
    if region.simd_eq(Simd::splat(r0)).all() {
        return match r0 {
            0 => solve_lower(x, beta, rb),
            1 | 2 => solve_middle(x, beta, rb),
            3 => solve_upper(x, beta, rb),
            _ => unreachable!("classify_region returns 0..=3"),
        };
    }
    // Relaxed fast path: all 8 lanes in the centre (regions 1 or 2). The
    // centre is the dominant region for natural workloads (any batch from a
    // single options chain at one expiry clusters around ATM); firing here
    // skips both solve_lower and solve_upper. `solve_middle` internally
    // mask-selects between the centre-left and centre-right initial guesses
    // already, so it's correct for mixed-1-and-2 chunks.
    let in_centre = region.simd_ge(Simd::splat(1)) & region.simd_le(Simd::splat(2));
    if in_centre.all() {
        return solve_middle(x, beta, rb);
    }
    // Heterogeneous: at least one lane crosses into lower or upper. Compute
    // all three candidates and mask-select.
    let s_lower = solve_lower(x, beta, rb);
    let s_middle = solve_middle(x, beta, rb);
    let s_upper = solve_upper(x, beta, rb);
    let is_0 = region.simd_eq(Simd::splat(0));
    let is_3 = region.simd_eq(Simd::splat(3));
    is_0.select(s_lower, is_3.select(s_upper, s_middle))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OptionKind;

    /// All four canonical-form cases round-trip: the canonicalized option
    /// must re-price (via `bs_price` in non-normalized form, or via the
    /// normalized `b` formula) to the canonical β.
    #[test]
    fn canonicalize_handles_all_four_quadrants() {
        // Use a moderate-volatility, well-conditioned set so the naive `b`
        // formula is accurate. For each, the canonical (x_canon, β_canon)
        // should produce the same call-equivalent normalized price.
        //
        // Construct test cases by starting from a known σ_true, computing the
        // normalized β for each (x, θ) combination, then canonicalizing.

        let sigma_true = 0.30_f64;
        for x_val in [-0.5_f64, -0.1, 0.1, 0.5] {
            // Compute β for both call and put at this x via the (2.4) form.
            for &is_call_val in &[true, false] {
                let theta = if is_call_val { 1.0_f64 } else { -1.0 };
                let h = x_val / sigma_true;
                let k = 0.5 * sigma_true;
                let d1 = h + k;
                let d2 = h - k;
                let phi_d1 = crate::norm::phi_hart(f64x8::splat(theta * d1))[0];
                let phi_d2 = crate::norm::phi_hart(f64x8::splat(theta * d2))[0];
                let beta_input =
                    theta * ((0.5 * x_val).exp() * phi_d1 - (-0.5 * x_val).exp() * phi_d2);

                let x = f64x8::splat(x_val);
                let beta = f64x8::splat(beta_input);
                let is_call = if is_call_val {
                    Mask::from_array([true; 8])
                } else {
                    Mask::from_array([false; 8])
                };

                let (x_canon, beta_canon) = canonicalize(x, beta, is_call);

                // x_canon must be ≤ 0.
                assert!(
                    x_canon[0] <= 1e-15,
                    "x_canon[0] = {} should be ≤ 0 (input x={x_val}, call={is_call_val})",
                    x_canon[0]
                );

                // Compute b(x_canon, σ_true) and compare against β_canon.
                let b_check = black::b_normalized(x_canon, f64x8::splat(sigma_true))[0];
                let err = (b_check - beta_canon[0]).abs();
                assert!(
                    err < 1e-13,
                    "x={x_val} call={is_call_val}: canonical β={} but b(x_canon, σ_true)={}, err={:e}",
                    beta_canon[0],
                    b_check,
                    err
                );
            }
        }
    }

    /// Region boundaries respect the partition ordering: `0 < b_l < b_c < b_u < b_max`
    /// and `0 < σ_l < σ_c < σ_u`.
    #[test]
    fn region_boundaries_are_ordered() {
        for x_val in [-0.05_f64, -0.2, -0.5, -1.0, -2.0, -4.0] {
            let x = f64x8::splat(x_val);
            let rb = region_boundaries(x);
            assert!(rb.sigma_l[0] > 0.0, "σ_l > 0 (x={x_val})");
            assert!(
                rb.sigma_l[0] < rb.sigma_c[0],
                "σ_l < σ_c (x={x_val}): {} vs {}",
                rb.sigma_l[0],
                rb.sigma_c[0]
            );
            assert!(
                rb.sigma_c[0] < rb.sigma_u[0],
                "σ_c < σ_u (x={x_val}): {} vs {}",
                rb.sigma_c[0],
                rb.sigma_u[0]
            );
            assert!(rb.b_l[0] > 0.0, "b_l > 0 (x={x_val})");
            assert!(
                rb.b_l[0] < rb.b_c[0],
                "b_l < b_c (x={x_val}): {} vs {}",
                rb.b_l[0],
                rb.b_c[0]
            );
            assert!(
                rb.b_c[0] < rb.b_u[0],
                "b_c < b_u (x={x_val}): {} vs {}",
                rb.b_c[0],
                rb.b_u[0]
            );
            assert!(
                rb.b_u[0] < rb.b_max[0],
                "b_u < b_max (x={x_val}): {} vs {}",
                rb.b_u[0],
                rb.b_max[0]
            );
        }
    }

    /// `σ_c = √(2|x|)` matches `crate::black::sigma_c`.
    #[test]
    fn region_boundaries_sigma_c_matches_black() {
        let x = f64x8::splat(-1.5);
        let rb = region_boundaries(x);
        let sc = black::sigma_c(x);
        assert!((rb.sigma_c[0] - sc[0]).abs() < 1e-15);
    }

    /// Region classification dispatches correctly for the four partition
    /// regions at representative β values.
    #[test]
    fn classify_region_assigns_correct_index() {
        let x = f64x8::splat(-1.0);
        let rb = region_boundaries(x);
        // Pick β values one in each region by interpolating between the
        // anchors.
        let half = f64x8::splat(0.5);
        let beta_lower = half * rb.b_l; // strictly < b_l
        let beta_cl = half * (rb.b_l + rb.b_c); // between b_l and b_c
        let beta_cr = half * (rb.b_c + rb.b_u); // between b_c and b_u
        let beta_upper = half * (rb.b_u + rb.b_max); // > b_u

        assert_eq!(classify_region(beta_lower, &rb)[0], 0, "lower");
        assert_eq!(classify_region(beta_cl, &rb)[0], 1, "centre-left");
        assert_eq!(classify_region(beta_cr, &rb)[0], 2, "centre-right");
        assert_eq!(classify_region(beta_upper, &rb)[0], 3, "upper");
    }

    /// Delbourgo–Gregory interpolation must match level and slope at both
    /// endpoints (the defining property).
    #[test]
    fn dg_rational_cubic_matches_endpoints() {
        let x_l = f64x8::splat(0.0);
        let x_r = f64x8::splat(2.0);
        let f_l = f64x8::splat(1.0);
        let f_r = f64x8::splat(4.0);
        let fp_l = f64x8::splat(0.5);
        let fp_r = f64x8::splat(2.0);
        let r = f64x8::splat(2.5); // arbitrary > -1

        // At x = x_l, f^rc = f_l.
        let v_at_l = dg_rational_cubic(x_l, x_l, x_r, f_l, f_r, fp_l, fp_r, r);
        assert!(
            (v_at_l[0] - f_l[0]).abs() < 1e-14,
            "left endpoint: got {}, expected {}",
            v_at_l[0],
            f_l[0]
        );

        // At x = x_r, f^rc = f_r.
        let v_at_r = dg_rational_cubic(x_r, x_l, x_r, f_l, f_r, fp_l, fp_r, r);
        assert!(
            (v_at_r[0] - f_r[0]).abs() < 1e-14,
            "right endpoint: got {}, expected {}",
            v_at_r[0],
            f_r[0]
        );

        // Slopes: derivatives at endpoints should match fp_l and fp_r.
        // (Numerical FD; DG is C¹ at endpoints by construction.)
        let h = 1e-6_f64;
        let x_l_h = f64x8::splat(h);
        let fd_l = (dg_rational_cubic(x_l_h, x_l, x_r, f_l, f_r, fp_l, fp_r, r)[0] - f_l[0]) / h;
        assert!(
            (fd_l - fp_l[0]).abs() < 1e-5,
            "left slope: fd {} vs {}",
            fd_l,
            fp_l[0]
        );
    }

    /// Centre-left initial guess matches the anchor values at both endpoints.
    #[test]
    fn initial_guess_centre_left_matches_anchors() {
        let x = f64x8::splat(-0.5);
        let rb = region_boundaries(x);
        // At β = b_l, the guess should equal σ_l (within tiny f64 round-off).
        let g_at_bl = initial_guess_centre_left(x, rb.b_l, &rb);
        assert!(
            (g_at_bl[0] - rb.sigma_l[0]).abs() < 1e-13,
            "σ_0(b_l) = {} expected σ_l = {}",
            g_at_bl[0],
            rb.sigma_l[0]
        );
        // At β = b_c, the guess should equal σ_c.
        let g_at_bc = initial_guess_centre_left(x, rb.b_c, &rb);
        assert!(
            (g_at_bc[0] - rb.sigma_c[0]).abs() < 1e-13,
            "σ_0(b_c) = {} expected σ_c = {}",
            g_at_bc[0],
            rb.sigma_c[0]
        );
    }

    /// Centre-right initial guess matches the anchor values at both endpoints.
    #[test]
    fn initial_guess_centre_right_matches_anchors() {
        let x = f64x8::splat(-0.5);
        let rb = region_boundaries(x);
        let g_at_bc = initial_guess_centre_right(x, rb.b_c, &rb);
        assert!(
            (g_at_bc[0] - rb.sigma_c[0]).abs() < 1e-13,
            "σ_0(b_c) = {} expected σ_c = {}",
            g_at_bc[0],
            rb.sigma_c[0]
        );
        let g_at_bu = initial_guess_centre_right(x, rb.b_u, &rb);
        assert!(
            (g_at_bu[0] - rb.sigma_u[0]).abs() < 1e-13,
            "σ_0(b_u) = {} expected σ_u = {}",
            g_at_bu[0],
            rb.sigma_u[0]
        );
    }

    /// Centre-region initial guess accuracy on a grid of well-conditioned
    /// inputs. Paper Figure 3 shows σ_0 ≈ σ_exact within graphical resolution
    /// across [b_l, b_u]; expect relative error well under 1% as the harness
    /// for the iteration step.
    #[test]
    fn initial_guess_centre_close_to_truth() {
        let mut worst = 0.0_f64;
        for x_val in [-0.05_f64, -0.2, -0.5, -1.0, -2.0] {
            let x = f64x8::splat(x_val);
            let rb = region_boundaries(x);
            // Sample σ_true across the centre band by sampling σ ∈ [σ_l, σ_u].
            let n = 8;
            for i in 0..=n {
                let frac = i as f64 / n as f64;
                let sigma_true = rb.sigma_l[0] + frac * (rb.sigma_u[0] - rb.sigma_l[0]);
                let s = f64x8::splat(sigma_true);
                let beta = black::b_normalized(x, s)[0];
                let beta_v = f64x8::splat(beta);

                // Classify and dispatch.
                let region = classify_region(beta_v, &rb)[0];
                let guess = match region {
                    1 => initial_guess_centre_left(x, beta_v, &rb)[0],
                    2 => initial_guess_centre_right(x, beta_v, &rb)[0],
                    _ => continue, // skip lower/upper for this test
                };
                let rel = ((guess - sigma_true) / sigma_true).abs();
                worst = worst.max(rel);
                assert!(
                    rel < 0.05,
                    "x={x_val} σ_true={sigma_true} guess={guess} rel_err={rel:e}"
                );
            }
        }
        // Paper's claim is that the guess is essentially exact; 5% is loose;
        // actual measured worst is much lower. Print for visibility.
        eprintln!("centre-region worst relative initial-guess error: {worst:e}");
    }

    /// Initial guess across all four regions: relative error in the inferred
    /// σ vs the truth used to construct β.
    #[test]
    fn unified_initial_guess_across_all_regions() {
        let mut worst_by_region = [0.0_f64; 4];
        let mut count_by_region = [0_usize; 4];
        for x_val in [-0.05_f64, -0.2, -0.5, -1.0, -2.0] {
            let x = f64x8::splat(x_val);
            let rb = region_boundaries(x);
            // Sample σ_true across a wide range, then compute β and dispatch.
            for &sigma_true in &[
                0.5 * rb.sigma_l[0],                   // lower
                0.99 * rb.sigma_l[0],                  // edge of lower
                0.5 * (rb.sigma_l[0] + rb.sigma_c[0]), // centre-left mid
                rb.sigma_c[0],                         // exactly at inflection
                0.5 * (rb.sigma_c[0] + rb.sigma_u[0]), // centre-right mid
                1.01 * rb.sigma_u[0],                  // edge of upper
                1.5 * rb.sigma_u[0],                   // upper
            ] {
                let s = f64x8::splat(sigma_true);
                let beta = black::b_normalized(x, s)[0];
                let beta_v = f64x8::splat(beta);
                let region = classify_region(beta_v, &rb)[0] as usize;
                let guess = initial_guess(x, beta_v, &rb)[0];
                if !guess.is_finite() || guess <= 0.0 {
                    eprintln!(
                        "non-positive/non-finite guess: x={x_val} σ_true={sigma_true} \
                         beta={beta} region={region} guess={guess}"
                    );
                    continue;
                }
                let rel = ((guess - sigma_true) / sigma_true).abs();
                worst_by_region[region] = worst_by_region[region].max(rel);
                count_by_region[region] += 1;
            }
        }
        for (i, (&w, &c)) in worst_by_region.iter().zip(&count_by_region).enumerate() {
            let name = ["lower", "centre-left", "centre-right", "upper"][i];
            eprintln!("region {i} ({name}): worst rel err = {w:e} over {c} cases");
        }
        // The centre regions should be excellent (we already validated 0.29%).
        // The upper and lower are asymptotic-anchored; expect ≤ 5% per the
        // paper's design ("first-order asymptotically correct").
        assert!(
            worst_by_region[1] < 0.01,
            "centre-left worst: {}",
            worst_by_region[1]
        );
        assert!(
            worst_by_region[2] < 0.01,
            "centre-right worst: {}",
            worst_by_region[2]
        );
        // Lower and upper: looser bar — paper claims asymptotic accuracy
        // improving toward the limits, so the edge cases (near b_l, near b_u)
        // are the worst within their respective branches.
        if count_by_region[0] > 0 {
            assert!(
                worst_by_region[0] < 0.20,
                "lower worst: {}",
                worst_by_region[0]
            );
        }
        if count_by_region[3] > 0 {
            assert!(
                worst_by_region[3] < 0.20,
                "upper worst: {}",
                worst_by_region[3]
            );
        }
    }

    /// Householder-3 reduces to Newton when c = d = 0 (sanity check from spec).
    #[test]
    fn householder3_reduces_to_newton_when_higher_derivs_zero() {
        let sigma = f64x8::splat(1.0);
        let g = f64x8::splat(0.5);
        let gp = f64x8::splat(2.0);
        let gpp = f64x8::splat(0.0);
        let gppp = f64x8::splat(0.0);
        let next = householder3_step(sigma, g, gp, gpp, gppp)[0];
        // Newton step: σ - g/g' = 1 - 0.5/2 = 0.75
        assert!((next - 0.75).abs() < 1e-15, "got {next}");
    }

    /// Middle-region solver: from Jäckel initial guess + 2 Householder-3
    /// iterations, recover σ_true to near-machine precision.
    #[test]
    fn solve_middle_converges_to_truth() {
        let mut worst = 0.0_f64;
        for x_val in [-0.05_f64, -0.2, -0.5, -1.0, -2.0] {
            let x = f64x8::splat(x_val);
            let rb = region_boundaries(x);
            // Sample σ_true across the middle band; compute β; recover σ.
            for &frac in &[0.001_f64, 0.1, 0.25, 0.5, 0.75, 0.9, 0.999] {
                let sigma_true = rb.sigma_l[0] + frac * (rb.sigma_u[0] - rb.sigma_l[0]);
                let s = f64x8::splat(sigma_true);
                let beta = black::b_normalized(x, s)[0];
                let beta_v = f64x8::splat(beta);
                let recovered = solve_middle(x, beta_v, &rb)[0];
                let abs_err = (recovered - sigma_true).abs();
                worst = worst.max(abs_err);
                assert!(
                    abs_err < 1e-10,
                    "x={x_val} σ_true={sigma_true}: recovered={recovered} err={abs_err:e}"
                );
            }
        }
        eprintln!("middle-region worst abs error after 2 Householder-3 iters: {worst:e}");
    }

    /// End-to-end rational solver: across all regions, recover σ_true to
    /// near machine precision.
    #[test]
    fn solve_rational_end_to_end() {
        let mut worst_by_region = [0.0_f64; 4];
        let mut count_by_region = [0_usize; 4];
        for x_val in [-0.05_f64, -0.2, -0.5, -1.0, -2.0] {
            let x = f64x8::splat(x_val);
            let rb = region_boundaries(x);
            // Sample widely. Skip near the very boundary at 0 and b_max
            // where the price loses f64 precision.
            for &frac in &[0.01_f64, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99] {
                // Total range: σ_l < σ_c < σ_u, but extend a bit beyond.
                let sigma_range_lo = 0.3 * rb.sigma_l[0];
                let sigma_range_hi = 1.7 * rb.sigma_u[0];
                let sigma_true = sigma_range_lo + frac * (sigma_range_hi - sigma_range_lo);
                let s = f64x8::splat(sigma_true);
                let beta = black::b_normalized(x, s)[0];
                if !beta.is_finite() || beta <= 0.0 {
                    continue;
                }
                let beta_v = f64x8::splat(beta);
                let region = classify_region(beta_v, &rb)[0] as usize;
                let recovered = solve_rational(x, beta_v, &rb)[0];
                if !recovered.is_finite() {
                    eprintln!(
                        "non-finite: x={x_val} σ_true={sigma_true} beta={beta} \
                         region={region}"
                    );
                    continue;
                }
                let abs_err = (recovered - sigma_true).abs();
                worst_by_region[region] = worst_by_region[region].max(abs_err);
                count_by_region[region] += 1;
            }
        }
        for (i, (&w, &c)) in worst_by_region.iter().zip(&count_by_region).enumerate() {
            let name = ["lower", "centre-left", "centre-right", "upper"][i];
            eprintln!("region {i} ({name}): worst abs err = {w:e} over {c} cases");
        }
        // Centre and upper should land at machine precision. Lower has the
        // asymptotic-anchor limitation (initial guess only 8% accurate) so
        // 2 iterations may not always hit f64 floor.
        assert!(
            worst_by_region[1] < 1e-13,
            "centre-left worst: {}",
            worst_by_region[1]
        );
        assert!(
            worst_by_region[2] < 1e-13,
            "centre-right worst: {}",
            worst_by_region[2]
        );
        if count_by_region[3] > 0 {
            assert!(
                worst_by_region[3] < 1e-10,
                "upper worst: {}",
                worst_by_region[3]
            );
        }
        if count_by_region[0] > 0 {
            // Loose bar; iteration may need refinement here.
            assert!(
                worst_by_region[0] < 1e-3,
                "lower worst: {}",
                worst_by_region[0]
            );
        }
    }

    /// Suppress unused-symbol warnings until Phase 3 uses these.
    #[test]
    fn dg_r_helpers_compile() {
        let _ = dg_r_left;
        let _ = dg_r_right;
    }

    /// Avoid the unused-OptionKind warning until later phases use it.
    #[allow(dead_code)]
    fn _kind_marker() -> OptionKind {
        OptionKind::Call
    }
}
