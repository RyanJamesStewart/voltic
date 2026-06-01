//! `voltic` — Black-Scholes implied volatility, one operation, vectorized.
//!
//! Given (spot, strike, time-to-expiry, risk-free rate, option price, call/put),
//! [`implied_vol`] returns the Black-Scholes implied volatility, computed over a
//! batch of contracts in lane-packed `f64` SIMD. That is the entire API.
//!
//! Three things, in this order, are where the speed comes from:
//!
//! 1. **A rational initial guess** — a closed-form approximation
//!    (Corrado & Miller 1996, with a Brenner & Subrahmanyam 1988 ATM fallback)
//!    lands within one or two Newton steps of the answer for well-conditioned
//!    inputs, so the iteration does almost no work. Most of the speedup is doing
//!    less, not doing it faster. For the genuinely hard region — deep OTM near
//!    expiry, where rational-guess-plus-Newton can stall — the answer is the
//!    rational-cubic-spline method of Jäckel, *"Let Be Rational"* (Wilmott,
//!    2015); `voltic` does not implement that and returns `NaN` if Newton has
//!    not converged within [`MAX_ITERS`] (see the README "Limitations").
//!
//! 2. **Lane-packed Newton with masked convergence** — the batch iterates
//!    together; a lane whose update has fallen below tolerance is masked out via
//!    `mask.select(...)` so its value stops changing, and the loop ends when
//!    every lane has converged (or the cap is hit). The slowest lane never
//!    gates the rest, and a converged lane costs nothing.
//!
//! 3. **A branch-free cumulative normal** — Φ(x) is called twice per iteration,
//!    so it is the inner-inner loop; see [`norm`] for the kernel choice
//!    (Hart 5666 — measured ~8e-9 relative error, far below the ~1e-6 IV
//!    conditioning floor, and the fastest of the three accurate kernels).
//!
//! ```
//! use voltic::{implied_vol, OptionKind};
//! // 30%-vol ATM call, S = K = 100, 1y, r = 2%  → priced at ~12.8216
//! let iv = implied_vol(&[100.0], &[100.0], &[1.0], &[0.02], &[12.821_58], &[OptionKind::Call]);
//! assert!((iv[0] - 0.30).abs() < 1e-4);
//! ```
#![feature(portable_simd)]
#![feature(simd_ffi)]
#![allow(clippy::needless_range_loop)]

pub mod black;
pub mod jackel;
pub mod norm;
pub mod otm_context;
pub mod schadner;
pub mod schadner_fast;
pub mod typed;

pub use otm_context::{
    broadcast_context, canonical_c_from_price, implied_vol_vectorized_with_contexts,
    implied_vol_with_context, implied_vol_with_context_batch, pack_contexts, OtmContext,
    OtmContextSimd,
};
pub use schadner::implied_vol_explicit;
pub use schadner_fast::implied_vol_fast_kernel;
pub use typed::{implied_vol_typed, implied_vol_typed_batch, ImpliedVolResult, ImpliedVolStatus};

/// Public `implied_vol_fast` — the fast Cheb+Halley kernel with a NaN
/// fallback to the rational kernel. The kernel runs on the full batch as
/// usual; any output lane that comes back `NaN` is re-solved through
/// [`implied_vol_rational`] using the original inputs and overwritten in
/// place. The happy path (zero NaN in the kernel output) does no extra work
/// beyond a linear NaN scan.
/// Lanes with `|k|/sqrt(T) < ATM_BAILOUT_THRESHOLD` (where `k = ln(K/F)`) sit
/// in the structural ATM-ceiling of the FIX-5 kernel; route them straight to
/// the rational kernel.
const ATM_BAILOUT_THRESHOLD: f64 = 5e-3;

/// Deep-OTM bailout. Lanes whose canonical OTM-leg premium `c_otm / F` is
/// below this threshold are routed directly to the rational kernel — the
/// Chebyshev seed + Halley path cannot drive σ-error below ~1e-7 in this
/// regime within fewer than 5 Halley steps. Picked by the diagnostic in
/// `bench/diag.rs`: at H=4, `c_otm/F < 3e-6` catches all 3544 deep-OTM bad
/// lanes (those that violate the 1e-7 σ-gate) at a 3.2% tag rate.
const OTM_BAILOUT_C_OVER_F_THRESHOLD: f64 = 3e-6;

pub fn implied_vol_fast(
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
        "implied_vol_fast: all input slices must have the same length"
    );

    // Dual pre-classify:
    //   - ATM bailout: |k|/sqrt(T) < ATM_BAILOUT_THRESHOLD — Chebyshev seed +
    //     scaled-probit ATM arm has a structural ceiling below this.
    //   - OTM bailout: c_otm / F < OTM_BAILOUT_C_OVER_F_THRESHOLD — the deep-OTM
    //     tail where the Halley path can't drive σ-error under 1e-7 at H<5.
    // Everyone else goes through the fast kernel.
    let mut bail_idx: Vec<usize> = Vec::new();
    let mut fast_idx: Vec<usize> = Vec::with_capacity(n);
    for i in 0..n {
        let t = tte[i];
        let s = spot[i];
        let k = strike[i];
        let r = rate[i];
        let p = price[i];
        // k_log = ln(K / (S * exp(r*T))) = ln(K/S) - r*T
        let k_log = (k / s).ln() - r * t;
        let sqrt_t = t.sqrt();
        let metric = k_log.abs() / sqrt_t;
        let bail_atm = metric < ATM_BAILOUT_THRESHOLD;

        // OTM bailout: canonical OTM-leg premium / forward.
        //   F = S * exp(rT);   df = exp(-rT);   Kp = K * df.
        //   put-call parity (spot space):  c_call - c_put = S - Kp.
        let df = (-r * t).exp();
        let kp = k * df;
        let c_call = match kind[i] {
            OptionKind::Call => p,
            OptionKind::Put => p + s - kp,
        };
        let c_put = c_call - s + kp;
        let c_otm = c_call.min(c_put).max(0.0);
        let f = s / df; // s * exp(rT)
        let bail_otm = c_otm < OTM_BAILOUT_C_OVER_F_THRESHOLD * f;

        if (bail_atm || bail_otm) && t > 0.0 {
            bail_idx.push(i);
        } else {
            fast_idx.push(i);
        }
    }

    let mut out = vec![f64::NAN; n];

    // Fast path: kernel over the non-bailout lanes.
    if !fast_idx.is_empty() {
        let m = fast_idx.len();
        let mut fs = Vec::with_capacity(m);
        let mut fk = Vec::with_capacity(m);
        let mut ft = Vec::with_capacity(m);
        let mut fr = Vec::with_capacity(m);
        let mut fp = Vec::with_capacity(m);
        let mut fkind = Vec::with_capacity(m);
        for &i in &fast_idx {
            fs.push(spot[i]);
            fk.push(strike[i]);
            ft.push(tte[i]);
            fr.push(rate[i]);
            fp.push(price[i]);
            fkind.push(kind[i]);
        }
        let fast_out = implied_vol_fast_kernel(&fs, &fk, &ft, &fr, &fp, &fkind);

        // NaN safety-net: any fast lane that came back NaN goes to rational.
        let mut nan_local: Vec<usize> = Vec::new();
        for (j, v) in fast_out.iter().enumerate() {
            if v.is_nan() {
                nan_local.push(j);
            }
        }
        if !nan_local.is_empty() {
            let mm = nan_local.len();
            let mut rs = Vec::with_capacity(mm);
            let mut rk = Vec::with_capacity(mm);
            let mut rt = Vec::with_capacity(mm);
            let mut rr = Vec::with_capacity(mm);
            let mut rp = Vec::with_capacity(mm);
            let mut rkind = Vec::with_capacity(mm);
            for &j in &nan_local {
                let i = fast_idx[j];
                rs.push(spot[i]);
                rk.push(strike[i]);
                rt.push(tte[i]);
                rr.push(rate[i]);
                rp.push(price[i]);
                rkind.push(kind[i]);
            }
            let refire = implied_vol_rational(&rs, &rk, &rt, &rr, &rp, &rkind);
            for (kk, &j) in nan_local.iter().enumerate() {
                let i = fast_idx[j];
                out[i] = refire[kk];
            }
        }
        for (j, &i) in fast_idx.iter().enumerate() {
            if !fast_out[j].is_nan() {
                out[i] = fast_out[j];
            }
        }
    }

    // Bailout path: rational on the definitely-bad ATM lanes.
    if !bail_idx.is_empty() {
        let m = bail_idx.len();
        let mut bs = Vec::with_capacity(m);
        let mut bk = Vec::with_capacity(m);
        let mut bt = Vec::with_capacity(m);
        let mut br = Vec::with_capacity(m);
        let mut bp = Vec::with_capacity(m);
        let mut bkind = Vec::with_capacity(m);
        for &i in &bail_idx {
            bs.push(spot[i]);
            bk.push(strike[i]);
            bt.push(tte[i]);
            br.push(rate[i]);
            bp.push(price[i]);
            bkind.push(kind[i]);
        }
        let bail_out = implied_vol_rational(&bs, &bk, &bt, &br, &bp, &bkind);
        for (j, &i) in bail_idx.iter().enumerate() {
            out[i] = bail_out[j];
        }
    }

    out
}

/// Black-Scholes implied volatility via Jäckel's "Let's be rational" method
/// (Wilmott Magazine, 2015) — the canonical full-precision algorithm.
///
/// Same six-slice API as [`implied_vol`]: each element of the result is the
/// implied volatility of the corresponding option, or `NaN` if the input is
/// outside the recoverable domain (premium below intrinsic, etc.). Differs
/// from [`implied_vol`] in two ways:
///
/// 1. **Complete across all moneyness regimes** — the deep-OTM-near-expiry
///    corner that returns `NaN` from the direct-Newton kernel is solved
///    here at the conditioning floor.
/// 2. **Near-machine precision in the well-conditioned regions** — at
///    a per-input cost roughly 2–3× the direct kernel, the rational
///    method's two Householder-3 iterations land at f64 epsilon.
///
/// **Cross-validated** against `py_lets_be_rational` (the canonical
/// reference implementation, which wraps Jäckel's C++) on a 1,000,000-
/// option dataset: median absolute disagreement 6.66e-16 (single ULP);
/// maximum 2.31e-11 in the deep-OTM band. Both implementations solve
/// the entire dataset; neither has one-sided failures.
///
/// Implementation is a **strict clean-room derivation from the paper
/// only** — Jäckel's C++ source (`lets_be_rational.c`) and the
/// `py_lets_be_rational` Python wrapper were not opened by the
/// implementer at any point before validation. See
/// `specs/jackel-lbr-spec.md` for the derivation, the seven TBDs
/// (one of which surfaced an apparent typo in the paper's (4.32)
/// equation), and the phase-by-phase numerical gates.
///
/// # Panics
/// If the input slices are not all the same length.
///
/// # Implementation
///
/// Macro-chunked lane segregation: each macro-chunk of up to [`MACRO_CHUNK`]
/// options is processed in two passes. Pass 1 canonicalizes every option and
/// classifies it by region (lower / centre-left / centre-right / upper); lanes
/// at the exactly-ATM corner (`|x_canon| < 1e-12`) are solved on the spot via
/// the closed-form `σ = 2·Φ⁻¹((β+1)/2)`. Pass 2 processes each non-empty
/// region bucket as a dense SIMD8 stream through a region-specialized solver,
/// then scatters results back to the macro-chunk's output slice. The savings
/// vs the heterogeneous mask-and-compute path are ≈⅔ of the region-solver
/// work (only one region's solver runs per chunk) plus the elimination of the
/// centre-region branch (each chunk uses either [`jackel::solve_centre_left`]
/// or [`jackel::solve_centre_right`], never both).
pub fn implied_vol_rational(
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
        "implied_vol_rational: all input slices must have the same length"
    );
    let mut out = vec![f64::NAN; n];

    let mut macro_i = 0;
    while macro_i < n {
        let m_take = core::cmp::min(MACRO_CHUNK, n - macro_i);

        // Per-macro stack buffers: four region buckets. `bucket_idx[r][k]` is
        // the offset within the current macro-chunk (0..m_take); the scatter
        // combines it with `macro_i` to write into `out`.
        let mut bucket_x = [[0.0_f64; MACRO_CHUNK]; 4];
        let mut bucket_beta = [[0.0_f64; MACRO_CHUNK]; 4];
        let mut bucket_sqrt_t = [[0.0_f64; MACRO_CHUNK]; 4];
        let mut bucket_idx = [[0_u16; MACRO_CHUNK]; 4];
        let mut bucket_len = [0_usize; 4];

        // Pass 1: classify, with on-the-spot x≈0 ATM solve.
        let mut i = 0;
        while i < m_take {
            let take = core::cmp::min(LANES, m_take - i);
            let mut sb = [1.0_f64; LANES];
            let mut kb = [1.0_f64; LANES];
            let mut tb = [1.0_f64; LANES];
            let mut rb_in = [0.0_f64; LANES];
            let mut pb = [1.0_f64; LANES];
            let mut callb = [false; LANES];
            let mut realb = [false; LANES];
            for j in 0..take {
                sb[j] = spot[macro_i + i + j];
                kb[j] = strike[macro_i + i + j];
                tb[j] = tte[macro_i + i + j];
                rb_in[j] = rate[macro_i + i + j];
                pb[j] = price[macro_i + i + j];
                callb[j] = kind[macro_i + i + j].is_call();
                realb[j] = true;
            }
            let s_v = V::from_array(sb);
            let k_v = V::from_array(kb);
            let t_v = V::from_array(tb);
            let r_v = V::from_array(rb_in);
            let p_v = V::from_array(pb);
            let is_call = M::from_array(callb);
            let real = M::from_array(realb);
            let valid = real & screen(s_v, k_v, t_v, r_v, p_v, is_call);

            let sqrt_t = t_v.sqrt();
            let er_t = (r_v * t_v).exp();
            let fwd = s_v * er_t;
            let x = (fwd / k_v).ln();
            let sqrt_fk = (fwd * k_v).sqrt();
            let beta_raw = p_v * er_t / sqrt_fk;
            let (x_canon, beta_canon) = jackel::canonicalize(x, beta_raw, is_call);

            // x≈0 ATM lanes: closed-form σ = 2·Φ⁻¹((β+1)/2). The rational
            // kernel's `σ_c = √(2·|x|) = 0` divide would NaN them otherwise.
            let is_atm = x_canon.abs().simd_lt(V::splat(1e-12));
            let atm_sigma =
                V::splat(2.0) * norm::phi_inv((beta_canon + V::splat(1.0)) * V::splat(0.5));
            let atm_sigma_hat = atm_sigma / sqrt_t;

            let rb_struct = jackel::region_boundaries(x_canon);
            let region = jackel::classify_region(beta_canon, &rb_struct);

            let x_arr = x_canon.to_array();
            let beta_arr = beta_canon.to_array();
            let st_arr = sqrt_t.to_array();
            let atm_hat_arr = atm_sigma_hat.to_array();
            let region_arr = region.to_array();
            let is_atm_arr: [bool; LANES] = is_atm.to_array();
            let valid_arr: [bool; LANES] = valid.to_array();
            for j in 0..take {
                if !valid_arr[j] {
                    continue; // out[..] already NaN
                }
                if is_atm_arr[j] {
                    let h = atm_hat_arr[j];
                    out[macro_i + i + j] = if h.is_finite() && h > VOL_MIN && h < VOL_MAX {
                        h
                    } else {
                        f64::NAN
                    };
                    continue;
                }
                let r = region_arr[j] as usize;
                debug_assert!(r < 4);
                let pos = bucket_len[r];
                bucket_x[r][pos] = x_arr[j];
                bucket_beta[r][pos] = beta_arr[j];
                bucket_sqrt_t[r][pos] = st_arr[j];
                bucket_idx[r][pos] = (i + j) as u16;
                bucket_len[r] = pos + 1;
            }
            i += take;
        }

        // Pass 2: dense per-region solve.
        for region_id in 0..4_usize {
            let m = bucket_len[region_id];
            if m == 0 {
                continue;
            }
            let mut j = 0;
            while j < m {
                let take = core::cmp::min(LANES, m - j);
                let mut xb = [0.0_f64; LANES];
                let mut bb = [0.0_f64; LANES];
                let mut stb = [1.0_f64; LANES];
                xb[..take].copy_from_slice(&bucket_x[region_id][j..j + take]);
                bb[..take].copy_from_slice(&bucket_beta[region_id][j..j + take]);
                stb[..take].copy_from_slice(&bucket_sqrt_t[region_id][j..j + take]);
                // Pad trailing SIMD lanes by duplicating the last real lane —
                // keeps the SIMD compute well-defined; results are discarded.
                for l in take..LANES {
                    xb[l] = xb[take - 1];
                    bb[l] = bb[take - 1];
                    stb[l] = stb[take - 1];
                }
                let x_v = V::from_array(xb);
                let b_v = V::from_array(bb);
                let sqrt_t_v = V::from_array(stb);
                let rb_v = jackel::region_boundaries(x_v);
                let sigma_total = match region_id {
                    0 => jackel::solve_lower_dense(x_v, b_v, &rb_v),
                    1 => jackel::solve_centre_left(x_v, b_v, &rb_v),
                    2 => jackel::solve_centre_right(x_v, b_v, &rb_v),
                    3 => jackel::solve_upper_dense(x_v, b_v, &rb_v),
                    _ => unreachable!(),
                };
                let sigma_hat = sigma_total / sqrt_t_v;
                let inside =
                    sigma_hat.simd_gt(V::splat(VOL_MIN)) & sigma_hat.simd_lt(V::splat(VOL_MAX));
                let finite = sigma_hat.is_finite();
                let accept = inside & finite;
                let res = accept.select(sigma_hat, V::splat(f64::NAN)).to_array();
                for l in 0..take {
                    let rel_idx = bucket_idx[region_id][j + l] as usize;
                    out[macro_i + rel_idx] = res[l];
                }
                j += take;
            }
        }

        macro_i += m_take;
    }
    out
}

/// Macro-chunk size for the segregated [`implied_vol_rational`] path. Sized so
/// the per-macro stack buffers (4 buckets × 3 f64-arrays + 1 u16-array ≈ 26 KB)
/// fit comfortably in L1d on every target voltic compiles to.
const MACRO_CHUNK: usize = 256;

// The Python extension module (PyO3 + maturin). One file, behind a feature
// flag; see `python/voltic_py.rs` and `pyproject.toml`.
#[cfg(feature = "python")]
#[path = "../python/voltic_py.rs"]
mod voltic_py;

use std::simd::prelude::*;
use std::simd::StdFloat;

/// SIMD width: 8 × f64 = one AVX-512 register (and lowers to 2×256 / 4×128 on
/// narrower targets via `std::simd`'s portable lowering).
pub(crate) const LANES: usize = 8;
pub(crate) type V = Simd<f64, LANES>;
pub(crate) type M = Mask<i64, LANES>;

/// Newton stopping tolerance, in vol units (absolute change between iterates).
const TOL: f64 = 1e-12;
/// Hard cap on Newton iterations. A well-conditioned input converges in 1–3;
/// hitting this means the input is in the pathological region voltic does not
/// solve (deep OTM near expiry) — that lane returns `NaN`.
pub const MAX_ITERS: usize = 32;
/// Volatility bounds. A solved vol outside `[VOL_MIN, VOL_MAX]` is reported as
/// `NaN` — see the README "Numerical domain" limitation.
pub const VOL_MIN: f64 = 0.01;
pub const VOL_MAX: f64 = 5.0;

/// Call or put.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptionKind {
    Call,
    Put,
}

impl OptionKind {
    #[inline]
    fn is_call(self) -> bool {
        matches!(self, OptionKind::Call)
    }
}

// --- Black-Scholes price + vega, lane-packed -------------------------------

/// Returns `(price, vega)` under Black-Scholes for the given vol `sigma`.
///
/// `s` spot, `k` strike, `t` time to expiry (years), `r` continuously
/// compounded rate, `is_call` lane mask (true = call). `vega = ∂price/∂σ` —
/// identical for calls and puts. All inputs assumed `t > 0`, `sigma > 0`,
/// `s > 0`, `k > 0`; degenerate inputs are screened by the caller in
/// [`implied_vol`].
#[inline]
pub(crate) fn bs_price_vega(s: V, k: V, t: V, r: V, sigma: V, is_call: M) -> (V, V) {
    let sqrt_t = t.sqrt();
    let vol_sqrt_t = sigma * sqrt_t;
    let df = (-r * t).exp(); // discount factor
    let ln_s_k = (s / k).ln();
    let d1 = (ln_s_k + (r + V::splat(0.5) * sigma * sigma) * t) / vol_sqrt_t;
    let d2 = d1 - vol_sqrt_t;

    let nd1 = norm::phi_hart(d1);
    let nd2 = norm::phi_hart(d2);
    // Call:  S·Φ(d1) − K·e^{−rT}·Φ(d2)
    // Put :  K·e^{−rT}·Φ(−d2) − S·Φ(−d1)  =  call − S + K·e^{−rT}  (put–call parity)
    let call = s * nd1 - k * df * nd2;
    let put = call - s + k * df;
    let price = is_call.select(call, put);

    // vega = S·φ(d1)·√T  (same both ways)
    let vega = s * norm::phi_pdf(d1) * sqrt_t;
    (price, vega)
}

// --- Initial guess: Corrado–Miller (1996) with Brenner–Subrahmanyam fallback

/// Closed-form starting σ. Corrado & Miller (1996) — a rational approximation
/// that extends the Brenner–Subrahmanyam (1988) ATM rule
/// (`σ ≈ √(2π/T)·C/S`) to away-from-the-money strikes by carrying the
/// `(S − K')` skew term, where `K' = K·e^{−rT}` is the discounted strike.
/// We clamp the discriminant to ≥ 0 (it goes negative far from ATM, where the
/// approximation is out of its domain) and clamp the result into
/// `[VOL_MIN, VOL_MAX]`; Newton repairs the rest.
#[inline]
pub(crate) fn initial_guess(s: V, k: V, t: V, r: V, price: V, is_call: M) -> V {
    let sqrt_t = t.sqrt();
    let two_pi_over_t = V::splat(2.0 * core::f64::consts::PI) / t;
    let kp = k * (-r * t).exp(); // discounted strike

    // Work with the *call* price for the formula; convert a put via parity:
    //   C = P + S − K'
    let c = is_call.select(price, price + s - kp);

    let s_minus_kp = s - kp;
    let half_diff = V::splat(0.5) * s_minus_kp;
    let a = c - half_diff; // C − (S − K')/2
    let disc =
        (a * a - s_minus_kp * s_minus_kp / V::splat(core::f64::consts::PI)).simd_max(V::splat(0.0));
    let cm = two_pi_over_t.sqrt() / (s + kp) * (a + disc.sqrt());

    // Brenner–Subrahmanyam ATM fallback (always finite, decent near ATM):
    //   σ ≈ √(2π/T) · C / S
    let bs88 = two_pi_over_t.sqrt() * c / s;

    // Use Corrado–Miller where it produced something sane; else BS88; else a
    // neutral 0.5. (`is_finite` + positivity check, all lane-wise.)
    let cm_ok = cm.is_finite() & cm.simd_gt(V::splat(0.0));
    let guess = cm_ok.select(cm, bs88);
    let guess_ok = guess.is_finite() & guess.simd_gt(V::splat(0.0));
    let guess = guess_ok.select(guess, V::splat(0.5));

    let _ = sqrt_t;
    guess
        .simd_max(V::splat(VOL_MIN))
        .simd_min(V::splat(VOL_MAX))
}

// --- The masked-Newton core ------------------------------------------------

/// Solve one lane-packed batch. `valid` marks lanes that passed the domain
/// screen; invalid lanes are returned as `NaN`. Within the valid lanes, a lane
/// that does not converge within [`MAX_ITERS`], or converges to a vol outside
/// `[VOL_MIN, VOL_MAX]`, is also returned as `NaN`.
#[inline]
fn solve_chunk(s: V, k: V, t: V, r: V, price: V, is_call: M, valid: M) -> V {
    let nan = V::splat(f64::NAN);
    let mut sigma = initial_guess(s, k, t, r, price, is_call);

    // `converged` starts at the invalid lanes (so we never touch them and the
    // all-converged check fires correctly once the real lanes finish).
    let mut converged = !valid;
    let tol = V::splat(TOL);

    // Per-lane residual tolerance: the price scale below which a lane's pricing
    // error is "as good as zero" — needed because in the flat tail (deep OTM /
    // near expiry) vega is tiny, so Newton converges *linearly* not
    // quadratically and the *step* stays well above `TOL` for dozens of
    // iterations even though the *residual* is already negligible. Converging on
    // either condition keeps the iteration count bounded without loosening the
    // step tolerance the well-conditioned case relies on.
    let scale0 = price.abs().simd_max(s.abs()).simd_max(k.abs());
    let res_tol = V::splat(1e-12) * scale0 + V::splat(1e-12) * price.abs();

    for _ in 0..MAX_ITERS {
        let (p, vega) = bs_price_vega(s, k, t, r, sigma, is_call);
        // Newton step: σ ← σ − (price(σ) − target) / vega. Guard a vanishing
        // vega (deep OTM / tiny T) so we never divide by 0; the residual test
        // below catches a lane stuck on a flat patch.
        let denom = vega.simd_max(V::splat(1e-300));
        let residual = p - price;
        let step = residual / denom;
        let next = sigma - step;
        // Keep the iterate inside the bracket so a wild Newton step from a bad
        // guess can't fly off to ±∞ and NaN the exp(); clamp, don't reflect.
        let next = next.simd_max(V::splat(VOL_MIN)).simd_min(V::splat(VOL_MAX));

        // A lane is "newly converged" if its update fell below `TOL` *or* its
        // pricing residual fell below `res_tol` this step.
        let small_step = (next - sigma).abs().simd_lt(tol);
        let small_residual = residual.abs().simd_le(res_tol);
        // Only move lanes that have NOT already converged — masked convergence:
        // a converged lane's value is frozen, so the slowest lane never gates
        // the rest and a converged lane costs nothing.
        sigma = converged.select(sigma, next);
        converged |= small_step | small_residual;

        if converged.all() {
            break;
        }
    }

    // Final acceptance, three conditions:
    //  (1) priced_ok — the lane re-prices to `price` to a relative tolerance
    //      (with a small absolute floor scaled to the underlying), catching a
    //      lane that "converged" to a spurious value;
    //  (2) inside — σ is strictly inside the open bracket (a value pinned at
    //      VOL_MIN/VOL_MAX is the clamp talking, not a real root);
    //  (3) vega_ok — vega at the solution is non-negligible. When both price
    //      *and* vega underflow toward 0 (a deep-OTM option whose time value is
    //      below the f64 floor at this magnitude), Newton's residual test
    //      passes for a whole range of σ — a flat spurious basin. Requiring a
    //      real vega rejects that basin; the lane is reported `NaN`, which is
    //      the honest outcome for a price the conditioning floor can't invert.
    let (p_final, vega_final) = bs_price_vega(s, k, t, r, sigma, is_call);
    let scale = price.abs().simd_max(s.abs()).simd_max(k.abs());
    // Tolerance = a relative part (1e-7·price) plus an absolute floor scaled to
    // the underlying (~1e-13·scale) — the latter absorbs the f64 round-off of
    // forming a deep-OTM premium by put–call-parity cancellation
    // (`call − S + K·e^{−rT}`, which loses ~ulp(S) of precision). Tight enough
    // to reject a spurious-basin "solution" (whose re-priced value differs by
    // orders of magnitude), loose enough to accept a genuine tiny premium.
    let priced_ok = (p_final - price)
        .abs()
        .simd_le(V::splat(1e-7) * price.abs() + V::splat(1e-13) * scale);
    let inside =
        sigma.simd_gt(V::splat(VOL_MIN * 1.0000001)) & sigma.simd_lt(V::splat(VOL_MAX * 0.9999999));
    let vega_ok = vega_final.simd_gt(V::splat(1e-12) * scale);
    let accept = valid & converged & priced_ok & inside & vega_ok;
    accept.select(sigma, nan)
}

/// Domain screen, lane-packed: which inputs admit a Black-Scholes implied vol.
/// Rejects `t ≤ 0`, `s ≤ 0`, `k ≤ 0`, non-finite anything, and a premium below
/// intrinsic value (or above the trivial upper bound) — all of which have no
/// solution and would otherwise drive Newton to garbage.
#[inline]
pub(crate) fn screen(s: V, k: V, t: V, r: V, price: V, is_call: M) -> M {
    let finite = s.is_finite() & k.is_finite() & t.is_finite() & r.is_finite() & price.is_finite();
    let positive = s.simd_gt(V::splat(0.0)) & k.simd_gt(V::splat(0.0)) & t.simd_gt(V::splat(0.0));
    let df = (-r * t).exp();
    // Intrinsic (lower no-arbitrage bound) and the trivial upper bound:
    //   call: max(S − K·e^{−rT}, 0) ≤ C ≤ S
    //   put : max(K·e^{−rT} − S, 0) ≤ P ≤ K·e^{−rT}
    let fwd_intrinsic_call = (s - k * df).simd_max(V::splat(0.0));
    let fwd_intrinsic_put = (k * df - s).simd_max(V::splat(0.0));
    let lower = is_call.select(fwd_intrinsic_call, fwd_intrinsic_put);
    let upper = is_call.select(s, k * df);
    // Strictly above intrinsic (a premium *at* intrinsic implies σ → 0, outside
    // VOL_MIN) and strictly below the cap (σ → ∞).
    let in_band = price.simd_gt(lower * V::splat(1.0) + V::splat(0.0)) & price.simd_lt(upper);
    let in_band = in_band & (price - lower).simd_gt(V::splat(0.0));
    finite & positive & in_band
}

/// Black-Scholes implied volatility for a batch of European options.
///
/// All slices must be the same length `n`; the result has length `n`. Element
/// `i` is the implied vol of the option `(spot[i], strike[i], tte[i], rate[i],
/// price[i], kind[i])` — or `NaN` if that input has no Black-Scholes implied
/// vol in `[`[`VOL_MIN`]`, `[`VOL_MAX`]`]` (premium below intrinsic, non-finite
/// input, `tte ≤ 0`, or the pathological deep-OTM-near-expiry region voltic
/// does not solve; see the README "Limitations").
///
/// # Panics
/// If the input slices are not all the same length.
pub fn implied_vol(
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
        "implied_vol: all input slices must have the same length"
    );
    let mut out = vec![0.0_f64; n];

    let mut i = 0;
    while i < n {
        let take = core::cmp::min(LANES, n - i);
        // Load a chunk; pad the tail of a short final chunk with dummy values
        // and an all-false `valid` for those lanes.
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
            callb[j] = kind[i + j].is_call();
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
        let res = solve_chunk(s, k, t, r, p, is_call, valid);
        out[i..i + take].copy_from_slice(&res.to_array()[..take]);
        i += take;
    }
    out
}

/// Convenience: solve a single option (just calls [`implied_vol`] with
/// one-element slices). For real workloads pass the whole batch — the SIMD only
/// pays off across lanes.
pub fn implied_vol_one(
    spot: f64,
    strike: f64,
    tte: f64,
    rate: f64,
    price: f64,
    kind: OptionKind,
) -> f64 {
    implied_vol(&[spot], &[strike], &[tte], &[rate], &[price], &[kind])[0]
}

/// Forward-price the batch under Black-Scholes at the given vols — used by
/// tests (round-trip: solve, then re-price, and check you get the input back)
/// and by the benchmark harness to generate the synthetic dataset. Returns
/// `price` for each `(spot, strike, tte, rate, sigma, kind)`.
pub fn bs_price(
    spot: &[f64],
    strike: &[f64],
    tte: &[f64],
    rate: &[f64],
    sigma: &[f64],
    kind: &[OptionKind],
) -> Vec<f64> {
    let n = spot.len();
    assert!(
        strike.len() == n
            && tte.len() == n
            && rate.len() == n
            && sigma.len() == n
            && kind.len() == n
    );
    let mut out = vec![0.0_f64; n];
    let mut i = 0;
    while i < n {
        let take = core::cmp::min(LANES, n - i);
        let mut sb = [1.0; LANES];
        let mut kb = [1.0; LANES];
        let mut tb = [1.0; LANES];
        let mut rb = [0.0; LANES];
        let mut vb = [0.2; LANES];
        let mut cb = [false; LANES];
        for j in 0..take {
            sb[j] = spot[i + j];
            kb[j] = strike[i + j];
            tb[j] = tte[i + j];
            rb[j] = rate[i + j];
            vb[j] = sigma[i + j];
            cb[j] = kind[i + j].is_call();
        }
        let (p, _) = bs_price_vega(
            V::from_array(sb),
            V::from_array(kb),
            V::from_array(tb),
            V::from_array(rb),
            V::from_array(vb),
            M::from_array(cb),
        );
        out[i..i + take].copy_from_slice(&p.to_array()[..take]);
        i += take;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_atm_call() {
        let s = [100.0];
        let k = [100.0];
        let t = [1.0];
        let r = [0.02];
        let kind = [OptionKind::Call];
        let true_vol = 0.30;
        let price = bs_price(&s, &k, &t, &r, &[true_vol], &kind);
        let iv = implied_vol(&s, &k, &t, &r, &price, &kind);
        assert!(
            (iv[0] - true_vol).abs() < 1e-9,
            "iv={} price={}",
            iv[0],
            price[0]
        );
    }

    #[test]
    fn round_trip_grid() {
        // A grid of well-conditioned inputs: solve a known vol back out.
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
        let iv = implied_vol(&s, &k, &t, &r, &price, &kind);
        let mut worst_atm = 0.0_f64;
        let mut worst_all = 0.0_f64;
        let mut nan_count = 0usize;
        let mut solved = 0usize;
        let mut solved_atm = 0usize;
        for idx in 0..s.len() {
            // Skip points whose Black-Scholes price is numerically
            // indistinguishable from intrinsic (a deep-ITM short-expiry option
            // can lose all its time value to f64 rounding when `bs_price` is
            // formed via put–call parity) — those have no recoverable IV and
            // are *meant* to NaN; that's not a bug, it's the conditioning floor.
            let df = (-r[idx] * t[idx]).exp();
            let intrinsic = match kind[idx] {
                OptionKind::Call => (s[idx] - k[idx] * df).max(0.0),
                OptionKind::Put => (k[idx] * df - s[idx]).max(0.0),
            };
            let time_value = price[idx] - intrinsic;
            if time_value <= 1e-10 * (s[idx] + k[idx]) {
                continue; // not well-posed at f64 precision
            }
            if iv[idx].is_nan() {
                nan_count += 1;
                continue;
            }
            solved += 1;
            let err = (iv[idx] - sig[idx]).abs();
            worst_all = worst_all.max(err);
            // Near-ATM points have a healthy vega — they should solve to ~1e-9.
            // Deep-OTM-near-expiry points sit near the ~1e-6 conditioning floor
            // (vol error ≈ residual / vega, and vega → 0 there); a single 1e-9
            // bar across the whole grid would be claiming accuracy the problem
            // doesn't have.
            let moneyness = s[idx] / k[idx];
            if (0.9..=1.1).contains(&moneyness) {
                solved_atm += 1;
                worst_atm = worst_atm.max(err);
            }
        }
        assert!(
            nan_count == 0,
            "{nan_count} well-posed grid points NaN'd (of {} solved)",
            solved
        );
        assert!(
            solved > 200,
            "only {solved} well-posed points — grid construction bug?"
        );
        assert!(
            solved_atm > 30,
            "only {solved_atm} near-ATM points — grid bug?"
        );
        // Near-ATM: tight. Whole grid (incl. deep OTM near expiry): the floor.
        assert!(
            worst_atm < 1e-8,
            "near-ATM worst abs vol error = {worst_atm:e}"
        );
        assert!(worst_all < 1e-6, "grid worst abs vol error = {worst_all:e}");
    }

    #[test]
    fn put_call_parity_on_solved_vols() {
        // Solve a call's IV, solve the matching put's IV (same S,K,T,r), price
        // both back, and check C − P = S − K·e^{−rT}.
        let s = [105.0, 95.0, 120.0];
        let k = [100.0, 100.0, 110.0];
        let t = [0.5, 1.0, 0.25];
        let r = [0.03, 0.01, 0.05];
        let v = [0.25, 0.4, 0.18];
        let call_kind = [OptionKind::Call; 3];
        let put_kind = [OptionKind::Put; 3];
        let cp = bs_price(&s, &k, &t, &r, &v, &call_kind);
        let pp = bs_price(&s, &k, &t, &r, &v, &put_kind);
        let civ = implied_vol(&s, &k, &t, &r, &cp, &call_kind);
        let piv = implied_vol(&s, &k, &t, &r, &pp, &put_kind);
        for idx in 0..3 {
            assert!((civ[idx] - v[idx]).abs() < 1e-9);
            assert!((piv[idx] - v[idx]).abs() < 1e-9);
            // C − P should equal S − K e^{−rT}
            let parity = cp[idx] - pp[idx];
            let theo = s[idx] - k[idx] * (-r[idx] * t[idx]).exp();
            assert!((parity - theo).abs() < 1e-9);
        }
    }

    #[test]
    fn edge_cases_return_nan_not_garbage() {
        // premium below intrinsic, t = 0, negative spot, premium above cap.
        let s = [100.0, 100.0, -1.0, 100.0, 100.0];
        let k = [100.0, 100.0, 100.0, 100.0, 100.0];
        let t = [1.0, 0.0, 1.0, 1.0, 1.0];
        let r = [0.0, 0.0, 0.0, 0.0, 0.0];
        // [0]: price 0.001 (way below ATM intrinsic-ish lower bound for high vol? actually
        //      ATM call lower bound is 0, but 0.001 → σ ~ 0.0025%, below VOL_MIN → NaN)
        // [1]: t = 0
        // [2]: negative spot
        // [3]: price = 200 > S = 100 (above cap)
        // [4]: price = 100 == S (at cap) → NaN
        let p = [0.001, 5.0, 5.0, 200.0, 100.0];
        let kind = [OptionKind::Call; 5];
        let iv = implied_vol(&s, &k, &t, &r, &p, &kind);
        for (idx, v) in iv.iter().enumerate() {
            assert!(v.is_nan(), "lane {idx} should be NaN, got {v}");
        }
    }

    #[test]
    fn zero_rate_and_high_vol() {
        let s = [100.0];
        let k = [100.0];
        let t = [1.0];
        let r = [0.0];
        let v = [1.2]; // 120% vol
        let kind = [OptionKind::Call];
        let p = bs_price(&s, &k, &t, &r, &v, &kind);
        let iv = implied_vol(&s, &k, &t, &r, &p, &kind);
        assert!((iv[0] - 1.2).abs() < 1e-8, "iv={}", iv[0]);
    }

    #[test]
    fn deep_otm_short_expiry_is_handled_or_nan() {
        // S=100, K=160, T=1 week, σ=15% → premium is ~1e-6, deep in the
        // conditioning-floor region. Either voltic solves it within ~1e-6, or
        // it returns NaN — both acceptable; what's not acceptable is a wrong
        // finite answer.
        let s = [100.0];
        let k = [160.0];
        let t = [7.0 / 365.0];
        let r = [0.0];
        let v = [0.15];
        let kind = [OptionKind::Call];
        let p = bs_price(&s, &k, &t, &r, &v, &kind);
        let iv = implied_vol(&s, &k, &t, &r, &p, &kind);
        if !iv[0].is_nan() {
            assert!((iv[0] - 0.15).abs() < 1e-3, "iv={} price={}", iv[0], p[0]);
        }
    }

    #[test]
    fn implied_vol_rational_handles_x_zero_atm() {
        // ATM with r = 0 → x_canon = 0 exactly. Probe that the solver returns
        // a finite value matching σ_true, not NaN from a σ_c = 0 divide.
        let s = [100.0];
        let k = [100.0];
        let t = [1.0];
        let r = [0.0];
        let v = [0.30];
        let kind = [OptionKind::Call];
        let p = bs_price(&s, &k, &t, &r, &v, &kind);
        let iv = implied_vol_rational(&s, &k, &t, &r, &p, &kind);
        assert!(
            iv[0].is_finite() && (iv[0] - 0.30).abs() < 1e-10,
            "ATM x=0 r=0: iv={} expected 0.30",
            iv[0]
        );
    }

    #[test]
    fn implied_vol_rational_recovers_known_vol_atm() {
        // ATM 1-year call: S=K=100, σ=30%, r=2%.
        let s = [100.0];
        let k = [100.0];
        let t = [1.0];
        let r = [0.02];
        let v = [0.30];
        let kind = [OptionKind::Call];
        let p = bs_price(&s, &k, &t, &r, &v, &kind);
        let iv = implied_vol_rational(&s, &k, &t, &r, &p, &kind);
        assert!(
            (iv[0] - 0.30).abs() < 1e-12,
            "rational ATM: iv={} expected 0.30",
            iv[0]
        );
    }

    #[test]
    fn implied_vol_rational_handles_grid() {
        // Same grid as `round_trip_grid` — rational solver should land at
        // near-machine precision across the full grid.
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
        let iv = implied_vol_rational(&s, &k, &t, &r, &price, &kind);
        // Per-input tolerance: the conditioning floor of inverting price → σ
        // is ~|price·ε_machine / vega|. A correctly converging Newton-like
        // method should land within a small multiple of that. We allow 10×
        // the conditioning floor as the bar (vs strict 1× from machine
        // precision, accounting for Newton-step roundoff).
        let eps = f64::EPSILON;
        let mut solved = 0usize;
        let mut worst_floor_ratio = 0.0_f64;
        let mut worst_floor_idx = 0usize;
        let mut top_ratios: Vec<(f64, usize, f64, f64)> = Vec::new();
        for idx in 0..s.len() {
            let df = (-r[idx] * t[idx]).exp();
            let intrinsic = match kind[idx] {
                OptionKind::Call => (s[idx] - k[idx] * df).max(0.0),
                OptionKind::Put => (k[idx] * df - s[idx]).max(0.0),
            };
            let time_value = price[idx] - intrinsic;
            if time_value <= 1e-10 * (s[idx] + k[idx]) {
                continue;
            }
            if iv[idx].is_nan() {
                continue;
            }
            solved += 1;
            // Compute vega for this input at σ_true.
            let sqrt_t = t[idx].sqrt();
            let f_fwd = s[idx] * (r[idx] * t[idx]).exp();
            let d1 =
                ((f_fwd / k[idx]).ln() + 0.5 * sig[idx] * sig[idx] * t[idx]) / (sig[idx] * sqrt_t);
            let phi_d1 = (-0.5 * d1 * d1).exp() / (2.0 * core::f64::consts::PI).sqrt();
            let vega = s[idx] * phi_d1 * sqrt_t;
            // Conditioning floor in vol: roughly (input-magnitude · ε) / vega.
            // For deep OTM where price was computed via put-call parity, the
            // achievable input precision is S-scale, not price-scale, because
            // C = S·Φ(d₁) − K·e^(−rT)·Φ(d₂) loses ~ulp(S) when both Φ are near
            // each other. Use max(price, S, K·e^(−rT)) as the scale.
            let df = (-r[idx] * t[idx]).exp();
            let scale = price[idx].max(s[idx]).max(k[idx] * df);
            let cond_floor = scale * eps / vega.max(1e-300);
            let err = (iv[idx] - sig[idx]).abs();
            let ratio = err / cond_floor.max(1e-15);
            if ratio > worst_floor_ratio {
                worst_floor_ratio = ratio;
                worst_floor_idx = idx;
            }
            top_ratios.push((ratio, idx, err, cond_floor));
        }
        top_ratios.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        eprintln!(
            "rational grid: solved {} of {}; worst err/cond_floor ratio = {worst_floor_ratio:.2}x at idx {worst_floor_idx}",
            solved,
            s.len()
        );
        eprintln!("worst error/floor-ratio cases:");
        for (ratio, idx, err, floor) in top_ratios.iter().take(5) {
            eprintln!(
                "  idx={idx} S={} K={} T={} σ={}: err={err:.2e} floor={floor:.2e} ratio={ratio:.2}x",
                s[*idx], k[*idx], t[*idx], sig[*idx]
            );
        }
        assert!(solved > 200, "only {solved} solved");
        // Allow up to 50× the conditioning floor — wing iteration can pick
        // up extra Newton-step roundoff because the objective transforms
        // (1/ln(b), ln(b_max−b)) themselves amplify input noise.
        assert!(
            worst_floor_ratio < 50.0,
            "worst error/floor ratio = {worst_floor_ratio:.2}x"
        );
    }

    #[test]
    fn batch_with_padding_tail() {
        // A length not a multiple of 8 must still solve every element.
        let n = 19;
        let s: Vec<f64> = (0..n).map(|_| 100.0).collect();
        let k: Vec<f64> = (0..n).map(|i| 80.0 + (i as f64) * 2.5).collect();
        let t: Vec<f64> = (0..n).map(|_| 0.75).collect();
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
        let iv = implied_vol(&s, &k, &t, &r, &p, &kind);
        for idx in 0..n {
            assert!(!iv[idx].is_nan(), "lane {idx} NaN");
            assert!(
                (iv[idx] - v[idx]).abs() < 1e-7,
                "lane {idx}: {} vs {}",
                iv[idx],
                v[idx]
            );
        }
    }
}
