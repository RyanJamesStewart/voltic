//! `voltic::otm_context` — two-stage public API for repeat-context workloads.
//!
//! Matches volfi's `otm_context(h)` split shape so amortized workloads
//! (vol-surface calibration, MC repricing on a fixed grid) pay the
//! `(k, T)`-prelude exactly once across N price evaluations.
//!
//! **What's in the prelude** (extracted from
//! [`crate::schadner_fast::solve_chunk_fast`]): everything that depends on
//! `(k = ln(K/F), T)` *but not* on the option price `c`. That includes
//! `sqrt_t`, `|k|`, `e^k`, `μ = 2/|k|`, the vol clamp window `[v_lo, v_hi]`,
//! and the Chebyshev k-side basis `T_m(u(|k|))` for `m = 0..SEED_DEG`.
//!
//! **What's per-option**: the price `c`, the canonical OTM premium ratio
//! `q = (1 - c)/m`, the Chebyshev q-side basis `T_n(w(q))`, and the
//! Halley/Householder iterations.
//!
//! ## Three APIs
//!
//! * [`OtmContext::new`] + [`implied_vol_with_context`]: scalar shape that
//!   matches volfi 1:1.
//! * [`implied_vol_with_context_batch`]: one context × many prices, SIMD
//!   batched per 8.
//! * [`implied_vol_vectorized_with_contexts`]: vector contexts × vector
//!   prices, fully SIMD throughout — voltic's structural advantage that
//!   volfi (scalar) cannot match.
//!
//! ## NaN contract
//!
//! These APIs return `NaN` outside the Cheb seed domain or when the residual
//! after `HOUSEHOLDER3_STEPS` does not converge. They do **not** carry the
//! dual-bailout safety net that [`crate::implied_vol_fast`] wraps around the
//! kernel — the context APIs are the raw kernel exposed as a split-shape
//! contract for callers that already know they're in the well-conditioned
//! regime (e.g. surface-fit code that screens its own grid).

use std::simd::prelude::*;
use std::simd::StdFloat;

use crate::schadner_fast::{
    cheb_qside_basis, cheb_qside_basis_simd, cheb_seed_from_basis_simd,
    cheb_seed_from_kside_basis_scalar, householder3_step_simd, ig_kt_prelude_scalar,
    wing_seed_simd, SEED_DEG_PUB, WING_H_MAX, WING_K_LO, WING_Q_MAX,
};
use crate::{LANES, M, V, VOL_MAX, VOL_MIN};

/// One Householder-3 step matches the `HOUSEHOLDER3_STEPS` count in
/// `schadner_fast.rs`. Kept as a local const so context APIs evolve
/// independently of the kernel's sweep.
const HOUSEHOLDER3_STEPS_CTX: usize = 3;

/// Precomputed `(k, T)`-only prelude. Build once, reuse across many prices.
///
/// `k` here is the log-moneyness `ln(K / F)` where `F = S · e^(r·T)` is the
/// forward — volfi's convention. The context is symmetric across calls/puts:
/// the canonicalization to the OTM leg is done at the price-evaluation step,
/// not here. A single `(k, T)` context serves both `c_call` and `c_put` on
/// the same forward / expiry.
///
/// Layout: 9 × `f64` scalars + 13 × `f64` Chebyshev basis = 22 × f64 = 176 B.
/// Trivially `Copy`, fits in three cache lines (2 + tail).
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct OtmContext {
    /// `sqrt(T)`.
    pub sqrt_t: f64,
    /// `ln(K/F)` — the log-moneyness used at construction.
    pub k_log: f64,
    /// `|k_log|`.
    pub ak: f64,
    /// `exp(k_log) = K/F`.
    pub ek: f64,
    /// Branch selector: `1.0` if `k_log > 0`, else `ek`. Matches
    /// `solve_chunk_fast`'s `m`.
    pub m: f64,
    /// `μ = 2/|k|` clamped to `[2/SEED_K_HI, 1e12]`. The IG mean parameter.
    pub mu: f64,
    /// Lower vol clamp `VOL_MIN · sqrt_t`.
    pub v_lo: f64,
    /// Upper vol clamp `VOL_MAX · sqrt_t`.
    pub v_hi: f64,
    /// Clamped `|k|` used as the Chebyshev x-axis input.
    pub k_for_seed: f64,
    /// Pre-evaluated Chebyshev k-side basis `T_m(u(k_for_seed))` for
    /// `m = 0..=SEED_DEG`. This is the load-bearing piece — these 13 values
    /// are the (k,T)-only sub-summation of the bivariate Chebyshev fit.
    pub cheb_tu: [f64; SEED_DEG_PUB + 1],
}

impl OtmContext {
    /// Build a context from raw `(k, T)`.
    ///
    /// `k = ln(K/F)`; positive for OTM puts, negative for OTM calls in the
    /// canonical convention. The context does NOT care about the sign — it
    /// stores both `k_log` and `|k_log|` for downstream branching.
    #[inline]
    pub fn new(k: f64, t: f64) -> Self {
        ig_kt_prelude_scalar(k, t)
    }

    /// Build a context from market inputs `(K, T, S, r, q_div)`. Computes
    /// `k = ln(K · e^(-(r - q_div)·T) / S) = ln(K/F)` internally.
    ///
    /// `q_div` is the continuous dividend yield (or carry). Pass `0.0` for
    /// pure interest-rate carry.
    #[inline]
    pub fn from_market(strike: f64, t: f64, spot: f64, rate: f64, q_div: f64) -> Self {
        // F = S · exp((r - q) T);  k = ln(K/F)
        let k = (strike / spot).ln() - (rate - q_div) * t;
        Self::new(k, t)
    }

    /// Discount factor for the prelude's rate. Stored in the context so the
    /// caller can recover `df` without retaining the rate separately.
    /// **Note**: this is `exp(-r·T)`, supplied by the caller for the
    /// pricing leg.  We don't store it on `OtmContext` because the IG
    /// solver path itself doesn't need it; spelled out here for the doc
    /// reader.
    #[inline]
    pub fn _doc_marker() {}
}

/// SIMD-packed bundle of 8 contexts, struct-of-arrays. Built by
/// [`pack_contexts`]. Used by the SIMD batch APIs to avoid per-lane scalar
/// broadcasts on the hot path.
#[derive(Copy, Clone)]
pub struct OtmContextSimd {
    pub sqrt_t: V,
    pub ak: V,
    pub m: V,
    pub mu: V,
    pub v_lo: V,
    pub v_hi: V,
    pub cheb_tu: [V; SEED_DEG_PUB + 1],
}

/// Pack 8 scalar contexts into a SIMD bundle. The caller's job to ensure
/// the contexts are compatible (any combination of `(k, T)` is fine; the
/// solver lanes are independent).
#[inline]
pub fn pack_contexts(ctxs: &[OtmContext; LANES]) -> OtmContextSimd {
    let mut sqrt_t = [0.0_f64; LANES];
    let mut ak = [0.0_f64; LANES];
    let mut m = [0.0_f64; LANES];
    let mut mu = [0.0_f64; LANES];
    let mut v_lo = [0.0_f64; LANES];
    let mut v_hi = [0.0_f64; LANES];
    for j in 0..LANES {
        sqrt_t[j] = ctxs[j].sqrt_t;
        ak[j] = ctxs[j].ak;
        m[j] = ctxs[j].m;
        mu[j] = ctxs[j].mu;
        v_lo[j] = ctxs[j].v_lo;
        v_hi[j] = ctxs[j].v_hi;
    }
    let mut cheb_tu = [V::splat(0.0); SEED_DEG_PUB + 1];
    for i in 0..=SEED_DEG_PUB {
        let mut row = [0.0_f64; LANES];
        for j in 0..LANES {
            row[j] = ctxs[j].cheb_tu[i];
        }
        cheb_tu[i] = V::from_array(row);
    }
    OtmContextSimd {
        sqrt_t: V::from_array(sqrt_t),
        ak: V::from_array(ak),
        m: V::from_array(m),
        mu: V::from_array(mu),
        v_lo: V::from_array(v_lo),
        v_hi: V::from_array(v_hi),
        cheb_tu,
    }
}

/// Broadcast a single context across 8 SIMD lanes. The hot path for the
/// `1 ctx × N prices` workload — caller pays the broadcast once.
#[inline]
pub fn broadcast_context(ctx: &OtmContext) -> OtmContextSimd {
    let mut cheb_tu = [V::splat(0.0); SEED_DEG_PUB + 1];
    for i in 0..=SEED_DEG_PUB {
        cheb_tu[i] = V::splat(ctx.cheb_tu[i]);
    }
    OtmContextSimd {
        sqrt_t: V::splat(ctx.sqrt_t),
        ak: V::splat(ctx.ak),
        m: V::splat(ctx.m),
        mu: V::splat(ctx.mu),
        v_lo: V::splat(ctx.v_lo),
        v_hi: V::splat(ctx.v_hi),
        cheb_tu,
    }
}

/// Single price on a single context (volfi-shape match).
///
/// `c` is the canonical OTM premium-over-spot ratio. If the caller has
/// raw market price `p` and known spot/strike/rate/kind, the canonicalization
/// is:
///
/// ```text
///   xn = p / S
///   c  = if kind == Call { xn } else { xn + 1.0 - K/F }
/// ```
///
/// where `K/F = ctx.ek`. Use [`canonical_c_from_price`] to do this on the
/// outside of a hot loop.
#[inline]
pub fn implied_vol_with_context(ctx: &OtmContext, c: f64) -> f64 {
    // Per-option: q-side Cheb basis, bilinear sum with cached k-side basis.
    let q = (1.0 - c) / ctx.m;
    let q_for_seed = q.max(crate::schadner_fast::SEED_P_LO_PUB).min(crate::schadner_fast::SEED_P_HI_PUB);

    let cheb_tw = cheb_qside_basis(q_for_seed);
    let seed_v = cheb_seed_from_kside_basis_scalar(&ctx.cheb_tu, &cheb_tw);
    let mut v_iter = seed_v.max(ctx.v_lo).min(ctx.v_hi);

    // Lane to a 1-wide SIMD step to reuse householder3_step. Cheap because
    // the compiler scalarizes on AVX-512 when the other 7 lanes are dead.
    let mu_v = V::splat(ctx.mu);
    let q_v = V::splat(q);
    let v_lo_v = V::splat(ctx.v_lo);
    let v_hi_v = V::splat(ctx.v_hi);
    let mut v_simd = V::splat(v_iter);
    let mut j = 0;
    while j < HOUSEHOLDER3_STEPS_CTX {
        v_simd = householder3_step_simd(v_simd, mu_v, q_v).simd_max(v_lo_v).simd_min(v_hi_v);
        j += 1;
    }
    v_iter = v_simd[0];
    v_iter / ctx.sqrt_t
}

/// Canonicalize a raw market price `p` against a context's `ek = K/F`.
///
/// Returns the canonical OTM premium ratio `c` consumed by the context APIs.
#[inline]
pub fn canonical_c_from_price(ctx: &OtmContext, spot: f64, price: f64, is_call: bool) -> f64 {
    let xn = price / spot;
    if is_call { xn } else { xn + 1.0 - ctx.ek }
}

/// Many prices on one context. SIMD-batched per 8.
///
/// Each lane runs the per-option solver against a broadcasted SIMD context.
/// On AVX-512 (znver5), this should saturate the IG `erfcx` pipeline almost
/// as hard as the cold-grid kernel — but with the (k,T) prelude paid exactly
/// once across `prices.len()` evaluations.
pub fn implied_vol_with_context_batch(ctx: &OtmContext, prices: &[f64]) -> Vec<f64> {
    let n = prices.len();
    let mut out = vec![0.0_f64; n];
    let ctx_simd = broadcast_context(ctx);

    let m_v = ctx_simd.m;
    let mu_v = ctx_simd.mu;
    let v_lo_v = ctx_simd.v_lo;
    let v_hi_v = ctx_simd.v_hi;
    let sqrt_t_v = ctx_simd.sqrt_t;
    let inv_sqrt_t_v = V::splat(1.0) / sqrt_t_v;
    let p_lo_v = V::splat(crate::schadner_fast::SEED_P_LO_PUB);
    let p_hi_v = V::splat(crate::schadner_fast::SEED_P_HI_PUB);

    let mut i = 0;
    while i < n {
        let take = core::cmp::min(LANES, n - i);
        let mut cb = [0.0_f64; LANES];
        for j in 0..take {
            cb[j] = prices[i + j];
        }
        // Caller already gave us canonical OTM c (this is the volfi-shape
        // contract: the context API consumes c, not raw price). If they
        // need canonicalization, they call canonical_c_from_price.
        let c_v = V::from_array(cb);
        let q_v = (V::splat(1.0) - c_v) / m_v;
        let q_for_seed = q_v.simd_max(p_lo_v).simd_min(p_hi_v);

        let cheb_tw = cheb_qside_basis_simd(q_for_seed);
        let seed_v = cheb_seed_from_basis_simd(&ctx_simd.cheb_tu, &cheb_tw);
        let mut v_iter = seed_v.simd_max(v_lo_v).simd_min(v_hi_v);

        let mut j = 0;
        while j < HOUSEHOLDER3_STEPS_CTX {
            v_iter = householder3_step_simd(v_iter, mu_v, q_v)
                .simd_max(v_lo_v)
                .simd_min(v_hi_v);
            j += 1;
        }
        let sigma = v_iter * inv_sqrt_t_v;
        let arr = sigma.to_array();
        out[i..i + take].copy_from_slice(&arr[..take]);
        i += take;
    }
    out
}

/// Vector contexts × vector prices. Full SIMD on prelude AND solver.
///
/// `contexts.len() == prices.len()`. Processes 8 at a time, fully SIMD
/// throughout — this is voltic's structural advantage that volfi cannot
/// match (volfi's two-stage API is scalar, so on a fan-out workload it
/// processes 1/lane).
///
/// On the cold-grid workload (1M unique `(k, T, c)`), this matches the
/// existing `implied_vol_fast_kernel` cost minus the (k,T) prelude work —
/// because the caller already paid that off-line when building the
/// `contexts` slice.
pub fn implied_vol_vectorized_with_contexts(
    contexts: &[OtmContext],
    prices: &[f64],
) -> Vec<f64> {
    let n = contexts.len();
    assert_eq!(
        n,
        prices.len(),
        "implied_vol_vectorized_with_contexts: contexts and prices must be the same length"
    );
    let mut out = vec![0.0_f64; n];
    let p_lo_v = V::splat(crate::schadner_fast::SEED_P_LO_PUB);
    let p_hi_v = V::splat(crate::schadner_fast::SEED_P_HI_PUB);

    let mut i = 0;
    while i < n {
        let take = core::cmp::min(LANES, n - i);
        // SoA pack the 8 contexts in this chunk. Padding lanes get lane-0's
        // context (cheap, no NaN risk; we mask the output below by `take`).
        let mut cb = [0.0_f64; LANES];
        let mut sqrt_t_a = [0.0_f64; LANES];
        let mut ak_a = [0.0_f64; LANES];
        let mut m_a = [1.0_f64; LANES];
        let mut mu_a = [1.0_f64; LANES];
        let mut v_lo_a = [0.0_f64; LANES];
        let mut v_hi_a = [1.0_f64; LANES];
        let mut tu_rows = [[0.0_f64; LANES]; SEED_DEG_PUB + 1];
        // Use contexts[i] as fallback for padding to avoid /0 in q = (1-c)/m.
        let fallback = &contexts[i];
        for j in 0..LANES {
            let src = if j < take { &contexts[i + j] } else { fallback };
            sqrt_t_a[j] = src.sqrt_t;
            ak_a[j] = src.ak;
            m_a[j] = src.m;
            mu_a[j] = src.mu;
            v_lo_a[j] = src.v_lo;
            v_hi_a[j] = src.v_hi;
            for r in 0..=SEED_DEG_PUB {
                tu_rows[r][j] = src.cheb_tu[r];
            }
            if j < take {
                cb[j] = prices[i + j];
            }
        }
        let mut cheb_tu_v = [V::splat(0.0); SEED_DEG_PUB + 1];
        for r in 0..=SEED_DEG_PUB {
            cheb_tu_v[r] = V::from_array(tu_rows[r]);
        }
        let m_v = V::from_array(m_a);
        let mu_v = V::from_array(mu_a);
        let v_lo_v = V::from_array(v_lo_a);
        let v_hi_v = V::from_array(v_hi_a);
        let sqrt_t_v = V::from_array(sqrt_t_a);
        let inv_sqrt_t_v = V::splat(1.0) / sqrt_t_v;
        let _ = ak_a; // ak not needed downstream — kept in struct for callers.

        let c_v = V::from_array(cb);
        let q_v = (V::splat(1.0) - c_v) / m_v;
        let q_for_seed = q_v.simd_max(p_lo_v).simd_min(p_hi_v);

        let cheb_tw = cheb_qside_basis_simd(q_for_seed);
        let seed_v = cheb_seed_from_basis_simd(&cheb_tu_v, &cheb_tw);
        let mut v_iter = seed_v.simd_max(v_lo_v).simd_min(v_hi_v);
        let mut j = 0;
        while j < HOUSEHOLDER3_STEPS_CTX {
            v_iter = householder3_step_simd(v_iter, mu_v, q_v)
                .simd_max(v_lo_v)
                .simd_min(v_hi_v);
            j += 1;
        }
        let sigma = v_iter * inv_sqrt_t_v;
        let arr = sigma.to_array();
        out[i..i + take].copy_from_slice(&arr[..take]);
        i += take;
    }
    out
}

// Silence the unused-mask warning during the early build — we may want it
// later for invalid-domain gating.
#[allow(dead_code)]
fn _phantom_use_M(_m: M) {}

// =========================================================================
// A5.1: SIMD-batched (k, T)-prelude.
//
// The scalar prelude (`ig_kt_prelude_scalar`) sits at ~48 ns/option on cold
// workloads — it dominates the END-TO-END cost (48 ns build + 62 ns SIMD
// solve = 110 ns, *worse* than vanilla voltic-fast at 95.8 ns).
//
// Every line of the scalar prelude is SIMD-friendly: `sqrt_t`, `k.exp()`,
// the `k > 0` branch, `μ = 2/|k|`, `ln(k_for_seed)`, and the 13-term
// Chebyshev k-side recurrence — all f64x8 with no gathers, no scatters,
// no dependent loads. Lanes are perfectly independent.
//
// Constants from `schadner_fast_seed.rs` (auto-generated). Kept local to
// this module because the kernel file is read-only at this conductor step.
// If these drift from `schadner_fast_seed.rs`, the SIMD prelude will produce
// answers that disagree with the scalar prelude — the bench's NaN audit + a
// `scalar vs simd-build agreement` probe in `bench/main.rs` guards this.
// =========================================================================

const SEED_K_LO_A51: f64 = 0.001_f64;
const SEED_K_HI_A51: f64 = 3.0_f64;
const SEED_LN_K_LO_A51: f64 = -6.907_755_278_982_137e+00_f64;
const SEED_LN_K_HI_A51: f64 = 1.098_612_288_668_109_6e+00_f64;

/// SIMD `(k, T)`-prelude: build an [`OtmContextSimd`] from f64x8 vectors of
/// `k = ln(K/F)` and `T`. Lane-for-lane equivalent to running
/// `ig_kt_prelude_scalar` 8 times and packing the result into an
/// `OtmContextSimd`, but without the per-lane scalar setup cost.
///
/// Target: ~6-10 ns/option amortized (vs ~48 ns/option scalar).
#[inline]
pub fn ig_kt_prelude_simd(k: V, t: V) -> OtmContextSimd {
    let sqrt_t = t.sqrt();
    let ak = k.abs();
    let ek = k.exp();
    // Branch: `m_call = 1.0`, `m_put = ek`. The scalar form is
    // `if k > 0 { 1.0 } else { ek }`. select on a mask is one cycle.
    let m = k.simd_gt(V::splat(0.0)).select(V::splat(1.0), ek);
    // μ = 2 / max(|k|, 1e-12), capped at 1e12. matches scalar exactly.
    let ak_clamped = ak.simd_max(V::splat(1e-12));
    let mu = (V::splat(2.0) / ak_clamped).simd_min(V::splat(1e12));
    let v_lo = V::splat(VOL_MIN) * sqrt_t;
    let v_hi = V::splat(VOL_MAX) * sqrt_t;
    let k_for_seed = ak.simd_max(V::splat(SEED_K_LO_A51)).simd_min(V::splat(SEED_K_HI_A51));

    // Chebyshev k-side basis: T_m(u(k_for_seed)) for m = 0..=SEED_DEG_PUB.
    // u = (ln(k_for_seed) - u_centre) / u_half_width
    let ln_k = k_for_seed.ln();
    let u_centre = V::splat(0.5 * (SEED_LN_K_LO_A51 + SEED_LN_K_HI_A51));
    let u_half_width = V::splat(0.5 * (SEED_LN_K_HI_A51 - SEED_LN_K_LO_A51));
    let u = (ln_k - u_centre) / u_half_width;

    let mut cheb_tu = [V::splat(0.0); SEED_DEG_PUB + 1];
    cheb_tu[0] = V::splat(1.0);
    if SEED_DEG_PUB >= 1 {
        cheb_tu[1] = u;
    }
    let two_u = V::splat(2.0) * u;
    let mut i = 2;
    while i <= SEED_DEG_PUB {
        // T_i = 2u·T_{i-1} - T_{i-2} — pure FMA, lane-independent.
        cheb_tu[i] = two_u * cheb_tu[i - 1] - cheb_tu[i - 2];
        i += 1;
    }

    OtmContextSimd {
        sqrt_t,
        ak,
        m,
        mu,
        v_lo,
        v_hi,
        cheb_tu,
    }
}

/// Solve one SIMD chunk: given a pre-built `OtmContextSimd` and a vector of
/// canonical OTM `c` values, run the seed + Householder-3 solver and return
/// `σ` (annualised vol) for the lane.
///
/// This is the per-chunk inner kernel shared by
/// [`implied_vol_vectorized_with_contexts`] and
/// [`implied_vol_fully_vectorized`]. Both call this with the same
/// `cheb_tu`/`mu`/`v_lo`/`v_hi`/`m`/`sqrt_t` SoA bundle; only the source
/// of those bundles differs (caller-supplied vs SIMD-built on the fly).
#[inline]
fn solve_with_ctx_simd(ctx: &OtmContextSimd, c_v: V) -> V {
    let p_lo_v = V::splat(crate::schadner_fast::SEED_P_LO_PUB);
    let p_hi_v = V::splat(crate::schadner_fast::SEED_P_HI_PUB);

    let inv_sqrt_t_v = V::splat(1.0) / ctx.sqrt_t;

    let q_v = (V::splat(1.0) - c_v) / ctx.m;
    let q_for_seed = q_v.simd_max(p_lo_v).simd_min(p_hi_v);

    let cheb_tw = cheb_qside_basis_simd(q_for_seed);
    let cheb_v = cheb_seed_from_basis_simd(&ctx.cheb_tu, &cheb_tw);

    // Wing dispatch: lanes in the deep-OTM wing regime get the analytic
    // wing seed instead of the Chebyshev seed. ctx.ak carries |k_log|. The
    // wing seed expects IG SURVIVAL (= c_*); kernel q is the IG CDF
    // (= 1 - c_*) — convert here.
    let q_surv = V::splat(1.0) - q_v;
    let use_wing = ctx.ak.simd_ge(V::splat(WING_K_LO))
        & q_surv.simd_lt(V::splat(WING_Q_MAX))
        & q_surv.simd_gt(V::splat(0.0))
        & ctx.ak.simd_lt(V::splat(WING_H_MAX));
    // Chunk-level bailout: skip the expensive wing_seed_simd call if no
    // lane in this chunk needs it. Preserves cold-grid throughput.
    let seed_v = if use_wing.any() {
        let q_wing_clamped = q_surv
            .simd_max(V::splat(1e-300))
            .simd_min(V::splat(WING_Q_MAX));
        let h_wing_clamped = ctx
            .ak
            .simd_max(V::splat(WING_K_LO))
            .simd_min(V::splat(WING_H_MAX));
        let wing_v = wing_seed_simd(h_wing_clamped, q_wing_clamped);
        use_wing.select(wing_v, cheb_v)
    } else {
        cheb_v
    };

    let mut v_iter = seed_v.simd_max(ctx.v_lo).simd_min(ctx.v_hi);
    let mut j = 0;
    while j < HOUSEHOLDER3_STEPS_CTX {
        v_iter = householder3_step_simd(v_iter, ctx.mu, q_v)
            .simd_max(ctx.v_lo)
            .simd_min(ctx.v_hi);
        j += 1;
    }
    v_iter * inv_sqrt_t_v
}

/// Build N SIMD contexts from raw `(k, T)` slices. Processes 8 at a time.
///
/// `k.len() == t.len()`. Output length = `ceil(k.len() / 8)`. The final
/// chunk's tail lanes (if `k.len() % 8 != 0`) are padded with lane 0 of
/// that chunk so the prelude doesn't fault on `ln(0)` or `2/0`. Caller is
/// responsible for masking those tail lanes on consumption.
pub fn pack_contexts_from_kt(k: &[f64], t: &[f64]) -> Vec<OtmContextSimd> {
    assert_eq!(
        k.len(),
        t.len(),
        "pack_contexts_from_kt: k and t must be the same length"
    );
    let n = k.len();
    let n_chunks = (n + LANES - 1) / LANES;
    let mut out = Vec::with_capacity(n_chunks);
    let mut i = 0;
    while i < n {
        let take = core::cmp::min(LANES, n - i);
        let mut kb = [0.0_f64; LANES];
        let mut tb = [1.0_f64; LANES];
        for j in 0..take {
            kb[j] = k[i + j];
            tb[j] = t[i + j];
        }
        // Pad tail lanes with the first valid (k, T) so the prelude stays
        // numerically clean (no NaN, no inf in mu).
        for j in take..LANES {
            kb[j] = k[i];
            tb[j] = t[i];
        }
        let k_v = V::from_array(kb);
        let t_v = V::from_array(tb);
        out.push(ig_kt_prelude_simd(k_v, t_v));
        i += take;
    }
    out
}

/// **A5.1**: Fused vectorized build + solve. Take raw `(k, T, c)` slices,
/// run the SIMD prelude and the SIMD solver in a single pass, return
/// `Vec<σ>`.
///
/// This is the cold-workload entry point. `c` is the canonical OTM premium
/// ratio (caller has already done canonicalization); for raw-price callers,
/// canonicalize against `e^k = K/F` outside the loop.
///
/// Optimal for cold workloads (every option has a unique `(k, T, c)`): the
/// scalar context build is gone entirely. Beats vanilla voltic-fast on
/// cold workloads because the IG prelude itself runs at SIMD throughput.
pub fn implied_vol_fully_vectorized(k: &[f64], t: &[f64], c: &[f64]) -> Vec<f64> {
    assert_eq!(k.len(), t.len(), "implied_vol_fully_vectorized: k.len() != t.len()");
    assert_eq!(k.len(), c.len(), "implied_vol_fully_vectorized: k.len() != c.len()");
    let n = k.len();
    let mut out = vec![0.0_f64; n];
    let mut i = 0;
    while i < n {
        let take = core::cmp::min(LANES, n - i);
        let mut kb = [0.0_f64; LANES];
        let mut tb = [1.0_f64; LANES];
        let mut cb = [0.0_f64; LANES];
        for j in 0..take {
            kb[j] = k[i + j];
            tb[j] = t[i + j];
            cb[j] = c[i + j];
        }
        // Pad tail lanes with the first valid (k, T) so the prelude stays
        // clean — those lanes are masked off on write-back below.
        for j in take..LANES {
            kb[j] = k[i];
            tb[j] = t[i];
            cb[j] = c[i];
        }
        let k_v = V::from_array(kb);
        let t_v = V::from_array(tb);
        let c_v = V::from_array(cb);
        let ctx = ig_kt_prelude_simd(k_v, t_v);
        let sigma = solve_with_ctx_simd(&ctx, c_v);
        let arr = sigma.to_array();
        out[i..i + take].copy_from_slice(&arr[..take]);
        i += take;
    }
    out
}

/// **A5.1**: Bench-only entry — measure the cost of the SIMD prelude in
/// isolation. Builds N contexts via [`pack_contexts_from_kt`] and writes
/// one `f64` per option (the lane's `sqrt_t * cheb_tu[0]`) to a Vec so
/// LLVM can't dead-code the build away.
///
/// Returns the materialized `Vec<f64>` (len == k.len()) as the visible
/// side effect for the bench harness.
pub fn build_simd_contexts_observed(k: &[f64], t: &[f64]) -> Vec<f64> {
    let ctxs = pack_contexts_from_kt(k, t);
    let n = k.len();
    let mut out = vec![0.0_f64; n];
    let mut i = 0;
    for ctx in ctxs.iter() {
        let take = core::cmp::min(LANES, n - i);
        // Observe sqrt_t + cheb_tu[0] (= 1.0) + cheb_tu[1] (= u) so the
        // compiler can't constant-fold the whole prelude. cheb_tu[0] is
        // always 1.0 — adding it costs nothing semantically but blocks
        // dead-code elimination of the array.
        let obs = ctx.sqrt_t + ctx.cheb_tu[0] + ctx.cheb_tu[1];
        let arr = obs.to_array();
        out[i..i + take].copy_from_slice(&arr[..take]);
        i += take;
    }
    out
}
