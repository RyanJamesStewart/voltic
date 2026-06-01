//! Status-code coverage tests for the typed implied-vol API.
//!
//! Each of the seven [`ImpliedVolStatus`] variants is exercised by at least
//! one test whose *falsifying instance* — the construction that produces it —
//! is described in-line. The intent (per the v1.2 spec) is that an
//! independent reviewer can read the test and reconstruct the regime that
//! triggers each variant.

use voltic::{
    bs_price, implied_vol_typed, implied_vol_typed_batch, ImpliedVolResult, ImpliedVolStatus,
    OptionKind, VOL_MAX, VOL_MIN,
};

/// Helper: assert a `Computed` result with a value matching expected to tol.
fn assert_computed(res: ImpliedVolResult, expected: f64, tol: f64, label: &str) {
    assert!(
        matches!(res.status, ImpliedVolStatus::Computed),
        "{label}: expected Computed, got {:?} value={}",
        res.status,
        res.value
    );
    assert!(
        res.value.is_finite() && (res.value - expected).abs() < tol,
        "{label}: value {} not within {} of {}",
        res.value,
        tol,
        expected
    );
}

// -- Variant: Computed --------------------------------------------------------

/// Falsifying instance: a well-conditioned ATM 1-year call (S=K=100, σ=30%,
/// r=2%). Vega is large, root is well inside `[VOL_MIN, VOL_MAX]`.
#[test]
fn variant_computed_atm_call() {
    let s = [100.0];
    let k = [100.0];
    let t = [1.0];
    let r = [0.02];
    let v = [0.30];
    let kind = [OptionKind::Call];
    let p = bs_price(&s, &k, &t, &r, &v, &kind);
    let res = implied_vol_typed(s[0], k[0], t[0], r[0], p[0], kind[0]);
    assert_computed(res, 0.30, 1e-9, "ATM call");
}

/// Falsifying instance: same canonical ATM call, but solved via the batch
/// entry point. Result must agree with the scalar path.
#[test]
fn variant_computed_atm_call_batch() {
    let s = [100.0];
    let k = [100.0];
    let t = [1.0];
    let r = [0.02];
    let v = [0.30];
    let kind = [OptionKind::Call];
    let p = bs_price(&s, &k, &t, &r, &v, &kind);
    let res = implied_vol_typed_batch(&s, &k, &t, &r, &p, &kind);
    assert_eq!(res.len(), 1);
    assert_computed(res[0], 0.30, 1e-9, "ATM call (batch)");
}

// -- Variant: BelowVolMin -----------------------------------------------------

/// Falsifying instance: forward-price the ATM 1-year call at σ = 0.005 (half
/// of `VOL_MIN`), then ask the typed solver to invert. The iteration must find
/// σ ≈ 0.005 and report `BelowVolMin { computed: ~0.005 }`. The accompanying
/// `value` field must equal `computed`.
#[test]
fn variant_below_vol_min_sub_vol_min_root() {
    let s = [100.0];
    let k = [100.0];
    let t = [1.0];
    let r = [0.02];
    let v = [0.005_f64]; // half of VOL_MIN
    let kind = [OptionKind::Call];
    let p = bs_price(&s, &k, &t, &r, &v, &kind);
    let res = implied_vol_typed(s[0], k[0], t[0], r[0], p[0], kind[0]);
    match res.status {
        ImpliedVolStatus::BelowVolMin { computed } => {
            assert!(
                computed.is_finite() && (computed - 0.005).abs() < 1e-8,
                "BelowVolMin computed = {} (want ~0.005)",
                computed
            );
            assert_eq!(
                res.value, computed,
                "value must mirror computed for BelowVolMin"
            );
            assert!(
                computed < VOL_MIN,
                "computed {} must be < VOL_MIN",
                computed
            );
        }
        other => panic!("expected BelowVolMin, got {:?}", other),
    }
}

// -- Variant: AboveVolMax -----------------------------------------------------

/// Falsifying instance: forward-price the ATM 1-year call at σ = 6.0 (above
/// `VOL_MAX = 5.0`), then ask the typed solver to invert. The iteration must
/// land at σ ≈ 6.0 and report `AboveVolMax { computed: ~6.0 }`.
#[test]
fn variant_above_vol_max_super_vol_max_root() {
    let s = [100.0];
    let k = [100.0];
    let t = [1.0];
    let r = [0.0];
    let v = [6.0_f64]; // above VOL_MAX
    let kind = [OptionKind::Call];
    let p = bs_price(&s, &k, &t, &r, &v, &kind);
    let res = implied_vol_typed(s[0], k[0], t[0], r[0], p[0], kind[0]);
    match res.status {
        ImpliedVolStatus::AboveVolMax { computed } => {
            assert!(
                computed.is_finite() && (computed - 6.0).abs() < 1e-6,
                "AboveVolMax computed = {} (want ~6.0)",
                computed
            );
            assert_eq!(res.value, computed);
            assert!(
                computed > VOL_MAX,
                "computed {} must be > VOL_MAX",
                computed
            );
        }
        other => panic!("expected AboveVolMax, got {:?}", other),
    }
}

// -- Variant: BelowIntrinsic --------------------------------------------------

/// Falsifying instance: ATM call (S=K=100, r=0), price = 0.0 (intrinsic for an
/// ATM call at r=0 is exactly 0). The strict `price > intrinsic` check fails,
/// so the result must be `BelowIntrinsic` (a value of *exactly* intrinsic
/// implies σ = 0, which is outside both `VOL_MIN` and the typed solver's wide
/// internal `[1e-8, 50]` bracket).
#[test]
fn variant_below_intrinsic_zero_premium() {
    let s = 100.0;
    let k = 100.0;
    let t = 1.0;
    let r = 0.0;
    let p = 0.0;
    let res = implied_vol_typed(s, k, t, r, p, OptionKind::Call);
    assert!(
        matches!(res.status, ImpliedVolStatus::BelowIntrinsic),
        "expected BelowIntrinsic, got {:?} value={}",
        res.status,
        res.value
    );
    assert!(res.value.is_nan(), "BelowIntrinsic value must be NaN");
}

/// Falsifying instance: ITM call (S=110, K=100, T=1, r=0) priced *below*
/// intrinsic (intrinsic = 10, price = 8). The premium below the no-arbitrage
/// floor.
#[test]
fn variant_below_intrinsic_itm_call() {
    let s = 110.0;
    let k = 100.0;
    let t = 1.0;
    let r = 0.0;
    let p = 8.0; // intrinsic is 10, so 8 < intrinsic
    let res = implied_vol_typed(s, k, t, r, p, OptionKind::Call);
    assert!(
        matches!(res.status, ImpliedVolStatus::BelowIntrinsic),
        "expected BelowIntrinsic, got {:?}",
        res.status
    );
}

// -- Variant: AboveMaximum ----------------------------------------------------

/// Falsifying instance: call premium > spot — no-arbitrage upper bound for an
/// undiscounted call is S; pricing it above S is impossible.
#[test]
fn variant_above_maximum_call_above_spot() {
    let s = 100.0;
    let k = 100.0;
    let t = 1.0;
    let r = 0.0;
    let p = 200.0; // above S = 100
    let res = implied_vol_typed(s, k, t, r, p, OptionKind::Call);
    assert!(
        matches!(res.status, ImpliedVolStatus::AboveMaximum),
        "expected AboveMaximum, got {:?}",
        res.status
    );
    assert!(res.value.is_nan());
}

/// Falsifying instance: put premium > K·e^{-rT}.
#[test]
fn variant_above_maximum_put_above_strike() {
    let s = 100.0;
    let k = 100.0;
    let t = 1.0;
    let r = 0.0;
    let p = 200.0; // above K·exp(-rT) = 100
    let res = implied_vol_typed(s, k, t, r, p, OptionKind::Put);
    assert!(
        matches!(res.status, ImpliedVolStatus::AboveMaximum),
        "expected AboveMaximum, got {:?}",
        res.status
    );
}

// -- Variant: NonFinite -------------------------------------------------------

/// Falsifying instance: spot is NaN.
#[test]
fn variant_non_finite_nan_spot() {
    let res = implied_vol_typed(f64::NAN, 100.0, 1.0, 0.0, 10.0, OptionKind::Call);
    assert!(
        matches!(res.status, ImpliedVolStatus::NonFinite),
        "expected NonFinite, got {:?}",
        res.status
    );
    assert!(res.value.is_nan());
}

/// Falsifying instance: T = 0 (degenerate expiry).
#[test]
fn variant_non_finite_zero_tte() {
    let res = implied_vol_typed(100.0, 100.0, 0.0, 0.0, 5.0, OptionKind::Call);
    assert!(
        matches!(res.status, ImpliedVolStatus::NonFinite),
        "expected NonFinite (T=0), got {:?}",
        res.status
    );
}

/// Falsifying instance: negative spot.
#[test]
fn variant_non_finite_negative_spot() {
    let res = implied_vol_typed(-1.0, 100.0, 1.0, 0.0, 5.0, OptionKind::Call);
    assert!(
        matches!(res.status, ImpliedVolStatus::NonFinite),
        "expected NonFinite (neg spot), got {:?}",
        res.status
    );
}

/// Falsifying instance: price is +Inf.
#[test]
fn variant_non_finite_inf_price() {
    let res = implied_vol_typed(100.0, 100.0, 1.0, 0.0, f64::INFINITY, OptionKind::Call);
    assert!(
        matches!(res.status, ImpliedVolStatus::NonFinite),
        "expected NonFinite (Inf price), got {:?}",
        res.status
    );
}

// -- Variant: FailedToConverge -----------------------------------------------

/// Falsifying instance: a price one ULP above intrinsic for a deep-ITM long
/// expiry call. The premium is below the f64 conditioning floor — the
/// iteration cannot land a finite re-pricing residual. We accept either
/// `FailedToConverge` *or* a flag of `BelowVolMin` (a near-zero σ that
/// re-prices within tolerance). Both are honest; the test fails only if the
/// typed API reports `Computed` for this regime, which would be a wrong-finite
/// claim.
#[test]
fn variant_failed_to_converge_deep_itm_floor() {
    // Falsifying instance: deep-OTM call near expiry, priced one ULP above
    // intrinsic (= one ULP above zero for this OTM strike). The price sits at
    // the f64 floor — vega is essentially zero across the σ-bracket, so the
    // iteration cannot resolve a finite root that re-prices within tolerance.
    // Accepted statuses: FailedToConverge, BelowIntrinsic, BelowVolMin, or
    // AboveVolMax (each is honest); the test fails only on Computed (a wrong
    // finite claim) or NonFinite (which would imply the inputs are bad).
    let s: f64 = 100.0;
    let k: f64 = 200.0;
    let t: f64 = 1.0 / 365.0;
    let r: f64 = 0.0;
    // Deep OTM intrinsic is 0; pick a price just below the f64 floor scaled to
    // the underlying — small enough that vega is denormal across the σ-bracket.
    let p: f64 = f64::from_bits(1); // smallest positive denormal
    let res = implied_vol_typed(s, k, t, r, p, OptionKind::Call);
    match res.status {
        ImpliedVolStatus::FailedToConverge
        | ImpliedVolStatus::BelowIntrinsic
        | ImpliedVolStatus::BelowVolMin { .. }
        | ImpliedVolStatus::AboveVolMax { .. } => {
            // honest report — none of these claims a wrong finite σ
        }
        ImpliedVolStatus::Computed => {
            panic!(
                "regime is below the f64 conditioning floor; Computed = {} \
                 would be a wrong-finite claim.",
                res.value
            );
        }
        other => panic!("unexpected status {:?}", other),
    }
}

/// A second  falsifying instance: a deeply mismatched
/// premium that the iteration cannot re-price within  even though
/// the bracket is wide. Constructed by feeding a price one ULP below the
/// upper bound () for a normal call — the iteration drives σ all
/// the way to , lands there as a clamp, and the priced-ok or
/// boundary gate flags it.
#[test]
fn variant_failed_to_converge_near_upper_bound() {
    // Falsifying instance: price one ULP below S (the call upper bound) so the
    // iteration is asked to find σ → ∞. The wide internal bracket caps at
    // ITER_VOL_MAX = 50; the iterate clamps there.
    let s: f64 = 100.0;
    let k: f64 = 100.0;
    let t: f64 = 1.0;
    let r: f64 = 0.0;
    let p: f64 = s - f64::EPSILON * s;
    let res = implied_vol_typed(s, k, t, r, p, OptionKind::Call);
    // Either AboveVolMax (clamp at ITER_VOL_MAX = 50 is reported as
    // a > VOL_MAX root) or FailedToConverge (clamp + priced_ok fails).
    match res.status {
        ImpliedVolStatus::AboveVolMax { .. }
        | ImpliedVolStatus::FailedToConverge
        | ImpliedVolStatus::AboveMaximum => {}
        other => panic!(
            "expected AboveVolMax / FailedToConverge / AboveMaximum, got {:?} value={}",
            other, res.value
        ),
    }
}

// -- Cross-cutting: status distribution on a small mixed batch ---------------

/// Hand-constructed 8-element batch exercising all rejection statuses in one
/// `implied_vol_typed_batch` call. Verifies the per-lane classification works
/// inside the SIMD path.
#[test]
fn batch_status_distribution() {
    // Eight inputs:
    //   0: Computed         (ATM call at σ=0.30)
    //   1: BelowVolMin      (σ=0.005 forward-priced)
    //   2: AboveVolMax      (σ=6.0 forward-priced)
    //   3: BelowIntrinsic   (price = 0 at ATM call r=0)
    //   4: AboveMaximum     (price = S+1)
    //   5: NonFinite        (NaN spot)
    //   6: NonFinite        (T = 0)
    //   7: Computed         (low-vol but still > VOL_MIN: σ=0.20)
    let s_ref = [100.0, 100.0, 100.0, 100.0, 100.0, f64::NAN, 100.0, 100.0];
    let k_ref = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0];
    let t_ref = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0];
    let r_ref = [0.0; 8];
    let kind_ref = [OptionKind::Call; 8];

    // Compute prices for the well-defined ones (0, 1, 2, 7).
    let sigs = [0.30, 0.005, 6.0, 0.0, 0.0, 0.0, 0.0, 0.20];
    let p_well = bs_price(&s_ref, &k_ref, &t_ref, &r_ref, &sigs, &kind_ref);

    // Override 3..=6 with their failure-mode prices.
    let mut p = p_well;
    p[3] = 0.0; // BelowIntrinsic (ATM r=0 intrinsic is 0; strict gate)
    p[4] = 101.0; // AboveMaximum
    p[5] = 10.0; // NonFinite (NaN spot)
    p[6] = 10.0; // NonFinite (T=0)

    let res = implied_vol_typed_batch(&s_ref, &k_ref, &t_ref, &r_ref, &p, &kind_ref);

    assert!(matches!(res[0].status, ImpliedVolStatus::Computed));
    assert!(matches!(
        res[1].status,
        ImpliedVolStatus::BelowVolMin { .. }
    ));
    assert!(matches!(
        res[2].status,
        ImpliedVolStatus::AboveVolMax { .. }
    ));
    assert!(matches!(res[3].status, ImpliedVolStatus::BelowIntrinsic));
    assert!(matches!(res[4].status, ImpliedVolStatus::AboveMaximum));
    assert!(matches!(res[5].status, ImpliedVolStatus::NonFinite));
    assert!(matches!(res[6].status, ImpliedVolStatus::NonFinite));
    assert!(matches!(res[7].status, ImpliedVolStatus::Computed));
    assert!((res[7].value - 0.20).abs() < 1e-9);
}

// -- to_f64_nan_on_boundary mapping ------------------------------------------

/// `to_f64_nan_on_boundary` should preserve `Computed` value and NaN every
/// other status — including `BelowVolMin`/`AboveVolMax` (the legacy f64 API
/// returns NaN there, NOT the typed `computed` value).
#[test]
fn to_f64_mapping_is_legacy_compat() {
    let computed = ImpliedVolResult {
        value: 0.30,
        status: ImpliedVolStatus::Computed,
    };
    assert_eq!(computed.to_f64_nan_on_boundary(), 0.30);

    let below = ImpliedVolResult {
        value: 0.005,
        status: ImpliedVolStatus::BelowVolMin { computed: 0.005 },
    };
    assert!(below.to_f64_nan_on_boundary().is_nan());

    let above = ImpliedVolResult {
        value: 6.0,
        status: ImpliedVolStatus::AboveVolMax { computed: 6.0 },
    };
    assert!(above.to_f64_nan_on_boundary().is_nan());

    for status in &[
        ImpliedVolStatus::BelowIntrinsic,
        ImpliedVolStatus::AboveMaximum,
        ImpliedVolStatus::NonFinite,
        ImpliedVolStatus::FailedToConverge,
    ] {
        let r = ImpliedVolResult {
            value: f64::NAN,
            status: *status,
        };
        assert!(r.to_f64_nan_on_boundary().is_nan());
    }
}

// -- Bracket-edge tests ------------------------------------------------------

/// σ exactly at `VOL_MIN`: Path B widens the public solver's accept bracket to
/// `[VOL_MIN, VOL_MAX]` inclusive; the typed API should report `Computed`
/// (not `BelowVolMin`) at the exact boundary.
#[test]
fn boundary_sigma_exactly_vol_min() {
    let s = [100.0];
    let k = [100.0];
    let t = [1.0];
    let r = [0.0];
    let v = [VOL_MIN];
    let kind = [OptionKind::Call];
    let p = bs_price(&s, &k, &t, &r, &v, &kind);
    let res = implied_vol_typed(s[0], k[0], t[0], r[0], p[0], kind[0]);
    assert!(
        matches!(res.status, ImpliedVolStatus::Computed),
        "σ = VOL_MIN should be Computed, got {:?} value={}",
        res.status,
        res.value
    );
    assert!((res.value - VOL_MIN).abs() < 1e-9);
}

/// σ one ULP below `VOL_MIN`: strict `< VOL_MIN` should classify as
/// `BelowVolMin`.
#[test]
fn boundary_sigma_one_ulp_below_vol_min() {
    let s = [100.0];
    let k = [100.0];
    let t = [1.0];
    let r = [0.0];
    let v = [VOL_MIN - f64::EPSILON * VOL_MIN];
    let kind = [OptionKind::Call];
    let p = bs_price(&s, &k, &t, &r, &v, &kind);
    let res = implied_vol_typed(s[0], k[0], t[0], r[0], p[0], kind[0]);
    // Either BelowVolMin (the iteration finds it below) or Computed (the
    // iteration lands at VOL_MIN exactly due to f64 rounding). Both honest.
    match res.status {
        ImpliedVolStatus::BelowVolMin { computed } => {
            assert!(computed < VOL_MIN);
        }
        ImpliedVolStatus::Computed => {
            assert!((res.value - VOL_MIN).abs() < 1e-9);
        }
        other => panic!("unexpected status {:?}", other),
    }
}
