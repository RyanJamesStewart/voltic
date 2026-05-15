//! Property tests: round-trip identity, put–call parity, and a check against a
//! small `py_vollib`-generated reference table.
//!
//! The cross-language proptest-vs-`py_vollib` comparison (the spec's stated
//! ideal) requires `py_vollib` installed; running an embedded Python from
//! `cargo test` is brittle, so instead the benchmark/CI step on Crucible runs
//! `scripts/gen_reference.py` once to emit `tests/reference_pairs.csv`
//! (`spot,strike,tte,rate,price,kind,vol_py_vollib`) and [`reference_table`]
//! below checks voltic against it. The round-trip and parity proptests need no
//! Python and run on every `cargo test`.

use voltic::{bs_price, implied_vol, OptionKind};
use proptest::prelude::*;

/// A strategy producing a *well-conditioned* option: parameters in plausible
/// ranges, then **filtered** to those whose Black-Scholes premium is
/// meaningfully above intrinsic (time value > 1e-6 · spot) — i.e. an IV that
/// the conditioning floor can actually invert. The deep-OTM-near-expiry corner
/// (where the premium underflows below that floor and voltic returns `NaN` by
/// design) is *excluded here on purpose*; it's covered by the named edge-case
/// tests in `src/lib.rs`, not by the round-trip property.
fn well_conditioned() -> impl Strategy<Value = (f64, f64, f64, f64, f64, OptionKind)> {
    (
        50.0f64..200.0, // spot
        0.6f64..1.6,    // strike / spot ratio (S/K ∈ [0.625, 1.667])
        0.02f64..2.0,   // tte (years) — at least ~1 week
        -0.01f64..0.08, // rate
        0.05f64..1.5,   // vol
        prop_oneof![Just(OptionKind::Call), Just(OptionKind::Put)],
    )
        .prop_map(|(s, ratio, t, r, v, kind)| {
            let k = s / ratio;
            (s, k, t, r, v, kind)
        })
        .prop_filter(
            "premium must be meaningfully above intrinsic",
            |&(s, k, t, r, v, kind)| {
                let price = bs_price(&[s], &[k], &[t], &[r], &[v], &[kind])[0];
                let df = (-r * t).exp();
                let intrinsic = match kind {
                    OptionKind::Call => (s - k * df).max(0.0),
                    OptionKind::Put => (k * df - s).max(0.0),
                };
                price.is_finite() && (price - intrinsic) > 1e-6 * s
            },
        )
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 2000, ..ProptestConfig::default() })]

    /// Solve the IV of a price that was itself produced by Black-Scholes from a
    /// known σ; voltic must recover that σ to high accuracy.
    #[test]
    fn round_trip_recovers_sigma((s, k, t, r, v, kind) in well_conditioned()) {
        let price = bs_price(&[s], &[k], &[t], &[r], &[v], &[kind]);
        let iv = implied_vol(&[s], &[k], &[t], &[r], &price, &[kind]);
        // The dataset is exact (price from σ to full f64), so the only error is
        // Newton's residual + the Φ kernel — well under 1e-8 in the
        // well-conditioned region. (`NaN` would mean the screen or the cap
        // rejected a well-posed input; that's a bug.)
        prop_assert!(!iv[0].is_nan(), "NaN for well-conditioned S={s} K={k} T={t} r={r} v={v} {kind:?} price={}", price[0]);
        prop_assert!((iv[0] - v).abs() < 1e-5, "S={s} K={k} T={t} r={r} v={v} {kind:?}: solved {} (Δ={})", iv[0], (iv[0]-v).abs());
    }

    /// Solving a call and the matching put (same S, K, T, r) and re-pricing
    /// both must satisfy C − P = S − K·e^{−rT}.
    #[test]
    fn put_call_parity((s, k, t, r, v, _kind) in well_conditioned()) {
        let cp = bs_price(&[s], &[k], &[t], &[r], &[v], &[OptionKind::Call]);
        let pp = bs_price(&[s], &[k], &[t], &[r], &[v], &[OptionKind::Put]);
        let civ = implied_vol(&[s], &[k], &[t], &[r], &cp, &[OptionKind::Call]);
        let piv = implied_vol(&[s], &[k], &[t], &[r], &pp, &[OptionKind::Put]);
        prop_assert!(!civ[0].is_nan() && !piv[0].is_nan());
        prop_assert!((civ[0] - v).abs() < 1e-5);
        prop_assert!((piv[0] - v).abs() < 1e-5);
        let parity = cp[0] - pp[0];
        let theo = s - k * (-r * t).exp();
        prop_assert!((parity - theo).abs() < 1e-8 * (s + k).max(1.0), "parity {parity} vs {theo}");
    }

    /// A batch of arbitrary well-conditioned options solves identically whether
    /// you pass them one at a time or as one vectorized call (no lane-packing
    /// artifact at the chunk boundary).
    #[test]
    fn batch_equals_singletons(batch in prop::collection::vec(well_conditioned(), 1..40)) {
        let s: Vec<f64> = batch.iter().map(|x| x.0).collect();
        let k: Vec<f64> = batch.iter().map(|x| x.1).collect();
        let t: Vec<f64> = batch.iter().map(|x| x.2).collect();
        let r: Vec<f64> = batch.iter().map(|x| x.3).collect();
        let v: Vec<f64> = batch.iter().map(|x| x.4).collect();
        let kd: Vec<OptionKind> = batch.iter().map(|x| x.5).collect();
        let price = bs_price(&s, &k, &t, &r, &v, &kd);
        let batched = implied_vol(&s, &k, &t, &r, &price, &kd);
        for i in 0..s.len() {
            let one = implied_vol(&[s[i]], &[k[i]], &[t[i]], &[r[i]], &[price[i]], &[kd[i]]);
            // bit-for-bit: same code path, same inputs.
            let (a, b) = (batched[i].to_bits(), one[0].to_bits());
            prop_assert_eq!(a, b, "lane {}: batched {} vs single {}", i, batched[i], one[0]);
        }
    }
}

/// Check voltic against a `py_vollib`-generated reference table if present.
/// Run `python scripts/gen_reference.py` (needs `py_vollib`) to (re)generate
/// `tests/reference_pairs.csv`. When the file is absent (e.g. local dev without
/// Python) the test is a no-op with a printed note — the round-trip proptests
/// above still gate correctness; the *accuracy-vs-the-reference* number is what
/// this file pins, and it's the one the README quotes.
#[test]
fn reference_table() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/reference_pairs.csv");
    let Ok(text) = std::fs::read_to_string(path) else {
        eprintln!("reference_pairs.csv not present — skipping py_vollib comparison (run scripts/gen_reference.py to generate it)");
        return;
    };
    let mut max_abs = 0.0_f64;
    let mut n = 0;
    let mut n_nan = 0;
    for (lineno, line) in text.lines().enumerate() {
        if lineno == 0 || line.trim().is_empty() {
            continue; // header
        }
        let f: Vec<&str> = line.split(',').collect();
        assert!(f.len() >= 7, "bad reference line {lineno}: {line}");
        let s: f64 = f[0].parse().unwrap();
        let k: f64 = f[1].parse().unwrap();
        let t: f64 = f[2].parse().unwrap();
        let r: f64 = f[3].parse().unwrap();
        let price: f64 = f[4].parse().unwrap();
        let kind = match f[5].trim() {
            "c" => OptionKind::Call,
            "p" => OptionKind::Put,
            other => panic!("bad kind {other:?} on line {lineno}"),
        };
        let vol_ref: f64 = f[6].parse().unwrap();
        let got = implied_vol(&[s], &[k], &[t], &[r], &[price], &[kind])[0];
        n += 1;
        if got.is_nan() {
            // py_vollib produced a vol; voltic NaN'd. That's a real
            // disagreement on a point py_vollib could solve — fail unless it's
            // the deep-OTM-near-expiry region (which we *document* as NaN).
            n_nan += 1;
            let near_expiry = t < 14.0 / 365.0;
            let deep = !(0.7..=1.4).contains(&(s / k));
            assert!(near_expiry && deep, "voltic NaN'd a point py_vollib solved and that point is not in the documented unsupported region: S={s} K={k} T={t} r={r} price={price} {kind:?} (py_vollib σ={vol_ref})");
            continue;
        }
        max_abs = max_abs.max((got - vol_ref).abs());
    }
    eprintln!("reference comparison: {n} points, {n_nan} NaN'd (documented region), max |σ_voltic − σ_py_vollib| = {max_abs:.3e}");
    // py_vollib wraps Jäckel's reference, machine-precision; voltic Newton-to-
    // tol against it should agree to ~1e-9 in vol space across well-conditioned
    // inputs (looser, ~1e-6, for the deep-OTM-near-expiry points it *does*
    // solve). Anything claiming tighter than ~1e-10 would be a harness bug.
    assert!(
        max_abs < 1e-6,
        "max abs vol error vs py_vollib = {max_abs:.3e} — too large; harness or algorithm bug"
    );
}
