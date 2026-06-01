//! `bench/wing_grid.rs` — volfi-style v×Δ fixed-grid throughput benchmark.
//!
//! The volfi paper's evaluation grid:
//!   v ∈ {0.01, 0.05, 0.10, 0.15, ..., 2.00}  (41 values)
//!   Δ ∈ {0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99} (13)
//!   533 (v, Δ) points; repeated REPS times for a measurable timing window.
//!
//! Each (v, Δ) → (h, c_*) via:
//!   d1 = Φ⁻¹(Δ);  d2 = d1 - v;  h = |k_log| = |d1·v - 0.5·v²| (need sign).
//!   c_* = Φ(d1) - K/F · Φ(d2)  with K/F = exp(-h) for OTM call (k>0).
//!
//! We translate to voltic's (S, K, T, r, price) by fixing S=1, T=1, r=0,
//! K = exp(k_log), price = c_* · S.
//!
//! Reports: ns/option for implied_vol_fast on the wing-dominated grid,
//! and max abs err vs σ_true (= v / sqrt(T) = v with T=1).

use std::time::Instant;
use voltic::{implied_vol_fast, OptionKind};

const V_GRID_START: f64 = 0.01;
const V_GRID_STEP: f64 = 0.05;
const V_GRID_N: usize = 41; // 0.01, 0.05, 0.10, ..., 2.00 (overstep is fine)
const D_GRID: &[f64] = &[
    0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99,
];
const REPS: usize = 5000;

fn phi_inv(p: f64) -> f64 {
    // Acklam's algorithm, scalar; accurate enough for grid construction.
    // Adapted from public domain Wichura AS241 polynomial.
    let q = p - 0.5;
    if q.abs() <= 0.425 {
        let r = q * q;
        q * ((((-39.69683028665376 * r + 220.9460984245205) * r - 275.9285104469687) * r
            + 138.357751867269)
            * r
            - 30.66479806614716)
            * r
            + 2.506628277459239
                / (((((-54.47609879822406 * r + 161.5858368580409) * r - 155.6989798598866) * r
                    + 66.80131188771972)
                    * r
                    - 13.28068155288572)
                    * r
                    + 1.0)
    } else {
        // Tail. Use scipy-style approximation for ~6 digit accuracy; sufficient.
        let r = if q < 0.0 { p } else { 1.0 - p };
        let lr = (-r.ln()).sqrt();
        let z = (((((2.938163982698783 * lr + 4.374664141464968) * lr - 2.549732539343734) * lr
            - 2.400758277161838)
            * lr
            - 0.3223964580411365)
            * lr
            - 0.007784894002430293)
            / ((((3.754408661907416 * lr + 2.445134137142996) * lr + 0.3224671290700398) * lr
                + 0.007784695709041462)
                * lr
                + 1.0);
        if q < 0.0 {
            -z
        } else {
            z
        }
    }
}

type GridCols = (
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<OptionKind>,
    Vec<f64>,
);

fn build_grid() -> GridCols {
    let mut s = Vec::new();
    let mut k = Vec::new();
    let mut t = Vec::new();
    let mut r = Vec::new();
    let mut price = Vec::new();
    let mut kind = Vec::new();
    let mut sigma_true = Vec::new();
    let spot: f64 = 1.0;
    let tte: f64 = 1.0;
    let rate: f64 = 0.0;
    let sqrt_t: f64 = tte.sqrt();
    for vi in 0..V_GRID_N {
        let v = V_GRID_START + V_GRID_STEP * vi as f64;
        if !(V_GRID_START..=2.0).contains(&v) {
            continue;
        }
        for &dlt in D_GRID.iter() {
            // d1 = Φ⁻¹(Δ); d2 = d1 − v.
            // k_log = ln(K/F) = −(d1·v − 0.5·v²) = 0.5 v² − d1·v.
            // For OTM call (positive k), Δ ∈ (0, 0.5): d1 < 0, k > 0.
            let d1 = phi_inv(dlt);
            let _d2 = d1 - v;
            let k_log = 0.5 * v * v - d1 * v;
            // We want OTM CALLS (positive k_log, Δ < 0.5) and OTM PUTS
            // (negative k_log, |k_log| via Δ > 0.5). Both feed the wing.
            let strike = (k_log).exp();
            let sigma = v / sqrt_t;
            // Use OTM-leg pricing: if k_log > 0, call; else put.
            let is_call = k_log >= 0.0;
            let opt_kind = if is_call {
                OptionKind::Call
            } else {
                OptionKind::Put
            };
            // Price via voltic's bs_price so the round-trip is internally consistent.
            let p = voltic::bs_price(&[spot], &[strike], &[tte], &[rate], &[sigma], &[opt_kind])[0];
            // Filter: price must be above f64 noise so the inverse is meaningful.
            if !(p.is_finite() && p > 1e-15) {
                continue;
            }
            s.push(spot);
            k.push(strike);
            t.push(tte);
            r.push(rate);
            price.push(p);
            kind.push(opt_kind);
            sigma_true.push(sigma);
        }
    }
    (s, k, t, r, price, kind, sigma_true)
}

fn main() {
    let (s, k, t, r, price, kind, sigma_true) = build_grid();
    let n = s.len();
    eprintln!("built grid: {n} cases");

    // Sanity: solve once, report max abs err.
    let solved = implied_vol_fast(&s, &k, &t, &r, &price, &kind);
    let mut max_abs = 0.0_f64;
    let mut nan_count = 0;
    for i in 0..n {
        if solved[i].is_nan() {
            nan_count += 1;
            continue;
        }
        let e = (solved[i] - sigma_true[i]).abs();
        if e > max_abs {
            max_abs = e;
        }
    }
    println!("=== volfi v×Δ grid benchmark ===");
    println!("cases: {n}");
    println!("max |σ_solved − σ_true|: {max_abs:.3e}");
    println!("NaN count: {nan_count}");

    // Warmup.
    let w = implied_vol_fast(&s, &k, &t, &r, &price, &kind);
    std::hint::black_box(w);

    // Timed loop: REPS passes.
    let mut samples = Vec::new();
    for _ in 0..7 {
        let t0 = Instant::now();
        let mut total = 0usize;
        for _ in 0..REPS {
            let v = implied_vol_fast(&s, &k, &t, &r, &price, &kind);
            total += v.len();
            std::hint::black_box(&v);
        }
        let dt = t0.elapsed();
        std::hint::black_box(total);
        samples.push(dt.as_secs_f64() / (n as f64 * REPS as f64) * 1e9);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = samples[samples.len() / 2];
    println!(
        "median ns/option (median of 7, {} reps each): {median:.1}",
        REPS
    );
    println!("options/sec: {:.3e}", 1e9 / median);

    // Stratified report by whether the lane goes through the wing predicate.
    let mut wing_lanes = 0usize;
    let mut wing_max_err = 0.0_f64;
    let mut cheb_lanes = 0usize;
    let mut cheb_max_err = 0.0_f64;
    for i in 0..n {
        let k_log = (k[i] / s[i]).ln() - r[i] * t[i];
        let h = k_log.abs();
        let q_cdf = {
            // Approximation: q = (1-c)/m; computed via the kernel formula.
            let ek = k_log.exp();
            let m = if k_log > 0.0 { 1.0 } else { ek };
            let xn = price[i] / s[i];
            let c = if matches!(kind[i], OptionKind::Call) {
                xn
            } else {
                xn + 1.0 - ek
            };
            (1.0 - c) / m
        };
        let q_surv = 1.0 - q_cdf;
        let in_wing = (2.95..8.0).contains(&h) && q_surv > 0.0 && q_surv < 0.30;
        if !solved[i].is_nan() {
            let e = (solved[i] - sigma_true[i]).abs();
            if in_wing {
                wing_lanes += 1;
                if e > wing_max_err {
                    wing_max_err = e;
                }
            } else {
                cheb_lanes += 1;
                if e > cheb_max_err {
                    cheb_max_err = e;
                }
            }
        }
    }
    println!("wing lanes: {wing_lanes}  max abs err: {wing_max_err:.3e}");
    println!("cheb lanes: {cheb_lanes}  max abs err: {cheb_max_err:.3e}");
}
