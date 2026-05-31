//! `bench/cly_3d.rs` — CLY-3D (Cui-Liu-Yao 2021) standard grid benchmark.
//!
//! GRID SPECIFICATION (verbatim, from ThiopheneIV (arXiv:2605.22427)
//! Appendix A.1 "Dataset grids", cross-referenced with FlashIV
//! (arXiv:2605.29102) §4.2 Table 3):
//!
//!   "CLY-3D. S = 100, r = 0.03,
//!    K = linspace(105, 800, 40),
//!    T = linspace(0.01, 2, 40),
//!    sigma = linspace(0.01, 0.99, 40),
//!    retaining prices above 10^-20 as in the Cui–Liu–Yao comparison script."
//!
//! Factor structure: 40 × 40 × 40 = 64,000 raw grid points; after the
//! price > 1e-20 filter we get 51,321 cases (the canonical CLY-3D count
//! also reported in FlashIV Table 3 and ThiopheneIV Table 3).
//!
//! All cases are calls (S < K everywhere, so they are OTM calls), as in
//! the CLY 2021 comparison script. The grid deliberately puts deep-OTM
//! short-maturity low-vol cases adjacent to high-vol long-maturity cases,
//! covering the full price range from machine epsilon up to ~F.
//!
//! Source paper for the grid itself:
//!   Cui, Z.; Liu, Y.; Yao, Y. "Tighter Bounds for Implied Volatility with
//!   the Dirac Delta Family Method." J. Futures Markets 2025, 45, e70024.
//!
//! Operational definition transcribed from FlashIV / ThiopheneIV because
//! the underlying CLY paper is paywalled; both post-LBR papers cite the
//! identical 51,321-case count, so the spec is unambiguous.
//!
//! What we measure (median of 7 timed passes after warmup, `taskset -c 0`):
//!   * voltic `implied_vol_fast`                  — Schadner-cold + Cheb+Halley
//!   * voltic `implied_vol_vectorized_with_contexts` — cold per-row context API
//!   * (LBR / py_vollib_vectorized / volfi are run from the Python harness
//!      `bench/python/cly_3d_compare.py` on the same `cly3d_data.csv`)
//!
//! Output CSV `cly3d_data.csv` columns: `spot,strike,tte,rate,price,sigma_true,kind`
//! consumed by the Python harness for cross-solver comparison.

use std::time::Instant;
use voltic::{
    bs_price, canonical_c_from_price, implied_vol_fast,
    implied_vol_vectorized_with_contexts, OptionKind, OtmContext,
};

const SPOT: f64 = 100.0;
const RATE: f64 = 0.03;
const K_LO: f64 = 105.0;
const K_HI: f64 = 800.0;
const T_LO: f64 = 0.01;
const T_HI: f64 = 2.0;
const SIG_LO: f64 = 0.01;
const SIG_HI: f64 = 0.99;
const N_AXIS: usize = 40;
const PRICE_FLOOR: f64 = 1e-20;
const REPS: usize = 50;

fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![a];
    }
    let step = (b - a) / (n as f64 - 1.0);
    (0..n).map(|i| a + step * i as f64).collect()
}

#[allow(clippy::type_complexity)]
fn build_grid() -> (
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<OptionKind>,
    Vec<f64>,
) {
    let ks = linspace(K_LO, K_HI, N_AXIS);
    let ts = linspace(T_LO, T_HI, N_AXIS);
    let sigs = linspace(SIG_LO, SIG_HI, N_AXIS);

    let mut s_v = Vec::with_capacity(N_AXIS.pow(3));
    let mut k_v = Vec::with_capacity(N_AXIS.pow(3));
    let mut t_v = Vec::with_capacity(N_AXIS.pow(3));
    let mut r_v = Vec::with_capacity(N_AXIS.pow(3));
    let mut p_v = Vec::with_capacity(N_AXIS.pow(3));
    let mut sig_v = Vec::with_capacity(N_AXIS.pow(3));
    let mut kind_v = Vec::with_capacity(N_AXIS.pow(3));

    let kind = OptionKind::Call;
    for &kk in &ks {
        for &tt in &ts {
            for &ss in &sigs {
                // S=100, K in [105,800] => K > F = S·e^(rT) for small rT, so OTM call.
                let p = bs_price(&[SPOT], &[kk], &[tt], &[RATE], &[ss], &[kind])[0];
                if !p.is_finite() || p <= PRICE_FLOOR {
                    continue;
                }
                s_v.push(SPOT);
                k_v.push(kk);
                t_v.push(tt);
                r_v.push(RATE);
                p_v.push(p);
                sig_v.push(ss);
                kind_v.push(kind);
            }
        }
    }
    (s_v, k_v, t_v, r_v, p_v, kind_v, sig_v)
}

fn write_csv(
    path: &str,
    s: &[f64],
    k: &[f64],
    t: &[f64],
    r: &[f64],
    p: &[f64],
    sig: &[f64],
    kind: &[OptionKind],
) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "spot,strike,tte,rate,price,sigma_true,kind")?;
    for i in 0..s.len() {
        let kc = match kind[i] {
            OptionKind::Call => "c",
            OptionKind::Put => "p",
        };
        writeln!(
            f,
            "{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{}",
            s[i], k[i], t[i], r[i], p[i], sig[i], kc
        )?;
    }
    Ok(())
}

fn timed_solver_fast(
    n: usize,
    s: &[f64],
    k: &[f64],
    t: &[f64],
    r: &[f64],
    p: &[f64],
    kind: &[OptionKind],
) -> (f64, Vec<f64>) {
    // Warmup
    let warm = implied_vol_fast(s, k, t, r, p, kind);
    std::hint::black_box(&warm);

    let mut samples = Vec::with_capacity(7);
    let mut last: Vec<f64> = warm;
    for _ in 0..7 {
        let t0 = Instant::now();
        let mut total = 0usize;
        for _ in 0..REPS {
            let v = implied_vol_fast(s, k, t, r, p, kind);
            total += v.len();
            last = v;
        }
        let dt = t0.elapsed();
        std::hint::black_box(total);
        samples.push(dt.as_secs_f64() / (n as f64 * REPS as f64) * 1e9);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (samples[samples.len() / 2], last)
}

fn timed_solver_context(
    n: usize,
    s: &[f64],
    k: &[f64],
    t: &[f64],
    r: &[f64],
    p: &[f64],
    kind: &[OptionKind],
) -> (f64, Vec<f64>) {
    // Build one OtmContext per row (each (K,T) is unique on this grid) and
    // pre-canonicalize prices to OTM `c`.
    let contexts: Vec<OtmContext> = (0..n)
        .map(|i| OtmContext::from_market(k[i], t[i], s[i], r[i], 0.0))
        .collect();
    let canon: Vec<f64> = (0..n)
        .map(|i| {
            canonical_c_from_price(
                &contexts[i],
                s[i],
                p[i],
                matches!(kind[i], OptionKind::Call),
            )
        })
        .collect();

    // Warmup
    let warm = implied_vol_vectorized_with_contexts(&contexts, &canon);
    std::hint::black_box(&warm);

    let mut samples = Vec::with_capacity(7);
    let mut last: Vec<f64> = warm;
    for _ in 0..7 {
        let t0 = Instant::now();
        let mut total = 0usize;
        for _ in 0..REPS {
            let v = implied_vol_vectorized_with_contexts(&contexts, &canon);
            total += v.len();
            last = v;
        }
        let dt = t0.elapsed();
        std::hint::black_box(total);
        samples.push(dt.as_secs_f64() / (n as f64 * REPS as f64) * 1e9);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // `implied_vol_vectorized_with_contexts` returns sigma already (divides
    // by sqrt(T) internally), so no further conversion needed.
    (samples[samples.len() / 2], last)
}

fn report_errors(name: &str, solved: &[f64], sigma_true: &[f64]) -> (f64, usize, usize) {
    let n = solved.len();
    let mut max_abs = 0.0_f64;
    let mut nan_count = 0usize;
    let mut cat_count = 0usize;
    for i in 0..n {
        if !solved[i].is_finite() {
            nan_count += 1;
            continue;
        }
        let e = (solved[i] - sigma_true[i]).abs();
        if e > max_abs {
            max_abs = e;
        }
        if e >= 1e-3 {
            cat_count += 1;
        }
    }
    println!(
        "  {name}: cases {n}  max |sigma_err| {max_abs:.3e}  NaN {nan_count}  catastrophic(>=1e-3) {cat_count}"
    );
    (max_abs, nan_count, cat_count)
}

fn dump_per_row_errors(
    path: &str,
    s: &[f64],
    k: &[f64],
    t: &[f64],
    sig_true: &[f64],
    voltic_fast: &[f64],
    voltic_ctx: &[f64],
) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    writeln!(
        f,
        "spot,strike,tte,sigma_true,sigma_voltic_fast,sigma_voltic_ctx,err_voltic_fast,err_voltic_ctx"
    )?;
    for i in 0..s.len() {
        let ef = (voltic_fast[i] - sig_true[i]).abs();
        let ec = (voltic_ctx[i] - sig_true[i]).abs();
        writeln!(
            f,
            "{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.3e},{:.3e}",
            s[i], k[i], t[i], sig_true[i], voltic_fast[i], voltic_ctx[i], ef, ec
        )?;
    }
    Ok(())
}

fn main() {
    eprintln!("CLY-3D: building 40×40×40 = 64,000 raw cases, filtering price > 1e-20 ...");
    let (s, k, t, r, p, kind, sig) = build_grid();
    let n = s.len();
    println!("=== CLY-3D (Cui-Liu-Yao 2021) standard grid ===");
    println!(
        "spec: S=100, r=0.03, K=linspace(105,800,40), T=linspace(0.01,2,40), sigma=linspace(0.01,0.99,40)"
    );
    println!("retained cases (price > 1e-20): {n}");
    if n != 51_321 {
        println!(
            "WARN: case count {n} != 51,321 — check filter threshold or grid construction"
        );
    }

    // Dataset CSV (consumed by the Python comparison harness).
    let csv_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "cly3d_data.csv".to_string());
    write_csv(&csv_path, &s, &k, &t, &r, &p, &sig, &kind).expect("write csv");
    println!("dataset CSV written: {csv_path}");

    // ---- Solver 1: voltic implied_vol_fast (Cheb+Halley fast kernel) ----
    let (ns_fast, solved_fast) = timed_solver_fast(n, &s, &k, &t, &r, &p, &kind);
    let (max_fast, nan_fast, cat_fast) =
        report_errors("voltic implied_vol_fast", &solved_fast, &sig);
    println!(
        "  voltic implied_vol_fast: {ns_fast:.1} ns/option (median of 7, {REPS} reps each)"
    );

    // ---- Solver 2: voltic implied_vol_vectorized_with_contexts (cold) ----
    let (ns_ctx, solved_ctx) = timed_solver_context(n, &s, &k, &t, &r, &p, &kind);
    let (max_ctx, nan_ctx, cat_ctx) = report_errors(
        "voltic implied_vol_vectorized_with_contexts",
        &solved_ctx,
        &sig,
    );
    println!(
        "  voltic implied_vol_vectorized_with_contexts: {ns_ctx:.1} ns/option (median of 7, {REPS} reps each)"
    );

    // ---- Per-row error dump for Wren F's banded analysis ----
    let err_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "/tmp/cly3d_per_row_errors.csv".to_string());
    dump_per_row_errors(&err_path, &s, &k, &t, &sig, &solved_fast, &solved_ctx)
        .expect("write per-row csv");
    println!("per-row error CSV written: {err_path}");

    // ---- Emit a small JSON summary for the voltic rows (Python harness
    //      will merge in LBR / py_vollib_vectorized / volfi rows). ----
    println!("\n=== summary (voltic) ===");
    println!(
        "{{ \"voltic_fast\": {{ \"ns_per_option\": {ns_fast:.2}, \"max_abs_err\": {max_fast:.6e}, \"nan\": {nan_fast}, \"cat_ge_1e-3\": {cat_fast} }},"
    );
    println!(
        "  \"voltic_with_context_batch\": {{ \"ns_per_option\": {ns_ctx:.2}, \"max_abs_err\": {max_ctx:.6e}, \"nan\": {nan_ctx}, \"cat_ge_1e-3\": {cat_ctx} }} }}"
    );
}
