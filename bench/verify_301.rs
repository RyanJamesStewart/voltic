//! `bench/verify_301.rs` — verify the v1.1.0 NaN rows (13 on CLY-3D + 288 on
//! ATM-dense) classify as `BelowVolMin { computed: ~0.00999 }` under the v1.2
//! typed API.
//!
//! Reads `cly3d_data.csv` and `atm_dense_data.csv`, runs `implied_vol_fast`
//! and `implied_vol_typed_batch` over each. For every row where
//! `implied_vol_fast` returned NaN, dumps:
//!   - typed status
//!   - typed value (if Computed / BelowVolMin / AboveVolMax)
//!   - σ_true from the bench CSV (the AQFED-derived ground truth)
//!   - |typed_value - σ_true| (the diagnostic claim is these should match to ~1e-16)

use std::fs::File;
use std::io::{BufRead, BufReader};
use voltic::{implied_vol_fast, implied_vol_typed_batch, ImpliedVolStatus, OptionKind};

type CsvCols = (
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<OptionKind>,
);

fn parse_csv(path: &str) -> CsvCols {
    let f = File::open(path).expect("open csv");
    let r = BufReader::new(f);
    let mut s = Vec::new();
    let mut k = Vec::new();
    let mut t = Vec::new();
    let mut rt = Vec::new();
    let mut p = Vec::new();
    let mut sigma = Vec::new();
    let mut kind = Vec::new();
    let mut first = true;
    for line in r.lines() {
        let line = line.unwrap();
        if first {
            first = false;
            continue;
        }
        let cols: Vec<&str> = line.split(',').collect();
        s.push(cols[0].parse().unwrap());
        k.push(cols[1].parse().unwrap());
        t.push(cols[2].parse().unwrap());
        rt.push(cols[3].parse().unwrap());
        p.push(cols[4].parse().unwrap());
        sigma.push(cols[5].parse().unwrap());
        kind.push(if cols[6].trim() == "c" {
            OptionKind::Call
        } else {
            OptionKind::Put
        });
    }
    (s, k, t, rt, p, sigma, kind)
}

fn analyze(label: &str, csv: &str) {
    println!("==== {label} ({csv}) ====");
    let (s, k, t, r, p, sigma_true, kind) = parse_csv(csv);
    let n = s.len();
    println!("rows: {}", n);

    let f_fast = implied_vol_fast(&s, &k, &t, &r, &p, &kind);
    let typed = implied_vol_typed_batch(&s, &k, &t, &r, &p, &kind);

    // Find the rows where f_fast returned NaN.
    let nan_idx: Vec<usize> = (0..n).filter(|&i| f_fast[i].is_nan()).collect();
    println!("implied_vol_fast NaN count: {}", nan_idx.len());

    // Tally typed status on those.
    let mut below_vol_min = 0;
    let mut above_vol_max = 0;
    let mut computed = 0;
    let mut below_intrinsic = 0;
    let mut above_max = 0;
    let mut non_finite = 0;
    let mut failed = 0;

    let mut computed_vals: Vec<f64> = Vec::new();
    let mut diffs_vs_true: Vec<f64> = Vec::new();

    for &i in &nan_idx {
        match typed[i].status {
            ImpliedVolStatus::BelowVolMin { computed: c } => {
                below_vol_min += 1;
                computed_vals.push(c);
                diffs_vs_true.push((c - sigma_true[i]).abs());
            }
            ImpliedVolStatus::AboveVolMax { computed: c } => {
                above_vol_max += 1;
                computed_vals.push(c);
                diffs_vs_true.push((c - sigma_true[i]).abs());
            }
            ImpliedVolStatus::Computed => {
                computed += 1;
                computed_vals.push(typed[i].value);
                diffs_vs_true.push((typed[i].value - sigma_true[i]).abs());
            }
            ImpliedVolStatus::BelowIntrinsic => below_intrinsic += 1,
            ImpliedVolStatus::AboveMaximum => above_max += 1,
            ImpliedVolStatus::NonFinite => non_finite += 1,
            ImpliedVolStatus::FailedToConverge => failed += 1,
        }
    }

    println!("  typed status of f_fast-NaN rows:");
    println!("    BelowVolMin     : {}", below_vol_min);
    println!("    AboveVolMax     : {}", above_vol_max);
    println!("    Computed        : {}", computed);
    println!("    BelowIntrinsic  : {}", below_intrinsic);
    println!("    AboveMaximum    : {}", above_max);
    println!("    NonFinite       : {}", non_finite);
    println!("    FailedToConverge: {}", failed);

    if !computed_vals.is_empty() {
        let min_v = computed_vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_v = computed_vals
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let mean: f64 = computed_vals.iter().sum::<f64>() / (computed_vals.len() as f64);
        println!(
            "  computed σ range: [{:.10e}, {:.10e}], mean {:.10e}",
            min_v, max_v, mean
        );
    }
    if !diffs_vs_true.is_empty() {
        let max_diff = diffs_vs_true.iter().cloned().fold(0.0, f64::max);
        let median = {
            let mut ds = diffs_vs_true.clone();
            ds.sort_by(|a, b| a.partial_cmp(b).unwrap());
            ds[ds.len() / 2]
        };
        println!("  |typed σ − σ_true| (vs CSV ground truth):");
        println!("    median: {:.6e}", median);
        println!("    max   : {:.6e}", max_diff);
    }

    // Print first 5 rows for human inspection.
    println!("  first 5 affected rows:");
    for &i in nan_idx.iter().take(5) {
        let sv = match typed[i].status {
            ImpliedVolStatus::BelowVolMin { computed: c } => {
                format!("BelowVolMin{{ c={:.10e} }}", c)
            }
            ImpliedVolStatus::AboveVolMax { computed: c } => {
                format!("AboveVolMax{{ c={:.10e} }}", c)
            }
            ImpliedVolStatus::Computed => format!("Computed (value={:.10e})", typed[i].value),
            other => format!("{:?}", other),
        };
        println!(
            "    row {} | S={} K={} T={} r={} p={:.6e} kind={:?} σ_true={:.10e}",
            i, s[i], k[i], t[i], r[i], p[i], kind[i], sigma_true[i]
        );
        println!("           typed status: {}", sv);
    }
    println!();
}

fn main() {
    analyze("CLY-3D", "cly3d_data.csv");
    analyze("ATM-dense", "atm_dense_data.csv");
}
