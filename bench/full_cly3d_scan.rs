//! Full CLY-3D typed-API scan: status counts + worst σ deviation among Computed.
//! Compares typed σ to CSV σ_true (the AQFED-derived ground truth).
use std::fs::File;
use std::io::{BufRead, BufReader};
use voltic::{implied_vol_typed_batch, ImpliedVolStatus, OptionKind};

fn parse_csv(
    path: &str,
) -> (
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<OptionKind>,
) {
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

fn run(label: &str, csv: &str) {
    println!("==== {label} ({csv}) ====");
    let (s, k, t, r, p, sigma_true, kind) = parse_csv(csv);
    let n = s.len();
    let typed = implied_vol_typed_batch(&s, &k, &t, &r, &p, &kind);

    let mut counts = [0usize; 7]; // Computed, BelowVolMin, AboveVolMax, BelowIntrinsic, AboveMaximum, NonFinite, FailedToConverge
    let mut max_sigma_err = 0.0_f64;
    let mut max_sigma_err_idx: usize = 0;
    let mut sum_sq_err = 0.0_f64;
    let mut n_computed = 0;

    for i in 0..n {
        match typed[i].status {
            ImpliedVolStatus::Computed => {
                counts[0] += 1;
                let err = (typed[i].value - sigma_true[i]).abs();
                if err > max_sigma_err {
                    max_sigma_err = err;
                    max_sigma_err_idx = i;
                }
                sum_sq_err += err * err;
                n_computed += 1;
            }
            ImpliedVolStatus::BelowVolMin { .. } => counts[1] += 1,
            ImpliedVolStatus::AboveVolMax { .. } => counts[2] += 1,
            ImpliedVolStatus::BelowIntrinsic => counts[3] += 1,
            ImpliedVolStatus::AboveMaximum => counts[4] += 1,
            ImpliedVolStatus::NonFinite => counts[5] += 1,
            ImpliedVolStatus::FailedToConverge => counts[6] += 1,
        }
    }
    println!("  rows: {}", n);
    println!("  Computed       : {}", counts[0]);
    println!("  BelowVolMin    : {}", counts[1]);
    println!("  AboveVolMax    : {}", counts[2]);
    println!("  BelowIntrinsic : {}", counts[3]);
    println!("  AboveMaximum   : {}", counts[4]);
    println!("  NonFinite      : {}", counts[5]);
    println!("  FailedConverge : {}", counts[6]);
    println!(
        "  Computed worst |σ−σ_true| = {:.6e} at row {}",
        max_sigma_err, max_sigma_err_idx
    );
    if n_computed > 0 {
        let rmse = (sum_sq_err / n_computed as f64).sqrt();
        println!(
            "  Computed RMSE                = {:.6e}  (n={})",
            rmse, n_computed
        );
    }
    // Count Computed rows with σ error > 1e-6 (V1 "silent mislabel" threshold).
    let mut mislabels_1e6 = 0usize;
    let mut mislabels_1e9 = 0usize;
    for i in 0..n {
        if matches!(typed[i].status, ImpliedVolStatus::Computed) {
            let err = (typed[i].value - sigma_true[i]).abs();
            if err > 1e-6 {
                mislabels_1e6 += 1;
            }
            if err > 1e-9 {
                mislabels_1e9 += 1;
            }
        }
    }
    println!("  Computed σ error > 1e-6  : {}", mislabels_1e6);
    println!("  Computed σ error > 1e-9  : {}", mislabels_1e9);
    println!();
}

fn main() {
    run("CLY-3D", "cly3d_data.csv");
    run("ATM-dense", "atm_dense_data.csv");
}
