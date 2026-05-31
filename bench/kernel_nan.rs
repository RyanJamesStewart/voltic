//! Diagnostic: count kernel-level NaN (before the wrapper fallback) on the
//! same synthetic dataset the bench harness uses. Reports overall + per-band.
//! Conductor-scope only; not part of the public surface.
#![feature(portable_simd)]

#[path = "data.rs"]
mod data;

use voltic::implied_vol_fast_kernel;

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(data::DEFAULT_N);
    let ds = data::generate(n);
    let raw = implied_vol_fast_kernel(
        &ds.spot, &ds.strike, &ds.tte, &ds.rate, &ds.price, &ds.kind,
    );

    let mut tot_nan = 0usize;
    let mut max_err_accepted = 0.0_f64;
    let mut otm_nan = 0usize;
    let mut otm_n = 0usize;
    let mut atm_nan = 0usize;
    let mut atm_n = 0usize;
    let mut itm_nan = 0usize;
    let mut itm_n = 0usize;
    let mut other_nan = 0usize;
    let mut other_n = 0usize;
    let mut atm_max_err = 0.0_f64;
    let mut otm_max_err = 0.0_f64;
    let mut itm_max_err = 0.0_f64;

    for i in 0..n {
        let v = raw[i];
        let is_nan = v.is_nan();
        if is_nan { tot_nan += 1; }
        match ds.band(i) {
            data::Band::DeepOtm => {
                otm_n += 1;
                if is_nan { otm_nan += 1; } else {
                    otm_max_err = otm_max_err.max((v - ds.sigma_true[i]).abs());
                }
            }
            data::Band::NearAtm => {
                atm_n += 1;
                if is_nan { atm_nan += 1; } else {
                    atm_max_err = atm_max_err.max((v - ds.sigma_true[i]).abs());
                }
            }
            data::Band::DeepItm => {
                itm_n += 1;
                if is_nan { itm_nan += 1; } else {
                    itm_max_err = itm_max_err.max((v - ds.sigma_true[i]).abs());
                }
            }
            _ => {
                other_n += 1;
                if is_nan { other_nan += 1; }
            }
        }
        if !is_nan {
            max_err_accepted = max_err_accepted.max((v - ds.sigma_true[i]).abs());
        }
    }

    println!("n = {n}");
    println!("kernel NaN overall: {tot_nan} ({:.4}%)", 100.0 * tot_nan as f64 / n as f64);
    println!("kernel max |err| on accepted lanes: {:.3e}", max_err_accepted);
    println!("per-band kernel NaN%:");
    println!("  deep OTM (<0.7)  : {otm_nan:>7}/{otm_n:<7} = {:.4}%   max|err| accepted {:.3e}",
        100.0 * otm_nan as f64 / otm_n.max(1) as f64, otm_max_err);
    println!("  near ATM (0.95-1.05): {atm_nan:>7}/{atm_n:<7} = {:.4}%   max|err| accepted {:.3e}",
        100.0 * atm_nan as f64 / atm_n.max(1) as f64, atm_max_err);
    println!("  deep ITM (>1.3)  : {itm_nan:>7}/{itm_n:<7} = {:.4}%   max|err| accepted {:.3e}",
        100.0 * itm_nan as f64 / itm_n.max(1) as f64, itm_max_err);
    println!("  other            : {other_nan:>7}/{other_n:<7} = {:.4}%",
        100.0 * other_nan as f64 / other_n.max(1) as f64);
}
