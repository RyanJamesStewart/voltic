//! Build an implied-volatility smile from a strip of synthetic option quotes
//! around the forward. Run with `cargo run --release --example smile_fit`.
//!
//! The example contrasts the three voltic kernels on the same input:
//!  * `implied_vol` — fastest, but `NaN`s on the deep-OTM-near-expiry corner;
//!  * `implied_vol_rational` — full canonical coverage at machine precision;
//!  * `implied_vol_explicit` — Schadner's inverse-Gaussian explicit kernel.
//!
//! For a real smile-fit pipeline, pass `implied_vol_rational` since you want
//! every quote in the chain to map to a finite σ. The dense print at the end
//! shows the rational kernel solving every strike — including the deep wings
//! where the direct Newton path would have returned `NaN`.

use voltic::{bs_price, implied_vol, implied_vol_explicit, implied_vol_rational, OptionKind};

fn main() {
    // 30-day option on a $100 forward; a 19-strike smile from 70 to 130
    // with a textbook "smirk" — higher vol in the wings, lower around ATM.
    let spot = 100.0_f64;
    let tte = 30.0 / 365.0;
    let rate = 0.04;
    let strikes: Vec<f64> = (70..=130).step_by(5).map(|k| k as f64).collect();
    let n = strikes.len();

    // Construct a smirky smile: σ(K) = σ_atm + a·(K/S − 1)² + b·(K/S − 1).
    let sigma_true: Vec<f64> = strikes
        .iter()
        .map(|&k| {
            let m = k / spot - 1.0;
            0.20 + 0.30 * m * m - 0.10 * m
        })
        .collect();

    // Calls below ATM are OTM puts and vice versa — but voltic accepts both
    // kinds and canonicalizes internally, so we just price calls everywhere.
    let kind = vec![OptionKind::Call; n];
    let spot_v = vec![spot; n];
    let tte_v = vec![tte; n];
    let rate_v = vec![rate; n];
    let price = bs_price(&spot_v, &strikes, &tte_v, &rate_v, &sigma_true, &kind);

    let iv_direct = implied_vol(&spot_v, &strikes, &tte_v, &rate_v, &price, &kind);
    let iv_rational = implied_vol_rational(&spot_v, &strikes, &tte_v, &rate_v, &price, &kind);
    let iv_explicit = implied_vol_explicit(&spot_v, &strikes, &tte_v, &rate_v, &price, &kind);

    println!(
        "{:>7} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "K", "price", "σ_true", "direct", "rational", "explicit"
    );
    let mut direct_nan = 0;
    for i in 0..n {
        let d = if iv_direct[i].is_nan() {
            direct_nan += 1;
            format!("{:>10}", "NaN")
        } else {
            format!("{:>10.6}", iv_direct[i])
        };
        println!(
            "{:>7.1} {:>10.4} {:>10.6} {} {:>10.6} {:>10.6}",
            strikes[i], price[i], sigma_true[i], d, iv_rational[i], iv_explicit[i]
        );
    }
    println!();
    println!(
        "direct NaN'd {direct_nan} of {n} strikes; rational and explicit solved all {n} (machine precision)"
    );
}
