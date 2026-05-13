#![allow(clippy::excessive_precision)]
//! The "naive pure-Rust scalar Newton" baseline for the benchmark.
//!
//! This isolates *what SIMD bought* from *what Rust bought over Python*: it is
//! the same algorithm jackal uses (rational guess → Newton) but written one
//! option at a time with no vectorization — the implementation a competent
//! engineer would write in an afternoon without reaching for `std::simd`.
//! It uses the same West-2009 cumulative normal (scalar form) so the only
//! difference from jackal is the vectorization.

const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_68;
const SQRT_2PI: f64 = 2.506_628_274_631_000_5;

/// West 2009 cumulative normal, scalar.
fn phi(x: f64) -> f64 {
    let z = x.abs();
    if z > 37.0 {
        return if x > 0.0 { 1.0 } else { 0.0 };
    }
    let expo = (-0.5 * z * z).exp();
    let tail = if z < 7.071_067_811_865_475 {
        // Hart 5666 rational arm
        let np = (((((3.526_249_659_989_11e-2_f64 * z + 0.700_383_064_443_688) * z
            + 6.373_962_203_531_65)
            * z
            + 33.912_866_078_383)
            * z
            + 112.079_291_497_871)
            * z
            + 221.213_596_169_931)
            * z
            + 220.206_867_912_376;
        let dp = ((((((8.838_834_764_831_84e-2_f64 * z + 1.755_667_163_182_64) * z
            + 16.064_177_579_207)
            * z
            + 86.780_732_202_946_1)
            * z
            + 296.564_248_779_674)
            * z
            + 637.333_633_378_831)
            * z
            + 793.826_512_519_948)
            * z
            + 440.413_735_824_752;
        expo * np / dp
    } else {
        let mut b = z + 0.65;
        b = z + 4.0 / b;
        b = z + 3.0 / b;
        b = z + 2.0 / b;
        b = z + 1.0 / b;
        expo / b / SQRT_2PI
    };
    let res = if x > 0.0 { 1.0 - tail } else { tail };
    res.clamp(0.0, 1.0)
}

fn phi_pdf(x: f64) -> f64 {
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

fn bs_price(s: f64, k: f64, t: f64, r: f64, sigma: f64, is_call: bool) -> f64 {
    let sqrt_t = t.sqrt();
    let vst = sigma * sqrt_t;
    let df = (-r * t).exp();
    let d1 = ((s / k).ln() + (r + 0.5 * sigma * sigma) * t) / vst;
    let d2 = d1 - vst;
    let call = s * phi(d1) - k * df * phi(d2);
    if is_call {
        call
    } else {
        call - s + k * df
    }
}

fn vega(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
    let sqrt_t = t.sqrt();
    let d1 = ((s / k).ln() + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t);
    s * phi_pdf(d1) * sqrt_t
}

fn corrado_miller(s: f64, k: f64, t: f64, r: f64, price: f64, is_call: bool) -> f64 {
    let kp = k * (-r * t).exp();
    let c = if is_call { price } else { price + s - kp };
    let two_pi_over_t = 2.0 * core::f64::consts::PI / t;
    let a = c - 0.5 * (s - kp);
    let disc = (a * a - (s - kp).powi(2) / core::f64::consts::PI).max(0.0);
    let cm = two_pi_over_t.sqrt() / (s + kp) * (a + disc.sqrt());
    let bs88 = two_pi_over_t.sqrt() * c / s;
    let g = if cm.is_finite() && cm > 0.0 { cm } else { bs88 };
    let g = if g.is_finite() && g > 0.0 { g } else { 0.5 };
    g.clamp(0.01, 5.0)
}

/// Implied vol of one option; `NaN` on the same conditions jackal NaNs.
pub fn implied_vol_one(s: f64, k: f64, t: f64, r: f64, price: f64, is_call: bool) -> f64 {
    if !(s.is_finite() && k.is_finite() && t.is_finite() && r.is_finite() && price.is_finite()) {
        return f64::NAN;
    }
    if !(s > 0.0 && k > 0.0 && t > 0.0) {
        return f64::NAN;
    }
    let df = (-r * t).exp();
    let lower = if is_call {
        (s - k * df).max(0.0)
    } else {
        (k * df - s).max(0.0)
    };
    let upper = if is_call { s } else { k * df };
    if !(price > lower && price < upper) {
        return f64::NAN;
    }
    let mut sigma = corrado_miller(s, k, t, r, price, is_call);
    for _ in 0..32 {
        let p = bs_price(s, k, t, r, sigma, is_call);
        let v = vega(s, k, t, r, sigma).max(1e-300);
        let next = (sigma - (p - price) / v).clamp(0.01, 5.0);
        let done = (next - sigma).abs() < 1e-12;
        sigma = next;
        if done {
            break;
        }
    }
    let p_final = bs_price(s, k, t, r, sigma, is_call);
    if (p_final - price).abs() < 1e-7 * price.abs().max(1.0)
        && sigma > 0.01 * 1.0000001
        && sigma < 5.0 * 0.9999999
    {
        sigma
    } else {
        f64::NAN
    }
}

/// Solve a whole batch, scalar — the "baseline throughput" row.
pub fn implied_vol(
    spot: &[f64],
    strike: &[f64],
    tte: &[f64],
    rate: &[f64],
    price: &[f64],
    is_call: &[bool],
) -> Vec<f64> {
    (0..spot.len())
        .map(|i| implied_vol_one(spot[i], strike[i], tte[i], rate[i], price[i], is_call[i]))
        .collect()
}
