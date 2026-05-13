#![allow(dead_code)] // shared module included into multiple bench targets; each uses a subset
//! The synthetic benchmark dataset — one persisted generator, one seed.
//!
//! Every implementation in the benchmark (jackal scalar/vectorized, py_vollib,
//! py_vollib_vectorized, QuantLib, the naive-Rust-scalar baseline) runs against
//! *this* dataset. A skeptic can re-run with byte-identical inputs by re-running
//! the generator with [`SEED`].
//!
//! Distribution (drawn independently per option, then a `(price, σ_true)` pair
//! computed by Black-Scholes; an option is **kept only if its premium exceeds
//! intrinsic by more than 1e-6 · spot** — see "the recoverability filter"
//! below — otherwise it is resampled, so every row in the dataset has an IV the
//! problem's conditioning floor can actually invert):
//!   * spot   `S` ~ Uniform[50, 200]
//!   * strike `K` ~ Uniform[40, 240] (independent of S; S/K spans the
//!     deep-OTM / near-ATM / deep-ITM bands)
//!   * tte    `T` ~ exp(Uniform[ln(1/365), ln(2)]) (log-uniform, 1 day–2 y)
//!   * rate   `r` ~ Uniform[0, 0.06]
//!   * vol    `σ` ~ Uniform[0.05, 0.80]
//!   * kind       — alternating call/put by index parity
//!   * price      = Black-Scholes(S, K, T, r, σ, kind)
//!
//! **The recoverability filter.** A randomly sampled (deep-OTM, short-expiry,
//! low-vol) option can have a Black-Scholes premium below the f64-representable
//! floor for its magnitude — there is no IV to recover, only round-off. Such
//! draws are *resampled* (the RNG keeps advancing; the index parity for
//! call/put is preserved) so the benchmark measures solver throughput and
//! accuracy on the options that *have* a well-posed IV. The corner that's
//! filtered out is the same one the README's "Limitations" section names as
//! unsupported (Jäckel's rational-cubic-spline regime); a skeptic can reproduce
//! both the dataset and the filter from [`SEED`] + this file.
//!
//! RNG: SplitMix64 (a 64-bit-state, fully-specified, deterministic generator —
//! no external dep, byte-reproducible). The dataset is `N` options; the
//! benchmark uses `N = `[`DEFAULT_N`].

use jackal::OptionKind;

/// The one seed. Change this and the dataset changes; the README quotes it.
pub const SEED: u64 = 0x_5EED_BEEF_CAFE_F00D;
/// Dataset size used by the benchmark harness.
pub const DEFAULT_N: usize = 1_000_000;
/// Recoverability floor: an option is kept only if `premium − intrinsic`
/// exceeds this fraction of spot (below it, the IV is round-off, not signal).
pub const RECOVERABILITY_FLOOR_FRAC: f64 = 1e-6;

/// SplitMix64 — a tiny, fully-specified PRNG (Steele, Lea & Flood 2014). One
/// 64-bit word of state; `next_u64` is the standard finalizer mix.
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        SplitMix64(seed)
    }
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform f64 in [0, 1) — top 53 bits, the standard construction.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
    /// Uniform f64 in [lo, hi).
    #[inline]
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }
}

/// The columns of the dataset.
pub struct Dataset {
    pub spot: Vec<f64>,
    pub strike: Vec<f64>,
    pub tte: Vec<f64>,
    pub rate: Vec<f64>,
    pub price: Vec<f64>,
    pub sigma_true: Vec<f64>,
    pub kind: Vec<OptionKind>,
}

impl Dataset {
    pub fn len(&self) -> usize {
        self.spot.len()
    }
    pub fn is_empty(&self) -> bool {
        self.spot.is_empty()
    }
    /// Moneyness band of option `i`, by S/K.
    pub fn band(&self, i: usize) -> Band {
        let m = self.spot[i] / self.strike[i];
        if m < 0.7 {
            Band::DeepOtm
        } else if (0.95..=1.05).contains(&m) {
            Band::NearAtm
        } else if m > 1.3 {
            Band::DeepItm
        } else {
            Band::Other
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Band {
    /// S/K < 0.7
    DeepOtm,
    /// 0.95 ≤ S/K ≤ 1.05
    NearAtm,
    /// S/K > 1.3
    DeepItm,
    /// the rest (0.7–0.95, 1.05–1.3) — reported, just not one of the three
    /// stratified bands
    Other,
}

#[inline]
fn intrinsic(s: f64, k: f64, t: f64, r: f64, kind: OptionKind) -> f64 {
    let df = (-r * t).exp();
    match kind {
        OptionKind::Call => (s - k * df).max(0.0),
        OptionKind::Put => (k * df - s).max(0.0),
    }
}

/// Generate the dataset of `n` options from [`SEED`]. The `price` column is
/// computed by `jackal::bs_price`, so every row is exactly consistent with its
/// `sigma_true` — the accuracy a solver is measured against is "recover the σ
/// that produced this price", the right ground truth. Draws whose premium is
/// below the recoverability floor are resampled (see the module docs).
pub fn generate(n: usize) -> Dataset {
    let mut rng = SplitMix64::new(SEED);
    let mut spot = Vec::with_capacity(n);
    let mut strike = Vec::with_capacity(n);
    let mut tte = Vec::with_capacity(n);
    let mut rate = Vec::with_capacity(n);
    let mut sigma_true = Vec::with_capacity(n);
    let mut kind = Vec::with_capacity(n);
    let ln_lo = (1.0_f64 / 365.0).ln();
    let ln_hi = 2.0_f64.ln();
    let mut i = 0usize;
    while i < n {
        let s = rng.uniform(50.0, 200.0);
        let k = rng.uniform(40.0, 240.0);
        let t = rng.uniform(ln_lo, ln_hi).exp();
        let r = rng.uniform(0.0, 0.06);
        let v = rng.uniform(0.05, 0.80);
        let kd = if i.is_multiple_of(2) {
            OptionKind::Call
        } else {
            OptionKind::Put
        };
        // Compute the premium for *this* draw and accept only if it clears the
        // recoverability floor; otherwise skip (the RNG has already advanced).
        let p = jackal::bs_price(&[s], &[k], &[t], &[r], &[v], &[kd])[0];
        if !p.is_finite() || (p - intrinsic(s, k, t, r, kd)) <= RECOVERABILITY_FLOOR_FRAC * s {
            continue;
        }
        spot.push(s);
        strike.push(k);
        tte.push(t);
        rate.push(r);
        sigma_true.push(v);
        kind.push(kd);
        i += 1;
    }
    let price = jackal::bs_price(&spot, &strike, &tte, &rate, &sigma_true, &kind);
    Dataset {
        spot,
        strike,
        tte,
        rate,
        price,
        sigma_true,
        kind,
    }
}

/// Write the dataset to a CSV (`spot,strike,tte,rate,price,sigma_true,kind`)
/// so the Python comparison harness reads byte-identical inputs.
pub fn write_csv(ds: &Dataset, path: &str) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(f, "spot,strike,tte,rate,price,sigma_true,kind")?;
    for i in 0..ds.len() {
        writeln!(
            f,
            "{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{}",
            ds.spot[i],
            ds.strike[i],
            ds.tte[i],
            ds.rate[i],
            ds.price[i],
            ds.sigma_true[i],
            match ds.kind[i] {
                OptionKind::Call => "c",
                OptionKind::Put => "p",
            }
        )?;
    }
    Ok(())
}
