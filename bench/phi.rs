#![allow(dead_code, clippy::excessive_precision)]
//! The cumulative-normal kernel sub-benchmark: for each Φ(x) approximation
//! voltic implements, measure (a) max absolute error against a high-precision
//! reference and (b) throughput (ns per call, single core). Emitted as a CSV
//! (`kernel,max_abs_error,ns_per_call,options_per_sec`) consumed by
//! `scripts/plot_phi.py`.
//!
//! Reference values: a hardcoded table of Φ(x) at 41 points spanning x ∈
//! [−8, 8], each correct to all 16–17 f64 digits (computed offline with an
//! arbitrary-precision Φ). "Max abs error" is the worst |Φ̂(x) − Φ_ref(x)| over
//! those points — a fixed, reproducible measure, not a moving target.

use voltic::norm;
use std::simd::prelude::*;
use std::time::Instant;

/// (x, Φ(x)) — Φ correct to f64 precision (offline arbitrary-precision eval).
#[rustfmt::skip]
pub const REF: &[(f64, f64)] = &[
    (-8.0, 6.220960574271782e-16),
    (-7.0, 1.279812543885835e-12),
    (-6.0, 9.865876450376946e-10),
    (-5.0, 2.866515719235352e-7),
    (-4.0, 3.167124183311992e-5),
    (-3.5, 2.326290790355250e-4),
    (-3.0, 1.349898031630095e-3),
    (-2.5, 6.209665325776132e-3),
    (-2.326347874040841, 1.0e-2),
    (-2.0, 2.275013194817921e-2),
    (-1.96, 2.499789514822046e-2),
    (-1.75, 4.005915686381709e-2),
    (-1.5, 6.680720126885807e-2),
    (-1.281551565544601, 1.0e-1),
    (-1.0, 1.586552539314570e-1),
    (-0.75, 2.266273523768682e-1),
    (-0.5, 3.085375387259869e-1),
    (-0.25, 4.012936743170763e-1),
    (-0.1, 4.601721627229710e-1),
    (-0.01, 4.960106436853684e-1),
    (0.0, 5.0e-1),
    (0.01, 5.039893563146316e-1),
    (0.1, 5.398278372770290e-1),
    (0.25, 5.987063256829237e-1),
    (0.5, 6.914624612740131e-1),
    (0.75, 7.733726476231317e-1),
    (1.0, 8.413447460685429e-1),
    (1.281551565544601, 9.0e-1),
    (1.5, 9.331927987311419e-1),
    (1.644853626951472, 9.5e-1),
    (1.75, 9.599408431361829e-1),
    (1.959963984540054, 9.75e-1),
    (1.96, 9.750021048517795e-1),
    (2.0, 9.772498680518208e-1),
    (2.326347874040841, 9.9e-1),
    (2.5, 9.937903346742239e-1),
    (3.0, 9.986501019683699e-1),
    (3.5, 9.997673709209645e-1),
    (4.0, 9.999683287581669e-1),
    (5.0, 9.999997133484281e-1),
    (6.0, 9.999999990134123e-1),
    (7.0, 9.999999999987202e-1),
    (8.0, 9.999999999999994e-1),
];

type Kernel = fn(Simd<f64, 8>) -> Simd<f64, 8>;

const KERNELS: &[(&str, Kernel)] = &[
    ("Abramowitz-Stegun 26.2.17", norm::phi_as::<8>),
    ("Hart 5666", norm::phi_hart::<8>),
    ("West 2009", norm::phi_west::<8>),
    ("Cody 1969", norm::phi_cody::<8>),
];

/// (max absolute error, max *relative* error) of `f` over the reference table.
/// Relative error is the meaningful metric for Φ — it spans ~16 orders of
/// magnitude from the deep left tail to ≈1, and a relative error in the wing
/// (where d₂ lands for a deep-OTM option) is what propagates into the IV. The
/// plot uses the relative figure.
fn errors(f: Kernel) -> (f64, f64) {
    let mut worst_abs = 0.0_f64;
    let mut worst_rel = 0.0_f64;
    for &(x, truth) in REF {
        let got = f(Simd::splat(x))[0];
        let abs = (got - truth).abs();
        worst_abs = worst_abs.max(abs);
        if truth > 0.0 {
            worst_rel = worst_rel.max(abs / truth);
        }
    }
    (worst_abs, worst_rel)
}

/// ns per Φ(x) call: time a tight loop over a fixed batch of x's (8 lanes at a
/// time), after a discarded warmup, median of `repeats` passes; divide by the
/// element count. The x's are the REF points, tiled — a fixed, reproducible
/// input. `repeats` passes of `iters` batches each.
fn ns_per_call(f: Kernel, repeats: usize, iters: usize) -> f64 {
    // Build a batch of 8 lanes of varied x.
    let xs: Vec<Simd<f64, 8>> = (0..256)
        .map(|j| {
            let mut a = [0.0; 8];
            for (l, slot) in a.iter_mut().enumerate() {
                let (x, _) = REF[(j * 8 + l) % REF.len()];
                *slot = x;
            }
            Simd::from_array(a)
        })
        .collect();
    let n_elems = xs.len() * 8 * iters;
    // warmup
    {
        let mut acc = Simd::<f64, 8>::splat(0.0);
        for _ in 0..iters {
            for &v in &xs {
                acc += f(v);
            }
        }
        std::hint::black_box(acc);
    }
    let mut samples = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let t0 = Instant::now();
        let mut acc = Simd::<f64, 8>::splat(0.0);
        for _ in 0..iters {
            for &v in &xs {
                acc += f(std::hint::black_box(v));
            }
        }
        let dt = t0.elapsed();
        std::hint::black_box(acc);
        samples.push(dt.as_secs_f64());
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[repeats / 2] / n_elems as f64 * 1e9
}

/// Write the per-kernel CSV.
pub fn write_csv(path: &str) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(
        f,
        "kernel,max_abs_error,max_rel_error,ns_per_call,options_per_sec"
    )?;
    for &(name, kf) in KERNELS {
        let (abs, rel) = errors(kf);
        let ns = ns_per_call(kf, 9, 200);
        writeln!(f, "{name},{abs:.3e},{rel:.3e},{ns:.3},{:.3e}", 1e9 / ns)?;
    }
    Ok(())
}

/// Print the per-kernel table to stdout (also returns the rows).
pub fn report() -> Vec<(&'static str, f64, f64, f64)> {
    println!("\nCumulative-normal kernel frontier (single core):");
    println!(
        "{:<28} {:>12} {:>12} {:>12} {:>14}",
        "kernel", "max abs err", "max rel err", "ns/call", "calls/sec"
    );
    let mut rows = Vec::new();
    for &(name, kf) in KERNELS {
        let (abs, rel) = errors(kf);
        let ns = ns_per_call(kf, 9, 200);
        println!(
            "{name:<28} {abs:>12.3e} {rel:>12.3e} {ns:>12.3} {:>14.3e}",
            1e9 / ns
        );
        rows.push((name, abs, rel, ns));
    }
    rows
}
