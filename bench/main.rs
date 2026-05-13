//! `cargo run --release --bin bench` — generates the synthetic dataset, runs
//! the Rust implementations (jackal vectorized, jackal scalar entry point, and
//! the naive scalar-Newton baseline), and prints the benchmark table:
//! throughput (ns/option, options/sec), accuracy vs the σ that produced each
//! price (overall + stratified by moneyness band), and LOC per implementation.
//!
//! Methodology: a warmup pass is run and discarded, then the timed region is
//! the median of `REPEATS` full passes over the whole dataset, single-threaded.
//! The criterion benchmark (`cargo bench`, `benches/iv.rs`) is the rigorous
//! source of the headline ns/option; this binary is the reproducibility
//! convenience that also computes the accuracy columns and emits the CSV the
//! Python harness consumes.
//!
//! `cargo run --release --bin bench -- --csv data.csv` also writes the dataset
//! to `data.csv` for the Python comparison harness (`bench/python/`).
//! `--phi-csv phi.csv` additionally writes the cumulative-normal-kernel
//! frontier table for `scripts/plot_phi.py`.
#![feature(portable_simd)]

#[path = "data.rs"]
mod data;
#[path = "naive.rs"]
mod naive;
#[path = "phi.rs"]
mod phi;

use jackal::OptionKind;
use std::time::Instant;

const REPEATS: usize = 7;

fn kinds_as_bool(k: &[OptionKind]) -> Vec<bool> {
    k.iter().map(|x| matches!(x, OptionKind::Call)).collect()
}

/// Median of `REPEATS` timed passes (after one discarded warmup pass) of `f`.
/// Returns nanoseconds per option.
fn time_ns_per_option<F: FnMut() -> usize>(n: usize, mut f: F) -> f64 {
    // warmup
    let w = f();
    std::hint::black_box(w);
    let mut samples = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        let t0 = Instant::now();
        let r = f();
        let dt = t0.elapsed();
        std::hint::black_box(r);
        samples.push(dt.as_secs_f64());
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = samples[REPEATS / 2];
    median / n as f64 * 1e9
}

struct Acc {
    max_abs: f64,
    n_nan: usize,
    n: usize,
}

fn accuracy(solved: &[f64], truth: &[f64], ds: &data::Dataset, band: Option<data::Band>) -> Acc {
    let mut max_abs = 0.0_f64;
    let mut n_nan = 0;
    let mut n = 0;
    for i in 0..solved.len() {
        if let Some(b) = band {
            if ds.band(i) != b {
                continue;
            }
        }
        n += 1;
        if solved[i].is_nan() {
            n_nan += 1;
            continue;
        }
        max_abs = max_abs.max((solved[i] - truth[i]).abs());
    }
    Acc { max_abs, n_nan, n }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args
        .iter()
        .position(|a| a == "--n")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(data::DEFAULT_N);
    let csv_path = args
        .iter()
        .position(|a| a == "--csv")
        .and_then(|i| args.get(i + 1))
        .cloned();
    let phi_csv = args
        .iter()
        .position(|a| a == "--phi-csv")
        .and_then(|i| args.get(i + 1))
        .cloned();

    eprintln!(
        "generating {n} synthetic options (seed {:#018x}) …",
        data::SEED
    );
    let ds = data::generate(n);
    if let Some(p) = &csv_path {
        data::write_csv(&ds, p).expect("write csv");
        eprintln!("wrote dataset to {p}");
    }
    let kb = kinds_as_bool(&ds.kind);

    // --- jackal, vectorized (the headline) ---------------------------------
    let jackal_vec =
        jackal::implied_vol(&ds.spot, &ds.strike, &ds.tte, &ds.rate, &ds.price, &ds.kind);
    let jackal_vec_ns = time_ns_per_option(n, || {
        let r = jackal::implied_vol(&ds.spot, &ds.strike, &ds.tte, &ds.rate, &ds.price, &ds.kind);
        r.len()
    });

    // --- jackal via the single-option entry point, in a loop ---------------
    // (the "scalar" column for jackal — it's the same SIMD code, just one
    //  option at a time, so 7 of 8 lanes are wasted; the gap to `jackal_vec`
    //  is the per-call overhead, not a different algorithm. Reported as such.)
    let jackal_scalar_ns = time_ns_per_option(n, || {
        let mut acc = 0usize;
        for i in 0..n {
            let v = jackal::implied_vol_one(
                ds.spot[i],
                ds.strike[i],
                ds.tte[i],
                ds.rate[i],
                ds.price[i],
                ds.kind[i],
            );
            acc += (!v.is_nan()) as usize;
        }
        acc
    });

    // --- naive pure-Rust scalar Newton -------------------------------------
    let naive_solved = naive::implied_vol(&ds.spot, &ds.strike, &ds.tte, &ds.rate, &ds.price, &kb);
    let naive_ns = time_ns_per_option(n, || {
        naive::implied_vol(&ds.spot, &ds.strike, &ds.tte, &ds.rate, &ds.price, &kb).len()
    });

    // --- report ------------------------------------------------------------
    let ops = |ns: f64| 1e9 / ns;
    println!(
        "\n=== jackal benchmark — {n} synthetic options, seed {:#018x} ===\n",
        data::SEED
    );
    println!("Rust implementations (single-threaded, median of {REPEATS} passes after warmup):");
    println!("{:<28} {:>14} {:>18}", "impl", "ns/option", "options/sec");
    println!(
        "{:<28} {:>14.1} {:>18.3e}",
        "jackal (vectorized, f64x8)",
        jackal_vec_ns,
        ops(jackal_vec_ns)
    );
    println!(
        "{:<28} {:>14.1} {:>18.3e}",
        "jackal (scalar entry pt)",
        jackal_scalar_ns,
        ops(jackal_scalar_ns)
    );
    println!(
        "{:<28} {:>14.1} {:>18.3e}",
        "naive Rust scalar Newton",
        naive_ns,
        ops(naive_ns)
    );

    println!("\nAccuracy — max |solved σ − σ_true| (the σ that produced each price):");
    let overall_j = accuracy(&jackal_vec, &ds.sigma_true, &ds, None);
    let overall_n = accuracy(&naive_solved, &ds.sigma_true, &ds, None);
    println!(
        "  jackal  overall: max abs err {:.3e}   ({} of {} returned NaN)",
        overall_j.max_abs, overall_j.n_nan, overall_j.n
    );
    println!(
        "  naive   overall: max abs err {:.3e}   ({} of {} returned NaN)",
        overall_n.max_abs, overall_n.n_nan, overall_n.n
    );
    for (label, band) in [
        ("deep OTM (S/K < 0.7)", data::Band::DeepOtm),
        ("near ATM (0.95–1.05)", data::Band::NearAtm),
        ("deep ITM (S/K > 1.3)", data::Band::DeepItm),
    ] {
        let a = accuracy(&jackal_vec, &ds.sigma_true, &ds, Some(band));
        println!(
            "    jackal  {label:<22}: max abs err {:.3e}   ({} of {} NaN)",
            a.max_abs, a.n_nan, a.n
        );
    }

    // --- cumulative-normal kernel frontier ---------------------------------
    let _ = phi::report();
    if let Some(p) = &phi_csv {
        phi::write_csv(p).expect("write phi csv");
        eprintln!("wrote Φ-kernel frontier CSV to {p} (feed to scripts/plot_phi.py)");
    }

    println!("\nLOC (cloc-style, src + this harness):");
    println!("  jackal core (src/lib.rs + src/norm.rs):  see `tokei src/`");
    println!("  naive baseline (bench/naive.rs):          see `tokei bench/naive.rs`");

    // Sanity gate: nothing in jackal should be more accurate than the
    // reference round-trip allows; and nothing absurd in throughput.
    if overall_j.max_abs > 0.0 && overall_j.max_abs < 1e-12 {
        eprintln!("note: jackal max abs vol error {:.3e} is below the ~1e-10 conditioning floor — fine for a round-trip dataset (price was computed from σ_true to full f64), but accuracy *vs py_vollib* must still be reported and will be looser.", overall_j.max_abs);
    }
}
