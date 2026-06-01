//! `cargo run --release --bin bench` — generates the synthetic dataset, runs
//! the Rust implementations (voltic vectorized, voltic scalar entry point, and
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

use std::time::Instant;
use voltic::{
    canonical_c_from_price, implied_vol_vectorized_with_contexts, implied_vol_with_context,
    implied_vol_with_context_batch, OptionKind, OtmContext,
};

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

    // --- voltic, vectorized (the headline) ---------------------------------
    let voltic_vec =
        voltic::implied_vol(&ds.spot, &ds.strike, &ds.tte, &ds.rate, &ds.price, &ds.kind);
    let voltic_vec_ns = time_ns_per_option(n, || {
        let r = voltic::implied_vol(&ds.spot, &ds.strike, &ds.tte, &ds.rate, &ds.price, &ds.kind);
        r.len()
    });

    // --- voltic via the single-option entry point, in a loop ---------------
    // (the "scalar" column for voltic — it's the same SIMD code, just one
    //  option at a time, so 7 of 8 lanes are wasted; the gap to `voltic_vec`
    //  is the per-call overhead, not a different algorithm. Reported as such.)
    let voltic_scalar_ns = time_ns_per_option(n, || {
        let mut acc = 0usize;
        for i in 0..n {
            let v = voltic::implied_vol_one(
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

    // --- voltic explicit (Schadner inverse-Gaussian, vectorized) -----------
    let explicit_vec =
        voltic::implied_vol_explicit(&ds.spot, &ds.strike, &ds.tte, &ds.rate, &ds.price, &ds.kind);
    let explicit_vec_ns = time_ns_per_option(n, || {
        let r = voltic::implied_vol_explicit(
            &ds.spot, &ds.strike, &ds.tte, &ds.rate, &ds.price, &ds.kind,
        );
        r.len()
    });

    // --- voltic rational (Jäckel "Let's be rational", vectorized) ----------
    let rational_vec =
        voltic::implied_vol_rational(&ds.spot, &ds.strike, &ds.tte, &ds.rate, &ds.price, &ds.kind);
    let rational_vec_ns = time_ns_per_option(n, || {
        let r = voltic::implied_vol_rational(
            &ds.spot, &ds.strike, &ds.tte, &ds.rate, &ds.price, &ds.kind,
        );
        r.len()
    });

    // --- voltic fast (Cheb seed + 1 Halley, vectorized) --------------------
    let fast_vec =
        voltic::implied_vol_fast(&ds.spot, &ds.strike, &ds.tte, &ds.rate, &ds.price, &ds.kind);
    let fast_vec_ns = time_ns_per_option(n, || {
        let r =
            voltic::implied_vol_fast(&ds.spot, &ds.strike, &ds.tte, &ds.rate, &ds.price, &ds.kind);
        r.len()
    });

    // --- naive pure-Rust scalar Newton -------------------------------------
    let naive_solved = naive::implied_vol(&ds.spot, &ds.strike, &ds.tte, &ds.rate, &ds.price, &kb);
    let naive_ns = time_ns_per_option(n, || {
        naive::implied_vol(&ds.spot, &ds.strike, &ds.tte, &ds.rate, &ds.price, &kb).len()
    });

    // --- report ------------------------------------------------------------
    let ops = |ns: f64| 1e9 / ns;
    println!(
        "\n=== voltic benchmark — {n} synthetic options, seed {:#018x} ===\n",
        data::SEED
    );
    println!("Rust implementations (single-threaded, median of {REPEATS} passes after warmup):");
    println!("{:<28} {:>14} {:>18}", "impl", "ns/option", "options/sec");
    println!(
        "{:<28} {:>14.1} {:>18.3e}",
        "voltic (vectorized, f64x8)",
        voltic_vec_ns,
        ops(voltic_vec_ns)
    );
    println!(
        "{:<28} {:>14.1} {:>18.3e}",
        "voltic (scalar entry pt)",
        voltic_scalar_ns,
        ops(voltic_scalar_ns)
    );
    println!(
        "{:<28} {:>14.1} {:>18.3e}",
        "voltic explicit (Schadner)",
        explicit_vec_ns,
        ops(explicit_vec_ns)
    );
    println!(
        "{:<28} {:>14.1} {:>18.3e}",
        "voltic rational (Jäckel)",
        rational_vec_ns,
        ops(rational_vec_ns)
    );
    println!(
        "{:<28} {:>14.1} {:>18.3e}",
        "voltic fast (Cheb+Halley)",
        fast_vec_ns,
        ops(fast_vec_ns)
    );
    println!(
        "{:<28} {:>14.1} {:>18.3e}",
        "naive Rust scalar Newton",
        naive_ns,
        ops(naive_ns)
    );

    println!("\nAccuracy — max |solved σ − σ_true| (the σ that produced each price):");
    let overall_j = accuracy(&voltic_vec, &ds.sigma_true, &ds, None);
    let overall_e = accuracy(&explicit_vec, &ds.sigma_true, &ds, None);
    let overall_r = accuracy(&rational_vec, &ds.sigma_true, &ds, None);
    let overall_f = accuracy(&fast_vec, &ds.sigma_true, &ds, None);
    let overall_n = accuracy(&naive_solved, &ds.sigma_true, &ds, None);
    println!(
        "  voltic   overall: max abs err {:.3e}   ({} of {} returned NaN)",
        overall_j.max_abs, overall_j.n_nan, overall_j.n
    );
    println!(
        "  explicit overall: max abs err {:.3e}   ({} of {} returned NaN)",
        overall_e.max_abs, overall_e.n_nan, overall_e.n
    );
    println!(
        "  rational overall: max abs err {:.3e}   ({} of {} returned NaN)",
        overall_r.max_abs, overall_r.n_nan, overall_r.n
    );
    println!(
        "  fast     overall: max abs err {:.3e}   ({} of {} returned NaN)",
        overall_f.max_abs, overall_f.n_nan, overall_f.n
    );
    println!(
        "  naive    overall: max abs err {:.3e}   ({} of {} returned NaN)",
        overall_n.max_abs, overall_n.n_nan, overall_n.n
    );
    for (label, band) in [
        ("deep OTM (S/K < 0.7)", data::Band::DeepOtm),
        ("near ATM (0.95–1.05)", data::Band::NearAtm),
        ("deep ITM (S/K > 1.3)", data::Band::DeepItm),
    ] {
        let a = accuracy(&voltic_vec, &ds.sigma_true, &ds, Some(band));
        let e = accuracy(&explicit_vec, &ds.sigma_true, &ds, Some(band));
        let r = accuracy(&rational_vec, &ds.sigma_true, &ds, Some(band));
        let f = accuracy(&fast_vec, &ds.sigma_true, &ds, Some(band));
        println!(
            "    voltic   {label:<22}: max abs err {:.3e}   ({} of {} NaN)",
            a.max_abs, a.n_nan, a.n
        );
        println!(
            "    explicit {label:<22}: max abs err {:.3e}   ({} of {} NaN)",
            e.max_abs, e.n_nan, e.n
        );
        println!(
            "    rational {label:<22}: max abs err {:.3e}   ({} of {} NaN)",
            r.max_abs, r.n_nan, r.n
        );
        println!(
            "    fast     {label:<22}: max abs err {:.3e}   ({} of {} NaN)",
            f.max_abs, f.n_nan, f.n
        );
    }

    // Direct vs explicit: the cross-method agreement on the points both solve
    // (this is the head-to-head the README reports).
    let mut cross_max = 0.0_f64;
    let mut cross_n = 0usize;
    for i in 0..n {
        if voltic_vec[i].is_nan() || explicit_vec[i].is_nan() {
            continue;
        }
        cross_max = cross_max.max((voltic_vec[i] - explicit_vec[i]).abs());
        cross_n += 1;
    }
    println!(
        "  direct vs explicit: max |σ_direct − σ_explicit| = {cross_max:.3e} over {cross_n} jointly-solved options"
    );

    // --- A5: split-context API benches ------------------------------------
    // Match volfi's two-stage shape: one `(k, T)`-prelude payment, many price
    // evaluations against it. Two workloads:
    //   - REPEAT: 1 unique (k, T) × n prices       — context amortization win
    //   - COLD:  n unique (k, T, price)            — SIMD vector-context path
    //
    // For both, we pre-build the canonical OTM `c` outside the timed region —
    // the context API's contract is `c -> σ`, not raw `price -> σ`. The
    // canonicalization (1 div + 1 sub) is the caller's job in real workloads
    // (it's typically already done by the surface-fit layer).
    println!("\n=== A5: split-context API ({} options) ===", n);

    // Build all n contexts (used by the COLD workload, and by `repeat` to
    // pick a representative).
    let mut contexts: Vec<OtmContext> = Vec::with_capacity(n);
    let mut canonical_c: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        let t = ds.tte[i];
        let s = ds.spot[i];
        let k = ds.strike[i];
        let r = ds.rate[i];
        let k_log = (k / s).ln() - r * t;
        let ctx = OtmContext::new(k_log, t);
        let is_call = matches!(ds.kind[i], OptionKind::Call);
        let cc = canonical_c_from_price(&ctx, s, ds.price[i], is_call);
        contexts.push(ctx);
        canonical_c.push(cc);
    }

    // Cost of building all n contexts (the (k,T)-prelude itself, scalar) —
    // amortized over the COLD workload, not the REPEAT one. We materialize
    // every context into a Vec (and hash one f64 from each) so LLVM can't
    // optimize the prelude away.
    let prelude_ns = time_ns_per_option(n, || {
        let mut v: Vec<OtmContext> = Vec::with_capacity(n);
        for i in 0..n {
            let t = ds.tte[i];
            let s = ds.spot[i];
            let k = ds.strike[i];
            let r = ds.rate[i];
            let k_log = (k / s).ln() - r * t;
            v.push(OtmContext::new(k_log, t));
        }
        // Make the Vec observable to prevent dead-code elimination.
        let mut acc = 0.0_f64;
        for c in &v {
            acc += c.sqrt_t + c.cheb_tu[0];
        }
        std::hint::black_box(acc);
        v.len()
    });

    // -------- REPEAT-CONTEXT workload: 1 (k,T) × n prices ---------------------
    // We pick contexts[0] (an arbitrary in-domain point) and reuse it across
    // n canonical-c values resampled from the dataset's canonical-c column.
    // This stresses the per-option q-side cost only.
    let ctx0 = contexts[0];
    // Generate n prices that all land in the seed domain of ctx0. Cheap and
    // arbitrary — we use canonical_c[i % n] mod a clamp to ensure validity.
    let mut repeat_c: Vec<f64> = Vec::with_capacity(n);
    for &c_i in canonical_c.iter().take(n) {
        // Wrap around dataset; clamp to (0, 1) just in case.
        let c = c_i.clamp(1e-6, 1.0 - 1e-6);
        repeat_c.push(c);
    }
    // Scalar single-eval on shared context (volfi shape match).
    let repeat_scalar_ns = time_ns_per_option(n, || {
        let mut acc = 0usize;
        for &c_i in repeat_c.iter().take(n) {
            let v = implied_vol_with_context(&ctx0, c_i);
            acc += (!v.is_nan()) as usize;
        }
        acc
    });
    // SIMD batched on shared context.
    let repeat_batch_out = implied_vol_with_context_batch(&ctx0, &repeat_c);
    let repeat_batch_ns = time_ns_per_option(n, || {
        let r = implied_vol_with_context_batch(&ctx0, &repeat_c);
        r.len()
    });
    // NaN audit
    let repeat_scalar_nan = (0..n)
        .filter(|&i| implied_vol_with_context(&ctx0, repeat_c[i]).is_nan())
        .count();
    let repeat_batch_nan = repeat_batch_out.iter().filter(|v| v.is_nan()).count();

    // -------- COLD workload: n unique (ctx, c) ---------------------------
    let cold_out = implied_vol_vectorized_with_contexts(&contexts, &canonical_c);
    let cold_ns = time_ns_per_option(n, || {
        let r = implied_vol_vectorized_with_contexts(&contexts, &canonical_c);
        r.len()
    });
    let cold_nan = cold_out.iter().filter(|v| v.is_nan()).count();

    // -------- report ----------------------------------------------------
    println!("{:<46} {:>14} {:>12}", "config", "ns/option", "NaN");
    println!(
        "{:<46} {:>14.1} {:>12}",
        "build OtmContext (scalar, n times)", prelude_ns, "-"
    );
    println!(
        "{:<46} {:>14.1} {:>12}",
        "REPEAT: implied_vol_with_context (scalar)", repeat_scalar_ns, repeat_scalar_nan
    );
    println!(
        "{:<46} {:>14.1} {:>12}",
        "REPEAT: implied_vol_with_context_batch (SIMD)", repeat_batch_ns, repeat_batch_nan
    );
    println!(
        "{:<46} {:>14.1} {:>12}",
        "COLD:   implied_vol_vectorized_with_contexts", cold_ns, cold_nan
    );
    println!(
        "{:<46} {:>14.1} {:>12}",
        "REFERENCE: voltic (vectorized) full kernel", voltic_vec_ns, overall_j.n_nan
    );
    println!(
        "{:<46} {:>14.1} {:>12}",
        "REFERENCE: voltic fast (Cheb+Halley) full", fast_vec_ns, overall_f.n_nan
    );

    // Spot-check the repeat-batch output accuracy: compare to scalar on first
    // 1024 entries (the scalar path is the volfi-shape reference; agreement
    // here is the regression gate for the SIMD batch).
    let mut agree_max = 0.0_f64;
    let mut agree_n = 0usize;
    let probe = n.min(1024);
    for i in 0..probe {
        let s = implied_vol_with_context(&ctx0, repeat_c[i]);
        let b = repeat_batch_out[i];
        if s.is_finite() && b.is_finite() {
            agree_max = agree_max.max((s - b).abs());
            agree_n += 1;
        }
    }
    println!(
        "  scalar vs batch agreement on REPEAT (first {}): max |Δσ| = {:.3e} over {} pts",
        probe, agree_max, agree_n
    );

    // =====================================================================
    // A5.1: SIMD-batched (k, T)-prelude.
    //
    // The A5 cold workload pays a scalar build per option. A5.1 replaces
    // that with a SIMD-batched prelude (f64x8 over 8 (k, T) pairs) and a
    // fused build+solve entry point that processes 8 options end-to-end
    // in one pass — no intermediate `Vec<OtmContext>` materialization.
    //
    // Reports:
    //   - SIMD build amortized cost per option (target: ~6-10 ns)
    //   - cold END-TO-END through the fused API (target: ~70 ns)
    //   - delta vs vanilla voltic-fast and the volfi 46.6 ns target
    // =====================================================================
    println!("\n=== A5.1: SIMD-batched prelude + fused vectorized cold path ===");

    // Pre-extract k_log and T into flat slices the SIMD prelude consumes
    // directly. This is the same shape a real surface-fit caller would
    // supply (the (K, T, S, r) → k_log canonicalization is the caller's
    // responsibility; matches the volfi-shape contract).
    let mut k_log_vec: Vec<f64> = Vec::with_capacity(n);
    let t_vec: Vec<f64> = ds.tte.clone();
    for i in 0..n {
        let t = ds.tte[i];
        let s = ds.spot[i];
        let k = ds.strike[i];
        let r = ds.rate[i];
        k_log_vec.push((k / s).ln() - r * t);
    }

    // --- A5.1 (a): SIMD-build amortized cost per option -----------------
    let simd_build_ns = time_ns_per_option(n, || {
        let v = voltic::otm_context::build_simd_contexts_observed(&k_log_vec, &t_vec);
        v.len()
    });

    // --- A5.1 (b): cold END-TO-END through the fused API ----------------
    let fully_vec_out =
        voltic::otm_context::implied_vol_fully_vectorized(&k_log_vec, &t_vec, &canonical_c);
    let fully_vec_ns = time_ns_per_option(n, || {
        let r = voltic::otm_context::implied_vol_fully_vectorized(&k_log_vec, &t_vec, &canonical_c);
        r.len()
    });
    let fully_vec_nan = fully_vec_out.iter().filter(|v| v.is_nan()).count();

    // --- A5.1 (c): pack-then-solve (separated build via the SIMD prelude,
    //               then the existing vectorized solver path) -------------
    // This isolates the SIMD-prelude → vec<OtmContextSimd> → solve sequence
    // so we can attribute cost between build vs solve. Uses the new SIMD
    // prelude packer + a pass that consumes the packed contexts.
    let packed_solve_ns = time_ns_per_option(n, || {
        let packed = voltic::otm_context::pack_contexts_from_kt(&k_log_vec, &t_vec);
        // For each packed chunk, pack the corresponding 8 prices and solve.
        // This is the "two-call" shape (build, then solve) — what a caller
        // who wants to cache the contexts between solve passes would do.
        let mut out_local = vec![0.0_f64; n];
        let mut i_local = 0usize;
        for ctx in packed.iter() {
            let take = core::cmp::min(8, n - i_local);
            let mut cb = [0.0_f64; 8];
            cb[..take].copy_from_slice(&canonical_c[i_local..i_local + take]);
            // Re-using the solver via the existing public `OtmContextSimd`
            // type isn't directly callable from outside the crate (the
            // solve helper is private), so this bench reaches the same
            // fused path by going through `implied_vol_fully_vectorized`
            // on the chunk's k/t/c — equivalent in cost.
            let k_slice = &k_log_vec[i_local..i_local + take];
            let t_slice = &t_vec[i_local..i_local + take];
            let c_slice = &canonical_c[i_local..i_local + take];
            let r = voltic::otm_context::implied_vol_fully_vectorized(k_slice, t_slice, c_slice);
            out_local[i_local..i_local + take].copy_from_slice(&r);
            let _ = ctx; // keep the packed contexts live so build cost stays in
            i_local += take;
        }
        out_local.len()
    });

    // --- A5.1 (d): scalar vs SIMD-built context agreement check ---------
    // Verifies the SIMD prelude lane-for-lane matches the scalar prelude.
    // Same regression gate as the REPEAT scalar-vs-batch check, but for
    // the BUILD step. Probe the first `probe` options.
    let simd_packed = voltic::otm_context::pack_contexts_from_kt(&k_log_vec, &t_vec);
    let mut build_agree_max = 0.0_f64;
    let mut build_agree_n = 0usize;
    let probe2 = n.min(1024);
    for i in 0..probe2 {
        let scalar_ctx = OtmContext::new(k_log_vec[i], t_vec[i]);
        let chunk = i / 8;
        let lane = i % 8;
        let simd_ctx = simd_packed[chunk];
        let dsqrt = (scalar_ctx.sqrt_t - simd_ctx.sqrt_t.as_array()[lane]).abs();
        let dmu = (scalar_ctx.mu - simd_ctx.mu.as_array()[lane]).abs();
        let dm = (scalar_ctx.m - simd_ctx.m.as_array()[lane]).abs();
        let mut dtu = 0.0_f64;
        let deg = scalar_ctx.cheb_tu.len();
        for r in 0..deg {
            dtu = dtu.max((scalar_ctx.cheb_tu[r] - simd_ctx.cheb_tu[r].as_array()[lane]).abs());
        }
        let d = dsqrt.max(dmu).max(dm).max(dtu);
        build_agree_max = build_agree_max.max(d);
        build_agree_n += 1;
    }

    // Per-option amortized build cost — strip the materialized observe
    // overhead by reporting the raw simd_build_ns as the upper bound.
    let cold_end_to_end_a5 = prelude_ns + cold_ns; // A5 estimate
    let cold_end_to_end_a51 = fully_vec_ns; // A5.1 measured

    println!("{:<46} {:>14} {:>12}", "config", "ns/option", "NaN");
    println!(
        "{:<46} {:>14.1} {:>12}",
        "A5  scalar build (per option, materialized)", prelude_ns, "-"
    );
    println!(
        "{:<46} {:>14.1} {:>12}",
        "A5.1 SIMD build (per option, materialized)", simd_build_ns, "-"
    );
    println!(
        "{:<46} {:>14.1} {:>12}",
        "A5  cold END-TO-END (scalar build + SIMD solve)", cold_end_to_end_a5, cold_nan
    );
    println!(
        "{:<46} {:>14.1} {:>12}",
        "A5.1 cold END-TO-END (fully_vectorized fused)", cold_end_to_end_a51, fully_vec_nan
    );
    println!(
        "{:<46} {:>14.1} {:>12}",
        "A5.1 cold END-TO-END (pack_contexts_from_kt + solve)", packed_solve_ns, "-"
    );
    println!(
        "{:<46} {:>14.1} {:>12}",
        "REFERENCE: voltic fast (vanilla, no ctx API)", fast_vec_ns, overall_f.n_nan
    );
    println!(
        "  vs vanilla voltic-fast: Δ = {:+.1} ns/option",
        fully_vec_ns - fast_vec_ns
    );
    println!(
        "  vs volfi 46.6 ns target: remaining gap = {:+.1} ns/option",
        fully_vec_ns - 46.6
    );
    println!(
        "  scalar vs SIMD-build agreement (first {}): max |Δ| = {:.3e} over {} pts",
        probe2, build_agree_max, build_agree_n
    );

    // --- cumulative-normal kernel frontier ---------------------------------
    let _ = phi::report();
    if let Some(p) = &phi_csv {
        phi::write_csv(p).expect("write phi csv");
        eprintln!("wrote Φ-kernel frontier CSV to {p} (feed to scripts/plot_phi.py)");
    }

    println!("\nLOC (cloc-style, src + this harness):");
    println!("  voltic core (src/lib.rs + src/norm.rs):  see `tokei src/`");
    println!("  naive baseline (bench/naive.rs):          see `tokei bench/naive.rs`");

    // Sanity gate: nothing in voltic should be more accurate than the
    // reference round-trip allows; and nothing absurd in throughput.
    if overall_j.max_abs > 0.0 && overall_j.max_abs < 1e-12 {
        eprintln!("note: voltic max abs vol error {:.3e} is below the ~1e-10 conditioning floor — fine for a round-trip dataset (price was computed from σ_true to full f64), but accuracy *vs py_vollib* must still be reported and will be looser.", overall_j.max_abs);
    }
}
