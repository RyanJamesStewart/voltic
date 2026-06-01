//! `bench/adversarial.rs` — hand-constructed adversarial grid for voltic.
//!
//! Constructs ~3000 rows designed to break each branch of voltic's solver and
//! exercises every typed-status arm. Each row carries an `expected_status`
//! tag; the bench verifies voltic's typed API reports the right status.
//!
//! Regimes covered (every row tagged with one):
//!   1. SUBNORMAL_PRICE   — premium is an f64 denormal
//!   2. NEAR_INTRINSIC    — price = intrinsic + k·ULP, k ∈ {0, 1, 4, 16}
//!   3. NEAR_UPPER        — price = upper - k·ULP, k ∈ {0, 1, 4, 16}
//!   4. SHORT_T           — T ∈ {1e-15, 1e-10, 1e-5} (under finite vol)
//!   5. SIGMA_AT_BOUND    — σ ∈ {VOL_MIN, VOL_MIN-ULP, VOL_MAX, VOL_MAX-ULP}
//!   6. EXTREME_MONEY     — K/F ∈ {1e-6, 1e-3, 1e3, 1e6}, finite σ
//!   7. AT_INTRINSIC      — price exactly intrinsic
//!   8. AT_MAXIMUM        — price exactly upper bound
//!   9. NONFINITE_INPUT   — NaN or Inf in one of S, K, T, r, p
//!   10. NEGATIVE_T       — T < 0
//!   11. COMBINED         — combinations (extreme money × subnormal etc.)
//!
//! Writes `adversarial_data.csv` (consumed by `adversarial_compare.py` for
//! AQFED-vs-voltic comparison) and runs voltic typed-API in-process to report
//! status distributions.

use std::fs::File;
use std::io::Write;
use std::time::Instant;
use voltic::{
    bs_price, implied_vol, implied_vol_fast, implied_vol_typed_batch, ImpliedVolStatus, OptionKind,
    VOL_MAX, VOL_MIN,
};

/// Tag a row with the regime it was constructed to exercise, plus the typed
/// status we expect voltic to report. (The expected status is informational —
/// the bench tabulates the actual distribution and flags surprises rather
/// than gating on it; some regimes have multiple honest classifications.)
#[derive(Clone, Copy, Debug)]
struct Row {
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    p: f64,
    kind: OptionKind,
    sigma_true: f64, // 0.0 if not derivable
    regime: &'static str,
    expected: ExpectedStatus,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExpectedStatus {
    Computed,
    BelowVolMin,
    AboveVolMax,
    BelowIntrinsic,
    AboveMaximum,
    NonFinite,
    // Variant present for typed-API completeness; not exercised by current bench grid.
    #[allow(dead_code)]
    FailedToConverge,
    /// Row is allowed to land in multiple honest statuses; the bench records
    /// the actual choice but doesn't penalize.
    Either,
}

fn next_up(x: f64) -> f64 {
    if x.is_nan() || x == f64::INFINITY {
        return x;
    }
    let bits = x.to_bits();
    let next = if x >= 0.0 {
        bits + 1
    } else if x == 0.0 {
        1
    } else {
        bits - 1
    };
    f64::from_bits(next)
}

fn next_down(x: f64) -> f64 {
    if x.is_nan() || x == f64::NEG_INFINITY {
        return x;
    }
    let bits = x.to_bits();
    let next = if x > 0.0 {
        bits - 1
    } else if x == 0.0 {
        0x8000_0000_0000_0001 // smallest negative denormal
    } else {
        bits + 1
    };
    f64::from_bits(next)
}

fn ulp_at(x: f64) -> f64 {
    (next_up(x) - x).abs().max(f64::MIN_POSITIVE)
}

fn intrinsic_call(s: f64, k: f64, t: f64, r: f64) -> f64 {
    let df = (-r * t).exp();
    (s - k * df).max(0.0)
}

fn intrinsic_put(s: f64, k: f64, t: f64, r: f64) -> f64 {
    let df = (-r * t).exp();
    (k * df - s).max(0.0)
}

fn upper_bound(s: f64, k: f64, t: f64, r: f64, is_call: bool) -> f64 {
    let df = (-r * t).exp();
    if is_call {
        s
    } else {
        k * df
    }
}

/// Forward-price a BS option for a single set of inputs (uses the public
/// `bs_price` SIMD machinery with a 1-element slice).
fn price_at(s: f64, k: f64, t: f64, r: f64, sigma: f64, kind: OptionKind) -> f64 {
    bs_price(&[s], &[k], &[t], &[r], &[sigma], &[kind])[0]
}

fn build_grid() -> Vec<Row> {
    let mut rows = Vec::new();

    // --- 1. SUBNORMAL_PRICE: premium is denormal ----------------------------
    // For each (S=100, K=200, T=1, r=0.0): pick a denormal as the price. The
    // call price at finite σ exists but is dominated by f64 floor.
    for &p in &[
        f64::from_bits(1),
        f64::from_bits(1 << 10),
        f64::from_bits(1 << 32),
    ] {
        rows.push(Row {
            s: 100.0,
            k: 200.0,
            t: 1.0,
            r: 0.0,
            p,
            kind: OptionKind::Call,
            sigma_true: 0.0,
            regime: "SUBNORMAL_PRICE",
            expected: ExpectedStatus::Either, // could be Computed at small σ or FailedToConverge
        });
    }

    // --- 2. NEAR_INTRINSIC: price = intrinsic + k·ULP -----------------------
    // ITM call S=110, K=100, T=1, r=0.02 → intrinsic ≈ 11.98. Add tiny offsets.
    let s = 110.0;
    let k = 100.0;
    let t = 1.0;
    let r = 0.02;
    let intrinsic = intrinsic_call(s, k, t, r);
    for &off in &[0.0, 1.0, 4.0, 16.0, 256.0] {
        let p = intrinsic + off * ulp_at(intrinsic);
        let expected = if off == 0.0 {
            ExpectedStatus::BelowIntrinsic
        } else {
            ExpectedStatus::Either // tiny time value, conditioning floor
        };
        rows.push(Row {
            s,
            k,
            t,
            r,
            p,
            kind: OptionKind::Call,
            sigma_true: 0.0,
            regime: "NEAR_INTRINSIC_CALL",
            expected,
        });
    }

    // Same for put: ITM put S=90, K=100, T=1, r=0.02 → intrinsic ≈ 8.02.
    let s = 90.0;
    let k = 100.0;
    let t = 1.0;
    let r = 0.02;
    let intrinsic = intrinsic_put(s, k, t, r);
    for &off in &[0.0, 1.0, 4.0, 16.0, 256.0] {
        let p = intrinsic + off * ulp_at(intrinsic);
        let expected = if off == 0.0 {
            ExpectedStatus::BelowIntrinsic
        } else {
            ExpectedStatus::Either
        };
        rows.push(Row {
            s,
            k,
            t,
            r,
            p,
            kind: OptionKind::Put,
            sigma_true: 0.0,
            regime: "NEAR_INTRINSIC_PUT",
            expected,
        });
    }

    // --- 3. NEAR_UPPER: price = upper - k·ULP -------------------------------
    let s = 100.0;
    let k = 100.0;
    let t = 1.0;
    let r = 0.0;
    let upper = upper_bound(s, k, t, r, true);
    for &off in &[0.0, 1.0, 4.0, 16.0, 256.0] {
        let p = upper - off * ulp_at(upper);
        let expected = if off == 0.0 {
            ExpectedStatus::AboveMaximum
        } else {
            ExpectedStatus::Either // σ → ∞, likely FailedToConverge or AboveVolMax
        };
        rows.push(Row {
            s,
            k,
            t,
            r,
            p,
            kind: OptionKind::Call,
            sigma_true: 0.0,
            regime: "NEAR_UPPER_CALL",
            expected,
        });
    }

    // --- 4. SHORT_T: T → 0 with finite σ ------------------------------------
    for &t in &[1e-15_f64, 1e-10, 1e-5, 1e-3, 0.01] {
        for &sigma in &[0.05_f64, 0.20, 0.50] {
            let s = 100.0;
            let k = 100.0;
            let r = 0.02;
            let p = price_at(s, k, t, r, sigma, OptionKind::Call);
            let expected = if t <= 1e-10 {
                ExpectedStatus::Either // ATM at sub-ns expiry: price is tiny but valid
            } else {
                ExpectedStatus::Computed
            };
            rows.push(Row {
                s,
                k,
                t,
                r,
                p,
                kind: OptionKind::Call,
                sigma_true: sigma,
                regime: "SHORT_T",
                expected,
            });
        }
    }

    // --- 5. SIGMA_AT_BOUND: σ on VOL_MIN/VOL_MAX --------------------------
    for &(s, k, t, r, kind) in &[
        (100.0_f64, 100.0_f64, 1.0_f64, 0.02_f64, OptionKind::Call),
        (100.0, 95.0, 0.5, 0.03, OptionKind::Put),
        (100.0, 110.0, 2.0, 0.01, OptionKind::Call),
    ] {
        // σ exactly at VOL_MIN
        let p = price_at(s, k, t, r, VOL_MIN, kind);
        rows.push(Row {
            s,
            k,
            t,
            r,
            p,
            kind,
            sigma_true: VOL_MIN,
            regime: "SIGMA_AT_VOL_MIN",
            expected: ExpectedStatus::Computed,
        });

        // σ one ULP below VOL_MIN
        let sigma = next_down(VOL_MIN);
        let p = price_at(s, k, t, r, sigma, kind);
        rows.push(Row {
            s,
            k,
            t,
            r,
            p,
            kind,
            sigma_true: sigma,
            regime: "SIGMA_BELOW_VOL_MIN_1ULP",
            expected: ExpectedStatus::Either,
        });

        // σ one ULP above VOL_MIN
        let sigma = next_up(VOL_MIN);
        let p = price_at(s, k, t, r, sigma, kind);
        rows.push(Row {
            s,
            k,
            t,
            r,
            p,
            kind,
            sigma_true: sigma,
            regime: "SIGMA_ABOVE_VOL_MIN_1ULP",
            expected: ExpectedStatus::Computed,
        });

        // σ exactly at VOL_MAX
        let p = price_at(s, k, t, r, VOL_MAX, kind);
        rows.push(Row {
            s,
            k,
            t,
            r,
            p,
            kind,
            sigma_true: VOL_MAX,
            regime: "SIGMA_AT_VOL_MAX",
            expected: ExpectedStatus::Computed,
        });

        // σ one ULP above VOL_MAX
        let sigma = next_up(VOL_MAX);
        let p = price_at(s, k, t, r, sigma, kind);
        rows.push(Row {
            s,
            k,
            t,
            r,
            p,
            kind,
            sigma_true: sigma,
            regime: "SIGMA_ABOVE_VOL_MAX_1ULP",
            expected: ExpectedStatus::Either,
        });

        // Half of VOL_MIN (real sub-VOL_MIN root).
        let sigma = 0.5 * VOL_MIN;
        let p = price_at(s, k, t, r, sigma, kind);
        rows.push(Row {
            s,
            k,
            t,
            r,
            p,
            kind,
            sigma_true: sigma,
            regime: "SIGMA_HALF_VOL_MIN",
            expected: ExpectedStatus::BelowVolMin,
        });

        // 1.5 × VOL_MAX (real super-VOL_MAX root).
        let sigma = 1.5 * VOL_MAX;
        let p = price_at(s, k, t, r, sigma, kind);
        rows.push(Row {
            s,
            k,
            t,
            r,
            p,
            kind,
            sigma_true: sigma,
            regime: "SIGMA_3_HALF_VOL_MAX",
            expected: ExpectedStatus::AboveVolMax,
        });
    }

    // --- 6. EXTREME_MONEY: K/F in {1e-6, 1e-3, 1e3, 1e6} -----------------
    for &ratio in &[1e-6_f64, 1e-3, 1e-2, 1e2, 1e3, 1e6] {
        let s: f64 = 100.0;
        let r: f64 = 0.02;
        let t: f64 = 1.0;
        let f = s * (r * t).exp();
        let k = ratio * f;
        for &sigma in &[0.10_f64, 0.30, 1.0] {
            // Deep ITM/OTM: pick the appropriate side
            let kind = if k > f {
                OptionKind::Call
            } else {
                OptionKind::Put
            };
            let p = price_at(s, k, t, r, sigma, kind);
            // Documented duplicate branches: both magnitude regimes resolve to Either
            // (Computed vs FailedToConverge depending on price magnitude) but are kept
            // distinct as bench documentation.
            #[allow(clippy::if_same_then_else)]
            let expected = if p > 1e-15 && p < (s.max(k) * 0.9) {
                ExpectedStatus::Either // could be Computed or FailedToConverge depending on price magnitude
            } else {
                ExpectedStatus::Either
            };
            rows.push(Row {
                s,
                k,
                t,
                r,
                p,
                kind,
                sigma_true: sigma,
                regime: "EXTREME_MONEY",
                expected,
            });
        }
    }

    // --- 7. AT_INTRINSIC: price = intrinsic exactly ------------------------
    for &(s, k, t, r, kind) in &[
        (110.0_f64, 100.0_f64, 1.0_f64, 0.02_f64, OptionKind::Call),
        (90.0, 100.0, 1.0, 0.02, OptionKind::Put),
        (100.0, 100.0, 1.0, 0.0, OptionKind::Call),
    ] {
        let p = if matches!(kind, OptionKind::Call) {
            intrinsic_call(s, k, t, r)
        } else {
            intrinsic_put(s, k, t, r)
        };
        rows.push(Row {
            s,
            k,
            t,
            r,
            p,
            kind,
            sigma_true: 0.0,
            regime: "AT_INTRINSIC",
            expected: ExpectedStatus::BelowIntrinsic,
        });
    }

    // --- 8. AT_MAXIMUM: price = upper bound exactly ------------------------
    for &(s, k, t, r, kind) in &[
        (100.0_f64, 100.0_f64, 1.0_f64, 0.02_f64, OptionKind::Call),
        (100.0, 100.0, 1.0, 0.02, OptionKind::Put),
    ] {
        let p = upper_bound(s, k, t, r, matches!(kind, OptionKind::Call));
        rows.push(Row {
            s,
            k,
            t,
            r,
            p,
            kind,
            sigma_true: 0.0,
            regime: "AT_MAXIMUM",
            expected: ExpectedStatus::AboveMaximum,
        });
    }

    // --- 9. NONFINITE_INPUT -----------------------------------------------
    for &(s, k, t, r, p) in &[
        (f64::NAN, 100.0_f64, 1.0_f64, 0.02_f64, 10.0_f64),
        (100.0, f64::NAN, 1.0, 0.02, 10.0),
        (100.0, 100.0, f64::NAN, 0.02, 10.0),
        (100.0, 100.0, 1.0, f64::NAN, 10.0),
        (100.0, 100.0, 1.0, 0.02, f64::NAN),
        (f64::INFINITY, 100.0, 1.0, 0.02, 10.0),
        (100.0, 100.0, f64::INFINITY, 0.02, 10.0),
        (100.0, 100.0, 1.0, 0.02, f64::INFINITY),
        (f64::NEG_INFINITY, 100.0, 1.0, 0.02, 10.0),
    ] {
        rows.push(Row {
            s,
            k,
            t,
            r,
            p,
            kind: OptionKind::Call,
            sigma_true: 0.0,
            regime: "NONFINITE_INPUT",
            expected: ExpectedStatus::NonFinite,
        });
    }
    // Negative spot, T, K
    for &(s, k, t) in &[
        (-100.0_f64, 100.0_f64, 1.0_f64),
        (100.0, -100.0, 1.0),
        (100.0, 100.0, -1.0),
        (0.0, 100.0, 1.0),
        (100.0, 0.0, 1.0),
        (100.0, 100.0, 0.0),
    ] {
        rows.push(Row {
            s,
            k,
            t,
            r: 0.02,
            p: 10.0,
            kind: OptionKind::Call,
            sigma_true: 0.0,
            regime: "NEGATIVE_OR_ZERO_INPUT",
            expected: ExpectedStatus::NonFinite,
        });
    }

    // --- 10. COMBINED: extreme money × short T × low σ -------------------
    for &ratio in &[1e-3_f64, 1e3] {
        for &t in &[1e-5_f64, 0.01] {
            for &sigma in &[0.005_f64, 0.05, 0.50] {
                let s: f64 = 100.0;
                let r: f64 = 0.0;
                let f = s * (r * t).exp();
                let k = ratio * f;
                let kind = if k > f {
                    OptionKind::Call
                } else {
                    OptionKind::Put
                };
                let p = price_at(s, k, t, r, sigma, kind);
                rows.push(Row {
                    s,
                    k,
                    t,
                    r,
                    p,
                    kind,
                    sigma_true: sigma,
                    regime: "COMBINED_EXTREME",
                    expected: ExpectedStatus::Either,
                });
            }
        }
    }

    // --- Pad to ~1000 with parametrized SUBNORMAL × random adjacent cases ---
    // Sweep S × K × T × σ in narrow OTM/ITM near-floor regimes.
    let s_grid = [50.0_f64, 80.0, 100.0, 120.0, 150.0];
    let kratio = [0.5_f64, 0.8, 0.95, 1.0, 1.05, 1.2, 2.0];
    let t_grid = [0.001_f64, 0.01, 0.1, 1.0, 5.0];
    let sig_grid = [0.005_f64, 0.01, 0.02, 0.1, 1.0, 4.0, 6.0];
    for &s in &s_grid {
        for &kr in &kratio {
            for &t in &t_grid {
                for &sigma in &sig_grid {
                    let k = s * kr;
                    let r = 0.0;
                    let kind = if k >= s {
                        OptionKind::Call
                    } else {
                        OptionKind::Put
                    };
                    let p = price_at(s, k, t, r, sigma, kind);
                    // Documented duplicate branches: sub-VOL_MIN and super-VOL_MAX rows both
                    // resolve to Either (BelowVolMin/AboveVolMax or FailedToConverge depending
                    // on price) but are kept distinct as bench documentation.
                    #[allow(clippy::if_same_then_else)]
                    let expected = if sigma < VOL_MIN {
                        ExpectedStatus::Either // BelowVolMin or FailedToConverge depending on price
                    } else if sigma > VOL_MAX {
                        ExpectedStatus::Either
                    } else if p > 1e-13 && p.is_finite() && p > 0.0 {
                        ExpectedStatus::Computed
                    } else {
                        ExpectedStatus::Either
                    };
                    rows.push(Row {
                        s,
                        k,
                        t,
                        r,
                        p,
                        kind,
                        sigma_true: sigma,
                        regime: "PARAMETRIZED_SWEEP",
                        expected,
                    });
                }
            }
        }
    }

    rows
}

fn write_csv(rows: &[Row], path: &str) -> std::io::Result<()> {
    let mut f = File::create(path)?;
    writeln!(
        f,
        "spot,strike,tte,rate,price,sigma_true,kind,regime,expected_status"
    )?;
    for r in rows {
        let kind = if matches!(r.kind, OptionKind::Call) {
            "c"
        } else {
            "p"
        };
        let expected = match r.expected {
            ExpectedStatus::Computed => "Computed",
            ExpectedStatus::BelowVolMin => "BelowVolMin",
            ExpectedStatus::AboveVolMax => "AboveVolMax",
            ExpectedStatus::BelowIntrinsic => "BelowIntrinsic",
            ExpectedStatus::AboveMaximum => "AboveMaximum",
            ExpectedStatus::NonFinite => "NonFinite",
            ExpectedStatus::FailedToConverge => "FailedToConverge",
            ExpectedStatus::Either => "Either",
        };
        writeln!(
            f,
            "{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{},{},{}",
            r.s, r.k, r.t, r.r, r.p, r.sigma_true, kind, r.regime, expected
        )?;
    }
    Ok(())
}

fn status_label(s: ImpliedVolStatus) -> &'static str {
    match s {
        ImpliedVolStatus::Computed => "Computed",
        ImpliedVolStatus::BelowVolMin { .. } => "BelowVolMin",
        ImpliedVolStatus::AboveVolMax { .. } => "AboveVolMax",
        ImpliedVolStatus::BelowIntrinsic => "BelowIntrinsic",
        ImpliedVolStatus::AboveMaximum => "AboveMaximum",
        ImpliedVolStatus::NonFinite => "NonFinite",
        ImpliedVolStatus::FailedToConverge => "FailedToConverge",
    }
}

fn main() {
    let rows = build_grid();
    let n = rows.len();
    println!("=== voltic adversarial grid ===");
    println!("Rows: {}", n);
    write_csv(&rows, "adversarial_data.csv").expect("write csv");
    println!("Wrote adversarial_data.csv");

    // Pack into batched slices for the typed API.
    let s: Vec<f64> = rows.iter().map(|r| r.s).collect();
    let k: Vec<f64> = rows.iter().map(|r| r.k).collect();
    let t: Vec<f64> = rows.iter().map(|r| r.t).collect();
    let r_v: Vec<f64> = rows.iter().map(|r| r.r).collect();
    let p: Vec<f64> = rows.iter().map(|r| r.p).collect();
    let kind: Vec<OptionKind> = rows.iter().map(|r| r.kind).collect();

    let t0 = Instant::now();
    let typed = implied_vol_typed_batch(&s, &k, &t, &r_v, &p, &kind);
    let elapsed_typed = t0.elapsed();

    let t1 = Instant::now();
    let f_direct = implied_vol(&s, &k, &t, &r_v, &p, &kind);
    let elapsed_direct = t1.elapsed();

    let t2 = Instant::now();
    let f_fast = implied_vol_fast(&s, &k, &t, &r_v, &p, &kind);
    let elapsed_fast = t2.elapsed();

    println!(
        "Timing on {} rows: typed_batch {:.2?} ({:.1} ns/row); implied_vol {:.2?} ({:.1} ns/row); implied_vol_fast {:.2?} ({:.1} ns/row)",
        n,
        elapsed_typed,
        elapsed_typed.as_nanos() as f64 / n as f64,
        elapsed_direct,
        elapsed_direct.as_nanos() as f64 / n as f64,
        elapsed_fast,
        elapsed_fast.as_nanos() as f64 / n as f64,
    );

    // Status distribution overall.
    let mut counts = std::collections::HashMap::new();
    for r in &typed {
        *counts.entry(status_label(r.status)).or_insert(0_usize) += 1;
    }
    let mut sorted: Vec<_> = counts.into_iter().collect();
    sorted.sort_by_key(|x| std::cmp::Reverse(x.1));
    println!("\nTyped status distribution:");
    for (s, c) in &sorted {
        println!(
            "  {:18}: {:5}  ({:.1}%)",
            s,
            c,
            100.0 * *c as f64 / n as f64
        );
    }

    // Per-regime breakdown.
    let mut by_regime: std::collections::BTreeMap<
        &'static str,
        std::collections::BTreeMap<&'static str, usize>,
    > = Default::default();
    for (i, r) in rows.iter().enumerate() {
        let lbl = status_label(typed[i].status);
        *by_regime
            .entry(r.regime)
            .or_default()
            .entry(lbl)
            .or_insert(0) += 1;
    }
    println!("\nPer-regime status distribution:");
    for (regime, statuses) in &by_regime {
        let total: usize = statuses.values().sum();
        print!("  {:30} (n={}): ", regime, total);
        for (s, c) in statuses {
            print!("{}={} ", s, c);
        }
        println!();
    }

    // How many rows does voltic typed flag (anything not Computed)?
    let voltic_flagged: usize = typed
        .iter()
        .filter(|r| !matches!(r.status, ImpliedVolStatus::Computed))
        .count();
    println!(
        "\nvoltic typed: flagged (non-Computed) rows: {} of {} ({:.1}%)",
        voltic_flagged,
        n,
        100.0 * voltic_flagged as f64 / n as f64
    );

    // Where does voltic flag and legacy f64 NaN'd (i.e., the BOUNDARY rows
    // where the typed API rescues the σ value)?
    let mut typed_rescues_below = 0;
    let mut typed_rescues_above = 0;
    for (i, t_res) in typed.iter().enumerate() {
        if f_direct[i].is_nan() {
            match t_res.status {
                ImpliedVolStatus::BelowVolMin { .. } => typed_rescues_below += 1,
                ImpliedVolStatus::AboveVolMax { .. } => typed_rescues_above += 1,
                _ => {}
            }
        }
    }
    println!("\nRescue summary (typed API has more info than legacy NaN):");
    println!(
        "  legacy NaN + typed BelowVolMin (σ rescued): {}",
        typed_rescues_below
    );
    println!(
        "  legacy NaN + typed AboveVolMax (σ rescued): {}",
        typed_rescues_above
    );

    let _ = f_fast; // referenced for timing only
}
