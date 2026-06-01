//! `bench/adversarial_dump.rs` — dump voltic typed results for the adversarial
//! grid as JSON, consumed by `bench/python/adversarial_compare.py`.

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use voltic::{implied_vol_typed_batch, ImpliedVolStatus, OptionKind};

fn parse_adv(
    path: &str,
) -> (
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<OptionKind>,
    Vec<String>,
    Vec<String>,
) {
    let f = File::open(path).expect("open csv");
    let r = BufReader::new(f);
    let mut s = Vec::new();
    let mut k = Vec::new();
    let mut t = Vec::new();
    let mut rt = Vec::new();
    let mut p = Vec::new();
    let mut kind = Vec::new();
    let mut regime = Vec::new();
    let mut expected = Vec::new();
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
        kind.push(if cols[6].trim() == "c" {
            OptionKind::Call
        } else {
            OptionKind::Put
        });
        regime.push(cols[7].trim().to_string());
        expected.push(cols[8].trim().to_string());
    }
    (s, k, t, rt, p, kind, regime, expected)
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

fn status_computed(s: ImpliedVolStatus) -> Option<f64> {
    match s {
        ImpliedVolStatus::BelowVolMin { computed } => Some(computed),
        ImpliedVolStatus::AboveVolMax { computed } => Some(computed),
        _ => None,
    }
}

fn main() {
    let csv_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "adversarial_data.csv".to_string());
    let json_out = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "/tmp/voltic_typed_adversarial.json".to_string());

    let (s, k, t, r, p, kind, _regime, _expected) = parse_adv(&csv_path);
    let n = s.len();
    let typed = implied_vol_typed_batch(&s, &k, &t, &r, &p, &kind);

    let mut out = File::create(&json_out).expect("create json");
    write!(out, "[").unwrap();
    for (i, res) in typed.iter().enumerate() {
        if i > 0 {
            write!(out, ",\n").unwrap();
        }
        let value_str = if res.value.is_finite() {
            format!("{}", res.value)
        } else {
            "null".to_string()
        };
        let computed_str = if let Some(c) = status_computed(res.status) {
            format!("{}", c)
        } else {
            "null".to_string()
        };
        write!(
            out,
            "{{\"i\":{},\"status\":\"{}\",\"value\":{},\"computed\":{}}}",
            i,
            status_label(res.status),
            value_str,
            computed_str
        )
        .unwrap();
    }
    write!(out, "]\n").unwrap();
    eprintln!("wrote {} rows to {}", n, json_out);
}
