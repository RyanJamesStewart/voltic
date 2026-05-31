//! Run voltic::implied_vol_rational on the dataset CSV and write results.
//!
//! Used by the py_lets_be_rational cross-validation harness. The
//! Python side reads `--in` to drive py_lbr; we read it and write our own
//! solver's IVs to `--out`. Format: one row per option,
//! `idx,voltic_iv` (idx matching the input CSV row order).

use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

fn main() {
    let args: Vec<String> = env::args().collect();
    let in_path = args
        .iter()
        .position(|a| a == "--in")
        .and_then(|i| args.get(i + 1))
        .expect("--in <path>");
    let out_path = args
        .iter()
        .position(|a| a == "--out")
        .and_then(|i| args.get(i + 1))
        .expect("--out <path>");

    let f = File::open(in_path).expect("open input csv");
    let reader = BufReader::new(f);
    let mut spot: Vec<f64> = Vec::new();
    let mut strike: Vec<f64> = Vec::new();
    let mut tte: Vec<f64> = Vec::new();
    let mut rate: Vec<f64> = Vec::new();
    let mut price: Vec<f64> = Vec::new();
    let mut kind: Vec<voltic::OptionKind> = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        let line = line.expect("read line");
        if i == 0 {
            continue; // header
        }
        let parts: Vec<&str> = line.split(',').collect();
        spot.push(parts[0].parse().expect("parse spot"));
        strike.push(parts[1].parse().expect("parse strike"));
        tte.push(parts[2].parse().expect("parse tte"));
        rate.push(parts[3].parse().expect("parse rate"));
        price.push(parts[4].parse().expect("parse price"));
        // parts[5] is sigma_true; skip
        kind.push(match parts[6] {
            "c" => voltic::OptionKind::Call,
            "p" => voltic::OptionKind::Put,
            other => panic!("unknown kind {other}"),
        });
    }
    eprintln!("read {} options", spot.len());

    let iv = voltic::implied_vol_rational(&spot, &strike, &tte, &rate, &price, &kind);

    let out = File::create(out_path).expect("create output csv");
    let mut writer = BufWriter::new(out);
    writeln!(writer, "idx,voltic_iv").unwrap();
    for (idx, &v) in iv.iter().enumerate() {
        writeln!(writer, "{idx},{:.17e}", v).unwrap();
    }
    eprintln!("wrote IVs to {out_path}");
}
