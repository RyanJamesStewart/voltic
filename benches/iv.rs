//! Criterion benchmark for the headline ns/option number.
//!
//! Run with `taskset -c 0 cargo bench` after
//! `cpupower frequency-set --governor performance`. Criterion discards warmup
//! samples automatically. The dataset is the same persisted synthetic set the
//! `bench` binary uses (`bench/data.rs`, seed `data::SEED`).

#[path = "../bench/data.rs"]
mod data;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn iv_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("implied_vol");
    for &n in &[10_000usize, 100_000, 1_000_000] {
        let ds = data::generate(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("voltic_vectorized", n), &ds, |b, ds| {
            b.iter(|| {
                let r = voltic::implied_vol(
                    black_box(&ds.spot),
                    black_box(&ds.strike),
                    black_box(&ds.tte),
                    black_box(&ds.rate),
                    black_box(&ds.price),
                    black_box(&ds.kind),
                );
                black_box(r);
            });
        });
        group.bench_with_input(BenchmarkId::new("explicit_schadner", n), &ds, |b, ds| {
            b.iter(|| {
                let r = voltic::implied_vol_explicit(
                    black_box(&ds.spot),
                    black_box(&ds.strike),
                    black_box(&ds.tte),
                    black_box(&ds.rate),
                    black_box(&ds.price),
                    black_box(&ds.kind),
                );
                black_box(r);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, iv_bench);
criterion_main!(benches);
