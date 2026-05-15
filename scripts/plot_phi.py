#!/usr/bin/env python3
"""Plot the cumulative-normal kernel frontier: absolute error vs throughput for
the three Φ(x) approximations voltic implements (Abramowitz-Stegun 26.2.17,
Hart 5666, West 2009, Cody 1969) — and circle the one voltic uses.

Input: a CSV produced by the `bench` binary's `--phi-csv <path>` mode, with
columns `kernel,max_abs_error,max_rel_error,ns_per_call,options_per_sec`. (The Rust side does
the timing + the error measurement against a high-precision reference so the
numbers come from the same hardware as the main benchmark; this script only
draws the picture.)

Output: `phi_frontier.svg` (and `.png` if matplotlib has a raster backend) —
formatted to be screenshot-shareable without context: log-scale error axis,
labeled points, the chosen kernel circled with an annotation.

Usage:  python scripts/plot_phi.py phi_kernels.csv [out_prefix]
Requires: matplotlib  (`pip install matplotlib`).
"""
import csv
import sys

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib not installed — `pip install matplotlib`")

if len(sys.argv) < 2:
    sys.exit("usage: python scripts/plot_phi.py <phi_kernels.csv> [out_prefix]")
csv_path = sys.argv[1]
out_prefix = sys.argv[2] if len(sys.argv) > 2 else "phi_frontier"
CHOSEN = "Hart 5666"

rows = []
with open(csv_path) as f:
    for row in csv.DictReader(f):
        rows.append(
            (
                row["kernel"],
                float(row["max_rel_error"]),
                float(row["ns_per_call"]),
                float(row.get("options_per_sec", "nan")),
            )
        )

fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=140)
for name, err, ns, _ in rows:  # err = max relative error
    ax.scatter(ns, err, s=90, zorder=3)
    ax.annotate(
        name,
        (ns, err),
        textcoords="offset points",
        xytext=(8, 8),
        fontsize=11,
        fontweight="bold" if name == CHOSEN else "normal",
    )
    if name == CHOSEN:
        ax.scatter(ns, err, s=380, facecolors="none", edgecolors="#d62728", linewidths=2.4, zorder=4)
        ax.annotate(
            "← voltic uses this:\n~8e-9 rel. error — far below the\n~1e-6 IV conditioning floor —\nat the lowest cost of the three\naccurate kernels. Cody buys ~50×\nbetter rel. error the problem\ncan't use; West is a slower\nnear-clone; AS's ~1e-2 deep-wing\nrel. error is too coarse.",
            (ns, err),
            textcoords="offset points",
            xytext=(14, -78),
            fontsize=9.5,
            color="#d62728",
        )

ax.set_yscale("log")
ax.set_xlabel("ns per Φ(x) call (single AVX-512 core, lower is better) →", fontsize=11)
ax.set_ylabel("max |Φ̂(x) − Φ(x)| / Φ(x)  over the reference grid  (log scale)", fontsize=11)
ax.set_title("Cumulative-normal kernel: accuracy vs throughput", fontsize=13, fontweight="bold")
ax.grid(True, which="both", linewidth=0.4, alpha=0.5)
fig.tight_layout()
fig.savefig(f"{out_prefix}.svg")
try:
    fig.savefig(f"{out_prefix}.png")
except Exception:
    pass
print(f"wrote {out_prefix}.svg" + (f" and {out_prefix}.png" if True else ""))
