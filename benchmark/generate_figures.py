#!/usr/bin/env python3
"""Generate the performance bar charts embedded in RESULTS.md.

Reads numbers from BASELINE/AFTER constants below (sourced from the
benches recorded in benchmark/results/). Writes PNGs into
benchmark/figures/.

All numbers are devcontainer measurements (Rocky Linux 9, linux/arm64).
Baseline = pre-#94 (commit 8d17710d, Python 3.9); After = tip of this
branch on Python 3.11 (the production target). Memory is reported as
PSS (true footprint). Sourced from the cooldown-gated three-way matrix
in benchmark/results/ (baseline-* vs after-py311-*).

Run inside the devcontainer:
    python benchmark/generate_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Measurements (devcontainer; MALLOC_ARENA_MAX=2 set during the run)
# ---------------------------------------------------------------------------
# All runs use MALLOC_ARENA_MAX=2 so glibc ptmalloc2 arena overhead doesn't
# dominate the RSS measurements. See README "Memory measurement" section.
# JSON sources: results/baseline-arena2-* and results/after-arena2-*.

# CONUS (Tier C: 1.1 M flowpaths, 8 workers, 24 timesteps). Single clean run
# from the cooldown-gated matrix (results/baseline-C vs results/after-py311-C).
CONUS = {
    "wall_s":   {"before": 275.83, "after": 115.26},
    "cpu_s":    {"before": 386.87, "after": 233.27},
    "rss_gb":   {"before":  26.54, "after":  24.61},      # main process peak RSS
    "tree_gb":  {"before":  29.90, "after":  27.79},      # main + 8 workers, PSS
    "util_x":   {"before":   1.40, "after":   2.02},
}

# CONUS phase breakdown (t-route internal timing block; total is slightly
# less than wall because process startup/teardown is excluded).
CONUS_PHASES = {
    "graph_s":   {"before":  80.11, "after":  53.98},
    "routing_s": {"before": 169.48, "after":  40.82},
    "output_s":  {"before":  15.57, "after":  12.20},
    "forcing_s": {"before":   2.38, "after":   2.27},
}

# Tier A (nhf_subset_ohio: ~11,327 flowpaths, single worker, 1728 timesteps).
# Median of 5 timed runs (results/baseline-A vs results/after-py311-A).
TIER_A = {
    "wall_s":   {"before": 54.83, "after": 46.30},
    "cpu_s":    {"before": 55.81, "after": 47.25},
    "rss_mb":   {"before": 2008,  "after": 2031},
}

# Tier B (MC kernel replay only). Median of 15 replays of ~1.05 M invocations
# each (results/baseline-B vs results/after-py311-B).
TIER_B = {
    "kernel_ms": {"before": 3161.18, "after": 2787.56},
}


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 140,
    "font.size": 11,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.5,
})

C_BEFORE = "#b0b0b0"
C_AFTER  = "#2a8acb"


def _grouped_bars(ax, before_vals, after_vals, labels, ylabel, value_fmt="{:.1f}"):
    x = np.arange(len(labels))
    w = 0.36
    ax.bar(x - w/2, before_vals, w, label="Baseline (pre-#94)", color=C_BEFORE, edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, after_vals,  w, label="After (this PR, py3.11)", color=C_AFTER,  edgecolor="black", linewidth=0.5)
    for i, (b, a) in enumerate(zip(before_vals, after_vals)):
        ax.text(i - w/2, b, value_fmt.format(b), ha="center", va="bottom", fontsize=9)
        ax.text(i + w/2, a, value_fmt.format(a), ha="center", va="bottom", fontsize=9, color=C_AFTER, fontweight="bold")
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(max(before_vals), max(after_vals)) * 1.18)
    ax.legend(loc="upper right", framealpha=0.95)


# ---------------------------------------------------------------------------
# Chart: CONUS executive summary (the headline)
# ---------------------------------------------------------------------------
def chart_conus_summary():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    fig.suptitle("CONUS-scale routing performance (1.1 M flowpaths, 8 workers)",
                 fontsize=13, fontweight="bold", y=1.02)

    # Wall + CPU on one panel
    ax = axes[0]
    labels = ["Wall time", "CPU time"]
    bef = [CONUS["wall_s"]["before"], CONUS["cpu_s"]["before"]]
    aft = [CONUS["wall_s"]["after"],  CONUS["cpu_s"]["after"]]
    _grouped_bars(ax, bef, aft, labels, "seconds", "{:.0f}")
    sp_wall = bef[0] / aft[0]
    sp_cpu  = bef[1] / aft[1]
    ax.set_title(f"Wall: {sp_wall:.2f}x faster   |   CPU: {sp_cpu:.2f}x less")

    # Memory is essentially flat. Reported as PSS (proportional set size):
    # shared copy-on-write / memmap pages are split across the processes that
    # map them, so the tree total is the true physical footprint. (A naive RSS
    # sum double-counts those pages and can exceed physical RAM, which would
    # overstate the memory reduction.)
    ax = axes[1]
    labels = ["Main proc RSS", "Tree PSS\n(main + 8 workers)"]
    bef = [CONUS["rss_gb"]["before"], CONUS["tree_gb"]["before"]]
    aft = [CONUS["rss_gb"]["after"],  CONUS["tree_gb"]["after"]]
    _grouped_bars(ax, bef, aft, labels, "GB", "{:.1f}")
    ax.set_ylim(0, max(max(bef), max(aft)) * 1.35)
    ax.legend(loc="upper left", framealpha=0.95)
    sp_tree = bef[1] / aft[1]
    ax.set_title(f"Memory ~flat: tree PSS {sp_tree:.2f}x\n(true footprint ~28-30 GB either way)")

    # Parallel utilization
    ax = axes[2]
    labels = ["Parallel utilization"]
    bef = [CONUS["util_x"]["before"]]
    aft = [CONUS["util_x"]["after"]]
    _grouped_bars(ax, bef, aft, labels, "x (CPU / wall, max = 8)", "{:.2f}x")
    ax.axhline(8.0, color="red", linestyle="--", linewidth=1, alpha=0.4)
    ax.text(0, 8.0, "  ideal (8 cores)", color="red", fontsize=8, va="bottom", alpha=0.6)
    ax.set_ylim(0, 8.6)
    ax.set_title(f"Workers: {aft[0]/bef[0]:.2f}x more saturated")

    fig.tight_layout()
    out = OUT / "conus_summary.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out.relative_to(ROOT.parent)}")


# ---------------------------------------------------------------------------
# Chart: CONUS speedup attribution by phase
# ---------------------------------------------------------------------------
def chart_phase_attribution():
    fig, ax = plt.subplots(figsize=(9, 4.5))

    # CONUS phase breakdown (from the t-route internal timing block in
    # benchmark/results/baseline-C.conus.json vs
    # benchmark/results/after-py311-C.conus.json).
    phases = ["Network graph\nconstruction", "Routing\ncomputations",
              "Output\nwriting", "Forcing array\nconstruction"]
    before = [CONUS_PHASES["graph_s"]["before"],
              CONUS_PHASES["routing_s"]["before"],
              CONUS_PHASES["output_s"]["before"],
              CONUS_PHASES["forcing_s"]["before"]]
    after  = [CONUS_PHASES["graph_s"]["after"],
              CONUS_PHASES["routing_s"]["after"],
              CONUS_PHASES["output_s"]["after"],
              CONUS_PHASES["forcing_s"]["after"]]

    x = np.arange(len(phases))
    w = 0.36
    ax.bar(x - w/2, before, w, label="Baseline (pre-#94)", color=C_BEFORE, edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, after,  w, label="After (this PR, py3.11)", color=C_AFTER,  edgecolor="black", linewidth=0.5)

    for i, (b, a) in enumerate(zip(before, after)):
        ax.text(i - w/2, b, f"{b:.0f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + w/2, a, f"{a:.0f}", ha="center", va="bottom", fontsize=9,
                color=C_AFTER, fontweight="bold")
        if b > 0:
            sp = b / a
            ax.text(i, max(b, a) * 0.55, f"{sp:.1f}x", ha="center",
                    color="black", fontsize=10, fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=2))

    ax.set_xticks(x, phases)
    ax.set_ylabel("seconds")
    ax.set_title("CONUS wall time by phase  (1.1 M flowpaths, 8 workers)")
    ax.set_ylim(0, max(before) * 1.18)
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    out = OUT / "conus_phases.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out.relative_to(ROOT.parent)}")


# ---------------------------------------------------------------------------
# Chart: Tier A (nhf_subset_ohio) overview
# ---------------------------------------------------------------------------
def chart_tier_a():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    fig.suptitle("Tier A (nhf_subset_ohio, ~11 k flowpaths, single worker)",
                 fontsize=13, fontweight="bold", y=1.02)

    ax = axes[0]
    labels = ["Wall time", "CPU time"]
    bef = [TIER_A["wall_s"]["before"], TIER_A["cpu_s"]["before"]]
    aft = [TIER_A["wall_s"]["after"],  TIER_A["cpu_s"]["after"]]
    _grouped_bars(ax, bef, aft, labels, "seconds", "{:.1f}")
    sp_wall = bef[0] / aft[0]
    ax.set_title(f"Wall: {sp_wall:.2f}x faster")

    ax = axes[1]
    labels = ["Peak RSS"]
    bef = [TIER_A["rss_mb"]["before"]]
    aft = [TIER_A["rss_mb"]["after"]]
    _grouped_bars(ax, bef, aft, labels, "MB", "{:.0f}")
    delta = bef[0] - aft[0]
    if abs(delta) < 25:
        ax.set_title(f"Peak memory: ~{aft[0]:.0f} MB (unchanged)")
    else:
        sign = "saved" if delta > 0 else "regressed"
        if abs(delta) >= 256:
            ax.set_title(f"Peak memory: {bef[0]/aft[0]:.2f}x ({abs(delta)/1024:.2f} GB {sign})")
        else:
            ax.set_title(f"Peak memory: {bef[0]/aft[0]:.2f}x ({int(abs(delta))} MB {sign})")

    fig.tight_layout()
    out = OUT / "tier_a_summary.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out.relative_to(ROOT.parent)}")


# ---------------------------------------------------------------------------
# Chart: speedup waterfall, single bar showing total CONUS speedup
# ---------------------------------------------------------------------------
def chart_speedup_overview():
    fig, ax = plt.subplots(figsize=(10.5, 4.2))

    # Headline ratios across the three tiers + tree-PSS memory metric
    # (devcontainer measurements, MALLOC_ARENA_MAX=2). Memory is ~flat (~1.08x)
    # and shown for honesty next to the real wall/CPU wins.
    metrics = ["Tier A wall\n(nhf_subset_ohio)",
               "Tier B kernel\n(MC replay)",
               "Tier C wall\n(CONUS)",
               "Tier C CPU\n(CONUS)",
               "Tier C tree PSS\n(all 8 workers)"]
    befores = [TIER_A["wall_s"]["before"], TIER_B["kernel_ms"]["before"],
               CONUS["wall_s"]["before"],  CONUS["cpu_s"]["before"],
               CONUS["tree_gb"]["before"]]
    afters  = [TIER_A["wall_s"]["after"],  TIER_B["kernel_ms"]["after"],
               CONUS["wall_s"]["after"],   CONUS["cpu_s"]["after"],
               CONUS["tree_gb"]["after"]]
    speedups = [b / a for b, a in zip(befores, afters)]

    colors = [C_AFTER if s >= 1.2 else "#7fb4d9" if s >= 1.05 else "#bcd6ea" for s in speedups]
    bars = ax.bar(metrics, speedups, color=colors, edgecolor="black", linewidth=0.6)

    for bar, sp in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width()/2, sp, f"{sp:.2f}x",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_ylabel("Speedup (baseline / after)")
    ax.set_title("Overall improvement (higher is better)")
    ax.set_ylim(0, max(speedups) * 1.18)
    fig.tight_layout()
    out = OUT / "speedup_overview.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out.relative_to(ROOT.parent)}")


def main():
    print("Generating figures...")
    chart_speedup_overview()
    chart_conus_summary()
    chart_phase_attribution()
    chart_tier_a()
    print(f"\nDone. Wrote PNGs to {OUT.relative_to(ROOT.parent)}/")


if __name__ == "__main__":
    sys.exit(main())
