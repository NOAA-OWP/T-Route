"""Render max_loop_size sweep results as figures/max_loop_size_sweep.png.

Reads results/max_loop_size_sweep.json (produced by sweep_max_loop_size.py)
and produces a 2-panel figure: wall+CPU vs max_loop_size, and peak RSS vs
max_loop_size. Annotates the configured default and the empirical minimum.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
JSON_PATH = BENCH_DIR / "results" / "max_loop_size_sweep.json"
OUT_PATH = BENCH_DIR / "figures" / "max_loop_size_sweep.png"


def main() -> int:
    if not JSON_PATH.exists():
        sys.exit(f"ERROR: {JSON_PATH} not found. Run "
                 "python benchmark/sweep_max_loop_size.py first.")
    data = json.loads(JSON_PATH.read_text())
    pts = data["points"]
    mls = np.array([p["max_loop_size"] for p in pts])
    n_chunks = np.array([p["n_chunks"] for p in pts])
    wall_med = np.array([p["wall_s"]["median"] for p in pts])
    wall_min = np.array([p["wall_s"]["min"] for p in pts])
    cpu_med = np.array([p["cpu_s"]["median"] for p in pts])
    rss_med = np.array([p["rss_mb"]["median"] for p in pts])

    # Empirical optimum
    opt_idx = int(np.argmin(wall_med))
    opt_mls = mls[opt_idx]
    opt_wall = wall_med[opt_idx]

    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 140,
        "font.size": 11,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.5,
    })

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    fig.suptitle(
        f"max_loop_size sweep (Tier A: nhf_subset_ohio, 144 hourly forcing "
        f"files, MALLOC_ARENA_MAX=2)",
        fontsize=12, fontweight="bold", y=1.02,
    )

    # Panel 1: Wall + CPU
    ax = axes[0]
    ax.plot(mls, wall_med, marker="o", color="#2a8acb",
            label="Wall time (median)", linewidth=2)
    ax.plot(mls, cpu_med, marker="s", color="#cc6633",
            label="CPU time (median)", linewidth=2, linestyle="--")
    ax.set_xscale("log", base=2)
    ax.set_xticks(mls)
    ax.set_xticklabels([str(int(x)) for x in mls])
    ax.set_xlabel("max_loop_size (forcing files per chunk)")
    ax.set_ylabel("seconds")
    ax.set_title(f"Wall + CPU vs chunk size  "
                 f"(min wall = {opt_wall:.1f}s at mls={opt_mls})")
    ax.axvline(opt_mls, color="#2a8acb", linestyle=":", alpha=0.4)
    ax.axvline(24, color="black", linestyle=":", alpha=0.3)
    ax.text(24, ax.get_ylim()[1] * 0.97, " current\n default ",
            color="black", fontsize=8, va="top", ha="left", alpha=0.6)
    ax.legend(loc="best", framealpha=0.95)

    # Annotate each wall point
    for x, y in zip(mls, wall_med):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color="#2a8acb")

    # Panel 2: Peak RSS
    ax = axes[1]
    ax.plot(mls, rss_med, marker="o", color="#3aa843", linewidth=2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(mls)
    ax.set_xticklabels([str(int(x)) for x in mls])
    ax.set_xlabel("max_loop_size (forcing files per chunk)")
    ax.set_ylabel("Peak main-proc RSS (MB)")
    rss_pct = (rss_med.max() - rss_med.min()) / rss_med.min() * 100
    ax.set_title(f"Peak RSS vs chunk size  "
                 f"(spread = {rss_pct:.1f}%, "
                 f"{rss_med.min():.0f} -> {rss_med.max():.0f} MB)")
    ax.axvline(24, color="black", linestyle=":", alpha=0.3)
    for x, y in zip(mls, rss_med):
        ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color="#3aa843")

    fig.tight_layout()
    OUT_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PATH}")

    # Print a markdown-ready table for RESULTS.md
    print("\nMarkdown table for RESULTS.md:\n")
    print("| `max_loop_size` | Chunks | Wall (s, median) | CPU (s, median) | "
          "Peak RSS (MB) |")
    print("|---:|---:|---:|---:|---:|")
    for p in pts:
        print(f"| {p['max_loop_size']} | {p['n_chunks']} | "
              f"{p['wall_s']['median']:.2f} | "
              f"{p['cpu_s']['median']:.2f} | "
              f"{p['rss_mb']['median']:.0f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
