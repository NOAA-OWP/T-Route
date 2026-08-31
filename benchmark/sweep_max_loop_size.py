"""Sweep max_loop_size to characterize its effect on wall, CPU, and peak RSS.

max_loop_size in t-route chunks the forcing-file list: each chunk loads
max_loop_size hourly forcing files, builds a qlat array, runs routing on
that many timesteps, then frees and moves to the next chunk. Smaller
chunks trade more per-chunk overhead for lower transient memory; larger
chunks do the opposite.

This script runs the Tier A workload (nhf_subset_ohio.yaml, 144 hourly
forcing files) for a sweep of max_loop_size values, captures wall/CPU/RSS
per run, and writes a JSON summary to results/sweep_max_loop_size.json.

Run inside the devcontainer with MALLOC_ARENA_MAX=2 set (see
benchmark/README.md). Companion script plot_max_loop_size.py renders
figures/max_loop_size_sweep.png from the JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

BENCH_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCH_DIR / "data"
RESULTS_DIR = BENCH_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# 1, 2, 4, 8, 12, 24, 48, 72, 144 all divide 144 evenly so each sweep
# point uses uniformly-sized chunks (no short trailing chunk).
DEFAULT_SWEEP = (1, 2, 4, 8, 12, 24, 48, 72, 144)


def build_config(max_loop_size: int, output_dir: Path) -> Path:
    """Render nhf_subset_ohio.yaml with the given max_loop_size override."""
    src = BENCH_DIR / "nhf_subset_ohio.yaml"
    cfg = yaml.safe_load(src.read_text())
    sp = cfg["network_topology_parameters"]["supernetwork_parameters"]
    sp["geo_file_path"] = str((BENCH_DIR / sp["geo_file_path"]).resolve())
    fp = cfg["compute_parameters"]["forcing_parameters"]
    fp["qlat_input_folder"] = str((BENCH_DIR / fp["qlat_input_folder"]).resolve())
    fp["max_loop_size"] = int(max_loop_size)
    cfg["output_parameters"]["stream_output"]["stream_output_directory"] = str(output_dir)
    out = output_dir / "_resolved.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def run_once(config_path: Path, log_path: Path) -> dict:
    """Run nwm_routing once; return wall/cpu/rss for exactly that child."""
    cmd = [sys.executable, "-m", "nwm_routing", "-V5", "-f", str(config_path)]
    with open(log_path, "wb") as log:
        t0 = time.perf_counter()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        _, status, ru = os.wait4(proc.pid, 0)
        wall = time.perf_counter() - t0
    proc.returncode = os.waitstatus_to_exitcode(status)
    if proc.returncode != 0:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-30:])
        raise RuntimeError(f"nwm_routing failed (exit {proc.returncode}):\n{tail}")
    rss_bytes = ru.ru_maxrss if sys.platform == "darwin" else ru.ru_maxrss * 1024
    return {
        "wall_s": wall,
        "cpu_s": ru.ru_utime + ru.ru_stime,
        "rss_mb": rss_bytes / (1024 * 1024),
    }


def bench_point(max_loop_size: int, runs: int, warmup: int) -> dict:
    """One sweep point: warmup + timed runs, returns aggregated stats."""
    timed: list[dict] = []
    with tempfile.TemporaryDirectory(prefix=f"sweep_mls{max_loop_size}_") as td:
        out_dir = Path(td)
        cfg = build_config(max_loop_size, out_dir)
        for i in range(warmup + runs):
            log = out_dir / f"run_{i}.log"
            metrics = run_once(cfg, log)
            kind = "warmup" if i < warmup else "timed "
            print(f"  [{kind} mls={max_loop_size:>3d} run{i}] "
                  f"wall={metrics['wall_s']:6.2f}s  "
                  f"cpu={metrics['cpu_s']:6.2f}s  "
                  f"rss={metrics['rss_mb']:7.1f} MB", flush=True)
            if i >= warmup:
                timed.append(metrics)
            # Wipe outputs so successive runs don't accumulate
            for nc in out_dir.glob("troute_output_*.nc"):
                nc.unlink()

    def agg(field: str) -> dict:
        vals = [r[field] for r in timed]
        return {
            "min": min(vals),
            "median": statistics.median(vals),
            "mean": statistics.fmean(vals),
            "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        }

    return {
        "max_loop_size": int(max_loop_size),
        "n_chunks": int(144 // max_loop_size + (1 if 144 % max_loop_size else 0)),
        "runs": runs,
        "warmup": warmup,
        "wall_s": agg("wall_s"),
        "cpu_s": agg("cpu_s"),
        "rss_mb": agg("rss_mb"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", type=int, nargs="+", default=list(DEFAULT_SWEEP),
                    help=f"max_loop_size values to sweep (default {DEFAULT_SWEEP})")
    ap.add_argument("--runs", type=int, default=2, help="timed runs per point")
    ap.add_argument("--warmup", type=int, default=1, help="discarded warmups per point")
    ap.add_argument("--label", default="max_loop_size_sweep")
    args = ap.parse_args()

    if not (DATA_DIR / "MANIFEST.json").exists():
        sys.exit("ERROR: benchmark dataset missing. Run: "
                 "python benchmark/prep_ohio_data.py --src /path/to/nhf_1.1.4.gpkg")

    arena_max = os.environ.get("MALLOC_ARENA_MAX")
    print(f"sweep: max_loop_size in {args.sweep}")
    print(f"runs: {args.runs} timed, {args.warmup} warmup per point")
    print(f"MALLOC_ARENA_MAX={arena_max!r} (set to 2 for honest RSS measurement)")
    print()

    points = []
    for mls in args.sweep:
        print(f"=== max_loop_size = {mls} ({144 // mls} chunks of 144 files) ===",
              flush=True)
        t0 = time.perf_counter()
        point = bench_point(mls, args.runs, args.warmup)
        elapsed = time.perf_counter() - t0
        points.append(point)
        print(f"  -> wall_med={point['wall_s']['median']:.2f}s  "
              f"cpu_med={point['cpu_s']['median']:.2f}s  "
              f"rss_med={point['rss_mb']['median']:.1f} MB  "
              f"(point elapsed {elapsed:.1f}s)\n", flush=True)

    out_path = RESULTS_DIR / f"{args.label}.json"
    out_path.write_text(json.dumps({
        "label": args.label,
        "malloc_arena_max": arena_max,
        "points": points,
    }, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
