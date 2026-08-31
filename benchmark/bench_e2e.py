#!/usr/bin/env python3
"""Tier A end-to-end benchmark for the Muskingum-Cunge kernel perf work.

Runs the full nhf_subset_ohio NHF routing case (`python -m nwm_routing
-V5 -f ...`) as a subprocess over `--warmup` + `--runs` repetitions,
capturing wall time, CPU time, and peak RSS for each run via os.wait4.
Reports min/median/mean/sd and compares the stream output against
benchmark/golden/ for the correctness gate.

Reproducible: fixed dataset (benchmark/data/, built by prep_ohio_data.py),
fixed config (benchmark/nhf_subset_ohio.yaml), cpu_pool=1 (deterministic,
kernel-dominated ~87%).

Usage:
    python benchmark/bench_e2e.py --save-golden        # capture the reference
    python benchmark/bench_e2e.py                      # benchmark vs golden
    python benchmark/bench_e2e.py --runs 5 --label step1-O3 --json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

BENCH_DIR = Path(__file__).resolve().parent
CONFIG = BENCH_DIR / "nhf_subset_ohio.yaml"
DATA_DIR = BENCH_DIR / "data"
GOLDEN_DIR = BENCH_DIR / "golden"
RESULTS_DIR = BENCH_DIR / "results"

COMPARE_VARS = ("flow", "velocity", "depth")


# --------------------------------------------------------------------------
# config + run
# --------------------------------------------------------------------------
def resolved_config(output_dir: Path, nts: int | None) -> Path:
    """Render nhf_subset_ohio.yaml with absolute paths + an isolated output dir."""
    if not (DATA_DIR / "MANIFEST.json").exists():
        sys.exit("ERROR: benchmark dataset missing. Run: "
                 "python benchmark/prep_ohio_data.py --src /path/to/nhf_1.1.4.gpkg")
    cfg = yaml.safe_load(CONFIG.read_text())
    sp = cfg["network_topology_parameters"]["supernetwork_parameters"]
    sp["geo_file_path"] = str((BENCH_DIR / sp["geo_file_path"]).resolve())
    fp = cfg["compute_parameters"]["forcing_parameters"]
    fp["qlat_input_folder"] = str((BENCH_DIR / fp["qlat_input_folder"]).resolve())
    if nts is not None:
        fp["nts"] = nts
    cfg["output_parameters"]["stream_output"]["stream_output_directory"] = str(output_dir)
    path = output_dir / "_resolved.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def run_once(config_path: Path, log_path: Path) -> dict:
    """Run nwm_routing once; return wall/cpu/rss for exactly that child."""
    cmd = [sys.executable, "-m", "nwm_routing", "-V5", "-f", str(config_path)]
    with open(log_path, "wb") as log:
        t0 = time.perf_counter()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        _, status, ru = os.wait4(proc.pid, 0)
        wall = time.perf_counter() - t0
    proc.returncode = os.waitstatus_to_exitcode(status)  # mark reaped for Popen
    if proc.returncode != 0:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-30:])
        raise RuntimeError(f"nwm_routing failed (exit {proc.returncode}):\n{tail}")
    # ru_maxrss: bytes on macOS, kibibytes on Linux.
    rss = ru.ru_maxrss if sys.platform == "darwin" else ru.ru_maxrss * 1024
    return {"wall_s": wall,
            "cpu_s": ru.ru_utime + ru.ru_stime,
            "rss_mb": rss / (1024 * 1024)}


# --------------------------------------------------------------------------
# correctness
# --------------------------------------------------------------------------
def load_output(out_dir: Path) -> dict:
    """Concatenate all troute_output_*.nc into per-variable arrays."""
    files = sorted(out_dir.glob("troute_output_*.nc"))
    if not files:
        raise RuntimeError(f"no troute_output_*.nc found in {out_dir}")
    chunks: dict[str, list] = {v: [] for v in COMPARE_VARS}
    for f in files:
        with xr.open_dataset(f) as ds:
            for v in COMPARE_VARS:
                chunks[v].append(ds[v].values)
    return {v: np.concatenate(chunks[v], axis=-1) for v in COMPARE_VARS}


def compare(new: dict, golden: dict) -> tuple[dict, float, int]:
    """Per-variable max abs/rel error vs golden. Returns (report, worst_rel,
    total new NaNs)."""
    report, worst_rel, new_nan_total = {}, 0.0, 0
    for v in COMPARE_VARS:
        a, b = new[v], golden[v]
        if a.shape != b.shape:
            report[v] = {"status": f"SHAPE {a.shape} vs {b.shape}"}
            continue
        diff = np.abs(a - b)
        rel = diff / np.maximum(np.abs(b), 1e-6)
        new_nan = int((~np.isfinite(a) & np.isfinite(b)).sum())
        max_abs = float(np.nanmax(diff)) if diff.size else 0.0
        max_rel = float(np.nanmax(rel)) if rel.size else 0.0
        report[v] = {"max_abs": max_abs, "max_rel": max_rel, "new_nan": new_nan}
        worst_rel = max(worst_rel, max_rel)
        new_nan_total += new_nan
    return report, worst_rel, new_nan_total


# --------------------------------------------------------------------------
# stats + reporting
# --------------------------------------------------------------------------
def summarize(samples: list[dict]) -> dict:
    out = {}
    for key in ("wall_s", "cpu_s", "rss_mb"):
        xs = [s[key] for s in samples]
        out[key] = {
            "min": min(xs), "median": statistics.median(xs),
            "mean": statistics.fmean(xs),
            "stdev": statistics.stdev(xs) if len(xs) > 1 else 0.0,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=5, help="timed repetitions")
    ap.add_argument("--warmup", type=int, default=1, help="discarded warmup runs")
    ap.add_argument("--nts", type=int, default=None, help="override nts (dev loop)")
    ap.add_argument("--save-golden", action="store_true",
                    help="store this run's output as the correctness reference")
    ap.add_argument("--golden-dir", default=None,
                    help="reference-output dir for the correctness gate "
                         "(default benchmark/golden). regression_check.sh points "
                         "this at a baseline branch's output so the candidate is "
                         "compared against it without touching the committed golden")
    ap.add_argument("--label", default=None, help="label for the JSON record")
    ap.add_argument("--json", action="store_true",
                    help="write results/<label>.json")
    args = ap.parse_args()
    golden_dir = Path(args.golden_dir).resolve() if args.golden_dir else GOLDEN_DIR

    work = Path(tempfile.mkdtemp(prefix="bench_e2e_"))
    samples: list[dict] = []
    last_out: Path | None = None
    try:
        total = args.warmup + args.runs
        for i in range(total):
            tag = f"warmup{i+1}" if i < args.warmup else f"run{i-args.warmup+1}"
            out_dir = work / tag
            out_dir.mkdir()
            cfg = resolved_config(out_dir, args.nts)
            m = run_once(cfg, out_dir / "run.log")
            kind = "warmup " if i < args.warmup else "timed   "
            print(f"  [{kind}{tag:8s}] wall={m['wall_s']:7.2f}s  "
                  f"cpu={m['cpu_s']:7.2f}s  rss={m['rss_mb']:8.1f}MB")
            if i >= args.warmup:
                samples.append(m)
                if last_out is not None:
                    shutil.rmtree(last_out, ignore_errors=True)
                last_out = out_dir
            else:
                shutil.rmtree(out_dir, ignore_errors=True)

        stats = summarize(samples)
        print(f"\nnhf_subset_ohio end-to-end  ({args.runs} timed runs, "
              f"{args.warmup} warmup discarded)")
        print(f"  {'':10s} {'min':>10s} {'median':>10s} {'mean':>10s} {'stdev':>9s}")
        for key, unit in (("wall_s", "wall (s)"), ("cpu_s", "cpu (s)"),
                          ("rss_mb", "rss (MB)")):
            s = stats[key]
            print(f"  {unit:10s} {s['min']:10.2f} {s['median']:10.2f} "
                  f"{s['mean']:10.2f} {s['stdev']:9.3f}")

        # correctness
        if last_out is None:
            sys.exit("ERROR: no timed run produced output (need --runs >= 1)")
        new = load_output(last_out)
        status = "n/a"
        report: dict = {}
        if args.save_golden:
            golden_dir.mkdir(parents=True, exist_ok=True)
            for f in golden_dir.glob("troute_output_*.nc"):
                f.unlink()
            for f in sorted(last_out.glob("troute_output_*.nc")):
                shutil.copy2(f, golden_dir / f.name)
            print(f"\ngolden saved -> {golden_dir}")
            status = "golden-saved"
        elif any(golden_dir.glob("troute_output_*.nc")):
            golden = load_output(golden_dir)
            report, worst_rel, new_nan = compare(new, golden)
            print("\ncorrectness vs golden (candidate minus golden, per variable):")
            for v in COMPARE_VARS:
                r = report[v]
                if "status" in r:
                    print(f"  {v:10s} {r['status']}")
                else:
                    print(f"  {v:10s} max_abs={r['max_abs']:.3e}  "
                          f"max_rel={r['max_rel']:.3e}  new_nan={r['new_nan']}")
            print("  reading the numbers: max_abs is the absolute diff in physical")
            print("  units (flow m^3/s, velocity m/s, depth m). max_rel is relative,")
            print("  |cand-gold| / max(|gold|, 1e-6). FLOW max_rel is the reliable")
            print("  metric (1e-6 ~ identical, ~1 ~ 100% off at some reach); velocity")
            print("  and depth max_rel blow up at near-dry nodes (gold ~ 0 in the")
            print("  denominator), so judge those two by max_abs. new_nan counts")
            print("  values that became NaN/Inf (the only PASS/FAIL gate here).")
            status = "PASS" if new_nan == 0 else "FAIL (new NaN)"
            print(f"  -> {status}  (gate: no new NaN/Inf; flow worst rel {worst_rel:.3e})")
        else:
            print("\ncorrectness: no golden yet (run with --save-golden first)")

        # markdown row + json
        med = stats
        print("\nRESULTS.md row:")
        print(f"| {args.label or 'TBD'} | `{_git_sha()}` | "
              f"{med['wall_s']['median']:.2f} | {med['cpu_s']['median']:.2f} | "
              f"{med['rss_mb']['median']:.0f} | "
              f"{report.get('flow', {}).get('max_rel', 0.0):.2e} | {status} |")

        if args.json:
            RESULTS_DIR.mkdir(exist_ok=True)
            label = args.label or dt.datetime.now().strftime("%Y%m%d-%H%M%S")
            rec = {"label": label, "git_sha": _git_sha(),
                   "utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
                   "runs": args.runs, "warmup": args.warmup,
                   "stats": stats, "correctness": report, "status": status}
            (RESULTS_DIR / f"{label}.json").write_text(json.dumps(rec, indent=2))
            print(f"json -> {RESULTS_DIR / (label + '.json')}")

        return 0 if status in ("PASS", "golden-saved", "n/a") else 1
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=BENCH_DIR,
            text=True).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
