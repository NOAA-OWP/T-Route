#!/usr/bin/env python3
"""Benchmark the cost of the data-assimilation capabilities.

Each capability is measured as an A/B against the same case with that capability
switched off, so the number reported is the marginal cost of the feature rather
than the cost of the whole run. The arms differ by exactly one configuration
block; everything else (network, forcing, compute method, timestep) is held fixed.

Arms:
  baseline       routing only, no assimilation
  nudging        streamflow nudging at gages
  diversion      nudging plus the Old River style flow transfer
  reservoir_da   reservoir persistence and RFC assimilation (needs lakes)

Two things are worth watching over time, and both are reported:

  wall time      the direct cost of the feature.
  gage links     how many routing links carry an assimilated observation. This
                 drives how finely the execution plan splits reaches, which is
                 cached and reused across forcing windows, so it affects every
                 window and not just the first. A regression here shows up as a
                 routing slowdown with no obvious cause, which is exactly what the
                 one-to-many gage crosswalk did before it was fixed.

Usage:
    python benchmark/bench_da.py                       # all arms, 3 runs each
    python benchmark/bench_da.py --arms baseline nudging
    python benchmark/bench_da.py --runs 5 --json results.json
    python benchmark/bench_da.py --config path/to/case.yaml
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml

BENCH_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = BENCH_DIR / "nhf_subset_ohio.yaml"

# Old River: donor Mississippi link -> gage measuring the diverted flow.
# Only applied when the case's domain actually contains the donor link.
DIVERSION_CROSSWALK = {1270479816524705: "07381482"}


def _da_block(arm: str, base: dict, obs_dir: Path) -> dict:
    """The data_assimilation_parameters for one arm, on top of the case's own."""
    da = copy.deepcopy(base) or {}
    da.setdefault("streamflow_da", {})
    # Nudging reads this unconditionally; with it unset the run dies inside pathlib
    # with a TypeError rather than a configuration error. An empty directory keeps
    # the arm measuring what it is meant to measure: the cost of building the gage
    # crosswalk and splitting the execution plan at gages. It does NOT measure the
    # in-kernel assimilation itself, which needs an observation archive.
    da["usgs_timeslices_folder"] = str(obs_dir)
    if arm == "baseline":
        da["streamflow_da"]["streamflow_nudging"] = False
        da.pop("diversion_da", None)
    elif arm == "nudging":
        da["streamflow_da"]["streamflow_nudging"] = True
        da.pop("diversion_da", None)
    elif arm == "diversion":
        da["streamflow_da"]["streamflow_nudging"] = True
        da["diversion_da"] = {
            "diversion_gage_crosswalk": DIVERSION_CROSSWALK,
            # climatology so the arm does not depend on a timeslice archive being
            # present; it exercises the same kernel path as observed diversion.
            "persist_historical_median": True,
        }
    elif arm == "reservoir_da":
        da["streamflow_da"]["streamflow_nudging"] = False
        da.setdefault("reservoir_da", {}).setdefault("reservoir_persistence_da", {}).update(
            {"reservoir_persistence_usgs": True, "reservoir_persistence_usace": True}
        )
    else:
        raise ValueError(f"unknown arm {arm!r}")
    return da


def resolved_config(src: Path, arm: str, out_dir: Path) -> Path:
    cfg = yaml.safe_load(src.read_text())
    # The rendered config lands in out_dir, so any path relative to the source
    # config has to be made absolute first or it resolves against the wrong root.
    root = src.parent
    sp = cfg["network_topology_parameters"]["supernetwork_parameters"]
    sp["geo_file_path"] = str((root / sp["geo_file_path"]).resolve())
    fp = cfg["compute_parameters"]["forcing_parameters"]
    fp["qlat_input_folder"] = str((root / fp["qlat_input_folder"]).resolve())
    cp = cfg["compute_parameters"]
    obs_dir = out_dir / "empty_timeslices"
    obs_dir.mkdir(parents=True, exist_ok=True)
    cp["data_assimilation_parameters"] = _da_block(
        arm, cp.get("data_assimilation_parameters"), obs_dir
    )
    cfg.setdefault("output_parameters", {}).setdefault("stream_output", {})[
        "stream_output_directory"
    ] = str(out_dir)
    path = out_dir / f"_bench_{arm}.yaml"
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
    proc.returncode = os.waitstatus_to_exitcode(status)
    if proc.returncode != 0:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-25:])
        raise RuntimeError(f"{config_path.name} failed (exit {proc.returncode}):\n{tail}")
    # ru_maxrss: bytes on macOS, kibibytes on Linux.
    rss = ru.ru_maxrss if sys.platform == "darwin" else ru.ru_maxrss * 1024
    return {"wall_s": wall, "cpu_s": ru.ru_utime + ru.ru_stime, "rss_mb": rss / 1024**2}


def gage_link_count(log_path: Path) -> int | None:
    """How many routing links carry an observation, read back from the run log.

    Reported because it is the quantity that silently drove the execution plan's
    reach splitting; a jump here is a regression even when wall time looks flat.
    """
    for line in log_path.read_text(errors="replace").splitlines():
        # "gages: N USGS gage(s) placed on M routing link(s)"
        if "gage(s) placed on" in line:
            try:
                return int(line.split("placed on")[1].split()[0])
            except (ValueError, IndexError):
                return None
    return None


def summarize(samples: list[dict]) -> dict:
    out = {}
    for key in ("wall_s", "cpu_s", "rss_mb"):
        vals = [s[key] for s in samples]
        out[key] = {
            "min": min(vals),
            "median": statistics.median(vals),
            "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        }
    return out


def _inapplicable_arms(config: Path, arms: list[str]) -> dict[str, str]:
    """Arms whose feature cannot engage on this domain, with the reason."""
    out: dict[str, str] = {}
    if "diversion" not in arms:
        return out
    cfg = yaml.safe_load(config.read_text())
    geo = cfg["network_topology_parameters"]["supernetwork_parameters"]["geo_file_path"]
    geo_path = (config.parent / geo) if not Path(geo).is_absolute() else Path(geo)
    try:
        import geopandas as gpd

        fps = gpd.read_file(geo_path, layer="flowpaths", ignore_geometry=True,
                            columns=["fp_id"])
        present = set(fps["fp_id"].dropna().astype("int64"))
    except Exception as exc:  # pragma: no cover - depends on local data
        out["diversion"] = f"could not read flowpaths from {geo_path.name}: {exc}"
        return out
    missing = [fp for fp in DIVERSION_CROSSWALK if fp not in present]
    if missing:
        out["diversion"] = (
            f"donor link {missing[0]} is not in this domain, so no flow would be "
            "diverted; run against a domain containing the control structure"
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--arms", nargs="*",
                    default=["baseline", "nudging", "diversion"],
                    help="reservoir_da needs a domain with lakes")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=BENCH_DIR / "output_bench_da")
    args = ap.parse_args()

    if not args.config.exists():
        print(f"config not found: {args.config}", file=sys.stderr)
        print("build the dataset first: python benchmark/prep_ohio_data.py --src <gpkg>",
              file=sys.stderr)
        return 2

    # An arm whose feature cannot engage on this domain reports nothing rather than a
    # number that looks like a measurement. The diversion needs its donor link present.
    skip_reason = _inapplicable_arms(args.config, args.arms)
    for arm, why in skip_reason.items():
        print(f"[{arm}] skipped: {why}")

    results: dict[str, dict] = {}
    for arm in args.arms:
        if arm in skip_reason:
            continue
        out_dir = args.out_dir / arm
        cfg = resolved_config(args.config, arm, out_dir)
        log = out_dir / "run.log"
        print(f"[{arm}] warmup x{args.warmup}, runs x{args.runs}", flush=True)
        for _ in range(args.warmup):
            run_once(cfg, log)
        samples = [run_once(cfg, log) for _ in range(args.runs)]
        results[arm] = summarize(samples)
        results[arm]["gage_links"] = gage_link_count(log)

    base = results.get("baseline", {}).get("wall_s", {}).get("median")
    print(f"\n{'arm':<14}{'wall median':>13}{'sd':>8}{'rss MB':>10}{'gage links':>12}{'vs baseline':>13}")
    for arm, r in results.items():
        med = r["wall_s"]["median"]
        delta = f"{100.0 * (med - base) / base:+.1f}%" if base else "-"
        links = r["gage_links"] if r["gage_links"] is not None else "-"
        print(f"{arm:<14}{med:>13.2f}{r['wall_s']['sd']:>8.2f}"
              f"{r['rss_mb']['median']:>10.0f}{links:>12}{delta:>13}")

    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
