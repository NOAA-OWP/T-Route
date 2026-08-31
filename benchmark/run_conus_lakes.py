#!/usr/bin/env python3
"""Validate an NHF release against t-route: full CONUS routing WITH lakes.

This is an acceptance test for new NextGen Hydrofabric releases, intended for
the NHF team: point it at a CONUS geopackage and it answers "does this release
still route through t-route?". It exercises the paths a release is most likely
to break:

  * the full network build (~1.1 M flowpaths) including ID handling,
  * waterbody preprocessing with ``break_network_at_waterbodies`` enabled
    (every lake the release can anchor is modeled as a level-pool reservoir),
  * a short parallel routing run, validated for finite outputs.

Unlike ``prep_conus.py`` (which clones the geopackage, repairs non-finite
channel parameters, and empties the lakes layer to build a stable performance
benchmark), this script reads the source geopackage AS-IS and READ-ONLY: a
validation run must surface data problems, not mask them. Watch the warnings
t-route emits during the network build; they are the per-category census of
lakes that could not be modeled and why (non-numeric lake_id, no fp_id,
missing level-pool parameters, no single inlet -> outlet chain).

Usage:
    python benchmark/run_conus_lakes.py --src /path/to/nhf_conus.gpkg

    # quicker (fewer timesteps), custom scratch dir:
    python benchmark/run_conus_lakes.py --src nhf.gpkg --nts 12 --workdir /tmp/nhf_check

Exit codes: 0 = routed, outputs finite; 1 = run or validation failed;
2 = source data failed the pre-checks (non-finite channel parameters).

Requires ~16 GB RAM and a few minutes (the network build dominates).
"""
from __future__ import annotations

import argparse
import datetime as dt
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

BENCH_DIR = Path(__file__).resolve().parent
CONUS_CONFIG = BENCH_DIR / "conus.yaml"
FORCING_START = dt.datetime(2000, 1, 1, 0, 0)
_FINITE = "BETWEEN -1e300 AND 1e300"

# Kept in sync with troute.nhf_preprocess._FLOWPATHS_CHANNEL_COLS.
try:  # pragma: no cover, prefer the live list if troute is importable
    from troute.nhf_preprocess import _FLOWPATHS_CHANNEL_COLS as CHANNEL_COLS
except Exception:
    CHANNEL_COLS = (
        "length_km", "n", "slope", "topwdth", "btmwdth",
        "topwdthcc", "ncc", "chslp", "musx", "musk", "mainstem_lp",
    )


def precheck_channel_params(src: Path) -> int:
    """Count non-finite Muskingum-Cunge channel parameters in ``flowpaths``.

    t-route's loader fails loud on these; checking up front gives the NHF team
    a per-column report instead of a mid-run traceback.
    """
    db = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    cur = db.cursor()
    fp_cols = {r[1] for r in cur.execute("PRAGMA table_info(flowpaths)")}
    total_bad = 0
    for col in CHANNEL_COLS:
        if col not in fp_cols:
            print(f"  flowpaths.{col}: MISSING COLUMN")
            total_bad += 1
            continue
        bad = cur.execute(
            f'SELECT count(*) FROM flowpaths '
            f'WHERE "{col}" IS NULL OR "{col}" NOT {_FINITE}'
        ).fetchone()[0]
        if bad:
            print(f"  flowpaths.{col}: {bad} non-finite value(s)")
            total_bad += bad
    db.close()
    if not total_bad:
        print("  channel parameters: all finite")
    return total_bad


def collect_forcing_ids(src: Path) -> list[int]:
    """Every divide id that lateral-inflow forcing must cover: the union of
    ``flowpaths.fp_id`` and ``reference_flowpaths.div_id`` (sub-divides that
    virtual flowpaths reference)."""
    db = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    ids = {int(r[0]) for r in db.execute(
        "SELECT fp_id FROM flowpaths WHERE fp_id IS NOT NULL")}
    ids |= {int(r[0]) for r in db.execute(
        "SELECT DISTINCT div_id FROM reference_flowpaths WHERE div_id IS NOT NULL")}
    db.close()
    return sorted(ids)


def synth_forcing(forcing_ids: list[int], dst_dir: Path, hours: int,
                  qlat: float) -> None:
    """Write ``hours`` hourly CHRTOUT CSVs with constant lateral inflow.

    Constant synthetic forcing is enough for validation: the network
    structure, not the forcing values, drives the code paths under test.
    """
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True)
    body = "\n".join(f"{fid},{qlat}" for fid in forcing_ids)
    for i in range(hours):
        stamp = (FORCING_START + dt.timedelta(hours=i)).strftime("%Y%m%d%H%M")
        (dst_dir / f"{stamp}.CHRTOUT_DOMAIN1.csv").write_text(
            f"feature_id,{stamp}\n{body}\n")
    print(f"  channel_forcing: {hours} files x {len(forcing_ids):,} divides "
          f"(constant q_lateral={qlat})")


def write_config(src: Path, forcing_dir: Path, out_dir: Path, nts: int,
                 cpu_pool: int | None, workdir: Path) -> Path:
    """Render the pinned CONUS config against the source geopackage.

    ``conus.yaml`` already pins ``network_type: NHF`` and
    ``break_network_at_waterbodies: True``; only the paths and run length are
    overridden, so a validation run uses the same settings as the production
    benchmark.
    """
    cfg = yaml.safe_load(CONUS_CONFIG.read_text())
    sp = cfg["network_topology_parameters"]["supernetwork_parameters"]
    sp["geo_file_path"] = str(src.resolve())
    assert cfg["network_topology_parameters"]["waterbody_parameters"][
        "break_network_at_waterbodies"] is True
    fp = cfg["compute_parameters"]["forcing_parameters"]
    fp["qlat_input_folder"] = str(forcing_dir.resolve())
    fp["nts"] = nts
    if cpu_pool is not None:
        cfg["compute_parameters"]["cpu_pool"] = cpu_pool
    cfg["output_parameters"]["stream_output"]["stream_output_directory"] = str(
        out_dir.resolve())
    config_path = workdir / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return config_path


def validate_output(out_dir: Path) -> bool:
    """All output values must be finite; reservoir nodes report pool elevation
    in ``depth``, so value ranges are wide but NaN/Inf means failure."""
    files = sorted(out_dir.glob("troute_output_*.nc"))
    print(f"  output files: {len(files)}")
    if not files:
        print("  FAIL: no output produced")
        return False
    ok = True
    for v in ("flow", "velocity", "depth"):
        arrs = [xr.open_dataset(f)[v].values for f in files]
        a = np.concatenate(arrs, axis=-1)
        n_bad = int((~np.isfinite(a)).sum())
        print(f"  {v:9s} shape={a.shape}  non-finite={n_bad}  "
              f"min={np.nanmin(a):.4g}  max={np.nanmax(a):.4g}")
        ok &= n_bad == 0
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True,
                    help="NHF CONUS geopackage to validate (read-only)")
    ap.add_argument("--workdir", type=Path,
                    default=BENCH_DIR / "data" / "conus_lakes",
                    help="scratch directory for forcing/config/output "
                         "(default: benchmark/data/conus_lakes, gitignored)")
    ap.add_argument("--forcing-hours", type=int, default=3,
                    help="hourly forcing files to synthesize (default 3)")
    ap.add_argument("--nts", type=int, default=24,
                    help="routing timesteps; must be <= forcing-hours * 12 "
                         "(default 24 = 2 simulated hours)")
    ap.add_argument("--qlat", type=float, default=0.05,
                    help="constant synthetic lateral inflow (default 0.05)")
    ap.add_argument("--cpu-pool", type=int, default=None,
                    help="override the cpu_pool pinned in conus.yaml")
    args = ap.parse_args()

    if not args.src.is_file():
        sys.exit(f"ERROR: source geopackage not found: {args.src}")
    if args.nts > args.forcing_hours * 12:
        sys.exit(f"ERROR: nts={args.nts} exceeds forcing coverage "
                 f"({args.forcing_hours}h x 12 = {args.forcing_hours * 12})")

    out_dir = args.workdir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("troute_output_*.nc"):
        f.unlink()

    print(f"=== 1) pre-checking channel parameters in {args.src} ===")
    if precheck_channel_params(args.src):
        print("\nFAIL: the release carries non-finite Muskingum-Cunge channel "
              "parameters; t-route's loader will reject it. Fix the data (or "
              "use prep_conus.py, which repairs to column medians, if you only "
              "need a performance benchmark).")
        return 2

    print("=== 2) collecting forcing divide ids ===")
    forcing_ids = collect_forcing_ids(args.src)
    print(f"  {len(forcing_ids):,} divides")

    print("=== 3) synthesizing forcing ===")
    synth_forcing(forcing_ids, args.workdir / "channel_forcing",
                  args.forcing_hours, args.qlat)

    print("=== 4) routing (watch the waterbody warnings: they are the lake "
          "census for this release) ===")
    config_path = write_config(args.src, args.workdir / "channel_forcing",
                               out_dir, args.nts, args.cpu_pool, args.workdir)
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", "nwm_routing", "-V5", "-f", str(config_path)])
    wall = time.perf_counter() - t0
    print(f"\n  nwm_routing exit={proc.returncode}  wall={wall:.1f}s")
    if proc.returncode != 0:
        print("FAIL: routing crashed")
        return 1

    print("=== 5) validating output ===")
    if not validate_output(out_dir):
        print("\nFAIL: output validation")
        return 1
    print("\nPASS: NHF release routes through t-route with lakes enabled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
