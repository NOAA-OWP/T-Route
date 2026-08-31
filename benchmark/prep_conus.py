#!/usr/bin/env python3
"""Prepare the (git-ignored) CONUS-scale benchmark dataset.

The CONUS workload is the full NHF hydrofabric (~1.1 M flowpaths). This
script, run once:

  1. Clones the source geopackage (a verbatim ~6 GB file copy, fast, low
     memory) and repairs any non-finite Muskingum-Cunge channel parameter
     to that column's finite median. GeoPackage spatialite triggers are
     dropped first so the in-place repair UPDATE can run; t-route only
     reads the file, so dropped triggers are harmless.
  2. Synthesizes constant-q_lateral channel forcing for every flowpath.
     The network structure, not the forcing values, drives load
     distribution and hotpaths, so synthesized forcing keeps the
     dataset self-contained.

Usage:
    python benchmark/prep_conus.py --src /path/to/nhf_1.1.4.gpkg
    python benchmark/prep_conus.py --src /path/to/nhf_1.1.4.gpkg --force
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sqlite3
import sys
from pathlib import Path

import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
CONUS_DIR = BENCH_DIR / "data" / "conus"
MANIFEST = CONUS_DIR / "MANIFEST.json"

try:  # pragma: no cover
    from troute.nhf_preprocess import _FLOWPATHS_CHANNEL_COLS as CHANNEL_COLS
except Exception:
    CHANNEL_COLS = (
        "length_km", "n", "slope", "topwdth", "btmwdth",
        "topwdthcc", "ncc", "chslp", "musx", "musk", "mainstem_lp",
    )

FORCING_START = dt.datetime(2000, 1, 1, 0, 0)
# A finite value is anything inside this range; NULL / NaN / +/-Inf are not.
_FINITE = "BETWEEN -1e300 AND 1e300"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clone_and_clean(
    src: Path, dst: Path,
) -> tuple[dict[str, int], list[int], int]:
    """Verbatim-copy the geopackage, drop triggers, repair non-finite
    channel parameters in `flowpaths`.

    Returns:
        repaired:       per-column count of non-finite rows replaced by the
                        column's finite median ({col_name: row_count}).
        fp_ids:         every flowpaths.fp_id, in source order.
        lakes_emptied:  number of rows deleted from the lakes layer (0 if
                        the geopackage had no lakes layer).
    """
    print(f"  copying {src.stat().st_size / 1e9:.1f} GB geopackage ...")
    shutil.copyfile(src, dst)

    db = sqlite3.connect(dst)
    cur = db.cursor()
    triggers = [r[0] for r in
                cur.execute("SELECT name FROM sqlite_master WHERE type='trigger'")]
    for name in triggers:
        cur.execute(f'DROP TRIGGER "{name}"')
    print(f"  dropped {len(triggers)} GeoPackage triggers")

    # Empty the lakes layer. The nhf_1.1.4 lakes layer has NaN `fp_id`
    # values that crash t-route's NHF waterbody preprocessing; the Tier A
    # subset has no lakes layer either, so routing every segment through
    # the MC kernel keeps the two tiers consistent (and kernel-dominated).
    lakes_emptied = 0
    has_lakes = cur.execute(
        "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='lakes'"
    ).fetchone()[0]
    if has_lakes:
        lakes_emptied = cur.execute("SELECT count(*) FROM lakes").fetchone()[0]
        cur.execute("DELETE FROM lakes")
        print(f"  lakes: emptied ({lakes_emptied} rows), routed as MC channels")

    fp_cols = {r[1] for r in cur.execute("PRAGMA table_info(flowpaths)")}
    repaired: dict[str, int] = {}
    for col in CHANNEL_COLS:
        if col not in fp_cols:
            continue
        bad = cur.execute(
            f'SELECT count(*) FROM flowpaths '
            f'WHERE "{col}" IS NULL OR "{col}" NOT {_FINITE}'
        ).fetchone()[0]
        if bad:
            vals = [r[0] for r in cur.execute(
                f'SELECT "{col}" FROM flowpaths '
                f'WHERE "{col}" IS NOT NULL AND "{col}" {_FINITE}')]
            median = float(np.median(vals))
            cur.execute(
                f'UPDATE flowpaths SET "{col}"=? '
                f'WHERE "{col}" IS NULL OR "{col}" NOT {_FINITE}', (median,))
            repaired[col] = bad
            print(f"  flowpaths.{col}: repaired {bad} non-finite -> "
                  f"median {median:.6g}")
    db.commit()
    if not repaired:
        print("  flowpaths: no non-finite channel parameters found")

    fp_ids = [r[0] for r in cur.execute("SELECT fp_id FROM flowpaths")]
    db.commit()
    db.close()
    return repaired, fp_ids, lakes_emptied


def build_forcing(fp_ids: list[int], dst_dir: Path, hours: int,
                  qlat: float) -> list[Path]:
    """Write `hours` hourly synthetic CHRTOUT CSVs (constant q_lateral)."""
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True)
    body = "\n".join(f"{fid},{qlat}" for fid in fp_ids)
    written: list[Path] = []
    for i in range(hours):
        stamp = (FORCING_START + dt.timedelta(hours=i)).strftime("%Y%m%d%H%M")
        path = dst_dir / f"{stamp}.CHRTOUT_DOMAIN1.csv"
        path.write_text(f"feature_id,{stamp}\n{body}\n")
        written.append(path)
    print(f"  channel_forcing: wrote {len(written)} files x {len(fp_ids):,} "
          f"segments (constant q_lateral={qlat})")
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True,
                    help="source CONUS geopackage (NHF v1.1.4 .gpkg)")
    ap.add_argument("--forcing-hours", type=int, default=2,
                    help="hourly forcing files to generate (default 2; "
                         "nts = hours * 12)")
    ap.add_argument("--qlat", type=float, default=0.05,
                    help="synthetic constant lateral inflow (default 0.05)")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if data/conus already exists")
    args = ap.parse_args()

    if MANIFEST.exists() and not args.force:
        print(f"CONUS dataset already prepared ({MANIFEST}). "
              f"Use --force to rebuild.")
        return 0
    if not args.src.is_file():
        sys.exit(f"ERROR: source geopackage not found: {args.src}")

    CONUS_DIR.mkdir(parents=True, exist_ok=True)
    dst_gpkg = CONUS_DIR / "nhf_conus.gpkg"

    print(f"Preparing CONUS benchmark dataset from {args.src}")
    repaired, fp_ids, lakes_emptied = clone_and_clean(args.src, dst_gpkg)
    print(f"  flowpaths: {len(fp_ids):,} segments")
    forcing = build_forcing(fp_ids, CONUS_DIR / "channel_forcing",
                            args.forcing_hours, args.qlat)

    print("Writing MANIFEST.json ...")
    manifest = {
        "source": str(args.src),
        "source_bytes": args.src.stat().st_size,
        "prepared_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "flowpaths": len(fp_ids),
        "forcing_hours": args.forcing_hours,
        "nts": args.forcing_hours * 12,
        "qlat": args.qlat,
        "repaired": repaired,
        "lakes_emptied": lakes_emptied,
        "gpkg_bytes": dst_gpkg.stat().st_size,
        "forcing_checksums": {p.name: _sha256(p) for p in forcing},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")

    try:
        import geopandas as gpd
        from troute.nhf_preprocess import _validate_flowpaths_channel_params
        fps = gpd.read_file(dst_gpkg, layer="flowpaths", ignore_geometry=True)
        _validate_flowpaths_channel_params(fps)
        print("  validator: cleaned flowpaths pass the MC-kernel guard")
    except ImportError:
        print("  validator: troute not importable, skipped sanity check")

    print(f"\nDone. nts={manifest['nts']}  ->  benchmark/data/conus/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
