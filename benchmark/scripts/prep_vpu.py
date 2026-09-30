#!/usr/bin/env python3
"""Carve a whole VPU out of the NGWPC Hydrofabric into a t-route-readable gpkg.

Unlike prep_ohio_data.py, which keeps the upstream subgraph of ONE tailwater, this
keeps every flowpath in a VPU (a whole hydrologic region, many tailwaters). VPU 01
(North Atlantic) is the streamflow-DA forecast demo domain: ~24,862 reaches, ~906
gages, ~834 lakes carrying all reservoir-DA types.

The mechanics mirror prep_ohio_data.py's proven sqlite cascade (copy the gpkg, delete
rows outside the kept set, layer by layer) so t-route reads the result unchanged. The
only real differences: the kept set is `vpu_id = <VPU>` rather than a tailwater BFS,
and the lake / reservoir_da layers are KEPT and filtered, since t-route reads
reservoir DA from them, rather than emptied.

It also writes the exact divide-id list the forcing must cover, which is both the
coverage the NextGen delivery is checked against and the set t-route looks up in the
qlateral input.

    python benchmark/scripts/prep_vpu.py --src /path/to/nhf.gpkg --vpu 01

The dataset is gitignored; only the divide-id list and a small manifest are small
enough to keep.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sqlite3
import sys
from pathlib import Path

import numpy as np

BENCH_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BENCH_DIR / "data"

try:
    from troute.nhf_preprocess import _FLOWPATHS_CHANNEL_COLS as CHANNEL_COLS
except Exception:
    CHANNEL_COLS = (
        "length_km", "n", "slope", "topwdth", "btmwdth",
        "topwdthcc", "ncc", "chslp", "musx", "musk", "mainstem_lp",
    )
_FINITE = "BETWEEN -1e300 AND 1e300"


def _filter(cur, tbl: str, where_keep_sql: str, deleted: dict) -> None:
    """Delete rows NOT matching where_keep_sql from tbl, if the table exists."""
    if not cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (tbl,)
    ).fetchone():
        return
    before = cur.execute(f'SELECT count(*) FROM "{tbl}"').fetchone()[0]
    cur.execute(f'DELETE FROM "{tbl}" WHERE NOT ({where_keep_sql})')
    after = cur.execute(f'SELECT count(*) FROM "{tbl}"').fetchone()[0]
    deleted[tbl] = before - after
    print(f"  {tbl}: kept {after:,} (deleted {before - after:,})")


def carve_vpu(src: Path, dst: Path, vpu: str):
    print(f"  copying {src.stat().st_size / 1e9:.1f} GB geopackage ...")
    shutil.copyfile(src, dst)
    db = sqlite3.connect(dst)
    cur = db.cursor()
    for (name,) in [r for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='trigger'"
    )]:
        cur.execute(f'DROP TRIGGER IF EXISTS "{name}"')

    # keep_fp = every flowpath in the VPU. This is the one line that differs from the
    # upstream-of-a-tailwater carve; everything downstream is the same reference cascade.
    cur.execute("CREATE TEMP TABLE keep_fp(fp_id INTEGER PRIMARY KEY)")
    cur.execute("INSERT OR IGNORE INTO keep_fp "
                "SELECT fp_id FROM flowpaths WHERE vpu_id=? AND fp_id IS NOT NULL", (vpu,))
    n_fp = cur.execute("SELECT count(*) FROM keep_fp").fetchone()[0]
    if not n_fp:
        sys.exit(f"ERROR: no flowpaths with vpu_id={vpu!r} in {src.name}")
    print(f"  VPU {vpu}: {n_fp:,} flowpaths")

    deleted: dict[str, int] = {}
    _filter(cur, "flowpaths", "fp_id IN (SELECT fp_id FROM keep_fp)", deleted)
    _filter(cur, "reference_flowpaths",
            "fp_id IN (SELECT fp_id FROM keep_fp) "
            "OR div_id IN (SELECT fp_id FROM keep_fp)", deleted)
    _filter(cur, "gages", "fp_id IN (SELECT fp_id FROM keep_fp)", deleted)

    # divides via reference_flowpaths.div_id (the physical + sub-divide ids).
    cur.execute("CREATE TEMP TABLE keep_div AS "
                "SELECT DISTINCT div_id FROM reference_flowpaths WHERE div_id IS NOT NULL")
    cur.execute("INSERT OR IGNORE INTO keep_div "
                "SELECT fp_id FROM keep_fp")
    cur.execute("CREATE INDEX keep_div_idx ON keep_div(div_id)")
    _filter(cur, "divides", "div_id IN (SELECT div_id FROM keep_div)", deleted)

    # nexus via kept flowpaths' up/dn nexus ids (join, not vpu_id, so a boundary nexus a
    # kept flowpath drains to is not dropped).
    cur.execute("CREATE TEMP TABLE keep_nex(nex_id INTEGER PRIMARY KEY)")
    for col in ("dn_nex_id", "up_nex_id"):
        cur.execute(f"INSERT OR IGNORE INTO keep_nex "
                    f"SELECT DISTINCT {col} FROM flowpaths WHERE {col} IS NOT NULL")
    _filter(cur, "nexus", "nex_id IN (SELECT nex_id FROM keep_nex)", deleted)

    # virtual_flowpaths via reference_flowpaths.virtual_fp_id (only kept refs survive).
    cur.execute("CREATE TEMP TABLE keep_vfp(virtual_fp_id INTEGER PRIMARY KEY)")
    cur.execute("INSERT OR IGNORE INTO keep_vfp "
                "SELECT DISTINCT virtual_fp_id FROM reference_flowpaths "
                "WHERE virtual_fp_id IS NOT NULL")
    _filter(cur, "virtual_flowpaths",
            "virtual_fp_id IN (SELECT virtual_fp_id FROM keep_vfp)", deleted)

    cur.execute("CREATE TEMP TABLE keep_vnex(virtual_nex_id INTEGER PRIMARY KEY)")
    for col in ("dn_virtual_nex_id", "up_virtual_nex_id"):
        cur.execute(f"INSERT OR IGNORE INTO keep_vnex "
                    f"SELECT DISTINCT {col} FROM virtual_flowpaths WHERE {col} IS NOT NULL")
    _filter(cur, "virtual_nexus",
            "virtual_nex_id IN (SELECT virtual_nex_id FROM keep_vnex)", deleted)

    # Lakes and reservoir DA are KEPT (t-route reads reservoir DA from them), filtered
    # to lakes on kept virtual flowpaths. reservoir_da / lakes_polygons / lake_vfp_crosswalk
    # then follow the kept lakes by nhf_lake_id.
    _filter(cur, "lakes",
            "virtual_fp_id IN (SELECT virtual_fp_id FROM keep_vfp)", deleted)
    cur.execute("CREATE TEMP TABLE keep_lake(nhf_lake_id INTEGER PRIMARY KEY)")
    cur.execute("INSERT OR IGNORE INTO keep_lake "
                "SELECT DISTINCT nhf_lake_id FROM lakes WHERE nhf_lake_id IS NOT NULL")
    for tbl in ("reservoir_da", "lakes_polygons", "lake_vfp_crosswalk"):
        _filter(cur, tbl, "nhf_lake_id IN (SELECT nhf_lake_id FROM keep_lake)", deleted)

    # nhd is a 2.7 M-row reference layer t-route never reads; empty it so the subset is small.
    if cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='nhd'").fetchone():
        before = cur.execute("SELECT count(*) FROM nhd").fetchone()[0]
        cur.execute("DELETE FROM nhd")
        deleted["nhd"] = before
        print(f"  nhd: emptied ({before:,} rows; not read by t-route)")
    db.commit()

    # Repair non-finite MC channel params in the kept flowpaths (same guard as Ohio).
    fp_cols = {r[1] for r in cur.execute("PRAGMA table_info(flowpaths)")}
    repaired: dict[str, int] = {}
    for col in CHANNEL_COLS:
        if col not in fp_cols:
            continue
        bad = cur.execute(f'SELECT count(*) FROM flowpaths '
                          f'WHERE "{col}" IS NULL OR "{col}" NOT {_FINITE}').fetchone()[0]
        if bad:
            vals = [r[0] for r in cur.execute(
                f'SELECT "{col}" FROM flowpaths WHERE "{col}" IS NOT NULL AND "{col}" {_FINITE}')]
            median = float(np.median(vals))
            cur.execute(f'UPDATE flowpaths SET "{col}"=? '
                        f'WHERE "{col}" IS NULL OR "{col}" NOT {_FINITE}', (median,))
            repaired[col] = bad
            print(f"  flowpaths.{col}: repaired {bad} non-finite -> median {median:.6g}")
    db.commit()

    # The divide-id set the forcing must cover: reference_flowpaths.div_id plus the kept
    # flowpath ids. This is what the NextGen delivery is checked against.
    forcing_ids = sorted({r[0] for r in cur.execute(
        "SELECT DISTINCT div_id FROM reference_flowpaths WHERE div_id IS NOT NULL")}
        | {r[0] for r in cur.execute("SELECT fp_id FROM keep_fp")})
    lakes_kept = cur.execute("SELECT count(*) FROM lakes").fetchone()[0]
    rda_kept = cur.execute("SELECT count(*) FROM reservoir_da").fetchone()[0]
    gages_kept = cur.execute("SELECT count(*) FROM gages").fetchone()[0]

    cur.execute("VACUUM")
    db.commit()
    db.close()
    return {"deleted": deleted, "repaired": repaired, "forcing_ids": forcing_ids,
            "flowpaths": n_fp, "lakes": lakes_kept, "reservoir_da": rda_kept,
            "gages": gages_kept}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="full NHF CONUS gpkg")
    ap.add_argument("--vpu", default="01", help="VPU id to carve (default 01)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output gpkg (default data/vpu<VPU>/nhf_vpu<VPU>.gpkg)")
    ap.add_argument("--force", action="store_true", help="rebuild if the gpkg exists")
    args = ap.parse_args()

    if not args.src.exists():
        sys.exit(f"source not found: {args.src}")
    out = args.out or (DATA_DIR / f"vpu{args.vpu}" / f"nhf_vpu{args.vpu}.gpkg")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.force:
        sys.exit(f"{out} exists; pass --force to rebuild")

    print(f"Carving VPU {args.vpu} from {args.src} -> {out}")
    info = carve_vpu(args.src, out, args.vpu)

    ids_file = out.parent / f"vpu{args.vpu}_divide_ids.txt"
    ids_file.write_text("\n".join(str(i) for i in info["forcing_ids"]) + "\n")
    manifest = out.parent / "MANIFEST.json"
    manifest.write_text(json.dumps({
        "vpu": args.vpu,
        "source": str(args.src),
        "built_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "flowpaths": info["flowpaths"],
        "gages": info["gages"],
        "lakes": info["lakes"],
        "reservoir_da": info["reservoir_da"],
        "n_forcing_divides": len(info["forcing_ids"]),
        "divide_ids_file": ids_file.name,
    }, indent=2) + "\n")
    print(f"\n  gpkg   -> {out}  ({out.stat().st_size / 1e6:.0f} MB)")
    print(f"  divide ids ({len(info['forcing_ids']):,}) -> {ids_file}")
    print(f"  {info['flowpaths']:,} flowpaths, {info['gages']:,} gages, "
          f"{info['lakes']:,} lakes, {info['reservoir_da']:,} reservoir_da")
    print("  Give the divide-id list to the forcing provider so every routed divide is covered.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
