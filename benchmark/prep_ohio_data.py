#!/usr/bin/env python3
"""Prepare the Tier A benchmark dataset (``nhf_subset_ohio``) from the CONUS source.

The Tier A workload (``nhf_subset_ohio``) is the topological subgraph
upstream of one tailwater within the NextGen Hydrofabric. By default we
use ``fp_id = 1725641``, an Ohio River basin (VPU 05) outlet whose
upstream contributing area covers 11,327 flowpaths (~80,000 km² total
drainage area). It is a "real-world but tractable" benchmark: big
enough to expose per-segment overhead but single-worker-runnable in
~50 s.

This script derives ``data/domain/nhf_subset_ohio.gpkg`` + 144 hourly
synthesized-forcing CSVs from the same CONUS geopackage that
``prep_conus.py`` consumes. One source covers both tiers.

Usage:
    # Once you have the NextGen Hydrofabric CONUS geopackage:
    python benchmark/prep_ohio_data.py --src /path/to/nhf_1.1.4.gpkg

    # Carve a different tailwater (use any fp_id from the CONUS gpkg):
    python benchmark/prep_ohio_data.py --src /path/to/nhf_1.1.4.gpkg --tailwater 1234567

    # Rebuild even if data already prepared:
    python benchmark/prep_ohio_data.py --src /path/to/nhf_1.1.4.gpkg --force

The data set is gitignored; ``MANIFEST.json`` records sha256 checksums.

The source is the NextGen Hydrofabric v1.1.4 CONUS geopackage used by
the EDFS team for this analysis; reviewers with access can reproduce
the numbers in ``RESULTS.md`` directly.
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
DATA_DIR = BENCH_DIR / "data"
DOMAIN_DIR = DATA_DIR / "domain"
FORCING_DIR = DATA_DIR / "channel_forcing"
MANIFEST = DATA_DIR / "MANIFEST.json"

# Default tailwater fp_id. This is the Ohio River basin outlet whose
# upstream contributing area is the canonical nhf_subset_ohio dataset
# (11,327 flowpaths, ~80,000 km² total drainage area) used for Tier A
# timing.
DEFAULT_TAILWATER_FPID = 1275377122765895

# MC-kernel channel-parameter columns guarded by troute's load-time
# validator; kept in sync with troute.nhf_preprocess._FLOWPATHS_CHANNEL_COLS.
try:  # pragma: no cover, prefer the live list if troute is importable
    from troute.nhf_preprocess import _FLOWPATHS_CHANNEL_COLS as CHANNEL_COLS
except Exception:
    CHANNEL_COLS = (
        "length_km", "n", "slope", "topwdth", "btmwdth",
        "topwdthcc", "ncc", "chslp", "musx", "musk", "mainstem_lp",
    )

FORCING_START = dt.datetime(2000, 1, 1, 0, 0)
_FINITE = "BETWEEN -1e300 AND 1e300"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def carve_upstream_of(src: Path, dst: Path, tailwater_fpid: int
                      ) -> tuple[dict, list[int], list[int]]:
    """Copy ``src``, then keep only rows in the closed upstream subgraph
    rooted at ``tailwater_fpid`` (BFS up the ``fp_to_id -> fp_id`` graph).

    Returns ``(repaired counts, kept fp_ids, kept virtual_fp_ids)``.

    Implementation note: we copy the file (~6 GB) then DELETE non-matching
    rows in sqlite (GeoPackage IS sqlite). Triggers are dropped first so
    the row deletions can run; t-route only reads the result, so dropped
    triggers are harmless.
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

    # BFS upstream from the tailwater. For each frontier fp_id, find all
    # fp_ids whose fp_to_id equals it (i.e., the immediate upstream
    # neighbours) and add the new ones to the kept set.
    print(f"  BFS upstream from fp_id={tailwater_fpid:,} ...")
    if not cur.execute(
        "SELECT 1 FROM flowpaths WHERE fp_id = ? LIMIT 1",
        (tailwater_fpid,),
    ).fetchone():
        sys.exit(f"ERROR: tailwater fp_id={tailwater_fpid} not in source flowpaths layer")

    cur.execute("CREATE TEMP TABLE keep_fp(fp_id INTEGER PRIMARY KEY)")
    cur.execute("INSERT INTO keep_fp VALUES (?)", (tailwater_fpid,))
    frontier = [tailwater_fpid]
    while frontier:
        # Find immediate upstream neighbours: fp_id WHERE fp_to_id IN frontier
        # and not already kept. Batch in chunks to stay under sqlite's
        # variable limit.
        new_ids: list[int] = []
        for chunk_start in range(0, len(frontier), 500):
            chunk = frontier[chunk_start: chunk_start + 500]
            placeholders = ",".join("?" * len(chunk))
            new_ids.extend(r[0] for r in cur.execute(
                f"SELECT fp_id FROM flowpaths "
                f"WHERE fp_to_id IN ({placeholders}) "
                f"AND fp_id NOT IN (SELECT fp_id FROM keep_fp)",
                chunk,
            ))
        if not new_ids:
            break
        cur.executemany("INSERT OR IGNORE INTO keep_fp VALUES (?)",
                        [(i,) for i in new_ids])
        frontier = new_ids

    kept_fp = cur.execute("SELECT count(*) FROM keep_fp").fetchone()[0]
    print(f"  upstream subgraph: {kept_fp:,} flowpaths")

    # Delete every fp_id row not in keep_fp from each layer that has an
    # fp_id column. Layers without fp_id (gages, lakes, hydrolocations) are
    # left as-is; t-route handles missing rows fine.
    cur.execute("CREATE INDEX keep_fp_idx ON keep_fp(fp_id)")
    deleted_per_table: dict[str, int] = {}

    def _filter(tbl: str, where_sql: str) -> None:
        if not cur.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (tbl,),
        ).fetchone():
            return
        before = cur.execute(f'SELECT count(*) FROM "{tbl}"').fetchone()[0]
        cur.execute(f'DELETE FROM "{tbl}" WHERE {where_sql}')
        after = cur.execute(f'SELECT count(*) FROM "{tbl}"').fetchone()[0]
        deleted_per_table[tbl] = before - after
        print(f"  {tbl}: kept {after:,} (deleted {before - after:,})")

    # 1. Layers keyed directly on fp_id.
    _filter("flowpaths", "fp_id NOT IN (SELECT fp_id FROM keep_fp)")
    # reference_flowpaths needs BOTH fp_id and div_id inside the subset:
    # rows whose div_id points to a physical flowpath we dropped would
    # otherwise force the forcing CSV to cover ~390 k extra divs that
    # are not in our domain. flowpaths.fp_id == flowpaths.div_id in NHF
    # (1:1 in v1.1.4), so the kept fp_id set is also the kept div_id set.
    _filter("reference_flowpaths",
            "fp_id NOT IN (SELECT fp_id FROM keep_fp) "
            "OR div_id NOT IN (SELECT fp_id FROM keep_fp)")
    _filter("gages", "fp_id NOT IN (SELECT fp_id FROM keep_fp)")

    # Empty the lakes layer entirely (same as prep_conus.py): the
    # nhf_1.1.4 lakes layer has NaN fp_id values that crash t-route's
    # NHF waterbody preprocessing. The Tier A workload is intentionally
    # lakes-free anyway (all segments routed as MC channels), so this
    # keeps the two tiers consistent.
    if cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='lakes'"
    ).fetchone():
        before = cur.execute("SELECT count(*) FROM lakes").fetchone()[0]
        cur.execute("DELETE FROM lakes")
        deleted_per_table["lakes"] = before
        print(f"  lakes: emptied ({before} rows; routed as MC channels)")

    # Similarly, drop reservoir_da table for NHF v1.2.1+
    if cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='reservoir_da'"
    ).fetchone():
        before = cur.execute("SELECT count(*) FROM reservoir_da").fetchone()[0]
        cur.execute("DELETE FROM reservoir_da")
        deleted_per_table["reservoir_da"] = before
        print("  reservoir_da: emptied")

    # 2. divides joins through div_id. Build keep_div from flowpaths.div_id.
    cur.execute(
        "CREATE TEMP TABLE keep_div AS "
        "SELECT DISTINCT div_id FROM flowpaths WHERE div_id IS NOT NULL"
    )
    cur.execute("CREATE INDEX keep_div_idx ON keep_div(div_id)")
    _filter("divides", "div_id NOT IN (SELECT div_id FROM keep_div)")

    # 3. nexus joins through dn_fp_id (and the in_fp_id refs from kept
    #    flowpaths' dn_nex_id / up_nex_id). Build keep_nex from both
    #    directions so an outlet-flowpath's downstream nexus is kept.
    cur.execute(
        "CREATE TEMP TABLE keep_nex(nex_id INTEGER PRIMARY KEY)"
    )
    cur.execute(
        "INSERT OR IGNORE INTO keep_nex "
        "SELECT DISTINCT dn_nex_id FROM flowpaths WHERE dn_nex_id IS NOT NULL"
    )
    cur.execute(
        "INSERT OR IGNORE INTO keep_nex "
        "SELECT DISTINCT up_nex_id FROM flowpaths WHERE up_nex_id IS NOT NULL"
    )
    _filter("nexus", "nex_id NOT IN (SELECT nex_id FROM keep_nex)")

    # 4. virtual_flowpaths joins to physical flowpaths via reference_flowpaths.
    #    Keep VFPs that any kept reference_flowpaths row points to.
    cur.execute(
        "CREATE TEMP TABLE keep_vfp(virtual_fp_id INTEGER PRIMARY KEY)"
    )
    cur.execute(
        "INSERT OR IGNORE INTO keep_vfp "
        "SELECT DISTINCT virtual_fp_id FROM reference_flowpaths "
        "WHERE virtual_fp_id IS NOT NULL"
    )
    _filter("virtual_flowpaths",
            "virtual_fp_id NOT IN (SELECT virtual_fp_id FROM keep_vfp)")

    # 5. virtual_nexus joins through kept virtual_flowpaths.{dn,up}_virtual_nex_id.
    cur.execute(
        "CREATE TEMP TABLE keep_vnex(virtual_nex_id INTEGER PRIMARY KEY)"
    )
    cur.execute(
        "INSERT OR IGNORE INTO keep_vnex "
        "SELECT DISTINCT dn_virtual_nex_id FROM virtual_flowpaths "
        "WHERE dn_virtual_nex_id IS NOT NULL"
    )
    cur.execute(
        "INSERT OR IGNORE INTO keep_vnex "
        "SELECT DISTINCT up_virtual_nex_id FROM virtual_flowpaths "
        "WHERE up_virtual_nex_id IS NOT NULL"
    )
    _filter("virtual_nexus",
            "virtual_nex_id NOT IN (SELECT virtual_nex_id FROM keep_vnex)")

    # hydrolocations have no fp_id-style key; t-route reads them whole and
    # filters internally, so we leave the layer untouched.
    db.commit()

    vfp_ids = [r[0] for r in cur.execute(
        "SELECT virtual_fp_id FROM virtual_flowpaths"
    )]

    # Repair any non-finite MC-kernel channel parameters in the kept rows.
    fp_cols = {r[1] for r in cur.execute('PRAGMA table_info(flowpaths)')}
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
        print("  flowpaths: no non-finite channel parameters in subset")

    fp_ids_kept = [r[0] for r in cur.execute("SELECT fp_id FROM flowpaths")]

    # Collect the full set of div_ids that t-route will look up against
    # the forcing CSVs. NHF._build_div_weighting_matrix joins
    # virtual_flowpaths to reference_flowpaths on virtual_fp_id and reads
    # reference_flowpaths.div_id; the forcing CSV must cover every such
    # div_id, which includes the physical-flowpath div_ids plus the
    # "sub-divide" div_ids that virtual flowpaths reference. The union is
    # what we synthesize forcing for.
    forcing_ids = sorted({r[0] for r in cur.execute(
        "SELECT DISTINCT div_id FROM reference_flowpaths WHERE div_id IS NOT NULL"
    )} | set(fp_ids_kept))

    cur.execute("VACUUM")
    db.commit()
    db.close()
    return (
        {"deleted": deleted_per_table, "repaired": repaired},
        fp_ids_kept,
        vfp_ids,
        forcing_ids,
    )


def synth_forcing(fp_ids: list[int], dst_dir: Path, hours: int,
                  qlat: float) -> list[Path]:
    """Write ``hours`` hourly CHRTOUT CSVs with constant q_lateral.

    The MC kernel is forcing-shape-sensitive, not forcing-value-sensitive,
    for performance purposes: identical array shapes + dtypes produce
    representative timing. Synthesizing keeps the dataset self-contained
    (no external CHRTOUT dependency).
    """
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
                    help="NextGen Hydrofabric CONUS geopackage "
                         "(same file prep_conus.py consumes)")
    ap.add_argument("--tailwater", type=int, default=DEFAULT_TAILWATER_FPID,
                    help=f"fp_id of the tailwater to carve upstream from "
                         f"(default: {DEFAULT_TAILWATER_FPID}, "
                         f"Ohio River basin, ~11 k flowpaths)")
    ap.add_argument("--forcing-hours", type=int, default=144,
                    help="hourly forcing files to generate "
                         "(default: 144; nts = hours * 12 = 1728)")
    ap.add_argument("--qlat", type=float, default=0.05,
                    help="synthetic constant lateral inflow per segment "
                         "(default: 0.05)")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if data already prepared")
    args = ap.parse_args()

    if MANIFEST.exists() and not args.force:
        print(f"Dataset already prepared ({MANIFEST}). Use --force to rebuild.")
        return 0
    if not args.src.is_file():
        sys.exit(f"ERROR: source geopackage not found: {args.src}\n"
                 f"Provide the path to the NextGen Hydrofabric v1.1.4 "
                 f"CONUS geopackage (the same file prep_conus.py consumes).")

    DOMAIN_DIR.mkdir(parents=True, exist_ok=True)
    dst_gpkg = DOMAIN_DIR / "nhf_subset_ohio.gpkg"

    print(f"Preparing Tier A dataset (upstream of fp_id={args.tailwater:,}) "
          f"from {args.src}")
    carve_info, fp_ids, vfp_ids, forcing_ids = carve_upstream_of(
        args.src, dst_gpkg, args.tailwater,
    )
    print(f"  flowpaths in subset: {len(fp_ids):,}")
    print(f"  virtual_flowpaths:   {len(vfp_ids):,}")
    print(f"  forcing div_ids:     {len(forcing_ids):,}  "
          f"(union of flowpaths.div_id and reference_flowpaths.div_id)")
    forcing = synth_forcing(
        forcing_ids, FORCING_DIR, args.forcing_hours, args.qlat,
    )

    print("Writing MANIFEST.json ...")
    manifest = {
        "source": str(args.src),
        "source_bytes": args.src.stat().st_size,
        "tailwater_fp_id": args.tailwater,
        "prepared_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "flowpaths": len(fp_ids),
        "virtual_flowpaths": len(vfp_ids),
        "forcing_hours": args.forcing_hours,
        "nts": args.forcing_hours * 12,
        "qlat": args.qlat,
        "carve": carve_info,
        "gpkg_bytes": dst_gpkg.stat().st_size,
        "checksums": {"domain/nhf_subset_ohio.gpkg": _sha256(dst_gpkg)},
        "forcing_checksums": {p.name: _sha256(p) for p in forcing[:4]},
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

    print(f"\nDone. nts={manifest['nts']}  ->  benchmark/data/")
    print(
        "\nNOTE: synthesized forcing means routing output values will not\n"
        "      match a 'real-forcing' golden. Save a new correctness\n"
        "      reference once with `bench_e2e.py --save-golden`."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
