import argparse
import sqlite3
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import pandas as pd
import pyogrio


def get_upstream_fp_ids(gpkg_path: str, outlet_fp_id: int) -> list[int]:
    """Recursively find all upstream fp_ids via fp_to_id."""
    conn = sqlite3.connect(gpkg_path)
    rows = conn.execute(
        """
        WITH RECURSIVE upstream(fp_id) AS (
            SELECT fp_id FROM flowpaths WHERE fp_id = ?
            UNION ALL
            SELECT f.fp_id FROM flowpaths f JOIN upstream u ON f.fp_to_id = u.fp_id
        )
        SELECT fp_id FROM upstream
        """,
        (outlet_fp_id,),
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def get_downstream_fp_ids(gpkg_path: str, seed_fp_ids: list[int], max_depth: int) -> list[int]:
    """Recursively find all fp_ids within ``max_depth`` hops downstream of the
    seeds, following ``fp_to_id``. The inverse of :func:`get_upstream_fp_ids`,
    used to carve a small domain below a set of features (e.g. the Great Lakes,
    whose full upstream/downstream basin is far too large to subset by outlet).
    """
    seeds = ",".join(str(int(x)) for x in seed_fp_ids)
    conn = sqlite3.connect(gpkg_path)
    rows = conn.execute(
        f"""
        WITH RECURSIVE downstream(fp_id, depth) AS (
            SELECT fp_id, 0 FROM flowpaths WHERE fp_id IN ({seeds})
            UNION ALL
            SELECT f.fp_to_id, d.depth + 1 FROM flowpaths f
                JOIN downstream d ON f.fp_id = d.fp_id
                WHERE f.fp_to_id IS NOT NULL AND d.depth < ?
        )
        SELECT DISTINCT fp_id FROM downstream
        """,
        (max_depth,),
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def extract_layers(gpkg_path: str, fp_ids: list[int]) -> dict[str, gpd.GeoDataFrame]:
    """Extract all layers for the given flowpath IDs."""
    fp_set = set(fp_ids)
    layers = [name for name, _ in pyogrio.list_layers(gpkg_path)]

    def _in_clause(values) -> Optional[str]:
        cleaned = []
        for value in values:
            if pd.isna(value):
                continue
            cleaned.append(int(value))
        if not cleaned:
            return None
        unique_vals = sorted(set(cleaned))
        return "(" + ",".join(str(v) for v in unique_vals) + ")"

    def _read_layer(layer_name: str, where_sql: str):
        return gpd.read_file(gpkg_path, sql=f'SELECT * FROM "{layer_name}" WHERE {where_sql}')

    nex_ids: set[int] = set()
    vfp_ids: set[int] = set()
    lake_ids: set[int] = set()

    # Flowpaths
    if "flowpaths" in layers:
        fp_clause = _in_clause(fp_set)
        fp = _read_layer("flowpaths", f"fp_id IN {fp_clause}" if fp_clause else "1=0")
        print(f"  flowpaths: {len(fp)}")
    else:
        print("  no flowpaths layer, skipping")
        fp = None

    # Divides (div_id == fp_id)
    if "divides" in layers:
        fp_clause = _in_clause(fp_set)
        div = _read_layer("divides", f"div_id IN {fp_clause}" if fp_clause else "1=0")
        print(f"  divides: {len(div)}")
    else:
        print("  no divides layer, skipping")
        div = None

    # Nexus: collect all nex_ids referenced by selected flowpaths
    if "nexus" in layers:
        if fp is not None and not fp.empty:
            nex_ids = set(fp["dn_nex_id"].dropna().astype(int))
            nex_ids |= set(fp["up_nex_id"].dropna().astype(int))
        nex_clause = _in_clause(nex_ids)
        nex = _read_layer("nexus", f"nex_id IN {nex_clause}" if nex_clause else "1=0")
        print(f"  nexus: {len(nex)}")
    else:
        print("  no nexus layer, skipping")
        nex = None

    # Reference flowpaths: by div_id
    if "reference_flowpaths" in layers:
        fp_clause = _in_clause(fp_set)
        ref_fp = _read_layer("reference_flowpaths", f"div_id IN {fp_clause}" if fp_clause else "1=0")
        print(f"  reference_flowpaths: {len(ref_fp)} ({ref_fp['virtual_fp_id'].notna().sum()} VFP rows)")
    else:
        print("  no reference_flowpaths layer, skipping")
        ref_fp = None

    # Virtual flowpaths: by virtual_fp_id from reference_flowpaths
    if "virtual_flowpaths" in layers:
        if ref_fp is not None and not ref_fp.empty:
            vfp_ids = set(ref_fp.loc[ref_fp["virtual_fp_id"].notna(), "virtual_fp_id"].astype(int))
        vfp_clause = _in_clause(vfp_ids)
        vfp = _read_layer("virtual_flowpaths", f"virtual_fp_id IN {vfp_clause}" if vfp_clause else "1=0")
        print(f"  virtual_flowpaths: {len(vfp)}")
    else:
        print("  no virtual_flowpaths layer, skipping")
        vfp = None

    # Virtual nexus: from dn_virtual_nex_id of selected VFPs
    if "virtual_nexus" in layers:
        vnex_ids = set()
        if vfp is not None and not vfp.empty:
            vnex_ids = set(vfp["dn_virtual_nex_id"].dropna().astype(int))
        vnex_clause = _in_clause(vnex_ids)
        vnex = _read_layer("virtual_nexus", f"virtual_nex_id IN {vnex_clause}" if vnex_clause else "1=0")
        print(f"  virtual_nexus: {len(vnex)}")
    else:
        print("  no virtual_nexus layer, skipping")
        vnex = None

    # Hydrolocations: by dn_nex_id
    if "hydrolocations" in layers:
        nex_clause = _in_clause(nex_ids)
        hydroloc = _read_layer("hydrolocations", f"dn_nex_id IN {nex_clause}" if nex_clause else "1=0")
        print(f"  hydrolocations: {len(hydroloc)}")
    else:
        print("  no hydrolocations layer, skipping")
        hydroloc = None

    # Waterbodies: by fp_id
    if "waterbodies" in layers:
        fp_clause = _in_clause(fp_set)
        wb = _read_layer("waterbodies", f"fp_id IN {fp_clause}" if fp_clause else "1=0")
        print(f"  waterbodies: {len(wb)}")
    else:
        print("  no waterbodies layer, skipping")
        wb = None

    # lakes: by virtual_fp_id
    if "lakes" in layers:
        vfp_clause = _in_clause(vfp_ids)
        lk = _read_layer("lakes", f"virtual_fp_id IN {vfp_clause}" if vfp_clause else "1=0")
        lake_ids = set(lk["nhf_lake_id"].dropna().astype(int).tolist())
        print(f"  lakes: {len(lk)}")
    else:
        print("  no lakes layer, skipping")
        lk = None

    # reservoir_da: by nhf_lake_id
    if "reservoir_da" in layers:
        lake_clause = _in_clause(lake_ids)
        rda = _read_layer("reservoir_da", f"nhf_lake_id IN {lake_clause}" if lake_clause else "1=0")
        print(f"  reservoir_da: {len(rda)}")
    else:
        print("  no reservoir_da layer, skipping")
        rda = None

    # Gages: by fp_id
    if "gages" in layers:
        fp_clause = _in_clause(fp_set)
        gages = _read_layer("gages", f"fp_id IN {fp_clause}" if fp_clause else "1=0")
        print(f"  gages: {len(gages)}")
    else:
        print("  no gages layer, skipping")
        gages = None

    return {
        "flowpaths": fp,
        "divides": div,
        "nexus": nex,
        "reference_flowpaths": ref_fp,
        "virtual_flowpaths": vfp,
        "virtual_nexus": vnex,
        "hydrolocations": hydroloc,
        "waterbodies": wb,
        "lakes": lk,
        "reservoir_da": rda,
        "gages": gages,
    }



def write_gpkg(layers: dict[str, gpd.GeoDataFrame], out_path: Path):
    """Write all layers to a GeoPackage."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing file to avoid appending to old layers
    if out_path.exists():
        out_path.unlink()

    # Separate geo vs non-geo layers
    geo_layers = {}
    non_geo_layers = {}
    for name, gdf in layers.items():
        if gdf is None:
            continue
        if gdf.empty:
            continue
        if "geometry" in gdf.columns and not gdf.geometry.isna().all():
            geo_layers[name] = gdf
        else:
            non_geo_layers[name] = gdf

    # Write geo layers first (first one creates the file)
    first = True
    for name, gdf in geo_layers.items():
        gdf.to_file(out_path, layer=name, driver="GPKG", mode="w" if first else "a")
        first = False

    # Write non-geo layers as attribute tables via sqlite3
    if non_geo_layers:
        conn = sqlite3.connect(str(out_path))
        for name, gdf in non_geo_layers.items():
            df = pd.DataFrame(gdf.drop(columns=["geometry"], errors="ignore"))
            df.to_sql(name, conn, if_exists="replace", index=False)
            conn.execute("""
                INSERT OR IGNORE INTO gpkg_contents (table_name, data_type, identifier)
                VALUES (?, 'attributes', ?)
            """, (name, name))
        conn.commit()
        conn.close()

    print(f"  Wrote {out_path}")


def subset_nhf(source_gpkg: Union[str, Path], out_gpkg: Union[str, Path], outlet_fp_id: int) -> None:
    print(f"Source: {source_gpkg}")
    print(f"Output: {out_gpkg}")
    print()

    print("Step 1: Identifying upstream flowpaths...\n")
    fp_ids = get_upstream_fp_ids(str(source_gpkg), outlet_fp_id)
    print(f"  Found {len(fp_ids)} upstream flowpaths")

    print("Step 2: Extracting layers...\n")
    layers = extract_layers(str(source_gpkg), fp_ids)

    print("Step 3: Writing output GeoPackage...\n")
    write_gpkg(layers, Path(out_gpkg))

def main():
    parser = argparse.ArgumentParser(
        description="Extract all NHF components upstream of a flowpath of interest."
    )

    parser.add_argument(
        "--source-gpkg",
        help="Path to the NHF geopackage that data will be taken from.",
    )

    parser.add_argument(
        "--out-gpkg",
        help="Path to the geopackage that NHF subset data will be written to.",
    )

    parser.add_argument(
        "--outlet-fp-id",
        type=int,
        help="Case directory name",
    )

    args = parser.parse_args()

    subset_nhf(args.source_gpkg, args.out_gpkg, args.outlet_fp_id)


if __name__ == "__main__":
    main()
