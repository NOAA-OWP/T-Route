import argparse
import sqlite3
from pathlib import Path
from typing import Union

import fiona
import geopandas as gpd
import pandas as pd


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


def extract_layers(gpkg_path: str, fp_ids: list[int]) -> dict[str, gpd.GeoDataFrame]:
    """Extract all layers for the given flowpath IDs."""
    fp_set = set(fp_ids)
    layers = fiona.listlayers(gpkg_path)

    # Flowpaths
    if "flowpaths" in layers:
        fp = gpd.read_file(gpkg_path, layer="flowpaths")
        fp = fp[fp["fp_id"].isin(fp_set)]
        print(f"  flowpaths: {len(fp)}")
    else:
        print("  no flowpaths layer, skipping")
        fp = None

    # Divides (div_id == fp_id)
    if "divides" in layers:
        div = gpd.read_file(gpkg_path, layer="divides")
        div = div[div["div_id"].isin(fp_set)]
        print(f"  divides: {len(div)}")
    else:
        print("  no divides layer, skipping")
        div = None

    # Nexus: collect all nex_ids referenced by selected flowpaths
    if "nexus" in layers:
        nex_ids = set(fp["dn_nex_id"].dropna().astype(int))
        nex_ids |= set(fp["up_nex_id"].dropna().astype(int))
        nex = gpd.read_file(gpkg_path, layer="nexus")
        nex = nex[nex["nex_id"].isin(nex_ids)]
        print(f"  nexus: {len(nex)}")
    else:
        print("  no nexus layer, skipping")
        nex = None

    # Reference flowpaths: by div_id
    if "reference_flowpaths" in layers:
        ref_fp = gpd.read_file(gpkg_path, layer="reference_flowpaths")
        ref_fp = ref_fp[ref_fp["div_id"].isin(fp_set)]
        print(f"  reference_flowpaths: {len(ref_fp)} ({ref_fp['virtual_fp_id'].notna().sum()} VFP rows)")
    else:
        print("  no reference_flowpaths layer, skipping")
        ref_fp = None

    # Virtual flowpaths: by virtual_fp_id from reference_flowpaths
    if "virtual_flowpaths" in layers:
        vfp_ids = set(ref_fp.loc[ref_fp["virtual_fp_id"].notna(), "virtual_fp_id"].astype(int))
        vfp = gpd.read_file(gpkg_path, layer="virtual_flowpaths")
        vfp = vfp[vfp["virtual_fp_id"].isin(vfp_ids)]
        print(f"  virtual_flowpaths: {len(vfp)}")
    else:
        print("  no virtual_flowpaths layer, skipping")
        vfp = None

    # Virtual nexus: from dn_virtual_nex_id of selected VFPs
    if "virtual_nexus" in layers:
        vnex_ids = set(vfp["dn_virtual_nex_id"].dropna().astype(int))
        vnex = gpd.read_file(gpkg_path, layer="virtual_nexus")
        vnex = vnex[vnex["virtual_nex_id"].isin(vnex_ids)]
        print(f"  virtual_nexus: {len(vnex)}")
    else:
        print("  no virtual_nexus layer, skipping")
        vnex = None

    # Hydrolocations: by dn_nex_id
    if "hydrolocations" in layers:
        hydroloc = gpd.read_file(gpkg_path, layer="hydrolocations")
        hydroloc = hydroloc[hydroloc["dn_nex_id"].isin(nex_ids)]
        print(f"  hydrolocations: {len(hydroloc)}")
    else:
        print("  no hydrolocations layer, skipping")
        hydroloc = None

    # Waterbodies: by fp_id
    if "waterbodies" in layers:
        wb = gpd.read_file(gpkg_path, layer="waterbodies")
        wb = wb[wb["fp_id"].isin(fp_set)]
        print(f"  waterbodies: {len(wb)}")
    else:
        print("  no waterbodies layer, skipping")
        wb = None

    # Gages: by fp_id
    if "gages" in layers:
        gages = gpd.read_file(gpkg_path, layer="gages")
        gages = gages[gages["fp_id"].isin(fp_set)]
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
