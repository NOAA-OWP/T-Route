"""
Extract Conecuh River Basin subset from NHF 1.1.2 GeoPackage.

Subsets the basin upstream of fp_id=1594183 (USGS Gage 02374250),
then updates forcing CSVs with new div_id values.

Usage:
    cd test/nhf/conecuh_case
    uv run python extract_conecuh_from_nhf.py
"""

import sqlite3
from pathlib import Path

import geopandas as gpd
import pandas as pd

# Paths
SCRIPT_DIR = Path(__file__).parent
NHF_GPKG = SCRIPT_DIR.parent.parent.parent / "nhf_1.1.2_no_div_attr.gpkg"
OLD_GPKG = SCRIPT_DIR / "domain" / "02374250.gpkg"
OUT_GPKG = SCRIPT_DIR / "domain" / "02374250.gpkg"
FORCING_DIR = SCRIPT_DIR / "channel_forcing"

OUTLET_FP_ID = 1594183


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

    # Flowpaths
    fp = gpd.read_file(gpkg_path, layer="flowpaths")
    fp = fp[fp["fp_id"].isin(fp_set)]
    print(f"  flowpaths: {len(fp)}")

    # Divides (div_id == fp_id)
    div = gpd.read_file(gpkg_path, layer="divides")
    div = div[div["div_id"].isin(fp_set)]
    print(f"  divides: {len(div)}")

    # Nexus: collect all nex_ids referenced by selected flowpaths
    nex_ids = set(fp["dn_nex_id"].dropna().astype(int))
    nex_ids |= set(fp["up_nex_id"].dropna().astype(int))
    nex = gpd.read_file(gpkg_path, layer="nexus")
    nex = nex[nex["nex_id"].isin(nex_ids)]
    print(f"  nexus: {len(nex)}")

    # Reference flowpaths: by div_id
    ref_fp = gpd.read_file(gpkg_path, layer="reference_flowpaths")
    ref_fp = ref_fp[ref_fp["div_id"].isin(fp_set)]
    print(f"  reference_flowpaths: {len(ref_fp)} ({ref_fp['virtual_fp_id'].notna().sum()} VFP rows)")

    # Virtual flowpaths: by virtual_fp_id from reference_flowpaths
    vfp_ids = set(ref_fp.loc[ref_fp["virtual_fp_id"].notna(), "virtual_fp_id"].astype(int))
    vfp = gpd.read_file(gpkg_path, layer="virtual_flowpaths")
    vfp = vfp[vfp["virtual_fp_id"].isin(vfp_ids)]
    print(f"  virtual_flowpaths: {len(vfp)}")

    # Virtual nexus: from dn_virtual_nex_id of selected VFPs
    vnex_ids = set(vfp["dn_virtual_nex_id"].dropna().astype(int))
    vnex = gpd.read_file(gpkg_path, layer="virtual_nexus")
    vnex = vnex[vnex["virtual_nex_id"].isin(vnex_ids)]
    print(f"  virtual_nexus: {len(vnex)}")

    # Hydrolocations: by dn_nex_id
    hydroloc = gpd.read_file(gpkg_path, layer="hydrolocations")
    hydroloc = hydroloc[hydroloc["dn_nex_id"].isin(nex_ids)]
    print(f"  hydrolocations: {len(hydroloc)}")

    # Waterbodies: by fp_id
    wb = gpd.read_file(gpkg_path, layer="waterbodies")
    wb = wb[wb["fp_id"].isin(fp_set)]
    print(f"  waterbodies: {len(wb)}")

    # Gages: by fp_id
    gages = gpd.read_file(gpkg_path, layer="gages")
    gages = gages[gages["fp_id"].isin(fp_set)]
    print(f"  gages: {len(gages)}")

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


def build_div_id_mapping(old_gpkg: str, new_fp: gpd.GeoDataFrame) -> dict[int, int]:
    """Build old_div_id → new_div_id mapping via (total_da_sqkm, length_km) match."""
    old_fp = gpd.read_file(old_gpkg, layer="flowpaths")

    # Match on (total_da_sqkm, length_km) tuple — exact matches confirmed
    old_fp["_key"] = list(zip(old_fp["total_da_sqkm"].round(6), old_fp["length_km"].round(6)))
    new_fp = new_fp.copy()
    new_fp["_key"] = list(zip(new_fp["total_da_sqkm"].round(6), new_fp["length_km"].round(6)))

    old_key_to_div = dict(zip(old_fp["_key"], old_fp["div_id"].astype(int)))
    new_key_to_div = dict(zip(new_fp["_key"], new_fp["div_id"].astype(int)))

    mapping = {}  # old_div_id → new_div_id
    matched = 0
    for key, old_div in old_key_to_div.items():
        if key in new_key_to_div:
            mapping[old_div] = new_key_to_div[key]
            matched += 1

    print(f"  Matched {matched}/{len(old_key_to_div)} flowpaths by (total_da_sqkm, length_km)")
    if matched < len(old_key_to_div):
        print(f"  WARNING: {len(old_key_to_div) - matched} unmatched flowpaths")

    return mapping


def update_forcing_files(forcing_dir: Path, div_mapping: dict[int, int]):
    """Update feature_id values in forcing CSVs using old→new div_id mapping."""
    csv_files = sorted(forcing_dir.glob("*.csv"))
    print(f"  Updating {len(csv_files)} forcing files...")

    updated = 0
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        if "feature_id" not in df.columns:
            continue
        df["feature_id"] = df["feature_id"].map(div_mapping).fillna(df["feature_id"]).astype(int)
        df.to_csv(csv_path, index=False)
        updated += 1

    print(f"  Updated {updated} files")


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


def main():
    print(f"Source: {NHF_GPKG}")
    print(f"Output: {OUT_GPKG}")
    print()

    # Step 1: Find upstream features
    print("Step 1: Identifying upstream flowpaths...")
    fp_ids = get_upstream_fp_ids(str(NHF_GPKG), OUTLET_FP_ID)
    print(f"  Found {len(fp_ids)} upstream flowpaths")
    print()

    # Step 2: Extract layers
    print("Step 2: Extracting layers...")
    layers = extract_layers(str(NHF_GPKG), fp_ids)
    print()

    # Step 3: Build div_id mapping and update forcing
    print("Step 3: Building old→new div_id mapping...")
    div_mapping = build_div_id_mapping(str(OLD_GPKG), layers["flowpaths"])
    print()

    print("Step 4: Updating forcing files...")
    update_forcing_files(FORCING_DIR, div_mapping)
    print()

    # Step 5: Write output
    print("Step 5: Writing output GeoPackage...")
    write_gpkg(layers, OUT_GPKG)
    print()

    # Verification
    print("Verification:")
    print(f"  flowpaths: {len(layers['flowpaths'])} (expect 1426)")
    print(f"  divides: {len(layers['divides'])} (expect 1426)")
    print(f"  nexus: {len(layers['nexus'])} (expect 769)")
    print(f"  reference_flowpaths: {len(layers['reference_flowpaths'])} (expect 3527)")
    print(f"  virtual_flowpaths: {len(layers['virtual_flowpaths'])} (expect 882)")
    print(f"  virtual_nexus: {len(layers['virtual_nexus'])} (expect 882)")
    print()

    # Spot-check: verify outlet div_id in forcing
    sample_csv = next(FORCING_DIR.glob("*.csv"))
    sample = pd.read_csv(sample_csv)
    new_outlet_div = div_mapping.get(3490271)  # old outlet div_id
    if new_outlet_div and new_outlet_div in sample["feature_id"].values:
        print(f"  Forcing spot-check: old 3490271 → new {new_outlet_div} FOUND in {sample_csv.name}")
    else:
        print(f"  Forcing spot-check: outlet mapping not confirmed (old 3490271 → {new_outlet_div})")


if __name__ == "__main__":
    main()
