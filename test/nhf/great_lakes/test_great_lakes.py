from pathlib import Path

import pytest
import geopandas as gpd

from ..utils.integration_helpers import (
    assert_peak_bounds,
    delete_outputs,
    has_files,
    run_troute,
    skip_if_not_built,
)
from ..utils.make_configs import Config
from ..utils.make_da import DAConfig, build_da_dataset, write_lake_ontario_outflow
from ..utils.make_forcing import build_forcing_dataset
from ..utils.subset_nhf import get_downstream_fp_ids, extract_layers, write_gpkg

OUTLET_FP_ID = 1288454913281725
START_TIME = "2000-01-01 00:00"
END_TIME = "2000-01-03 00:00"
FORCING_MODE = "constant"
CONSTANT_QLAT = 0.0
USGS_STATION_IDS = ["04127885", "04159130"]
CAN_STATION_IDS = ["02HA013"]
DA_DISCHARGE = 2000.0

PEAK_BOUNDS: dict[int, tuple[float, float]] = {
    1278348162056612: (0.9*2000, 1.1*2000),  # Superior
    1276364270499315: (0.9*2000, 1.1*2000),  # Huron-Michigan
    1286192735893685: (0.9*2000, 1.1*2000),  # Erie
    1287248237297035: (0.9*2000, 1.1*2000),  # Ontario
}


DATA_DIR = Path(__file__).parent / "data"
CFG = Config(DATA_DIR, START_TIME, END_TIME)
CFG.data_assimilation_parameters.reservoir_persistence_usgs = True
CFG.data_assimilation_parameters.reservoir_persistence_greatLake = True
CFG.data_assimilation_parameters.usgs_timeslices_folder = "usgs_timeslice"
CFG.data_assimilation_parameters.canada_timeslices_folder = "canadian_timeslices"
CFG.data_assimilation_parameters.LakeOntario_outflow = "ontario/ontario_outflow.csv"

### Correct NHF versions prior to 1.2.1 ###

FP_LINKAGE = {
    "4800002": 1278348162056612,
    "4800004": 1276364270499315,
    "4800006": 1286192735893685,
    "4800007": 1287248237297035
}
VFP_LINKAGE = {
    "4800002": 1278346877373953,
    "4800004": 1276364270423160,
    "4800006": 1286154743979494,
    "4800007": 1287248166320950
}


def patch_gpkg_lakes(gpkg_path: str) -> None:
    """Patch Great Lakes fp_id and virtual_fp_id values in-place."""
    linkages = {"fp_id": FP_LINKAGE, "virtual_fp_id": VFP_LINKAGE}
    lake_id_list = ", ".join(FP_LINKAGE.keys())  # same keys for both

    # Quick check: skip if all columns already match.
    gl = gpd.read_file(gpkg_path, layer="lakes", where=f"lake_id IN ({lake_id_list})")
    if all(
        row[col] == {int(k): v for k, v in mapping.items()}[int(row["lake_id"])]
        for col, mapping in linkages.items()
        for _, row in gl.iterrows()
    ):
        return

    gdf = gpd.read_file(gpkg_path, layer="lakes")
    dirty = False
    for col, mapping in linkages.items():
        old = gdf[col].copy()
        gdf[col] = gdf["lake_id"].map(mapping).fillna(gdf[col]).astype(gdf[col].dtype)
        changed = ~((gdf[col] == old) | (gdf[col].isna() & old.isna()))
        for lake_id, old_val, new_val in (
            gdf.loc[changed, ["lake_id", col]]
            .assign(old_col=old[changed])[["lake_id", "old_col", col]]
            .itertuples(index=False)
        ):
            print(f"  [patch_gpkg] lake_id {lake_id}: {col} {old_val} -> {new_val}")
        dirty |= changed.any()

    if dirty:
        gdf.to_file(gpkg_path, layer="lakes", driver="GPKG")

### ###

def make_domain_gpkg(nhf_gpkg: Path, out_path: Path):
    """Subset 'mainstem' for great lakes instead of full watershed."""
    patch_gpkg_lakes(nhf_gpkg)
    gl_fps = FP_LINKAGE.values()
    fp_ids = get_downstream_fp_ids(nhf_gpkg, gl_fps, 30)
    layers = extract_layers(nhf_gpkg, fp_ids)
    write_gpkg(layers, out_path)

def setup(source_gpkg: str | Path, refresh: bool = True):
    """Subset the NHF domain and generate forcing for a standard test case."""
    if refresh or not CFG.config_path.exists():
        CFG.write_yaml()

    if refresh or not CFG.domain_path.exists():
        make_domain_gpkg(source_gpkg, CFG.domain_path)

    if refresh or not has_files(CFG.channel_forcing_dir, CFG.qlat_file_pattern):
        build_forcing_dataset(
            FORCING_MODE,
            START_TIME,
            END_TIME,
            CFG.channel_forcing_dir,
            CFG.domain_path,
            runout_period=0,
            constant_qlat=CONSTANT_QLAT,
        )

    if refresh or not has_files(CFG.usgs_timeslices_dir, "*.usgsTimeSlice.ncdf"):
        build_da_dataset(DAConfig(
            da_type="usgs",
            station_ids=USGS_STATION_IDS,
            start_time=START_TIME,
            end_time=END_TIME,
            output_dir=CFG.usgs_timeslices_dir,
            discharge=DA_DISCHARGE,
        ))

    if refresh or not has_files(CFG.canada_timeslices_dir, "*.wscTimeSlice.ncdf"):
        build_da_dataset(DAConfig(
            da_type="canada",
            station_ids=CAN_STATION_IDS,
            start_time=START_TIME,
            end_time=END_TIME,
            output_dir=CFG.canada_timeslices_dir,
            discharge=DA_DISCHARGE,
        ))

    if refresh or not CFG.lake_ontario_outflow_path.exists():
        write_lake_ontario_outflow(CFG.lake_ontario_outflow_path, START_TIME, END_TIME, DA_DISCHARGE)


@pytest.mark.integration
def test_great_lakes():
    skip_if_not_built(CFG)
    delete_outputs(CFG.output_dir)
    run_troute(CFG.config_path)
    assert_peak_bounds(CFG.output_dir, PEAK_BOUNDS)
