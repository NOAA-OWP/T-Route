from pathlib import Path
import sqlite3

import pytest


from ..utils.integration_helpers import (
    assert_lakeout,
    assert_peak_bounds,
    delete_outputs,
    has_files,
    run_troute,
    skip_if_not_built,
)
from ..utils.make_configs import Config, DataAssimilationParameters
from ..utils.make_da import DAConfig, build_da_dataset
from ..utils.make_forcing import build_forcing_dataset
from ..utils.subset_nhf import subset_nhf

OUTLET_FP_ID = 1276182780176988
START_TIME = "2020-01-01 00:00"
END_TIME = "2020-01-01 01:00"
FORCING_MODE = "constant"
CONSTANT_QLAT = 100

# Reservoir DA configuration from the legacy setup.py
RESERVOIR_TYPE_MOD = {
    1276675272336236: 2,   # USGS
    1276673989193624: 3,   # USACE
    1276672890005227: 4,   # RFC
    1276185235011805: 7,   # USBR
}
RESERVOIR_FLOW_VALUES = {
    1276675272336236: 1.0,
    1276673989193624: 1.0,
    1276672890005227: 12345,
    1276185235011805: 1.0,
}
RESERVOIR_SITE_NOS = {
    1276675272336236: "USGS00000000002",
    1276673989193624: "USAC00000000003",
    1276672890005227: "RFC000000000004",
    1276185235011805: "USBR00000000007",
}
RESERVOIR_DN_FP = {
    1276675272336236: 1276674107160768,
    1276673989193624: 1276673989391626,
    1276672890005227: 1276672844213843,
    1276185235011805: 1276185303862487,
}
PEAK_BOUNDS: dict[int, tuple[float, float]] = {
    dn_fp: (0.9*expected, 1.1*expected)
    for lake_id, dn_fp in RESERVOIR_DN_FP.items()
    for expected in [RESERVOIR_FLOW_VALUES[lake_id]]
}

# Expected mean outflow bounds (value ± 5% or ±1.0) for lakeout feature_ids.
LAKEOUT_OUTFLOW_BOUNDS: dict[int, tuple[float, float]] = {
    lake_id: (0.9*expected, 1.1*expected)
    for lake_id, expected in RESERVOIR_FLOW_VALUES.items()
}

# Group reservoirs by DA type for build_da_dataset calls.
_DA_GROUPS: dict[str, list[tuple[int, str, float]]] = {}
for _lid, _dtype in RESERVOIR_TYPE_MOD.items():
    _da_name = {2: "usgs", 3: "usace", 4: "rfc", 7: "usbr"}[_dtype]
    _DA_GROUPS.setdefault(_da_name, []).append(
        (_lid, RESERVOIR_SITE_NOS[_lid], RESERVOIR_FLOW_VALUES[_lid])
    )

DATA_DIR = Path(__file__).parent / "data"
DA_PARAMS = DataAssimilationParameters(
    reservoir_persistence_usgs=True,
    reservoir_persistence_usace=True,
    reservoir_persistence_usbr=True,
    reservoir_rfc_forecasts=True,
    timeslice_lookback_hours=1,
    usgs_timeslices_folder="reservoir_da/usgs_da",
    usace_timeslices_folder="reservoir_da/usace_da",
    usbr_timeslices_folder="reservoir_da/usbr_da",
    reservoir_rfc_forecasts_time_series_path="reservoir_da/rfc_da",
    reservoir_rfc_forecasts_lookback_hours=28,
    reservoir_rfc_forecasts_offset_hours=0,
    reservoir_rfc_forecast_persist_days=11,
)
CFG = Config(DATA_DIR, START_TIME, END_TIME, data_assimilation_parameters=DA_PARAMS, lakeout_output="lakeout")

def modify_lakes_table(gpkg_path: Path) -> None:
    """Update da_type and assign synthetic site_nos in the reservoir_da table."""
    conn = sqlite3.connect(gpkg_path)
    cur = conn.cursor()
    for lake_id, da_type in RESERVOIR_TYPE_MOD.items():
        site_no = RESERVOIR_SITE_NOS[lake_id]
        cur.execute(
            "UPDATE reservoir_da SET da_type=?, site_no=? WHERE nhf_lake_id=?",
            (da_type, site_no, lake_id),
        )
    conn.commit()
    conn.close()


def setup(source_gpkg: str | Path, refresh: bool = True):
    """Subset the NHF domain and generate forcing/DA for a standard test case."""
    if refresh or not CFG.config_path.exists():
        CFG.write_yaml()

    if refresh or not CFG.domain_path.exists():
        subset_nhf(source_gpkg, CFG.domain_path, OUTLET_FP_ID)
        modify_lakes_table(CFG.domain_path)

    if refresh or not has_files(CFG.channel_forcing_dir, CFG.qlat_file_pattern):
        build_forcing_dataset(
            FORCING_MODE,
            START_TIME,
            END_TIME,
            CFG.channel_forcing_dir,
            CFG.domain_path,
            constant_qlat=CONSTANT_QLAT
        )

    # Generate DA data for each persistence type (USGS, USACE, USBR).
    for da_name, group in _DA_GROUPS.items():
        if da_name == "rfc":
            continue
        target_dir = getattr(CFG, f"{da_name}_timeslices_dir")
        suffix_map = {"usgs": "usgsTimeSlice", "usace": "usaceTimeSlice", "usbr": "usbrTimeSlice"}
        if refresh or not has_files(target_dir, f"*.{suffix_map[da_name]}.ncdf"):
            build_da_dataset(DAConfig(
                da_type=da_name,
                station_ids=[sid for _, sid, _ in group],
                start_time=START_TIME,
                end_time=END_TIME,
                output_dir=target_dir,
                discharge=group[0][2],
            ))

    # Generate RFC DA data.
    if "rfc" in _DA_GROUPS:
        rfc_group = _DA_GROUPS["rfc"]
        if refresh or not has_files(CFG.rfc_timeslices_dir, "*.RFCTimeSeries.ncdf"):
            for _, sid, flow in rfc_group:
                build_da_dataset(DAConfig(
                    da_type="rfc",
                    station_ids=[sid],
                    start_time=START_TIME,
                    end_time=END_TIME,
                    output_dir=CFG.rfc_timeslices_dir,
                    discharge=flow,
                ))


@pytest.mark.integration
def test_four_lakes():
    skip_if_not_built(CFG)
    delete_outputs(CFG.output_dir)
    run_troute(CFG.config_path)
    assert_peak_bounds(CFG.output_dir, PEAK_BOUNDS)
    assert_lakeout(
        CFG.lakeout_dir,
        expected_feature_count=4,
        outflow_bounds=LAKEOUT_OUTFLOW_BOUNDS,
    )