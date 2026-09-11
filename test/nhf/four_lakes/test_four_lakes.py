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

# Reservoir DA configuration. One DA type per lake, assigned to the domain's lakes in
# id order rather than to hardcoded ids: `nhf_lake_id` is assigned per hydrofabric
# build, and on nhf 1.2.2 NONE of the four ids this used to name still existed. The
# UPDATEs in modify_lakes_table then matched zero rows, silently, so no DA type was set,
# no observations were written, and the peaks came out at background -- which reads as a
# routing regression and is not one. See _lake_plan and its rowcount check.
DA_TYPE_BY_SLOT = [
    (2, 1.0, "USGS00000000002"),      # USGS persistence
    (3, 1.0, "USAC00000000003"),      # USACE persistence
    (4, 12345.0, "RFC000000000004"),  # RFC forecasts
    (7, 1.0, "USBR00000000007"),      # USBR persistence
]
_DA_NAME = {2: "usgs", 3: "usace", 4: "rfc", 7: "usbr"}


def _lake_plan(domain_path) -> dict[int, dict]:
    """Assign the DA slots above to this domain's lakes, and find each one's outlet.

    Returns ``nhf_lake_id -> {da_type, flow, site_no, out_fp}``. ``out_fp`` is the
    lake's OWN ``fp_id``: the reservoir replaces that flowpath in the routing network,
    so the assimilated outflow appears there. (Following ``dn_nex_id`` to the next
    flowpath down instead lands one hop too far, where the outflow has already mixed
    with local inflow and the assertion sees background.)
    """
    import geopandas as gpd

    lakes = gpd.read_file(domain_path, layer="lakes", ignore_geometry=True)

    lake_ids = sorted(int(i) for i in lakes["nhf_lake_id"].dropna())
    if len(lake_ids) != len(DA_TYPE_BY_SLOT):
        msg = (
            f"four_lakes expects exactly {len(DA_TYPE_BY_SLOT)} lakes in the domain, "
            f"found {len(lake_ids)}; the DA-type assignment below would be ambiguous."
        )
        raise AssertionError(msg)

    by_id = lakes.set_index("nhf_lake_id")
    plan = {}
    for lake_id, (da_type, flow, site_no) in zip(lake_ids, DA_TYPE_BY_SLOT, strict=True):
        out_fp = by_id.loc[lake_id, "fp_id"]
        if out_fp is None or out_fp != out_fp:  # None or NaN
            msg = f"lake {lake_id} has no fp_id; its outflow has nowhere to be scored"
            raise AssertionError(msg)
        plan[lake_id] = {
            "da_type": da_type, "flow": flow, "site_no": site_no, "out_fp": int(out_fp),
        }
    return plan


def _peak_bounds(plan) -> dict[int, tuple[float, float]]:
    return {p["out_fp"]: (0.9 * p["flow"], 1.1 * p["flow"]) for p in plan.values()}


def _lakeout_bounds(plan) -> dict[int, tuple[float, float]]:
    return {lid: (0.9 * p["flow"], 1.1 * p["flow"]) for lid, p in plan.items()}


def _da_groups(plan) -> dict[str, list[tuple[int, str, float]]]:
    groups: dict[str, list[tuple[int, str, float]]] = {}
    for lid, p in plan.items():
        groups.setdefault(_DA_NAME[p["da_type"]], []).append((lid, p["site_no"], p["flow"]))
    return groups


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
    plan = _lake_plan(gpkg_path)
    conn = sqlite3.connect(gpkg_path)
    cur = conn.cursor()
    for lake_id, p in plan.items():
        cur.execute(
            "UPDATE reservoir_da SET da_type=?, site_no=? WHERE nhf_lake_id=?",
            (p["da_type"], p["site_no"], lake_id),
        )
        # A zero-row UPDATE here is how this test silently stopped assimilating: the
        # run still completes, writes output, and fails later on flows that look like a
        # routing bug. Fail at the cause instead.
        if cur.rowcount != 1:
            conn.close()
            msg = (
                f"reservoir_da has no row for lake {lake_id} (UPDATE matched "
                f"{cur.rowcount} rows), so its DA would never engage."
            )
            raise AssertionError(msg)
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
    groups = _da_groups(_lake_plan(CFG.domain_path))
    for da_name, group in groups.items():
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
    if "rfc" in groups:
        rfc_group = groups["rfc"]
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
    plan = _lake_plan(CFG.domain_path)
    assert_peak_bounds(CFG.output_dir, _peak_bounds(plan))
    assert_lakeout(
        CFG.lakeout_dir,
        expected_feature_count=len(plan),
        outflow_bounds=_lakeout_bounds(plan),
    )