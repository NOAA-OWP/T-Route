from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest

from ..utils.integration_helpers import (

    delete_outputs,
    get_usgs_station_ids,
    has_files,
    load_output,
    run_troute,
    skip_if_not_built,
)
from ..utils.generate_reference_data import generate_reference_data
from ..utils.make_configs import Config, DataAssimilationParameters
from ..utils.make_forcing import build_forcing_dataset, create_hot_start_file
from ..utils.make_da import write_usgs_timeslices
from ..utils.subset_nhf import get_offnetwork_upstreams, extract_layers, write_gpkg

FP_IDS = [
    1269985531956909,
    1269989421970703,
    1270477057754811,
    1270477071348474,
    1270477162655082,
    1270478408627136,
    1270478480769474,
    1270478443545299,
    1270478544606477,
    1270479710852096,
    1270478559871774,
    1270481040988413,
    1270482252262935,
    1270482207072236,
    1269974759984431,
    1269977357089802,
    1269978640043814,
    1269978588129958,
    1269978508717011,
    1269978605187084,
    1269979803636892,
    1269979801723785,
    1269980994441113,
    1269981094844870,
    1269982274955620,
    1269982288653121,
    1269983535480997,
    1269983486734837,
    1269983407894074,
    1269983415902697,
    1269983275710118,
    1269985856917587,
    1269987150880464,
    1269987179564045,
    1269988420397162,
    1269988372956327,
    1269988326884596,
    1269989538245521,
    1269989622554125,
    1270477248837734,
    1270477334887319,
    1270478607299993,
    1270478625399418,
    1270479816524705,
    1270481161746904,
    1270482410241539,
    1270483792118807,
    1270485077827397,
    1270486400302327,
    1270487708931571,
    1270490204531832,
    1270490290599818,
    1270491588752225,
    1270491644483850,
    1270492969620749,
    1270495491254275,
    1270496783022913,
    1270498170206546,
    1270499546086799,
    1270499583677310,
    1270500815903050,
    1270502177889062,
    1270502329921098,
    1270989956036842,
    1270989989038756,
    1270991250585901,
    1270991287045452,
    1270992549474126,
    1270992623454142,
    1271018265769831,
    1271018298184438,
    1271020831654835,
    1271022201616581,
    1270999001416760,
]
START_TIME = "2011-04-14 00:00"
END_TIME = "2011-06-30 00:00"
FORCING_MODE = "retro"

RUNOUT_PERIOD = int(
    (pd.Timestamp(END_TIME) - pd.Timestamp(START_TIME)).total_seconds() / 3600 / 2
)
END_TIME_WITH_RUNOUT = (
    pd.Timestamp(END_TIME) + pd.Timedelta(hours=RUNOUT_PERIOD)
).strftime("%Y-%m-%d %H:%M")

DATA_DIR = Path(__file__).parent / "data"
CFG_DIVERSION = Config(
    DATA_DIR,
    START_TIME,
    END_TIME_WITH_RUNOUT,
    restart_dir_name="restart",
    data_assimilation_parameters=DataAssimilationParameters(
        usgs_timeslices_folder="usgs_da",
        streamflow_nudging=True,
        timeslice_lookback_hours=48,
        diversion_gage_crosswalk={1270479816524705: "07381482"}
    ),
)
CFG_NO_DIVERSION = Config(
    DATA_DIR,
    START_TIME,
    END_TIME_WITH_RUNOUT,
    restart_dir_name="restart",
    config_file_name="config_no_diversion.yaml",
    output_dir_name="output_no_diversion",
    data_assimilation_parameters=DataAssimilationParameters(
        usgs_timeslices_folder="usgs_da_no_diversion",
        streamflow_nudging=True,
        timeslice_lookback_hours=48,
    ),
)
CFG_HISTORICAL = Config(
    DATA_DIR,
    START_TIME,
    END_TIME_WITH_RUNOUT,
    restart_dir_name="restart",
    config_file_name="config_historical.yaml",
    output_dir_name="output_historical",
    data_assimilation_parameters=DataAssimilationParameters(
        # Same observation archive as CFG_NO_DIVERSION, which has the diversion
        # gage removed. Every other gage is still nudged, so the run is a real
        # forecast-mode shape and CFG_NO_DIVERSION is a matched control: the only
        # difference between the two is the diversion itself.
        usgs_timeslices_folder="usgs_da_no_diversion",
        streamflow_nudging=True,
        timeslice_lookback_hours=48,
        persist_historical_median=True,
        diversion_gage_crosswalk={1270479816524705: "07381482"}
    ),
)

GAGES_PATCH = {
    "07381482": {"fp_id": 1270478544606477, "virtual_fp_id": 1270478544606478},
    "07381490": {"fp_id": 1269985531956909, "virtual_fp_id": 1269985531956910},
}

def patch_gages(domain_path: Path) -> None:
    """Manually patch fp_id and virtual_fp_id in the gages table."""
    gages = gpd.read_file(domain_path, layer="gages")
    for site_no, fields in GAGES_PATCH.items():
        for field, value in fields.items():
            gages.loc[gages["site_no"] == site_no, field] = value
    gages.to_file(domain_path, layer="gages", driver="GPKG")

def setup(source_gpkg: str | Path, refresh: bool = True):
    """Subset the NHF domain and generate forcing for a standard test case."""
    offnetwork_upstreams = None
    if refresh or not CFG_DIVERSION.config_path.exists():
        CFG_DIVERSION.write_yaml()
    if refresh or not CFG_NO_DIVERSION.config_path.exists():
        CFG_NO_DIVERSION.write_yaml()
    if refresh or not CFG_HISTORICAL.config_path.exists():
        CFG_HISTORICAL.write_yaml()

    if refresh or not CFG_DIVERSION.domain_path.exists():
        offnetwork_upstreams = get_offnetwork_upstreams(source_gpkg, FP_IDS)
        layers = extract_layers(source_gpkg, FP_IDS + offnetwork_upstreams)
        write_gpkg(layers, CFG_DIVERSION.domain_path)
        patch_gages(CFG_DIVERSION.domain_path)

    if refresh or not has_files(CFG_DIVERSION.channel_forcing_dir, CFG_DIVERSION.qlat_file_pattern):
        if offnetwork_upstreams is None:
            offnetwork_upstreams = get_offnetwork_upstreams(source_gpkg, FP_IDS)
        build_forcing_dataset(
            FORCING_MODE,
            START_TIME,
            END_TIME,
            CFG_DIVERSION.channel_forcing_dir,
            CFG_DIVERSION.domain_path,
            RUNOUT_PERIOD,
            offnetwork_upstreams=offnetwork_upstreams,
        )

    restart_file = CFG_DIVERSION.root_dir / CFG_DIVERSION.restart_dir_name / "restart.pkl"
    if refresh or not restart_file.exists():
        if offnetwork_upstreams is None:
            offnetwork_upstreams = get_offnetwork_upstreams(source_gpkg, FP_IDS)
        create_hot_start_file(
            t_start=START_TIME,
            restart_dir=str(CFG_DIVERSION.root_dir / CFG_DIVERSION.restart_dir_name),
            hydrofabric_path=str(CFG_DIVERSION.domain_path),
            offnetwork_upstreams=offnetwork_upstreams
        )

    if refresh or not CFG_DIVERSION.reference_data_path.exists():
        generate_reference_data(
            hydrofabric_path=CFG_DIVERSION.domain_path,
            t_start=pd.Timestamp(START_TIME),
            t_end=pd.Timestamp(END_TIME),
            output_dir=CFG_DIVERSION.reference_data_path.parent,
            dv_only=True
        )

    if CFG_DIVERSION.usgs_timeslices_dir is not None and (
        refresh or not has_files(CFG_DIVERSION.usgs_timeslices_dir, "*.usgsTimeSlice.ncdf")
    ):
        lookback_hours = CFG_DIVERSION.data_assimilation_parameters.timeslice_lookback_hours or 0
        da_start = (pd.Timestamp(START_TIME) - pd.Timedelta(hours=lookback_hours)).strftime("%Y-%m-%d %H:%M")
        write_usgs_timeslices(
            station_ids=["07381482", "07289000"],
            start_time=da_start,
            end_time=END_TIME_WITH_RUNOUT,
            output_dir=CFG_DIVERSION.usgs_timeslices_dir,
            dv_only=True
        )
    if CFG_NO_DIVERSION.usgs_timeslices_dir is not None and (
        refresh or not has_files(CFG_NO_DIVERSION.usgs_timeslices_dir, "*.usgsTimeSlice.ncdf")
    ):
        lookback_hours = CFG_NO_DIVERSION.data_assimilation_parameters.timeslice_lookback_hours or 0
        da_start = (pd.Timestamp(START_TIME) - pd.Timedelta(hours=lookback_hours)).strftime("%Y-%m-%d %H:%M")
        write_usgs_timeslices(
            station_ids=["07289000"],
            start_time=da_start,
            end_time=END_TIME_WITH_RUNOUT,
            output_dir=CFG_NO_DIVERSION.usgs_timeslices_dir,
            dv_only=True
        )

# Gages bracketing the control structure, with the routing link each resolves to.
# Taken from the diagnostics behind the Old River report (its Figure 7).
MISSISSIPPI_BATON_ROUGE = ("07374000", 1269974759984431)  # downstream of the diversion
ATCHAFALAYA_SIMMESPORT = ("07381490", 1269985531956909)  # receives the diverted water

# Inputs are generated out of band by ``python -m test.nhf.prep_tests``, which calls
# the module-level ``setup`` above. The fixtures below only gate on that data being
# present and clear stale outputs, so an assertion can never pass on a previous run.

@pytest.fixture
def diversion_case(built_case):
    """Case with the Old River transfer active, driven by real observations."""
    return built_case(CFG_DIVERSION)

@pytest.fixture
def no_diversion_case(built_case):
    """Same domain and forcing with the transfer switched off: the control."""
    return built_case(CFG_NO_DIVERSION)

@pytest.fixture
def historical_case(built_case):
    """Forecast-mode shape: the diversion gage has no observation, so climatology
    supplies it. Every other gage is still nudged, exactly as in no_diversion_case,
    which is therefore the matched control."""
    return built_case(CFG_HISTORICAL)

def _peak_at(output_dir: Path, fp_id: int) -> float:
    """Peak simulated discharge at a routing link over the run."""
    ds = load_output(output_dir)
    try:
        return float(ds["flow"].sel(feature_id=fp_id).max())
    finally:
        ds.close()

@pytest.mark.integration
def test_diversion_moves_water_from_mississippi_to_atchafalaya(
    diversion_case, no_diversion_case
):
    """The defining behavior of the control structure, as an A/B.

    Without the transfer t-route carries all of the simulated flow past the
    structure, which overestimates the Mississippi below it and starves the
    Atchafalaya. Turning it on must push discharge in opposite directions at the
    two gages, which is the claim the report makes from this same case.
    """
    run_troute(no_diversion_case.config_path)
    _, ms_link = MISSISSIPPI_BATON_ROUGE
    _, atch_link = ATCHAFALAYA_SIMMESPORT
    ms_without = _peak_at(no_diversion_case.output_dir, ms_link)
    atch_without = _peak_at(no_diversion_case.output_dir, atch_link)

    run_troute(diversion_case.config_path)
    ms_with = _peak_at(diversion_case.output_dir, ms_link)
    atch_with = _peak_at(diversion_case.output_dir, atch_link)

    assert ms_with < ms_without, (
        "Mississippi peak below the structure should fall once flow is diverted "
        f"({ms_with:.1f} vs {ms_without:.1f} cms)"
    )
    assert atch_with > atch_without, (
        "Atchafalaya peak should rise once it receives the diverted flow "
        f"({atch_with:.1f} vs {atch_without:.1f} cms)"
    )

@pytest.mark.integration
def test_historical_median_diverts_without_timeslices(
    historical_case, no_diversion_case
):
    """Forecast mode: climatology alone still moves water.

    persist_historical_median populates the diversion gage's row when no
    observation exists, so the transfer keeps working past the end of the
    observation record. This is the mode a forecast actually runs in.
    """
    run_troute(no_diversion_case.config_path)
    _, ms_link = MISSISSIPPI_BATON_ROUGE
    ms_without = _peak_at(no_diversion_case.output_dir, ms_link)

    run_troute(historical_case.config_path)
    ms_with = _peak_at(historical_case.output_dir, ms_link)

    assert ms_with < ms_without, (
        "climatological diversion should still reduce the Mississippi peak "
        f"({ms_with:.1f} vs {ms_without:.1f} cms)"
    )

# Regenerate the diagnostics behind the report with:
# python -m test.nhf.utils.generate_diagnostics -f test/nhf/old_river/data/config.yaml
# python -m test.nhf.utils.generate_diagnostics -f test/nhf/old_river/data/config_no_da.yaml
