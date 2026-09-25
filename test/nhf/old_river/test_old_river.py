from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr

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
# The retrospective hot start at SPINUP_START carries neither the Vicksburg nudge nor the
# transfer, and the two take 25 h and 111 h to reach Baton Rouge; each case spins up from
# it to START_TIME and the demonstration runs start there from the state that run wrote.
SPINUP_START = "2011-04-14 00:00"
START_TIME = "2011-04-19 00:00"
END_TIME = "2011-06-30 00:00"
HOT_START = "restart.pkl"  # retrospective at SPINUP_START
RESTART_DIVERSION = "spinup_diversion.pkl"
RESTART_NO_DIVERSION = "spinup_no_diversion.pkl"
WATERBODY_DIVERSION = "spinup_diversion_waterbody.pkl"
WATERBODY_NO_DIVERSION = "spinup_no_diversion_waterbody.pkl"
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
    restart_file_name=RESTART_DIVERSION,
    waterbody_restart_file_name=WATERBODY_DIVERSION,
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
    restart_file_name=RESTART_NO_DIVERSION,
    waterbody_restart_file_name=WATERBODY_NO_DIVERSION,
    config_file_name="config_no_diversion.yaml",
    output_dir_name="output_no_diversion",
    data_assimilation_parameters=DataAssimilationParameters(
        usgs_timeslices_folder="usgs_da_no_diversion",
        streamflow_nudging=True,
        timeslice_lookback_hours=48,
    ),
)
# The diversion gage's record is cut here while the forcing and the other gage
# run on: the shape of a forecast at a gage whose feed has stopped.
OBS_CUT = "2011-05-31 00:00"
CFG_PERSIST = Config(
    DATA_DIR,
    START_TIME,
    END_TIME_WITH_RUNOUT,
    restart_dir_name="restart",
    restart_file_name=RESTART_DIVERSION,
    waterbody_restart_file_name=WATERBODY_DIVERSION,
    config_file_name="config_persist.yaml",
    output_dir_name="output_persist",
    data_assimilation_parameters=DataAssimilationParameters(
        usgs_timeslices_folder="usgs_da_persist",
        streamflow_nudging=True,
        timeslice_lookback_hours=48,
        diversion_gage_crosswalk={1270479816524705: "07381482"},
        # The cut period is 30 days; the default horizon (11 days, as for RFC) would
        # expire inside it, so this case sets one that outlasts it.
        diversion_persist_days=45,
    ),
)
# The same demonstration with streamflow nudging off: the diversion reads its own
# gage and nothing else assimilates, the shape of the run once nudging is removed.
# Its control has no DA at all, so the two differ by the transfer alone.
CFG_PERSIST_NO_NUDGING = Config(
    DATA_DIR,
    START_TIME,
    END_TIME_WITH_RUNOUT,
    restart_dir_name="restart",
    restart_file_name=RESTART_DIVERSION,
    waterbody_restart_file_name=WATERBODY_DIVERSION,
    config_file_name="config_persist_no_nudging.yaml",
    output_dir_name="output_persist_no_nudging",
    data_assimilation_parameters=DataAssimilationParameters(
        usgs_timeslices_folder="usgs_da_persist",
        streamflow_nudging=False,
        timeslice_lookback_hours=48,
        diversion_gage_crosswalk={1270479816524705: "07381482"},
        diversion_persist_days=45,
    ),
)
CFG_CONTROL_NO_NUDGING = Config(
    DATA_DIR,
    START_TIME,
    END_TIME_WITH_RUNOUT,
    restart_dir_name="restart",
    restart_file_name=RESTART_NO_DIVERSION,
    waterbody_restart_file_name=WATERBODY_NO_DIVERSION,
    config_file_name="config_control_no_nudging.yaml",
    output_dir_name="output_control_no_nudging",
    data_assimilation_parameters=DataAssimilationParameters(streamflow_nudging=False),
)
# Five-day runs from the retrospective hot start that write the states above.
CFG_SPINUP = Config(
    DATA_DIR,
    SPINUP_START,
    START_TIME,
    restart_dir_name="restart",
    restart_file_name=HOT_START,
    lite_restart_dir_name="restart",
    config_file_name="config_spinup.yaml",
    output_dir_name="output_spinup",
    data_assimilation_parameters=CFG_DIVERSION.data_assimilation_parameters,
)
CFG_SPINUP_NO_DIVERSION = Config(
    DATA_DIR,
    SPINUP_START,
    START_TIME,
    restart_dir_name="restart",
    restart_file_name=HOT_START,
    lite_restart_dir_name="restart",
    config_file_name="config_spinup_no_diversion.yaml",
    output_dir_name="output_spinup_no_diversion",
    data_assimilation_parameters=CFG_NO_DIVERSION.data_assimilation_parameters,
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
    # Cheap, and a stale file would run the notebook on settings the tests do not use.
    for cfg in (CFG_DIVERSION, CFG_NO_DIVERSION, CFG_PERSIST, CFG_PERSIST_NO_NUDGING,
                CFG_CONTROL_NO_NUDGING):
        cfg.write_yaml()

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
            SPINUP_START,
            END_TIME,
            CFG_DIVERSION.channel_forcing_dir,
            CFG_DIVERSION.domain_path,
            RUNOUT_PERIOD,
            offnetwork_upstreams=offnetwork_upstreams,
        )

    restart_file = CFG_DIVERSION.root_dir / CFG_DIVERSION.restart_dir_name / HOT_START
    if refresh or not restart_file.exists():
        if offnetwork_upstreams is None:
            offnetwork_upstreams = get_offnetwork_upstreams(source_gpkg, FP_IDS)
        create_hot_start_file(
            t_start=SPINUP_START,
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
        da_start = (pd.Timestamp(SPINUP_START) - pd.Timedelta(hours=lookback_hours)).strftime("%Y-%m-%d %H:%M")
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
        da_start = (pd.Timestamp(SPINUP_START) - pd.Timedelta(hours=lookback_hours)).strftime("%Y-%m-%d %H:%M")
        write_usgs_timeslices(
            station_ids=["07289000"],
            start_time=da_start,
            end_time=END_TIME_WITH_RUNOUT,
            output_dir=CFG_NO_DIVERSION.usgs_timeslices_dir,
            dv_only=True
        )

    if CFG_PERSIST.usgs_timeslices_dir is not None and (
        refresh or not has_files(CFG_PERSIST.usgs_timeslices_dir, "*.usgsTimeSlice.ncdf")
    ):
        lookback_hours = CFG_PERSIST.data_assimilation_parameters.timeslice_lookback_hours or 0
        da_start = (pd.Timestamp(SPINUP_START) - pd.Timedelta(hours=lookback_hours)).strftime("%Y-%m-%d %H:%M")
        write_usgs_timeslices(
            station_ids=["07381482", "07289000"],
            start_time=da_start,
            end_time=END_TIME_WITH_RUNOUT,
            output_dir=CFG_PERSIST.usgs_timeslices_dir,
            dv_only=True,
            end_time_by_station={"07381482": OBS_CUT},
        )

    # The demonstration's warm states: each case run from the retrospective hot start to
    # START_TIME, the lite restart the driver writes at that hour kept under the case's name.
    restart_dir = CFG_DIVERSION.root_dir / CFG_DIVERSION.restart_dir_name
    stamp = pd.Timestamp(START_TIME).strftime("%Y%m%d%H%M")
    for cfg, channel_name, waterbody_name in (
        (CFG_SPINUP, RESTART_DIVERSION, WATERBODY_DIVERSION),
        (CFG_SPINUP_NO_DIVERSION, RESTART_NO_DIVERSION, WATERBODY_NO_DIVERSION),
    ):
        if refresh or not (restart_dir / channel_name).exists():
            cfg.write_yaml()
            delete_outputs(cfg.output_dir)
            run_troute(cfg.config_path)
            (restart_dir / f"channel_restart_{stamp}").replace(restart_dir / channel_name)
            (restart_dir / f"waterbody_restart_{stamp}").replace(restart_dir / waterbody_name)
            for earlier_window in restart_dir.glob("*_restart_????????????"):
                earlier_window.unlink()

# Gages bracketing the control structure, with the routing link each resolves to.
# Taken from the diagnostics behind the Old River report (its Figure 7).
MISSISSIPPI_BATON_ROUGE = ("07374000", 1269974759984431)  # downstream of the diversion
ATCHAFALAYA_SIMMESPORT = ("07381490", 1269985531956909)  # receives the diverted water
DONOR_FP = 1270479816524705  # Mississippi flowpath the diversion is taken from
RECEIVING_HEADWATER_FP = 1270478544606477  # where 07381482 sits, per GAGES_PATCH
MISSISSIPPI_VICKSBURG = ("07289000", 1271020831654835)  # upstream of the structure

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
def persist_case(built_case):
    """Forecast shape: the diversion record is cut at OBS_CUT, the forcing and the
    upstream gage run on, and the last observation is held. no_diversion_case is
    the matched control."""
    return built_case(CFG_PERSIST)

@pytest.fixture
def persist_no_nudging_case(built_case):
    """persist_case with streamflow nudging off; control_no_nudging_case is its control."""
    return built_case(CFG_PERSIST_NO_NUDGING)

@pytest.fixture
def control_no_nudging_case(built_case):
    """No DA at all: the control for the nudging-off demonstration."""
    return built_case(CFG_CONTROL_NO_NUDGING)

def _timeslice_series(timeslice_dir: Path, site_no: str) -> pd.Series:
    """The gage's discharge on the hour, read back from the timeslice files."""
    values = {}
    for path in sorted(timeslice_dir.glob("*_??:00:00.15min.usgsTimeSlice.ncdf")):
        with xr.open_dataset(path) as ds:
            ids = [s.decode().strip() if isinstance(s, bytes) else str(s).strip()
                   for s in ds["stationId"].to_numpy()]
            if site_no not in ids:
                continue
            i = ids.index(site_no)
            stamp = ds["time"].to_numpy()[i]
            stamp = stamp.decode() if isinstance(stamp, bytes) else str(stamp)
            values[pd.Timestamp(stamp.replace("_", " "))] = float(ds["discharge"].to_numpy()[i])
    return pd.Series(values).sort_index()

def _flow_at(output_dir: Path, fp_id: int) -> pd.Series:
    ds = load_output(output_dir)
    try:
        return ds["flow"].sel(feature_id=fp_id).to_series()
    finally:
        ds.close()

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
@pytest.mark.parametrize(
    ("persist", "control"),
    [("persist_case", "no_diversion_case"),
     ("persist_no_nudging_case", "control_no_nudging_case")],
    ids=["nudging", "no_nudging"],
)
def test_persistence_holds_the_last_observation(persist, control, request):
    """Miscellaneous step 13 with the record ending mid-run, with and without nudging.

    13.1 over the observed period: the receiving headwater carries the gage record
    and the donor gives up the same amount. 13.2 over the cut period: the last
    specified value persists at both nodes instead of transitioning back to the
    simulated flow, with no step at the cut, and nothing changes upstream.

    Without the hold the diverted amount drops to zero within an hour of the cut and
    the receiving headwater drains within a few hours; with it both nodes stay on the
    last value.
    """
    persist_case = request.getfixturevalue(persist)
    no_diversion_case = request.getfixturevalue(control)
    run_troute(no_diversion_case.config_path)
    run_troute(persist_case.config_path)
    obs = _timeslice_series(persist_case.usgs_timeslices_dir, "07381482")
    cut = pd.Timestamp(OBS_CUT)
    last = float(obs[:cut].dropna().iloc[-1])

    head = _flow_at(persist_case.output_dir, RECEIVING_HEADWATER_FP)
    diverted = (_flow_at(no_diversion_case.output_dir, DONOR_FP)
                - _flow_at(persist_case.output_dir, DONOR_FP))
    index = head.index
    observed = (index > pd.Timestamp(START_TIME) + pd.Timedelta(hours=24)) & (index <= cut)
    held = (index > cut) & (index <= pd.Timestamp(END_TIME))
    # The intervals must exist, or the equalities below hold on nothing.
    assert observed.sum() >= 24 * 30 and held.sum() >= 24 * 29, (observed.sum(), held.sum())

    # 13.1: assimilated as an addition at the receiving node and a subtraction at
    # the donor, every written hour, after the first day.
    np.testing.assert_allclose(head[observed], obs.reindex(index)[observed], rtol=0.01)
    np.testing.assert_allclose(diverted[observed], obs.reindex(index)[observed], rtol=0.03)
    # 13.2: the last specified value persists at both nodes.
    np.testing.assert_allclose(head[held], last, rtol=0.01)
    np.testing.assert_allclose(diverted[held], last, rtol=0.03)
    # No step at the cut.
    after, before = cut + pd.Timedelta(hours=1), cut - pd.Timedelta(hours=1)
    step = abs(float(diverted[after]) - float(diverted[before]))
    assert step < 0.02 * last, f"diverted amount stepped by {step:.0f} cms at the cut"
    # The receiving reach routes the imposed flow smoothly: a link of the creek that
    # carries it settling on the depth search's degenerate root shows up here within a
    # day as a drop of more than half in an hour.
    # Checked through END_TIME only: the hold expires 45 days after the cut and both
    # sides stop at once, which is a step by design.
    _, simmesport = ATCHAFALAYA_SIMMESPORT
    downstream = _flow_at(persist_case.output_dir, simmesport)
    settled = downstream[observed | held]
    hourly_change = (settled.diff().abs() / settled.shift()).dropna()
    assert hourly_change.max() < 0.10, f"Simmesport moved {100 * hourly_change.max():.0f}% in one hour"
    # Elsewhere unchanged: the upstream gage's flow is the control's.
    _, vicksburg = MISSISSIPPI_VICKSBURG
    np.testing.assert_allclose(
        _flow_at(persist_case.output_dir, vicksburg),
        _flow_at(no_diversion_case.output_dir, vicksburg), rtol=1e-6,
    )

# Regenerate the diagnostics behind the report with:
# python -m test.nhf.utils.generate_diagnostics -f test/nhf/old_river/data/config.yaml
# python -m test.nhf.utils.generate_diagnostics -f test/nhf/old_river/data/config_no_da.yaml
