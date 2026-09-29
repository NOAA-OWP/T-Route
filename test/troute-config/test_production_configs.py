"""The five configurations production actually runs, exercised as configurations.

They live in ``test/troute_prod_configs`` with the deployment's paths replaced by repo
fixtures. Everything that decides behavior, especially the RFC selection window and the
persistence horizon, is as deployed and is read from the file rather than restated here.
"""
from __future__ import annotations

import shutil
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest
import yaml

from troute.config import Config
from troute.DataAssimilation import _read_timeseries_files, assemble_rfc_dataframes

_CONFIGS = sorted((Path(__file__).parent.parent / "troute_prod_configs").glob("*.yaml"))
_FIXTURE = next(
    (Path(__file__).parent.parent / "BMI" / "rfc_timeseries").glob("*.RFCTimeSeries.ncdf"),
    None,
)

pytestmark = pytest.mark.skipif(not _CONFIGS, reason="production configs not present")
_IDS = [f.stem.replace("01205500_troute_config_", "") for f in _CONFIGS]


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _rfc(cfg: dict) -> dict:
    da = cfg["compute_parameters"]["data_assimilation_parameters"]
    return da["reservoir_da"]["reservoir_rfc_da"]


def _redirect(cfg: dict, tmp: Path) -> dict:
    """Resolve the fixture paths against the repo, and send output to a temp dir.

    The configs carry repo relative paths so they can be read from anywhere; nothing
    that decides behavior is touched.
    """
    root = Path(__file__).parent.parent.parent
    sn = cfg["network_topology_parameters"]["supernetwork_parameters"]
    sn["geo_file_path"] = str(root / sn["geo_file_path"])
    rfc = _rfc(cfg)
    rfc["reservoir_rfc_forecasts_time_series_path"] = str(
        root / rfc["reservoir_rfc_forecasts_time_series_path"]
    )
    da = cfg["compute_parameters"]["data_assimilation_parameters"]
    da["usgs_timeslices_folder"] = str(root / da["usgs_timeslices_folder"])
    cfg["compute_parameters"]["forcing_parameters"]["qlat_input_folder"] = str(tmp)
    out = cfg["output_parameters"]
    out["lakeout_output"] = str(tmp)
    out["stream_output"]["stream_output_directory"] = str(tmp)
    return cfg


def _window(t0: pd.Timestamp, lookback: int, offset: int) -> list[str]:
    """The hourly stamps DataAssimilation builds from a config's two window settings."""
    end = t0 + timedelta(hours=offset)
    stamps, d = [], end - timedelta(hours=lookback)
    while d <= end:
        stamps.append(d.strftime("%Y-%m-%d_%H"))
        d += timedelta(hours=1)
    return stamps


def _run_hours(cfg: dict) -> float:
    fp = cfg["compute_parameters"]["forcing_parameters"]
    return fp["nts"] * fp["dt"] / 3600


# ------------------------------------------------------------------ schema

@pytest.mark.parametrize("path", _CONFIGS, ids=_IDS)
def test_every_production_config_parses(path, tmp_path):
    """Guards the schema against tightening under production's feet.

    ``ReservoirRfcParameters`` forbids extra keys so a misspelled policy cannot be
    dropped in silence; that must not reject a config that is already deployed.
    """
    Config(**_redirect(_load(path), tmp_path))


# ------------------------------------------------------------------ the window

@pytest.mark.parametrize("path", _CONFIGS, ids=_IDS)
def test_the_rfc_window_reaches_backward_only(path):
    """offset 0 in every deployed configuration: nothing issued after t0 is eligible.

    The shipped example configs use offset 28, which selects issues dated after t0.
    Production does not, so a run assimilates only what was in hand when it started.
    """
    rfc = _rfc(_load(path))
    assert rfc["reservoir_rfc_forecasts_offset_hours"] == 0
    t0 = pd.Timestamp("2026-07-13 09:00")
    stamps = _window(t0, rfc["reservoir_rfc_forecasts_lookback_hours"], 0)
    assert stamps[-1] == t0.strftime("%Y-%m-%d_%H")


@pytest.mark.parametrize("path", _CONFIGS, ids=_IDS)
def test_the_window_outlasts_the_interval_between_issues(path):
    """Issues arrive every 6 to 24 hours, so a window under 24 h finds none at most
    cycle times. Every deployed configuration uses 28."""
    assert _rfc(_load(path))["reservoir_rfc_forecasts_lookback_hours"] >= 24


@pytest.mark.skipif(_FIXTURE is None, reason="no RFC fixture available")
@pytest.mark.parametrize("path", _CONFIGS, ids=_IDS)
def test_production_settings_select_the_issue_in_hand(path, tmp_path):
    """The deployed window, driven through the real reader against a real file."""
    rfc = _rfc(_load(path))
    shutil.copy(_FIXTURE, tmp_path / _FIXTURE.name)
    t0 = pd.Timestamp(_FIXTURE.name.split(".", 1)[0].replace("_", " ") + ":00:00")
    got = _read_timeseries_files(
        str(tmp_path),
        _window(t0, rfc["reservoir_rfc_forecasts_lookback_hours"],
                rfc["reservoir_rfc_forecasts_offset_hours"]),
        t0,
        t0 + timedelta(days=rfc["reservoir_rfc_forecast_persist_days"]),
    )
    assert set(got["stationId"]) == {_FIXTURE.name.split(".")[2]}


# ------------------------------------------------------------------ the horizon

def _deadline(cfg: dict, issue_age_hours: float, t0: pd.Timestamp) -> pd.Timestamp:
    """The deadline the assembler derives, driven through the assembler itself."""
    rfc = _rfc(cfg)
    stamps = pd.date_range(t0, periods=4, freq="h")
    frame = pd.DataFrame({
        "stationId": "A1", "discharges": [1.0, 2.0, 3.0, 4.0], "Datetime": stamps,
        "totalCounts": 4, "timeseries_idx": 0, "file": "f", "use_rfc": True,
        "da_timestep": 3600, "issue_time": t0 - timedelta(hours=issue_age_hours),
    })
    crosswalk = pd.DataFrame(
        {"rfc_gage_id": ["A1"], "rfc_lake_id": [101]}
    ).set_index("rfc_lake_id")
    _, params = assemble_rfc_dataframes(frame, crosswalk, t0, rfc)
    return params.loc[101, "persist_until"]


@pytest.mark.parametrize("path", _CONFIGS, ids=_IDS)
def test_a_fresh_issue_puts_the_horizon_where_the_run_length_decides(path):
    """With an issue dated t0 the allowance is the full persist_days.

    At 11 days that is inert in the analysis and short range configurations and live
    in long range, which is where the horizon is observable at all.
    """
    cfg = _load(path)
    t0 = pd.Timestamp("2026-07-13 12:00")
    reached = (_deadline(cfg, 0, t0) - t0).total_seconds() / 3600 < _run_hours(cfg)
    assert reached == ("lr" in path.stem)


@pytest.mark.parametrize("path", _CONFIGS, ids=_IDS)
def test_an_older_issue_moves_the_horizon_earlier_by_its_age(path):
    """The horizon follows the forecast, so a stale issue expires sooner.

    This is what a run-anchored deadline could not express, and the reason the medium
    range run is nearer the edge than its length alone suggests.
    """
    cfg = _load(path)
    t0 = pd.Timestamp("2026-07-13 12:00")
    fresh, stale = _deadline(cfg, 0, t0), _deadline(cfg, 28, t0)
    assert (fresh - stale) == timedelta(hours=28)


def test_the_medium_range_run_outlives_a_stale_issue():
    """239 h of routing against 264 h of allowance looks safe until the issue is old.

    At the lookback edge the forecast expires about three hours before the run ends,
    so medium range is not as far from the horizon as its length suggests.
    """
    mrb = next(p for p in _CONFIGS if "mrb" in p.stem)
    cfg = _load(mrb)
    t0 = pd.Timestamp("2026-07-13 12:00")
    lookback = _rfc(cfg)["reservoir_rfc_forecasts_lookback_hours"]
    left = (_deadline(cfg, lookback, t0) - t0).total_seconds() / 3600
    assert left < _run_hours(cfg)
    assert _run_hours(cfg) - left == pytest.approx(3)


@pytest.mark.parametrize("path", _CONFIGS, ids=_IDS)
def test_the_old_river_diversion_outlasts_the_range(path):
    """Every range diverts at the Old River Control Structure, and the hold covers
    the whole range, so a record that ends at issue time persists to the last hour."""
    cfg = _load(path)
    da = cfg["compute_parameters"]["data_assimilation_parameters"]
    assert da["usgs_timeslices_folder"], "the diversion reads its gage from the TimeSlices"
    diversion = da["diversion_da"]
    assert diversion["diversion_gage_crosswalk"] == {1270479816524705: "07381482"}
    # The last report can be as old at t0 as the RFC window allows (28 h) and the hold
    # must still reach the last hour, so the horizon covers the range plus that age.
    assert diversion["diversion_persist_days"] * 24 >= _run_hours(cfg) + 28, (
        f"{path.stem}: the hold expires inside the range"
    )


@pytest.mark.parametrize("path", _CONFIGS, ids=_IDS)
def test_a_day_old_report_is_held_to_the_last_hour_of_the_range(path):
    """The fill on the range's own grid: a report 28 h before t0 holds through nts."""
    import numpy as np
    from troute.DataAssimilation import _fill_diversion_row

    cfg = _load(path)
    fp = cfg["compute_parameters"]["forcing_parameters"]
    da = cfg["compute_parameters"]["data_assimilation_parameters"]
    t0 = pd.Timestamp(cfg["compute_parameters"]["restart_parameters"]["start_datetime"])

    class _Network:
        _diversion_site_to_node = {"07381482": 1}

    _Network.t0 = t0
    seed = {1: (t0 - timedelta(hours=28), 14484.0)}
    out, _ = _fill_diversion_row(pd.DataFrame(), da["diversion_da"], _Network,
                                 {"dt": fp["dt"]}, seed_in=seed, nts=fp["nts"])
    row = out.loc[1].to_numpy()
    assert row.shape[0] == fp["nts"] + 1
    np.testing.assert_array_equal(row, 14484.0)


def test_the_run_lengths_are_the_nwm_configurations():
    """Pins what each file is, so a renamed or re-timed config is not read as another."""
    hours = {p.stem.replace("01205500_troute_config_", ""): _run_hours(_load(p))
             for p in _CONFIGS}
    assert hours["default_ana"] == pytest.approx(2)
    assert hours["default_sr"] == pytest.approx(17)
    assert hours["default_ext_ana"] == pytest.approx(27)
    assert hours["default_mrb"] == pytest.approx(239)
    assert hours["lr"] == pytest.approx(719)
