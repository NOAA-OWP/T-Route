"""RFC forecasts that do not cover the run must degrade, not crash.

``reservoir_rfc_forecasts`` is switched on in a config, but the timeseries that
reach the run are whatever the forecast retrieval happened to produce for that
cycle. Three shapes of it reach BMI construction: no file dated inside the lookback
window, no station transported over BMI, and a selected file whose span misses t0.

Enabling RFC DA is a claim that the forecasts are provisioned for the run, so all three
are fatal, and each names the reason and the remedy.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from troute.DataAssimilation import _read_timeseries_files, assemble_rfc_dataframes

_T0 = pd.Timestamp("2021-10-21 12:00:00")
_CROSSWALK = pd.DataFrame({"rfc_gage_id": ["KNFC1"], "rfc_lake_id": [17609317]}).set_index(
    "rfc_lake_id"
)
_PARAMS = {"reservoir_rfc_forecast_persist_days": 11}


def test_no_file_in_lookback_window_names_the_window(tmp_path):
    (tmp_path / "2021-10-21_12.60min.KNFC1.RFCTimeSeries.ncdf").touch()
    with pytest.raises(FileNotFoundError, match="lookback window 2099-01-01_00"):
        _read_timeseries_files(
            str(tmp_path), ["2099-01-01_00"], _T0, _T0 + pd.Timedelta(days=11)
        )


def test_empty_timeseries_frame_names_the_cause():
    with pytest.raises(ValueError, match="no RFC timeseries observations"):
        assemble_rfc_dataframes(pd.DataFrame(), _CROSSWALK, _T0, _PARAMS)


def test_forecast_span_missing_t0_names_both_spans():
    # Forecast ends the hour before t0, so there is no column to rebase the index on.
    dates = pd.date_range(_T0 - pd.Timedelta(hours=3), periods=3, freq="h")
    rfc_df = pd.DataFrame(
        {
            "stationId": "KNFC1",
            "discharges": [1.0, 2.0, 3.0],
            "Datetime": dates,
            "totalCounts": 3,
            "timeseries_idx": 0,
            "file": "f",
            "use_rfc": True,
            "da_timestep": 3600,
            "issue_time": _T0,
        }
    )
    with pytest.raises(ValueError, match="do not cover the simulation start"):
        assemble_rfc_dataframes(rfc_df, _CROSSWALK, _T0, _PARAMS)


def test_covering_forecast_still_assembles():
    dates = pd.date_range(_T0 - pd.Timedelta(hours=1), periods=4, freq="h")
    rfc_df = pd.DataFrame(
        {
            "stationId": "KNFC1",
            "discharges": [1.0, 2.0, 3.0, 4.0],
            "Datetime": dates,
            "totalCounts": 4,
            "timeseries_idx": 1,
            "file": "f",
            "use_rfc": True,
            "da_timestep": 3600,
            "issue_time": _T0,
        }
    )
    obs, params = assemble_rfc_dataframes(rfc_df, _CROSSWALK, _T0, _PARAMS)
    assert not obs.empty
    # Seeded AT t0's column with the first update one cadence out, as the standalone
    # driver does, so both read i0 + floor(model_time / cadence) at any dt.
    assert params["timeseries_idx"].iloc[0] == 1
    assert params["update_time"].iloc[0] == params["da_timestep"].iloc[0]
    # totalCounts is the inclusive last index: 4 samples rebased onto columns 0..3.
    assert params["totalCounts"].iloc[0] == 3


_RFC_FILES = Path(__file__).parents[1] / "BMI" / "rfc_timeseries"


def test_selected_file_missing_t0_names_the_file_and_span():
    # The file is dated inside the window, but its slice starts an hour after t0.
    # With only one candidate there is nothing to fall back to, so this stays fatal.
    with pytest.raises(ValueError, match="cover the simulation start"):
        _read_timeseries_files(
            str(_RFC_FILES), ["2021-10-21_12"], datetime(2021, 10, 19, 11),
            datetime(2021, 11, 1),
        )


def test_covering_file_is_read():
    out = _read_timeseries_files(
        str(_RFC_FILES), ["2021-10-21_12"], datetime(2021, 10, 21, 12), datetime(2021, 11, 1)
    )
    assert list(out["stationId"].unique()) == ["KNFC1"]
    assert (out["timeseries_idx"] == 48).all()


def test_bmi_da_forcing_shares_this_reader():
    """The duplicate reader in the BMI DA-forcing module is how both crashes got there."""
    src = Path(__file__).parents[2] / "src" / "model_DAforcing.py"
    text = src.read_text()
    assert "from troute.DataAssimilation import _read_timeseries_files" in text
    assert "def _read_timeseries_files" not in text


def test_persist_horizon_is_inclusive():
    """The truncation must match the consumer's persist gate.

    ``reservoir_RFC_da`` assimilates while ``current_time <= persist_seconds``, so the
    forecast value AT ``t0 + rfc_forecast_persist_days`` is reachable. Truncating with a
    strict ``<`` drops it and the final assimilable hour falls back to level pool. The
    fixture's tail must not be flat, or both sides return the same number from
    different array positions.
    """
    t0 = datetime(2021, 10, 21, 12)
    # Inside the file's span (2021-10-19_12 to 2021-10-31_12) so the cut actually bites.
    horizon = t0 + timedelta(days=5)
    out = _read_timeseries_files(str(_RFC_FILES), ["2021-10-21_12"], t0, horizon)
    assert out["Datetime"].max() == pd.Timestamp(horizon)
