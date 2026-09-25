"""The array transport has to carry everything assembly needs, issue time included.

The BMI path flattens the RFC frame into arrays and rebuilds it on the other side. The
persistence horizon is measured from the forecast's issue, so a transport that drops it
cannot produce a horizon at all, and the assembler refuses rather than inventing one.
"""
from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from troute.DataAssimilation import assemble_rfc_dataframes
from troute.network.bmi_array2df import _bmi_reassemble_rfc_timeseries

_T0 = pd.Timestamp("2021-10-21 12:00:00")


def _frame(issue_time: pd.Timestamp) -> pd.DataFrame:
    stamps = pd.date_range(_T0, periods=4, freq="h")
    return pd.DataFrame({
        "stationId": ["KNFC1"] * 4,
        "discharges": [1.0, 2.0, 3.0, 4.0],
        "synthetic_values": [0, 0, 0, 0],
        "totalCounts": [4] * 4,
        "timeSteps": [timedelta(seconds=3600)] * 4,
        "Datetime": stamps,
        "timeseries_idx": [0] * 4,
        "file": ["f.ncdf"] * 4,
        "use_rfc": [True] * 4,
        "da_timestep": [3600] * 4,
        "issue_time": [issue_time] * 4,
    })


def _round_trip(frame: pd.DataFrame) -> pd.DataFrame:
    """Flatten to arrays and rebuild, the way the BMI path does."""
    import bmi_df2array as df2a

    (da_timestep, total_counts, synthetic, discharges, idx, use_rfc, datetime_s,
     timesteps, station_array, station_lengths, file_array, file_lengths,
     issue_time) = df2a._bmi_disassemble_rfc_timeseries(frame, _T0)
    return _bmi_reassemble_rfc_timeseries(
        da_timestep, total_counts, synthetic, discharges, idx, use_rfc,
        datetime_s, timesteps, station_array, station_lengths, file_array,
        file_lengths, _T0, issue_time,
    )


def test_the_issue_time_survives_the_round_trip() -> None:
    issue = _T0 - timedelta(hours=6)
    assert (_round_trip(_frame(issue))["issue_time"] == issue).all()


def test_the_transported_frame_assembles() -> None:
    """Before the issue time crossed, every populated transfer hit the refusal."""
    issue = _T0 - timedelta(hours=6)
    _, params = assemble_rfc_dataframes(
        _round_trip(_frame(issue)),
        pd.DataFrame({"rfc_gage_id": ["KNFC1"], "rfc_lake_id": [1]}).set_index(
            "rfc_lake_id"
        ),
        _T0,
        {"reservoir_rfc_forecast_persist_days": 11},
    )
    assert params.loc[1, "persist_until"] == issue + timedelta(days=11)


def test_a_transport_without_the_issue_time_is_still_refused() -> None:
    """Older senders omit it, and a horizon cannot be guessed from what is left."""
    frame = _round_trip(_frame(_T0)).drop(columns=["issue_time"])
    with pytest.raises(ValueError, match="carry no issue time"):
        assemble_rfc_dataframes(
            frame,
            pd.DataFrame({"rfc_gage_id": ["KNFC1"], "rfc_lake_id": [1]}).set_index(
                "rfc_lake_id"
            ),
            _T0,
            {"reservoir_rfc_forecast_persist_days": 11},
        )
