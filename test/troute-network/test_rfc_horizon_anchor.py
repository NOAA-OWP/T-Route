"""The persistence horizon measures the forecast's age, not the run's.

``reservoir_rfc_forecast_persist_days`` caps how long a product may go on driving a
reservoir. Anchoring that on the run's t0 made it a property of the run instead: a
chain of hourly cycles inherited one deadline from its first cycle and stopped
assimilating for good once it passed, with fresh forecasts unused on disk.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from troute.DataAssimilation import assemble_rfc_dataframes

_T0 = pd.Timestamp("2021-10-21 12:00:00")
_LAKE = 101
_CROSSWALK = pd.DataFrame(
    {"rfc_gage_id": ["KNFC1"], "rfc_lake_id": [_LAKE]}
).set_index("rfc_lake_id")


def _assemble(issue_time: pd.Timestamp, persist_days: float, t0: pd.Timestamp = _T0):
    stamps = pd.date_range(t0 - timedelta(hours=1), periods=48, freq="h")
    frame = pd.DataFrame({
        "stationId": "KNFC1",
        "discharges": 1000.0 + np.arange(len(stamps), dtype=float),
        "Datetime": stamps,
        "totalCounts": len(stamps),
        "timeseries_idx": list(stamps).index(t0),
        "file": "f",
        "use_rfc": True,
        "da_timestep": 3600,
        "issue_time": issue_time,
    })
    return assemble_rfc_dataframes(
        frame, _CROSSWALK, t0, {"reservoir_rfc_forecast_persist_days": persist_days}
    )[1]


def test_the_deadline_follows_the_issue() -> None:
    params = _assemble(_T0, 2)
    assert params.loc[_LAKE, "persist_until"] == _T0 + timedelta(days=2)


@pytest.mark.parametrize("age_hours", [0, 12, 24])
def test_an_older_forecast_expires_sooner(age_hours: int) -> None:
    """A product already 24 h old has 24 h less of its allowance left."""
    issue = _T0 - timedelta(hours=age_hours)
    params = _assemble(issue, 2)
    assert params.loc[_LAKE, "persist_until"] == issue + timedelta(days=2)


def test_the_same_issue_gives_the_same_deadline_at_every_t0() -> None:
    """The property that makes a chain safe: each cycle derives one deadline, so a
    checkpoint has nothing to carry and cannot freeze one."""
    issue = _T0
    first = _assemble(issue, 11, t0=_T0)
    later = _assemble(issue, 11, t0=_T0 + timedelta(hours=6))
    assert first.loc[_LAKE, "persist_until"] == later.loc[_LAKE, "persist_until"]


def test_adopting_a_newer_issue_renews_the_allowance() -> None:
    old = _assemble(_T0 - timedelta(days=1), 2)
    new = _assemble(_T0, 2)
    assert new.loc[_LAKE, "persist_until"] > old.loc[_LAKE, "persist_until"]


def test_observations_with_no_issue_time_are_refused() -> None:
    """The BMI transport path carries no issue time, so it cannot be given a horizon."""
    frame = pd.DataFrame({
        "stationId": ["KNFC1"], "discharges": [1.0], "Datetime": [_T0],
        "totalCounts": [1], "timeseries_idx": [0], "file": ["f"],
        "use_rfc": [True], "da_timestep": [3600],
    })
    with pytest.raises(ValueError, match="carry no issue time"):
        assemble_rfc_dataframes(
            frame, _CROSSWALK, _T0, {"reservoir_rfc_forecast_persist_days": 11}
        )
