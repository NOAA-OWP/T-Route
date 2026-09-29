"""The persistence horizon must survive being chopped into forcing windows.

``reservoir_RFC_da`` gates assimilation on ``current_time <= persist_seconds``, and
``current_time`` is ``dt * timestep``, which restarts at every forcing window. Handing
the kernel a whole-horizon duration would therefore re-arm it every window. An absolute
deadline, packed as the seconds still remaining at each window's start, makes the
window-local comparison equivalent to the continuous one. The deadline itself is
derived from the forecast's issue time; these cases fix it so the windowing is what
they measure.

These step ``reservoir_RFC_da`` directly and mirror the kernel's inter-window carry by
hand, so they pin the gate arithmetic and nothing above it. The end-to-end check that
the packer and the compiled kernel are wired correctly is
``test_chunked_run_matches_continuous_through_the_kernel``.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from troute.DataAssimilation import _set_rfc_reservoir_da_params, assemble_rfc_dataframes

_T0 = datetime(2021, 10, 21, 12)
_CADENCE = 3600
_LAKE = 1
_GAGE = "TESTA"


def _frames(persist_days: float, hours: int = 48):
    """An RFC frame whose forecast tail is a ramp, so the sample in use is readable."""
    stamps = pd.date_range(_T0 - timedelta(hours=1), periods=hours, freq="h")
    rfc_df = pd.DataFrame({
        "stationId": _GAGE,
        "discharges": 1000.0 + np.arange(len(stamps), dtype=float),
        "Datetime": stamps,
        "totalCounts": len(stamps),
        "timeseries_idx": list(stamps).index(pd.Timestamp(_T0)),
        "file": "f",
        "use_rfc": True,
        "da_timestep": _CADENCE,
        # Issued at t0, so a horizon measured from the issue lands where these
        # cases expect it.
        "issue_time": pd.Timestamp(_T0),
    })
    crosswalk = pd.DataFrame(
        {"rfc_gage_id": [_GAGE], "rfc_lake_id": [_LAKE]}
    ).set_index("rfc_lake_id")
    return assemble_rfc_dataframes(
        rfc_df, crosswalk, pd.Timestamp(_T0),
        {"reservoir_rfc_forecast_persist_days": persist_days},
    )


def _remaining_seconds(params: pd.DataFrame, window_t0: datetime) -> float:
    """What the packer hands the kernel for a window starting at window_t0."""
    return float((params["persist_until"].iloc[0] - pd.Timestamp(window_t0)).total_seconds())


def test_deadline_is_absolute_and_shrinks_with_each_window():
    _, params = _frames(persist_days=1.0)
    assert params["persist_until"].iloc[0] == pd.Timestamp(_T0) + timedelta(days=1)
    # The horizon left at each window start counts down; it does not reset.
    assert _remaining_seconds(params, _T0) == 86400
    assert _remaining_seconds(params, _T0 + timedelta(hours=6)) == 86400 - 6 * 3600
    assert _remaining_seconds(params, _T0 + timedelta(hours=24)) == 0
    # Past the deadline it goes negative, which no window-local clock can satisfy.
    assert _remaining_seconds(params, _T0 + timedelta(hours=30)) < 0


@pytest.mark.parametrize("window_hours", [6, 12, 36])
def test_chunked_matches_continuous_across_the_horizon(window_hours):
    """Windows shorter AND longer than the horizon must both track the continuous run.

    A window longer than the horizon is the case that can switch the DA back on at the
    next window boundary after it has correctly stopped.
    """
    from troute.routing.fast_reach.reservoir_RFC_da import reservoir_RFC_da

    persist_days = 1.0
    obs, params = _frames(persist_days)
    series = obs.to_numpy(dtype="float32")[0]
    kernel = {
        "routing_period": 300.0, "reservoir_type": 4, "inflow": 2.0,
        "water_elevation": 10.0, "levelpool_outflow": 2.0,
        "levelpool_water_elevation": 10.0, "lake_area": 1e7,
        "max_water_elevation": 25.0, "rfc_file": "f",
    }
    dt = 300.0
    # A whole number of windows, and always past the 1 day horizon so the cutoff is
    # actually exercised. 36 h windows overshoot it, which is the re-arm case.
    total_hours = window_hours * -(-48 // window_hours)

    def step(idx, update_time, current_time, remaining):
        return reservoir_RFC_da(
            True, series, idx, int(params["totalCounts"].iloc[0]),
            current_time=current_time, update_time=update_time,
            DA_time_step=_CADENCE, rfc_forecast_persist_seconds=remaining, **kernel
        )

    # Continuous: one window, absolute clock, the whole horizon.
    idx, update_time = int(params["timeseries_idx"].iloc[0]), float(params["update_time"].iloc[0])
    continuous = []
    for s in range(1, int(total_hours * 3600 / dt) + 1):
        out, _, update_time, idx, res_type, *_ = step(
            idx, update_time, dt * s, persist_days * 86400
        )
        continuous.append((round(float(out), 4), res_type))

    # Chunked: window-local clock, remaining horizon repacked per window, and the
    # kernel's own update_time rebase applied at each boundary.
    idx, update_time = int(params["timeseries_idx"].iloc[0]), float(params["update_time"].iloc[0])
    steps_per_window = int(window_hours * 3600 / dt)
    chunked = []
    for window in range(total_hours // window_hours):
        remaining = _remaining_seconds(params, _T0 + timedelta(hours=window * window_hours))
        for s in range(1, steps_per_window + 1):
            out, _, update_time, idx, res_type, *_ = step(idx, update_time, dt * s, remaining)
            chunked.append((round(float(out), 4), res_type))
        # The kernel leaves its loop with timestep == nsteps+1, so it rebases by the
        # FULL window duration, not one step less.
        update_time -= steps_per_window * dt

    assert chunked == continuous
    # And the horizon actually ends: assimilating through day 1, level pool after, with
    # no window boundary switching it back on.
    day_one = int(24 * 3600 / dt)
    assert {t for _, t in continuous[:day_one]} == {4}
    assert {t for _, t in chunked[day_one:]} == {1}


def test_state_carry_round_trips_the_deadline():
    """_set_rfc_reservoir_da_params must not disturb the deadline it does not own."""
    _, params = _frames(persist_days=2.0)
    before = params["persist_until"].iloc[0]
    results = [(None,) * 8 + (([_LAKE], [7200.0], [30]),)]
    updated = _set_rfc_reservoir_da_params(params, results)
    assert updated["persist_until"].iloc[0] == before
    assert updated["update_time"].iloc[0] == 7200.0
    assert updated["timeseries_idx"].iloc[0] == 30


def test_a_state_without_the_deadline_is_refused():
    """Measuring the horizon from the window is the defect, so do not fall back to it.

    The packer runs per job per window and knows only the window's t0, so it cannot
    rebuild the deadline.
    """
    from troute.routing.compute import _prep_reservoir_da_dataframes

    obs, params = _frames(persist_days=2.0)
    legacy = params.drop(columns=["persist_until"])
    types_df = pd.DataFrame({"reservoir_type": [4]}, index=[_LAKE])
    empty = pd.DataFrame()
    with pytest.raises(ValueError, match="no persist_until"):
        _prep_reservoir_da_dataframes(
            *([empty] * 6), obs, legacy, *([empty] * 3), types_df, pd.Timestamp(_T0)
        )


