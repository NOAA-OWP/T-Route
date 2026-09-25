"""The persistence horizon, end to end through the compiled kernel.

The sibling ``test_rfc_persist_horizon`` module steps ``reservoir_RFC_da`` directly and
mirrors the kernel's inter-window carry by hand. That pins the gate arithmetic and
nothing above it: it would still pass if the deadline were packed wrong, if the seconds
argument were miswired into the Cython signature, or if the kernel's own carry changed.

This one runs the production path. A three-node network with one type-4 reservoir goes
through ``compute_nhd_routing_v02`` twice over the same timeline, once continuously and
once in windows, carrying state between windows exactly as the drivers do:

    q0   <- the kernel's own returned state, applied to the waterbody frame
    rfc  <- _set_rfc_reservoir_da_params(param_df, results)
    t0   <- advanced by the window length

With the horizon shorter than the run, the two must still agree at every step.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from troute import nhd_network
from troute.DataAssimilation import (
    _read_timeseries_files,
    _set_rfc_reservoir_da_params,
    assemble_rfc_dataframes,
)
from troute.routing.compute import compute_nhd_routing_v02

_FIXTURES = Path(__file__).parents[1] / "BMI" / "rfc_timeseries"
_GAGE = "KNFC1"
_T0 = datetime(2021, 10, 21, 12)
_LAKE = 30
_CONNECTIONS = {10: [_LAKE], _LAKE: [40], 40: []}
_CHANNELS = [10, 40]
_DT = 300
_QTS = 12  # forcing columns are hourly
_STEPS_PER_DAY = 86400 // _DT

_WB_COLS = ["LkArea", "LkMxE", "OrificeA", "OrificeC", "OrificeE",
            "WeirC", "WeirE", "WeirL", "ifd", "qd0", "h0"]


def _rfc_frames(persist_days: float):
    # Backward from t0, so the selected issue is the one in hand when the run starts.
    # An offset window reaches forward and can pick an issue dated after t0, which
    # carries its own later horizon.
    stamps, d = [], _T0 - timedelta(hours=28)
    while d <= _T0:
        stamps.append(d.strftime("%Y-%m-%d_%H"))
        d += timedelta(hours=1)
    raw = _read_timeseries_files(
        str(_FIXTURES), stamps, _T0, _T0 + timedelta(days=persist_days), routing_period=_DT
    )
    crosswalk = pd.DataFrame(
        {"rfc_gage_id": [_GAGE], "rfc_lake_id": [_LAKE]}
    ).set_index("rfc_lake_id")
    return assemble_rfc_dataframes(
        raw, crosswalk, pd.Timestamp(_T0),
        {"reservoir_rfc_forecast_persist_days": persist_days},
    )


def _base_frames(hours: int, inflow: float = 2.0):
    reaches = pd.DataFrame(
        {"bw": 10.0, "tw": 20.0, "twcc": 60.0, "dx": 2000.0, "n": 0.03, "ncc": 0.06,
         "cs": 1.0, "s0": 0.001, "alt": 100.0},
        index=_CHANNELS,
    )
    waterbodies = pd.DataFrame(
        [[1.0, 105.0, 1.0, 0.1, 100.0, 0.4, 103.0, 10.0, 0.9, 1.0, 102.0]],
        index=[_LAKE], columns=_WB_COLS,
    )
    types = pd.DataFrame({"reservoir_type": [4]}, index=[_LAKE])
    columns = [(_T0 + timedelta(hours=h)).strftime("%Y%m%d%H%M") for h in range(hours)]
    qlats = pd.DataFrame(0.0, index=_CHANNELS, columns=columns)
    qlats.loc[10] = inflow
    q0 = pd.DataFrame(
        {"qu0": 0.0, "qd0": 0.0, "h0": 0.1, "ql0": 0.0}, index=_CHANNELS
    )
    return reaches, waterbodies, types, qlats, q0


def _route(t0, nts, reaches, waterbodies, types, qlats, q0, rfc_df, rfc_params, plan,
           method="serial", cpu_pool=1):
    """One kernel call. Positional, because that is the production signature."""
    empty = pd.DataFrame()
    rconn = nhd_network.reverse_network(_CONNECTIONS)
    independent = nhd_network.reachable_network(rconn)
    return compute_nhd_routing_v02(
        _CONNECTIONS, rconn, {_LAKE: [_LAKE]}, {tw: [] for tw in independent},
        "V02-structured", method, 0, cpu_pool, t0, _DT, nts, _QTS, independent,
        reaches.copy(), q0, qlats, empty, 0.0, empty, empty, empty, empty, empty,
        empty, empty, empty, rfc_df, rfc_params, empty, empty, empty, {}, True, False,
        waterbodies, {}, types, True, plan, from_files=False, qlat_add_loc="middle",
    )


def _lake_outflow(results, nts):
    for r in results:
        ids = np.asarray(r[0])
        where = np.where(ids == _LAKE)[0]
        if where.size:
            return np.asarray(r[1])[where[0]].reshape(nts, -1)[:, 0].astype(float)
    raise KeyError("the kernel did not return the lake")


def _continuous_and_chunked(days: int, window_hours: int, persist_days: float,
                            method: str = "serial", cpu_pool: int = 1):
    hours = days * 24
    reaches, waterbodies, types, qlats, q0 = _base_frames(hours)
    rfc_df, rfc_params = _rfc_frames(persist_days)

    results, _ = _route(
        _T0, hours * _QTS, reaches, waterbodies.copy(), types, qlats, q0.copy(),
        rfc_df, rfc_params.copy(), [None, None, None], method, cpu_pool,
    )
    continuous = _lake_outflow(results, hours * _QTS)

    chunks, plan = [], [None, None, None]
    wb, carried_q0, params = waterbodies.copy(), q0.copy(), rfc_params.copy()
    window_t0, hour = _T0, 0
    while hour < hours:
        n = min(window_hours, hours - hour)
        results, plan = _route(
            window_t0, n * _QTS, reaches, wb, types, qlats[qlats.columns[hour:hour + n]],
            carried_q0, rfc_df, params, plan, method, cpu_pool,
        )
        chunks.append(_lake_outflow(results, n * _QTS))
        carried_q0 = pd.concat([
            pd.DataFrame(np.asarray(r[1])[:, [-4, -4, -2, -1]], index=r[0],
                         columns=["qu0", "qd0", "h0", "ql0"])
            for r in results
        ])
        wb.update(carried_q0)
        params = _set_rfc_reservoir_da_params(params, results)
        window_t0 += timedelta(hours=n)
        hour += n
    return continuous, np.concatenate(chunks)


@pytest.mark.parametrize("window_hours", [6, 24])
@pytest.mark.parametrize(
    ("method", "cpu_pool"),
    [("serial", 1), ("by-subnetwork-jit", 2), ("by-subnetwork-jit-clustered", 2)],
)
def test_chunked_run_matches_continuous_through_the_kernel(window_hours, method, cpu_pool):
    """A 1 day horizon over a 2 day run: the windows must not extend it.

    Run under every parallel_compute_method, because the horizon is packed per compute
    job and how the domain is split into jobs differs between them.
    """
    continuous, chunked = _continuous_and_chunked(
        days=2, window_hours=window_hours, persist_days=1,
        method=method, cpu_pool=cpu_pool,
    )
    assert np.array_equal(continuous, chunked)
    # The horizon really ends: assimilated flow on day 1, level pool on day 2. The
    # selected issue is dated t0, so a horizon measured from it lands a day in.
    assert _rfc_frames(1)[1]["issue_time"].iloc[0] == pd.Timestamp(_T0)
    assert continuous[:_STEPS_PER_DAY].max() > 10.0
    assert continuous[_STEPS_PER_DAY:].max() < 1.0


def test_every_parallel_method_gives_the_same_answer():
    """The DA result must not depend on how the domain was partitioned."""
    reference = None
    for method, cpu_pool in (
        ("serial", 1), ("by-subnetwork-jit", 2), ("by-subnetwork-jit-clustered", 2)
    ):
        continuous, _ = _continuous_and_chunked(
            days=2, window_hours=6, persist_days=1, method=method, cpu_pool=cpu_pool
        )
        if reference is None:
            reference = continuous
        else:
            assert np.array_equal(continuous, reference), method
