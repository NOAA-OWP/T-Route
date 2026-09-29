"""A run-of-river RFC dam passes its inflow while no forecast controls it.

The dam keeps zero active storage, releasing its inflow at a held level, and an RFC
forecast in control still overrides it. Routes a reach, a type-4 lake and a reach through
compute_nhd_routing_v02 with test_rfc_persist_kernel's RFC fixture.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from troute import nhd_network
from troute.AbstractNetwork import AbstractNetwork
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
_QTS = 12
_DAY = 86400 // _DT
_H0 = 102.0
_WB_COLS = ["LkArea", "LkMxE", "OrificeA", "OrificeC", "OrificeE",
            "WeirC", "WeirE", "WeirL", "ifd", "qd0", "h0", "pass_through"]


def _rfc_frames(persist_days: float):
    stamps, d = [], _T0 - timedelta(hours=28)
    while d <= _T0:
        stamps.append(d.strftime("%Y-%m-%d_%H"))
        d += timedelta(hours=1)
    raw = _read_timeseries_files(
        str(_FIXTURES), stamps, _T0, _T0 + timedelta(days=persist_days), routing_period=_DT
    )
    crosswalk = pd.DataFrame({"rfc_gage_id": [_GAGE], "rfc_lake_id": [_LAKE]}).set_index(
        "rfc_lake_id")
    return assemble_rfc_dataframes(
        raw, crosswalk, pd.Timestamp(_T0), {"reservoir_rfc_forecast_persist_days": persist_days},
    )


def _frames(hours: int, pass_through: float, inflow: float = 20.0):
    reaches = pd.DataFrame(
        {"bw": 10.0, "tw": 20.0, "twcc": 60.0, "dx": 2000.0, "n": 0.03, "ncc": 0.06,
         "cs": 1.0, "s0": 0.001, "alt": 100.0},
        index=_CHANNELS,
    )
    waterbodies = pd.DataFrame(
        [[1.0, 105.0, 1.0, 0.1, 100.0, 0.4, 103.0, 10.0, 0.9, 1.0, _H0, pass_through]],
        index=[_LAKE], columns=_WB_COLS,
    )
    types = pd.DataFrame({"reservoir_type": [4]}, index=[_LAKE])
    columns = [(_T0 + timedelta(hours=h)).strftime("%Y%m%d%H%M") for h in range(hours)]
    qlats = pd.DataFrame(0.0, index=_CHANNELS, columns=columns)
    # A flood through the lake, so a level pool and a pass-through tell apart.
    qlats.loc[10] = inflow * (1.0 + 4.0 * np.exp(-((np.arange(hours) - 18) / 6.0) ** 2))
    q0 = pd.DataFrame({"qu0": 0.0, "qd0": 0.0, "h0": 0.1, "ql0": 0.0}, index=_CHANNELS)
    return reaches, waterbodies, types, qlats, q0


def _route(t0, nts, frames, rfc, plan):
    """One kernel call with the production signature. *frames* is what _frames returns."""
    reaches, waterbodies, types, qlats, q0 = frames
    rfc_df, rfc_params = rfc
    empty = pd.DataFrame()
    rconn = nhd_network.reverse_network(_CONNECTIONS)
    independent = nhd_network.reachable_network(rconn)
    return compute_nhd_routing_v02(
        _CONNECTIONS, rconn, {_LAKE: [_LAKE]}, {tw: [] for tw in independent},
        "V02-structured", "serial", 0, 1, t0, _DT, nts, _QTS, independent,
        reaches.copy(), q0, qlats, empty, 0.0, empty, empty, empty, empty, empty,
        empty, empty, empty, rfc_df, rfc_params, empty, empty, empty, {}, True, False,
        waterbodies, {}, types, True, plan, from_files=False, qlat_add_loc="middle",
    )


def _series(results, nts, segment, column):
    for r in results:
        ids = np.asarray(r[0])
        where = np.where(ids == segment)[0]
        if where.size:
            return np.asarray(r[1])[where[0]].reshape(nts, -1)[:, column].astype(float)
    raise KeyError(f"the kernel did not return {segment}")


def _run(hours, pass_through, rfc_days=None, inflow=20.0):
    reaches, waterbodies, types, qlats, q0 = _frames(hours, pass_through, inflow)
    rfc_df, rfc_params = _rfc_frames(rfc_days) if rfc_days else (pd.DataFrame(), pd.DataFrame())
    nts = hours * _QTS
    results, _ = _route(_T0, nts, (reaches, waterbodies, types, qlats, q0),
                        (rfc_df, rfc_params), [None, None, None])
    return (_series(results, nts, 10, 0), _series(results, nts, _LAKE, 0),
            _series(results, nts, _LAKE, 2))


# With assume_short_ts the kernel hands a reservoir its upstream's previous-step flow, so
# the lake's inflow at step j is the upstream reach's flow at j - 1.
def _handed(upstream: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], upstream[:-1]])


def test_with_no_forecast_the_dam_passes_its_inflow_and_keeps_its_level():
    upstream, outflow, stage = _run(48, pass_through=1.0)
    assert np.allclose(outflow, _handed(upstream), rtol=1e-6)
    assert np.allclose(stage, _H0, atol=1e-4)


def test_a_lake_without_the_flag_still_routes_as_a_level_pool():
    upstream, outflow, stage = _run(48, pass_through=0.0)
    assert outflow.max() < 0.99 * upstream.max()   # the pool attenuates the flood
    assert stage.max() - stage.min() > 0.1


def test_a_forecast_in_control_overrides_the_pass_through():
    """A 1 day horizon over 2 days: forecast releases on day 1, inflow on day 2."""
    upstream, outflow, stage = _run(48, pass_through=1.0, rfc_days=1)
    inflow = _handed(upstream)
    assert outflow[:_DAY].max() > 10.0
    assert not np.allclose(outflow[:_DAY], inflow[:_DAY])
    assert np.allclose(outflow[_DAY + 1:], inflow[_DAY + 1:], rtol=1e-6)
    # The level RFC control left is held, not reset to the start.
    held = stage[_DAY + 1:]
    assert np.allclose(held, held[0], atol=1e-4)
    assert abs(held[0] - _H0) > 1e-3


def test_windows_agree_with_one_continuous_run():
    """The held level crosses window boundaries in q0, as the drivers carry it."""
    hours, window = 48, 6
    reaches, waterbodies, types, qlats, q0 = _frames(hours, 1.0)
    rfc_df, rfc_params = _rfc_frames(1)
    results, _ = _route(_T0, hours * _QTS, (reaches, waterbodies.copy(), types, qlats,
                        q0.copy()), (rfc_df, rfc_params.copy()), [None, None, None])
    continuous = _series(results, hours * _QTS, _LAKE, 0)

    class _Carry:
        _q0 = None

    chunks, plan = [], [None, None, None]
    wb, carried, params, t0, hour = waterbodies.copy(), q0.copy(), rfc_params.copy(), _T0, 0
    while hour < hours:
        results, plan = _route(t0, window * _QTS, (reaches, wb, types,
                               qlats[qlats.columns[hour:hour + window]], carried),
                               (rfc_df, params), plan)
        chunks.append(_series(results, window * _QTS, _LAKE, 0))
        carried = AbstractNetwork.new_q0(_Carry(), results)
        wb.update(carried)
        params = _set_rfc_reservoir_da_params(params, results)
        t0 += timedelta(hours=window)
        hour += window
    assert np.array_equal(continuous, np.concatenate(chunks))


def test_an_inflow_that_is_not_a_flow_falls_to_the_level_pool_guards():
    reaches, waterbodies, types, qlats, q0 = _frames(24, 1.0)
    qlats.iloc[0, 6] = np.nan
    nts = 24 * _QTS
    results, _ = _route(_T0, nts, (reaches, waterbodies, types, qlats, q0),
                        (pd.DataFrame(), pd.DataFrame()), [None, None, None])
    outflow = _series(results, nts, _LAKE, 0)
    stage = _series(results, nts, _LAKE, 2)
    assert np.isfinite(stage).all()
    assert np.isfinite(outflow[np.isfinite(_series(results, nts, 10, 0))]).all()


@pytest.mark.parametrize("start", [np.inf, -np.inf, np.nan])
def test_a_level_the_level_pool_repairs_stays_repaired(start):
    """The level pool resets a non-finite level to the orifice invert; the pass-through
    must not write the bad level back over the repair."""
    reaches, waterbodies, types, qlats, q0 = _frames(12, 1.0)
    waterbodies["h0"] = start
    nts = 12 * _QTS
    results, _ = _route(_T0, nts, (reaches, waterbodies, types, qlats, q0),
                        (pd.DataFrame(), pd.DataFrame()), [None, None, None])
    assert np.isfinite(_series(results, nts, _LAKE, 2)).all()
