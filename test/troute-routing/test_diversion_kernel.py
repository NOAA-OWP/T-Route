"""The diversion subtraction through the compiled kernel, against an analytic answer.

Every other diversion test is directional (the donor's peak falls, the receiver's
rises) or bookkeeping (the map resolves). None asks whether the amount removed equals
the amount observed. A subtraction left in the stored outflow is read back next step
as the segment's previous outflow ``qdp``, and Muskingum-Cunge carries it forward
through ``C3``, so the donor settles at ``U - d / (1 - C3)`` instead of ``U - d``: 1.25
times the observed diversion at Old River, where ``C3`` is 0.2.

Two chains with no lakes: 10 -> 20 -> 30 is the donor river with a steady inflow at 10
and the diversion taken at 20; 40 -> 50 is the receiving river, with the diversion gage
at the headwater 40. A steady inflow and a constant observation have one right answer.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

if TYPE_CHECKING:
    from numpy.typing import NDArray

from troute import nhd_network
from troute.routing.compute import compute_nhd_routing_v02

_T0 = datetime(2011, 5, 1)
_DT = 300
_QTS = 12  # forcing columns are hourly
_CONNECTIONS = {10: [20], 20: [30], 30: [], 40: [50], 50: []}
_CHANNELS = [10, 20, 30, 40, 50]
_DONOR, _GAGE = 20, 40
_INFLOW, _DIVERTED = 1000.0, 100.0

_WB_COLS = ["LkArea", "LkMxE", "OrificeA", "OrificeC", "OrificeE",
            "WeirC", "WeirE", "WeirL", "ifd", "qd0", "h0"]


def _frames(hours: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reaches = pd.DataFrame(
        # Short and steep enough that Km clamps to dt and X to its floor, which is
        # the Old River donor's regime (dx = 308 m): C3 = 0.2 exactly.
        {"bw": 40.0, "tw": 80.0, "twcc": 200.0, "dx": 1000.0, "n": 0.03, "ncc": 0.06,
         "cs": 1.0, "s0": 0.001, "alt": 10.0},
        index=_CHANNELS,
    )
    columns = [(_T0 + timedelta(hours=h)).strftime("%Y%m%d%H%M") for h in range(hours)]
    qlats = pd.DataFrame(0.0, index=_CHANNELS, columns=columns)
    qlats.loc[10] = _INFLOW
    q0 = pd.DataFrame({"qu0": 0.0, "qd0": 0.0, "h0": 0.5, "ql0": 0.0}, index=_CHANNELS)
    return reaches, qlats, q0


def _observations(nts: int) -> pd.DataFrame:
    # Positional columns, exactly nts + 1 wide: column 0 seeds t0, columns 1..nts are
    # the routing steps. The same width every window, so the value at t0 is column 0.
    return pd.DataFrame(_DIVERTED, index=[_GAGE], columns=range(nts + 1))


def _route(
    t0: datetime,
    nts: int,
    reaches: pd.DataFrame,
    qlats: pd.DataFrame,
    q0: pd.DataFrame,
    *,
    plan: object,
    qlat_add_loc: str,
    obs: pd.DataFrame | None = None,
    diversion_applied: dict[int, float] | None = None,
) -> tuple[object, object]:
    """One kernel call. Positional, because that is the production signature."""
    if obs is None:
        obs = _observations(nts)
    empty = pd.DataFrame()
    rconn = nhd_network.reverse_network(_CONNECTIONS)
    independent = nhd_network.reachable_network(rconn)
    waterbodies = pd.DataFrame(columns=_WB_COLS, dtype=float)
    types = pd.DataFrame(columns=["reservoir_type"], dtype=int)
    return compute_nhd_routing_v02(
        _CONNECTIONS, rconn, {}, {tw: [] for tw in independent},
        "V02-structured", "serial", 0, 1, t0, _DT, nts, _QTS, independent,
        reaches.copy(), q0, qlats, empty, 0.0, obs, empty,
        empty, empty, empty, empty, empty, empty, empty, empty, empty, empty, empty,
        {"da_decay_coefficient": 120}, True, False, waterbodies, {}, types, False, plan,
        from_files=False, qlat_add_loc=qlat_add_loc,
        diversion_da={_DONOR: _GAGE}, gage_segments={_GAGE},
        diversion_applied=diversion_applied or {},
    )


def _obs_window(obs: pd.DataFrame, start: int, nts: int) -> pd.DataFrame:
    """Columns start..start+nts of a run-long positional frame, relabeled from 0."""
    window = obs.iloc[:, start:start + nts + 1].copy()
    window.columns = range(nts + 1)
    return window


def _flow(results: object, link: int, nts: int) -> NDArray[np.float64]:
    for r in results:  # type: ignore[attr-defined]  the kernel returns a list of tuples
        ids = np.asarray(r[0])
        where = np.where(ids == link)[0]
        if where.size:
            return np.asarray(r[1])[where[0]].reshape(nts, -1)[:, 0].astype(float)
    raise KeyError(f"the kernel did not return link {link}")


def _nudge(results: object, link: int) -> NDArray[np.float64]:
    """The kernel's nudge at a gage link, one column per routing step from t0."""
    for r in results:  # type: ignore[attr-defined]
        gages = list(np.asarray(r[3][0]))
        if link in gages:
            return np.asarray(r[9])[gages.index(link)].astype(float)
    raise KeyError(f"the kernel returned no nudge for link {link}")


def _carry_q0(results: object) -> pd.DataFrame:
    return pd.concat([
        pd.DataFrame(np.asarray(r[1])[:, [-4, -4, -2, -1]], index=r[0],
                     columns=["qu0", "qd0", "h0", "ql0"])
        for r in results  # type: ignore[attr-defined]
    ])


@pytest.mark.parametrize("qlat_add_loc", ["middle", "bottom"])
def test_steady_diversion_removes_exactly_the_observed_amount(qlat_add_loc: str) -> None:
    """Settled donor = inflow minus observation; settled receiver = observation.

    A subtraction fed back through the state settles the donor at ``U - d / (1 - C3)``
    and the downstream link inherits the shortfall; both must sit on ``U - d`` within a
    cms. The receiver is a replacement, not a routed quantity, so it is exact either way.
    """
    hours = 120
    nts = hours * _QTS
    reaches, qlats, q0 = _frames(hours)
    results, _ = _route(_T0, nts, reaches, qlats, q0, plan=[None, None, None],
                        qlat_add_loc=qlat_add_loc)

    settled = slice(-_QTS, None)  # the last hour
    donor = _flow(results, _DONOR, nts)[settled]
    below = _flow(results, 30, nts)[settled]
    upstream = _flow(results, 10, nts)[settled]
    receiver = _flow(results, 50, nts)[settled]

    assert np.allclose(upstream, _INFLOW, atol=1.0), "the inflow did not settle"
    removed = upstream - donor
    assert np.allclose(removed, _DIVERTED, atol=1.0), (
        f"donor gave up {removed.mean():.1f} cms for an observed {_DIVERTED:.0f}: the "
        "subtraction is feeding back through the routing state"
    )
    assert np.allclose(below, _INFLOW - _DIVERTED, atol=1.0)
    assert np.allclose(receiver, _DIVERTED, atol=1.0)


@pytest.mark.parametrize("qlat_add_loc", ["middle", "bottom"])
def test_one_step_subtraction_leaves_no_tail(qlat_add_loc: str) -> None:
    """A subtraction at one step must not echo into later steps, on either side.

    If the subtracted outflow were the routing state, a single-step diversion would
    leave a geometric tail (C3, C3 squared, ...) in the donor's flow. With the state
    restored the donor returns to the inflow at the very next step.

    The receiving gage stops with it: no nudge after the observation, so what is left
    is the channel draining the water it was handed (the observation is the node's
    stored outflow, which Muskingum-Cunge carries through C3, here 0.95 a step). A
    decaying nudge would hold it near the observation for hours instead.
    """
    hours = 120
    nts = hours * _QTS
    pulse_step = nts - 24
    obs = pd.DataFrame(np.nan, index=[_GAGE], columns=range(nts + 1))
    obs.loc[_GAGE, pulse_step] = _DIVERTED
    reaches, qlats, q0 = _frames(hours)
    empty = pd.DataFrame()
    rconn = nhd_network.reverse_network(_CONNECTIONS)
    independent = nhd_network.reachable_network(rconn)
    waterbodies = pd.DataFrame(columns=_WB_COLS, dtype=float)
    types = pd.DataFrame(columns=["reservoir_type"], dtype=int)
    results, _ = compute_nhd_routing_v02(
        _CONNECTIONS, rconn, {}, {tw: [] for tw in independent},
        "V02-structured", "serial", 0, 1, _T0, _DT, nts, _QTS, independent,
        reaches.copy(), q0, qlats, empty, 0.0, obs, empty,
        empty, empty, empty, empty, empty, empty, empty, empty, empty, empty, empty,
        {"da_decay_coefficient": 120}, True, False, waterbodies, {}, types, False,
        [None, None, None], from_files=False, qlat_add_loc=qlat_add_loc,
        diversion_da={_DONOR: _GAGE}, gage_segments={_GAGE},
    )
    donor = _flow(results, _DONOR, nts)
    receiver = _flow(results, _GAGE, nts)
    # Returned row j is routing step j + 1, so the pulse lands on row pulse_step - 1.
    assert np.isclose(donor[pulse_step - 1], _INFLOW - _DIVERTED, atol=1.0)
    np.testing.assert_allclose(donor[pulse_step:pulse_step + 4], _INFLOW, atol=1.0)
    assert np.isclose(receiver[pulse_step - 1], _DIVERTED, atol=1.0)
    nudge = _nudge(results, _GAGE)
    assert np.isclose(nudge[pulse_step], _DIVERTED, atol=1.0)
    np.testing.assert_array_equal(nudge[pulse_step + 1:pulse_step + 24], 0.0)
    tail = receiver[pulse_step - 1:pulse_step + 23]
    assert np.all(np.diff(tail) < 0), "the receiver is being held up after the observation"
    assert tail[12] < 70, tail[12]
    assert tail[18] < 10, tail[18]


@pytest.mark.parametrize("qlat_add_loc", ["middle", "bottom"])
def test_windowed_run_matches_continuous(qlat_add_loc: str) -> None:
    """The remembered subtraction must cross a window boundary.

    The initial condition of the second window is the first window's last stored
    outflow, which already has the subtraction in it. The kernel seeds the remembered
    amount from observation column 0, the previous window's last step; without that
    the second window's first step routes from a state short by one subtraction.
    """
    hours, window = 96, 24
    reaches, qlats, q0 = _frames(hours)
    results, _ = _route(_T0, hours * _QTS, reaches, qlats, q0, plan=[None, None, None],
                        qlat_add_loc=qlat_add_loc)
    continuous = _flow(results, _DONOR, hours * _QTS)

    chunks, plan, carried, t0 = [], [None, None, None], q0.copy(), _T0
    for hour in range(0, hours, window):
        results, plan = _route(
            t0, window * _QTS, reaches, qlats[qlats.columns[hour:hour + window]],
            carried, plan=plan, qlat_add_loc=qlat_add_loc,
        )
        chunks.append(_flow(results, _DONOR, window * _QTS))
        carried = _carry_q0(results)
        t0 += timedelta(hours=window)
    windowed = np.concatenate(chunks)

    np.testing.assert_allclose(windowed, continuous, rtol=1e-5, atol=1e-3)


@pytest.mark.parametrize("boundary", ["clamp", "missing"])
def test_the_applied_amount_is_carried_across_the_boundary(boundary: str) -> None:
    """Column 0 stands in for the previous window's subtraction only when it equals it.

    Two cases where it does not: the observation at the boundary step exceeds the
    donor's flow, so the clamp removes less than column 0 says; and the next window's
    frame has no value at its first column although the previous step subtracted.
    Handing the applied amount over as state keeps the chunked run on the continuous
    one; seeding from column 0 puts the boundary step off by C3 times the difference.
    """
    hours, window = 48, 24
    nts, half = hours * _QTS, window * _QTS
    obs = pd.DataFrame(_DIVERTED, index=[_GAGE], columns=range(nts + 1))
    if boundary == "clamp":
        obs.loc[_GAGE, half] = 50 * _INFLOW  # more than the river carries
    reaches, qlats, q0 = _frames(hours)
    results, _ = _route(_T0, nts, reaches, qlats, q0, plan=[None, None, None],
                        qlat_add_loc="middle", obs=obs)
    continuous = _flow(results, _DONOR, nts)

    def chunked(carry_state: bool) -> NDArray[np.float64]:
        chunks, plan, carried, applied, t0 = [], [None, None, None], q0.copy(), {}, _T0
        for hour in range(0, hours, window):
            window_obs = _obs_window(obs, hour * _QTS, half)
            if boundary == "missing" and hour:
                window_obs.loc[_GAGE, 0] = np.nan
            results, plan = _route(
                t0, half, reaches, qlats[qlats.columns[hour:hour + window]], carried,
                plan=plan, qlat_add_loc="middle", obs=window_obs,
                diversion_applied=applied if carry_state else {},
            )
            chunks.append(_flow(results, _DONOR, half))
            carried = _carry_q0(results)
            applied = results.diversion_applied()  # type: ignore[attr-defined]
            t0 += timedelta(hours=window)
        return np.concatenate(chunks)

    np.testing.assert_allclose(chunked(True), continuous, rtol=1e-5, atol=1e-3)
    off = np.abs(chunked(False) - continuous)[half - 1:half + 2].max()
    assert off > 10.0, f"column 0 seeding should miss the boundary step, was off by {off:.2f}"


_JUNCTION = {10: [20], 15: [20], 20: [30], 30: [35], 35: [], 40: [50], 50: []}
_JUNCTION_CHANNELS = [10, 15, 20, 30, 35, 40, 50]


def _route_junction(
    method: str, target: int, *, t0: datetime, nts: int, qlats: pd.DataFrame,
    q0: pd.DataFrame, obs: pd.DataFrame, plan: object, applied: dict[int, float],
) -> tuple[object, object]:
    """The donor sits at a junction, where a subnetwork plan cuts, so a downstream job
    holds it as an off-network upstream it never routes."""
    reaches = pd.DataFrame(
        {"bw": 40.0, "tw": 80.0, "twcc": 200.0, "dx": 1000.0, "n": 0.03, "ncc": 0.06,
         "cs": 1.0, "s0": 0.001, "alt": 10.0},
        index=_JUNCTION_CHANNELS,
    )
    empty = pd.DataFrame()
    rconn = nhd_network.reverse_network(_JUNCTION)
    independent = nhd_network.reachable_network(rconn)
    waterbodies = pd.DataFrame(columns=_WB_COLS, dtype=float)
    types = pd.DataFrame(columns=["reservoir_type"], dtype=int)
    return compute_nhd_routing_v02(
        _JUNCTION, rconn, {}, {tw: [] for tw in independent},
        "V02-structured", method, target, 1, t0, _DT, nts, _QTS, independent,
        reaches, q0, qlats, empty, 0.0, obs, empty,
        empty, empty, empty, empty, empty, empty, empty, empty, empty, empty, empty,
        {"da_decay_coefficient": 120}, True, False, waterbodies, {}, types, False, plan,
        from_files=False, qlat_add_loc="middle", diversion_da={_DONOR: _GAGE},
        gage_segments={_GAGE}, diversion_applied=applied,
    )


@pytest.mark.parametrize(("method", "target"), [
    ("by-subnetwork-jit", 1), ("by-subnetwork-jit", 2), ("by-subnetwork-jit-clustered", 1),
])
def test_the_carried_state_comes_from_the_job_that_routes_the_donor(
    method: str, target: int,
) -> None:
    """A job that only holds the donor as a boundary row must not report its state.

    Such a job never routes the donor, so its remembered amount stays at the seed;
    reported anyway, the stale copy overwrote the routing job's amount and the next
    window restored the wrong subtraction.
    """
    hours, window = 48, 24
    nts, half = hours * _QTS, window * _QTS
    columns = [(_T0 + timedelta(hours=h)).strftime("%Y%m%d%H%M") for h in range(hours)]
    qlats = pd.DataFrame(0.0, index=_JUNCTION_CHANNELS, columns=columns)
    qlats.loc[10], qlats.loc[15] = _INFLOW, 10.0
    q0 = pd.DataFrame({"qu0": 0.0, "qd0": 0.0, "h0": 0.5, "ql0": 0.0}, index=_JUNCTION_CHANNELS)
    obs = pd.DataFrame(_DIVERTED, index=[_GAGE], columns=range(nts + 1))
    obs.loc[_GAGE, half] = 60.0  # the boundary step asks for less than the seed

    results, _ = _route_junction(method, target, t0=_T0, nts=nts, qlats=qlats, q0=q0, obs=obs,
                                 plan=[None, None, None], applied={})
    continuous = _flow(results, _DONOR, nts)

    chunks, plan, carried, applied, t0 = [], [None, None, None], q0.copy(), {}, _T0
    for hour in range(0, hours, window):
        results, plan = _route_junction(
            method, target, t0=t0, nts=half, qlats=qlats[qlats.columns[hour:hour + window]],
            q0=carried, obs=_obs_window(obs, hour * _QTS, half), plan=plan, applied=applied,
        )
        chunks.append(_flow(results, _DONOR, half))
        carried = _carry_q0(results)
        applied = results.diversion_applied()  # type: ignore[attr-defined]
        if hour == 0:
            assert applied == {_DONOR: pytest.approx(60.0, abs=1e-3)}
        t0 += timedelta(hours=window)
    np.testing.assert_allclose(np.concatenate(chunks), continuous, rtol=1e-5, atol=1e-3)
