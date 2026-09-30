"""The Muskingum-Cunge depth search must not publish a near-zero depth for a segment with
water to route.

Seeded at a near-zero previous depth, the secant's first step is negative and the halving
that follows exits below ``mindepth``, which the absolute-error test alone would read as a
root: a segment publishing a centimeter of depth passes little more than its own previous
outflow whatever arrives from upstream. The search restarts from the normal depth the flow
implies, and these cases pin that from a trickle to a flood, cold and warm, and along a
cold chain.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

if TYPE_CHECKING:
    from numpy.typing import NDArray

from troute import nhd_network
from troute.routing.compute import compute_nhd_routing_v02
from troute.routing.fast_reach.reach import compute_reach_kernel

_DT = 300.0
# A stream-order-2 creek as the NHF carries it, in 312 m links: bankfull depth 112 m by its
# geometry, so 8,556 cms sits at about 724 m. Far outside the channel, which is the point.
_CREEK = {"dx": 312.076, "bw": 2.4, "tw": 9.16122, "twcc": 27.4837, "n": 0.076, "ncc": 0.152,
          "cs": 0.03, "s0": 0.00013317}
_MAINSTEM = {"dx": 313.823, "bw": 16.0, "tw": 176.073, "twcc": 528.22, "n": 0.021, "ncc": 0.042,
             "cs": 0.04, "s0": 0.000372191}
_INFLOW = 8556.0


def _step(p: dict[str, float], qup: float, quc: float, qdp: float, depthp: float) -> tuple[float, float]:
    """One kernel step with no lateral inflow: (outflow, depth)."""
    r = compute_reach_kernel(_DT, qup, quc, qdp, 0.0, p["dx"], p["bw"], p["tw"], p["twcc"],
                             p["n"], p["ncc"], p["cs"], p["s0"], 0.0, depthp)
    return float(r["qdc"]), float(r["depthc"])


@pytest.mark.parametrize("qdp", [0.0, 500.0, 2000.0, 6000.0])
def test_a_cold_creek_link_handed_a_flood_routes_it(qdp: float) -> None:
    """From zero depth the answer must match the one found from a plausible depth."""
    q_cold, h_cold = _step(_CREEK, _INFLOW, _INFLOW, qdp, 0.0)
    q_warm, h_warm = _step(_CREEK, _INFLOW, _INFLOW, qdp, 300.0)
    assert h_warm > 100.0
    # The search stops at a 1 percent step, so the published depth wanders by about a
    # tenth along the flat residual; the outflow does not.
    assert h_cold == pytest.approx(h_warm, rel=0.15), (h_cold, h_warm)
    assert q_cold == pytest.approx(q_warm, rel=0.02), (q_cold, q_warm)


def test_a_cold_mainstem_link_handed_a_flood_routes_it() -> None:
    q_cold, h_cold = _step(_MAINSTEM, _INFLOW, _INFLOW, 500.0, 0.0)
    q_warm, h_warm = _step(_MAINSTEM, _INFLOW, _INFLOW, 500.0, 12.0)
    assert h_cold == pytest.approx(h_warm, rel=0.15)
    assert q_cold == pytest.approx(q_warm, rel=0.02)


def test_a_converged_link_is_untouched() -> None:
    """The restart only fires when the search collapsed; a settled link keeps its root."""
    q, h = _step(_CREEK, _INFLOW, _INFLOW, 8551.0, 724.15)
    assert h == pytest.approx(724.47, abs=0.05)
    assert q == pytest.approx(8555.0, abs=0.5)


def test_a_dry_channel_stays_dry() -> None:
    assert _step(_CREEK, 0.0, 0.0, 0.0, 0.0) == (0.0, 0.0)


def test_a_trickle_from_zero_depth_gets_its_own_depth() -> None:
    """Small flows have their own restart depth; the guard is not only for floods."""
    q, h = _step(_CREEK, 0.3, 0.3, 0.3, 0.0)
    q_ref, h_ref = _step(_CREEK, 0.3, 0.3, 0.3, 0.5)
    assert h == pytest.approx(h_ref, rel=0.15)
    assert q == pytest.approx(q_ref, rel=0.02)


# Two ordinary channels at ordinary flows: a small channel at 2 cms, whose collapsed
# depth lies close to the detection threshold, and a steep creek at 183 cms, whose
# bottom-width seed stalls on its own bracket.
_SMALL = {"dx": 400.0, "bw": 6.0, "tw": 24.0, "twcc": 36.0, "n": 0.035, "ncc": 0.07, "cs": 2.0,
          "s0": 0.0045}
_STEEP_CREEK = {"dx": 1298.4591761556376, "bw": 3.19046172932243, "tw": 4.429399087361535,
                "twcc": 25.88722950560005, "n": 0.032652956975075406, "ncc": 0.15431844489021518,
                "cs": 0.14735109051673106, "s0": 0.0031506677238497204}


# A wide floodplain (1,500 m) over an 80 m channel at 16 cms: its collapsed depth,
# 1.9 cm, sits below any width-based bound; only the residual catches it.
_FLOODPLAIN = {"dx": 150.0, "bw": 80.0, "tw": 320.0, "twcc": 1500.0, "n": 0.03, "ncc": 0.06,
               "cs": 1.4, "s0": 0.006}


@pytest.mark.parametrize(("p", "q", "qdp", "warm"), [
    (_SMALL, 2.0, 0.2, 0.3), (_STEEP_CREEK, 183.05714410385465, 18.305714410385466, 0.1),
    (_FLOODPLAIN, 16.0, 1.6, 0.2),
])
def test_ordinary_channels_from_zero_depth_match_their_warm_answer(
    p: dict[str, float], q: float, qdp: float, warm: float,
) -> None:
    q_cold, h_cold = _step(p, q, q, qdp, 0.0)
    q_warm, h_warm = _step(p, q, q, qdp, warm)
    assert h_cold > 0.05
    assert h_cold == pytest.approx(h_warm, rel=0.15), (h_cold, h_warm)
    assert q_cold == pytest.approx(q_warm, rel=0.06), (q_cold, q_warm)


# A bottom width wider than the top width: the kernel treats it as a trapezoid that keeps
# widening, so no fixed width bounds its depth. Its steady root is legitimate and must
# not be retried step after step.
_WIDENING = {"dx": 500.0, "bw": 10.0, "tw": 5.0, "twcc": 5.0, "n": 0.035, "ncc": 0.07,
             "cs": 0.1, "s0": 0.0001}


def test_a_widening_section_keeps_its_steady_root() -> None:
    q, h = _step(_WIDENING, 1000.0, 1000.0, 1000.0, 10.225)
    assert h == pytest.approx(10.225, rel=0.02)
    assert q == pytest.approx(1000.0, rel=0.01)
    q_cold, h_cold = _step(_WIDENING, 1000.0, 1000.0, 1000.0, 0.0)
    assert h_cold == pytest.approx(h, rel=0.15)
    assert q_cold == pytest.approx(q, rel=0.02)


_WIDE_FLOODPLAIN = {"dx": 500.0, "bw": 10.0, "tw": 20.0, "twcc": 1000.0, "n": 0.035, "ncc": 0.07,
                    "cs": 1.0, "s0": 0.005}


def test_lateral_inflow_alone_from_zero_depth_gets_the_warm_answer() -> None:
    """Two restart seeds can land on the same degenerate root; their agreement is not a root."""
    def step(depthp: float) -> tuple[float, float]:
        p = _WIDE_FLOODPLAIN
        r = compute_reach_kernel(_DT, 0.0, 0.0, 0.0, 10.0, p["dx"], p["bw"], p["tw"], p["twcc"],
                                 p["n"], p["ncc"], p["cs"], p["s0"], 0.0, depthp)
        return float(r["qdc"]), float(r["depthc"])

    q_cold, h_cold = step(0.0)
    q_warm, h_warm = step(1.0)
    assert h_cold == pytest.approx(h_warm, rel=0.15), (h_cold, h_warm)
    assert q_cold == pytest.approx(q_warm, rel=0.02), (q_cold, q_warm)


_LINKS = list(range(1, 17))  # a 16-link chain, the inflow imposed at link 1


def _route_chain(hours: int) -> NDArray[np.float64]:
    """Outlet flow of a cold 16-link creek chain fed a constant 8,556 cms at its head."""
    t0 = datetime(2011, 5, 1)
    nts = hours * 12
    conn = {a: [b] for a, b in pairwise(_LINKS)}
    conn[_LINKS[-1]] = []
    reaches = pd.DataFrame({**_CREEK, "alt": 0.0}, index=_LINKS)
    cols = [(t0 + timedelta(hours=h)).strftime("%Y%m%d%H%M") for h in range(hours)]
    qlats = pd.DataFrame(0.0, index=_LINKS, columns=cols)
    q0 = pd.DataFrame({"qu0": 0.0, "qd0": 0.0, "h0": 0.5, "ql0": 0.0}, index=_LINKS)
    obs = pd.DataFrame(_INFLOW, index=[_LINKS[0]], columns=range(nts + 1))
    empty = pd.DataFrame()
    rconn = nhd_network.reverse_network(conn)
    independent = nhd_network.reachable_network(rconn)
    wb_cols = ["LkArea", "LkMxE", "OrificeA", "OrificeC", "OrificeE", "WeirC", "WeirE", "WeirL",
               "ifd", "qd0", "h0"]
    waterbodies = pd.DataFrame(columns=wb_cols, dtype=float)
    types = pd.DataFrame(columns=["reservoir_type"], dtype=int)
    results, _ = compute_nhd_routing_v02(
        conn, rconn, {}, {tw: [] for tw in independent}, "V02-structured", "serial", 0, 1,
        t0, int(_DT), nts, 12, independent, reaches, q0, qlats, empty, 0.0, obs, empty,
        empty, empty, empty, empty, empty, empty, empty, empty, empty, empty, empty,
        {"da_decay_coefficient": 120}, True, False, waterbodies, {}, types, False,
        [None, None, None], from_files=False, qlat_add_loc="bottom", diversion_da={},
        gage_segments={_LINKS[0]},
    )
    for r in results:  # type: ignore[attr-defined]  the kernel returns a list of tuples
        ids = np.asarray(r[0])
        where = np.where(ids == _LINKS[-1])[0]
        if where.size:
            return np.asarray(r[1])[where[0]].reshape(nts, -1)[:, 0].astype(float)
    raise KeyError("the kernel did not return the outlet link")


def test_a_cold_chain_fills_within_hours_not_days() -> None:
    """Sixteen 312 m links of the creek, a constant 8,556 cms at the head from a cold
    start: the outlet must carry the inflow within six hours, not days."""
    outlet = _route_chain(12)
    assert outlet[6 * 12 - 1] > 0.95 * _INFLOW, outlet[6 * 12 - 1]
    np.testing.assert_allclose(outlet[-12:], _INFLOW, rtol=0.01)
