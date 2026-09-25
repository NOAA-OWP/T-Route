"""The Muskingum-Cunge kernel must not publish an abandoned Secant guess as depth.

Publishing it returned ``h`` as the next step's ``depthp``, re-seeding the same
failing search, so depth wandered indefinitely under steady forcing.
"""

from __future__ import annotations

from typing import Any

import pytest

reach = pytest.importorskip(
    "troute.routing.fast_reach.reach",
    reason="compiled Muskingum-Cunge kernel is not built",
)

# A fast recession: true normal depth is centimeters, the search starts meters
# away at bankfull, and only the restart from the normal-depth seed reaches it.
_CHANNEL: dict[str, float] = {
    "dt": 300.0,
    "dx": 100.0,
    "bw": 50.0,
    "tw": 75.0,
    "twcc": 150.0,
    "n": 0.02,
    "ncc": 0.30,
    "cs": 0.5,
    "s0": 1.0e-4,
}
_DEPTH0 = 6.25
_NORMAL_DEPTH = (1.0 * _CHANNEL["n"] / (_CHANNEL["bw"] * _CHANNEL["s0"] ** 0.5)) ** 0.6


def _step(
    depthp: float,
    velp: float = 0.5,
    qup: float = 1.0,
    quc: float = 1.0,
    qdp: float = 1.0,
    ql: float = 0.0,
) -> dict[str, Any]:
    c = _CHANNEL
    return reach.compute_reach_kernel(
        c["dt"], qup, quc, qdp, ql, c["dx"], c["bw"], c["tw"], c["twcc"],
        c["n"], c["ncc"], c["cs"], c["s0"], velp, depthp,
    )


def test_a_failing_search_restarts_and_settles() -> None:
    """Depth lands on the normal depth and stays there under steady forcing."""
    depthp, velp = _DEPTH0, 0.5
    depths: list[float] = []
    for _ in range(20):
        out = _step(depthp, velp)
        depthp, velp = out["depthc"], out["velc"]
        depths.append(depthp)

    assert depths[0] == pytest.approx(_NORMAL_DEPTH, rel=0.05)
    assert depths[3:] == pytest.approx([depths[-1]] * len(depths[3:]), rel=1e-6), (
        f"depth drifted under steady forcing: {depths}"
    )


def test_the_settled_depth_is_not_a_trap() -> None:
    """A step with new forcing leaves the settled depth immediately."""
    depthp, velp = _DEPTH0, 0.5
    for _ in range(8):
        out = _step(depthp, velp)
        depthp, velp = out["depthc"], out["velc"]
    assert depthp == pytest.approx(_NORMAL_DEPTH, rel=0.05)

    out = _step(depthp, velp, qup=400.0, quc=400.0, qdp=400.0)
    assert out["depthc"] > 10.0 * depthp, "depth stayed pinned after the forcing returned"


def test_asymmetric_inflows_solve_from_bankfull() -> None:
    """Asymmetric inflows and non-zero ql pin the solve, not the mean of equal inflows."""
    out = _step(_DEPTH0, qup=0.4, quc=1.0, qdp=0.7, ql=0.3)
    assert out["depthc"] == pytest.approx(0.1204259, rel=1e-6)
    assert out["qdc"] == pytest.approx(0.7313878, rel=1e-6)
    assert out["velc"] == pytest.approx(0.1214524, rel=1e-6)


def test_an_abandoned_search_holds_the_previous_depth() -> None:
    """When every restart fails too, the fallback publishes the previous depth, not a guess."""
    out = reach.compute_reach_kernel(
        300.0, 0.0, 0.0, 0.0, 51.27, 1000.0, 112.795, 135.354, 270.709,
        0.02, 0.12, 0.5, 1.0849e-7, 0.5, 0.5,
    )
    assert out["depthc"] == 0.5
    assert 0.0 < out["qdc"] < 51.27


def test_converging_case_still_solves() -> None:
    """A well-posed solve is untouched: it converges away from its initial guess."""
    out = reach.compute_reach_kernel(
        300.0, 0.5, 0.5, 0.5, 0.01, 1000.0, 10.0, 20.0, 30.0,
        0.035, 0.05, 1.0, 0.001, 0.5, 1.0,
    )
    assert 0.0 < out["depthc"] < 1.0
    assert out["qdc"] == pytest.approx(0.5023310, rel=1e-6)
