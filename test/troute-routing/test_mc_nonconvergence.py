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

# A fast recession: true normal depth is centimetres, the search starts metres
# away at bankfull, and the 1.33x expansion cannot reach it. Fails every step.
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


def test_abandoned_search_does_not_move_depth() -> None:
    """Depth is a fixed point while the search keeps failing under steady forcing."""
    depthp, velp = _DEPTH0, 0.5
    depths: list[float] = []
    for _ in range(20):
        out = _step(depthp, velp)
        depthp, velp = out["depthc"], out["velc"]
        depths.append(depthp)

    assert depths == pytest.approx([_DEPTH0] * len(depths), rel=1e-6), (
        f"depth drifted off the antecedent value under steady forcing: {depths}"
    )


def test_fallback_releases_once_the_solve_is_possible_again() -> None:
    """Holding depth is not a trap: a solvable step takes over immediately."""
    depthp, velp = _DEPTH0, 0.5
    for _ in range(8):
        out = _step(depthp, velp)
        depthp, velp = out["depthc"], out["velc"]
    assert depthp == pytest.approx(_DEPTH0, rel=1e-6)

    out = _step(depthp, velp, qup=400.0, quc=400.0, qdp=400.0)
    assert out["depthc"] != pytest.approx(_DEPTH0, rel=1e-3), (
        "depth stayed pinned after the forcing returned, so the fallback is a trap"
    )


def test_fallback_flow_uses_the_published_depth() -> None:
    """The fallback re-evaluates C1..C4 at the depth it publishes.

    Inflows are asymmetric and ql non-zero, so qdc depends on the coefficients
    rather than collapsing to the mean of equal inflows.
    """
    out = _step(_DEPTH0, qup=0.4, quc=1.0, qdp=0.7, ql=0.3)
    assert out["depthc"] == pytest.approx(_DEPTH0, rel=1e-6)
    assert out["qdc"] == pytest.approx(0.7587749, rel=1e-6)
    assert out["velc"] == pytest.approx(1.4641856, rel=1e-6)


def test_converging_case_still_solves() -> None:
    """A well-posed solve is untouched: it converges away from its initial guess."""
    out = reach.compute_reach_kernel(
        300.0, 0.5, 0.5, 0.5, 0.01, 1000.0, 10.0, 20.0, 30.0,
        0.035, 0.05, 1.0, 0.001, 0.5, 1.0,
    )
    assert 0.0 < out["depthc"] < 1.0
    assert out["qdc"] == pytest.approx(0.5023310, rel=1e-6)
