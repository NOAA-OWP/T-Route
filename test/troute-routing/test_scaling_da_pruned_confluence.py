"""Edge Case 1 and Edge Case 2 interacting at a partially pruned confluence.

At a confluence where one branch carries its own gage, Edge Case 1 removes that branch from
this gage's tree. Edge Case 2 still splits the parent increment across ALL branches in
proportion to their modeled flow, so the surviving branch is owed ``Q_b / sum(Q)`` and the
stopped branch's share must go UNALLOCATED -- the gage that owns it applies its own
correction there, and reassigning that share double-counts against it.

The defect this pins: ``branch_sum`` was summed over surviving branches only, and the
denominator is ``max(Q_parent, branch_sum)``. The routed parent flow is the only remaining
trace of the stopped branch, and on a rising limb it lags BELOW the surviving branch alone.
The clamp then selected the surviving branch's own flow as the denominator and handed it a
factor of 1.0 -- the entire parent increment, stopped branch's share included.

Steady state hid it: there ``Q_parent >= sum(Q_branch)``, so the denominator was the parent
flow and the split was already correct. Only the lagged regime exposes it, which is the same
regime the non-amplification clamp exists to handle (measured at 38% of active confluence
timesteps on the spun-up continental flood).

Everything here drives the production path (``apply_scaling_da``, which is the
compiled kernel); the NumPy reference this arithmetic was first stated in was
retired once the compiled port's equivalence was proven (see git history for
``scaling_da_numpy_reference.py`` and its 2000-case fuzz).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from troute.routing.fast_reach.scaling_da import apply_scaling_da
from troute.scaling_da import build_gage_trees_from_mappings

THETA = 0.77
Q_SURVIVING, Q_STOPPED = 12.0, 8.0
EXPECTED_SHARE = Q_SURVIVING / (Q_SURVIVING + Q_STOPPED)   # Edge Case 2: 0.6


def _trees():
    """Confluence 100 fed by 101 (survives) and 102 (has gage H, so it is pruned)."""
    rconn = {100: [101, 102], 101: [], 102: []}
    area = {100: 1000.0, 101: 600.0, 102: 400.0}
    built = build_gage_trees_from_mappings(rconn, {"G": 100, "H": 102}, area,
                                           theta_default=THETA)
    return {"G": built["G"]}


def _apply(q_parent, trees):
    idx = pd.date_range("2020-01-01", periods=2, freq="1h")
    q_model = pd.DataFrame([[q_parent, Q_SURVIVING, Q_STOPPED]] * 2,
                           index=idx, columns=[100, 101, 102], dtype=float)
    q_obs = pd.DataFrame([[q_parent + 10.0]] * 2, index=idx, columns=["G"], dtype=float)
    _, dq = apply_scaling_da(q_model, q_obs, {"G": 100}, trees, min_flow_cms=1e-6)
    return dq


def test_the_stopped_branch_is_recorded_against_its_confluence():
    """The tree must remember what the stop rule removed, or the split cannot be right."""
    t = _trees()["G"]
    assert list(np.asarray(t.seg_order).ravel()) == [100, 101]     # 102 pruned
    assert list(np.asarray(t.pruned_segs).ravel()) == [102]
    assert list(np.asarray(t.pruned_ptr).ravel()) == [0, 1, 1]     # attached to node 0


def test_positions_resolve_for_the_pruned_branch():
    t = _trees()["G"].with_positions({100: 0, 101: 1, 102: 2})
    assert list(np.asarray(t.pruned_positions).ravel()) == [2]
    assert list(np.asarray(t.pruned_pos_ptr).ravel()) == [0, 1, 1]


@pytest.mark.parametrize(
    ("q_parent", "label"),
    [(20.0, "steady: parent flow equals the branch sum"),
     (4.0, "rising limb: parent flow lags below the surviving branch alone")],
)
def test_split_is_proportional_in_both_regimes(q_parent, label):
    dq = _apply(q_parent, _trees())
    assert dq.iloc[0][101] / dq.iloc[0][100] == pytest.approx(EXPECTED_SHARE), label


def test_the_stopped_branch_share_is_not_reassigned():
    """The surviving branch must never receive the whole parent increment.

    This is the failure directly: before the fix the rising-limb case handed the
    surviving branch a share of 1.0.
    """
    dq = _apply(4.0, _trees())
    share = dq.iloc[0][101] / dq.iloc[0][100]
    assert share < 1.0
    assert share == pytest.approx(EXPECTED_SHARE)


def test_an_unpruned_confluence_is_unchanged():
    """No pruning: both branches receive proportional shares that sum to the parent."""
    rconn = {100: [101, 102], 101: [], 102: []}
    area = {100: 1000.0, 101: 600.0, 102: 400.0}
    trees = {"G": build_gage_trees_from_mappings(rconn, {"G": 100}, area,
                                                 theta_default=THETA)["G"]}
    assert np.asarray(trees["G"].pruned_segs).size == 0
    dq = _apply(20.0, trees)
    assert dq.iloc[0][101] + dq.iloc[0][102] == pytest.approx(dq.iloc[0][100])


def test_lag_edge_falls_back_to_the_latest_increment():
    """A shift past the series end must clamp to the latest increment, not
    zero: the LAST timestep seeds q0 for the forecast, and zeroing it silently
    nulled the prognostic hand-off in an earlier revision."""
    rconn = {100: [101], 101: []}
    area = {100: 1000.0, 101: 600.0}
    trees = {"G": build_gage_trees_from_mappings(rconn, {"G": 100}, area,
                                                 theta_default=THETA)["G"]}
    nt = 8
    idx = pd.date_range("2020-01-01", periods=nt, freq="1h")
    q_model = pd.DataFrame({100: 20.0, 101: 12.0}, index=idx, dtype=float)
    dq_o = np.full(nt, 5.0)
    factor = (600.0 / 1000.0) ** THETA
    for shift in (0, 1, 3):
        lag = {"G": (np.ones(2), np.array([0, shift], dtype=np.int64))}
        _, dq = apply_scaling_da(q_model, None, {"G": 100}, trees,
                                 min_flow_cms=1e-6, dq_o_by_site={"G": dq_o},
                                 lag_by_site=lag)
        assert dq.iloc[-1][101] == pytest.approx(5.0 * factor), (
            f"shift {shift} lost the seeded step"
        )
