"""Synthetic correctness tests for the simple-scaling DA kernel.

Ported verbatim (numeric parity) from the standalone proof-of-concept. The
kernel applies area-scaling (Eq. 2) along linear steps and the flow-ratio
split (Edge Case 2) at confluences, as the white paper prescribes.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from troute.routing.fast_reach.scaling_da import apply_scaling_da
from troute.scaling_da import build_one_gage_tree


def _series_index(n: int = 3) -> pd.DatetimeIndex:
    return pd.date_range("2018-02-01", periods=n, freq="h", name="time")


def test_recovery_at_gage() -> None:
    """Corrected Q at the gage equals the observation."""
    idx = _series_index(2)
    # Single segment, just the gage.
    q_model = pd.DataFrame({1: [5.0, 8.0]}, index=idx)
    q_obs = pd.DataFrame({"A": [10.0, 12.0]}, index=idx)
    tree = build_one_gage_tree(
        gage_fp=1,
        rconn={},
        area_sqkm={1: 100.0},
        stop_segs=frozenset(),
        theta=0.77,
    )
    q_corr, dq = apply_scaling_da(q_model, q_obs, {"A": 1}, {"A": tree})
    # Q_corrected at gage matches obs.
    np.testing.assert_array_almost_equal(q_corr[1].to_numpy(), [10.0, 12.0])
    # delta-Q at gage = obs - model
    np.testing.assert_array_almost_equal(dq[1].to_numpy(), [5.0, 4.0])


def test_eq2_area_scaling_telescopes_on_a_chain() -> None:
    """Each upstream segment gets dQ_o * (A_s/A_o)^theta (Eq. 2), independent of Q."""
    idx = _series_index(2)
    # Chain: 4 -> 3 -> 2 -> 1 (gage)
    q_model = pd.DataFrame(
        {1: [5.0, 8.0], 2: [4.0, 6.0], 3: [3.0, 5.0], 4: [1.0, 2.0]},
        index=idx,
    )
    q_obs = pd.DataFrame({"A": [10.0, 12.0]}, index=idx)  # dQ_o = [5, 4]
    areas = {1: 100.0, 2: 80.0, 3: 60.0, 4: 20.0}
    theta = 0.77
    tree = build_one_gage_tree(
        gage_fp=1,
        rconn={1: [2], 2: [3], 3: [4]},
        area_sqkm=areas,
        stop_segs=frozenset(),
        theta=theta,
    )
    _q_corr, dq = apply_scaling_da(q_model, q_obs, {"A": 1}, {"A": tree})
    dq_o = np.array([5.0, 4.0])
    for fp in (2, 3, 4):
        ratio = (areas[fp] / areas[1]) ** theta
        np.testing.assert_allclose(dq[fp].to_numpy(), dq_o * ratio, rtol=1e-9)


def test_flow_ratio_split_at_junction() -> None:
    """At a confluence the parent correction splits by modeled flow (Edge Case 2).

    The white paper applies area-scaling along a linear step (Eq. 2) and, at a
    confluence, splits the parent correction between branches in proportion to
    their modeled flow: dQ_branch = dQ_parent * Q_branch / Q_parent. Because the
    branch flows sum to the parent flow, the split conserves dQ across the
    junction.
    """
    idx = _series_index(1)
    # 3 and 4 merge at 2, then 2 -> 1 (gage): linear step 1->2, junction 2->{3,4}.
    q_model = pd.DataFrame({1: [20.0], 2: [16.0], 3: [6.0], 4: [10.0]}, index=idx)
    q_obs = pd.DataFrame({"A": [30.0]}, index=idx)  # dQ_o = 10
    tree = build_one_gage_tree(
        gage_fp=1,
        rconn={1: [2], 2: [3, 4]},
        area_sqkm={1: 100.0, 2: 80.0, 3: 30.0, 4: 50.0},
        stop_segs=frozenset(),
        theta=0.5,
    )
    _, dq = apply_scaling_da(q_model, q_obs, {"A": 1}, {"A": tree})
    # Linear step 1->2: area-scaling (A_2/A_1)^theta = (80/100)^0.5.
    dq2 = 10.0 * (80.0 / 100.0) ** 0.5
    np.testing.assert_allclose(dq[2].iloc[0], dq2, rtol=1e-9)
    # Junction 2->{3,4}: flow-ratio split of dQ(2).
    np.testing.assert_allclose(dq[3].iloc[0], dq2 * (6.0 / 16.0), rtol=1e-9)
    np.testing.assert_allclose(dq[4].iloc[0], dq2 * (10.0 / 16.0), rtol=1e-9)
    # Flow-ratio conserves dQ across the junction (Q_3 + Q_4 = Q_2).
    np.testing.assert_allclose(dq[3].iloc[0] + dq[4].iloc[0], dq[2].iloc[0], rtol=1e-9)


def test_flow_ratio_guards_dry_junction() -> None:
    """A confluence whose parent has near-zero modeled flow propagates no correction."""
    idx = _series_index(1)
    q_model = pd.DataFrame({1: [10.0], 2: [0.0], 3: [0.0], 4: [0.0]}, index=idx)
    q_obs = pd.DataFrame({"A": [20.0]}, index=idx)  # dQ_o = 10
    tree = build_one_gage_tree(
        gage_fp=1,
        rconn={1: [2], 2: [3, 4]},
        area_sqkm={1: 100.0, 2: 100.0, 3: 50.0, 4: 50.0},
        stop_segs=frozenset(),
        theta=0.5,
    )
    _, dq = apply_scaling_da(q_model, q_obs, {"A": 1}, {"A": tree})
    # Gage corrected; node 2 (linear, (100/100)^0.5 = 1) gets the full dQ_o;
    # the junction children sit on a dry parent -> no correction.
    np.testing.assert_allclose(dq[1].iloc[0], 10.0)
    np.testing.assert_allclose(dq[2].iloc[0], 10.0)
    assert dq[3].iloc[0] == 0.0
    assert dq[4].iloc[0] == 0.0


def test_flow_ratio_at_junction_with_stopped_branch() -> None:
    """A confluence keeps the flow-ratio split when one branch is stopped (Edge Case 1 + 2)."""
    idx = _series_index(1)
    # 3 and 4 merge at 2, then 2 -> 1 (gage); branch 4 is stopped (upstream gage/reservoir).
    q_model = pd.DataFrame({1: [20.0], 2: [16.0], 3: [6.0]}, index=idx)
    q_obs = pd.DataFrame({"A": [30.0]}, index=idx)  # dQ_o = 10
    tree = build_one_gage_tree(
        gage_fp=1,
        rconn={1: [2], 2: [3, 4]},
        area_sqkm={1: 100.0, 2: 80.0, 3: 30.0, 4: 50.0},
        stop_segs=frozenset({4}),
        theta=0.5,
    )
    assert tree.seg_order.tolist() == [1, 2, 3]  # branch 4 pruned by the stop rule
    # Node 2 is still a physical confluence (rconn[2] = [3, 4]).
    assert bool(tree.step_is_junction[tree.seg_order.tolist().index(3)])
    _, dq = apply_scaling_da(q_model, q_obs, {"A": 1}, {"A": tree})
    dq2 = 10.0 * (80.0 / 100.0) ** 0.5  # linear step 1->2
    np.testing.assert_allclose(dq[2].iloc[0], dq2, rtol=1e-9)
    # Surviving branch 3 takes the flow-ratio split Q_3/Q_2, NOT area-scaling.
    np.testing.assert_allclose(dq[3].iloc[0], dq2 * (6.0 / 16.0), rtol=1e-9)
    assert not np.isclose(dq[3].iloc[0], dq2 * (30.0 / 80.0) ** 0.5)


def test_flow_ratio_single_branch_not_amplified() -> None:
    """A lone surviving branch modeled higher than its routed parent is not amplified.

    Root-cause fix (vs the old output cap): the split denominator is
    max(Q_parent, sum branch flows), so a single surviving branch with Q_3 > Q_parent
    divides by Q_3 itself -> factor 1, dq3 == dq2 (not 3*dq2). Guards the telescoped
    multiplier against blow-up when routing lag drives Q_parent below its tributary
    (un-spun-up cold start, or a flood rising limb at CONUS scale).
    """
    idx = _series_index(1)
    # 1 (gage) -> 2 (confluence) -> {3, 4}, branch 4 stopped. Q_3 (12) > Q_2 (4).
    q_model = pd.DataFrame({1: [10.0], 2: [4.0], 3: [12.0]}, index=idx)
    q_obs = pd.DataFrame({"A": [20.0]}, index=idx)  # dQ_o = 10 at gage
    tree = build_one_gage_tree(
        gage_fp=1,
        rconn={1: [2], 2: [3, 4]},
        area_sqkm={1: 100.0, 2: 80.0, 3: 60.0, 4: 40.0},
        stop_segs=frozenset({4}),
        theta=0.5,
    )
    _, dq = apply_scaling_da(q_model, q_obs, {"A": 1}, {"A": tree})
    dq2 = 10.0 * (80.0 / 100.0) ** 0.5  # linear step 1->2
    # Junction 2->3: denom = max(Q_2=4, sum branch=12) = 12, factor = 12/12 = 1.
    np.testing.assert_allclose(dq[3].iloc[0], dq2, rtol=1e-9)
    assert dq[3].iloc[0] < 2.0 * dq2  # decisively not amplified


def test_flow_ratio_conserves_on_lagged_confluence() -> None:
    """The split never manufactures water when routing lag makes Q_parent < sum branches.

    THE water-balance guard. When two branches each appear to carry near the parent's
    routed flow (Q_3=Q_4=6 vs Q_2=4, a lagged confluence), the old per-branch
    `min(factor, 1)` cap clamped each to 1 -> dq3+dq4 = 2*dq2, creating a full dq2 of
    water. The root-cause fix divides by max(Q_parent, sum branches) = 12, so the
    factors sum to exactly 1 and dq3+dq4 == dq2 -- the correction is conserved across
    the junction, not amplified.
    """
    idx = _series_index(1)
    # 1 (gage) -> 2 (confluence) -> {3, 4}, both present. Q_3 + Q_4 (12) > Q_2 (4).
    q_model = pd.DataFrame({1: [10.0], 2: [4.0], 3: [6.0], 4: [6.0]}, index=idx)
    q_obs = pd.DataFrame({"A": [20.0]}, index=idx)  # dQ_o = 10 at gage
    tree = build_one_gage_tree(
        gage_fp=1,
        rconn={1: [2], 2: [3, 4]},
        area_sqkm={1: 100.0, 2: 80.0, 3: 40.0, 4: 40.0},
        stop_segs=frozenset(),
        theta=0.5,
    )
    _, dq = apply_scaling_da(q_model, q_obs, {"A": 1}, {"A": tree})
    dq2 = 10.0 * (80.0 / 100.0) ** 0.5
    # denom = max(4, 6+6=12) = 12; factor_3 = factor_4 = 6/12 = 0.5.
    np.testing.assert_allclose(dq[3].iloc[0], dq2 * 0.5, rtol=1e-9)
    np.testing.assert_allclose(dq[4].iloc[0], dq2 * 0.5, rtol=1e-9)
    # CONSERVED: dq3 + dq4 == dq2 (the old cap gave 2*dq2 -- a manufactured dq2).
    np.testing.assert_allclose(dq[3].iloc[0] + dq[4].iloc[0], dq2, rtol=1e-9)


def test_theta_is_one_scalar_per_tree() -> None:
    """A tree carries ONE theta, so a non-uniform tree cannot be constructed.

    The linear step telescopes to dQ_o*(A_s/A_o)^theta only for a constant exponent.
    That used to be a runtime check inside the correction (and a duplicate one in the
    compiled path); holding theta as a scalar on the tree makes it structural, so
    there is no longer a malformed-tree case to reject mid-run.
    """
    tree = build_one_gage_tree(
        gage_fp=1, rconn={1: [2]}, area_sqkm={1: 100.0, 2: 50.0}, stop_segs=frozenset(), theta=0.77
    )
    assert isinstance(tree.theta, float)
    assert tree.theta == 0.77
    assert not hasattr(tree, "seg_thetas")

def test_nonpositive_decay_tau_raises() -> None:
    idx = _series_index(2)
    q_model = pd.DataFrame({1: [5.0, 8.0]}, index=idx)
    q_obs = pd.DataFrame({"A": [10.0, np.nan]}, index=idx)
    tree = build_one_gage_tree(
        gage_fp=1, rconn={}, area_sqkm={1: 100.0}, stop_segs=frozenset(), theta=0.77
    )
    for bad in (0.0, -60.0):
        with pytest.raises(ValueError, match="obs_decay_tau_min must be"):
            apply_scaling_da(q_model, q_obs, {"A": 1}, {"A": tree}, obs_decay_tau_min=bad)


def test_overlapping_trees_emit_disjointness_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Two DA trees sharing a segment trigger the not-disjoint warning."""
    sda_logger = logging.getLogger("TROUTE")

    idx = _series_index(1)
    q_model = pd.DataFrame({1: [10.0], 2: [10.0], 3: [5.0]}, index=idx)
    q_obs = pd.DataFrame({"A": [12.0], "B": [12.0]}, index=idx)
    # Both trees include segment 3 (artificial overlap; the real stop rule prevents this).
    tree_a = build_one_gage_tree(1, {1: [3]}, {1: 100.0, 3: 50.0}, frozenset(), theta=0.77)
    tree_b = build_one_gage_tree(2, {2: [3]}, {2: 100.0, 3: 50.0}, frozenset(), theta=0.77)

    sda_logger.addHandler(caplog.handler)  # robust whether or not "TROUTE" propagates
    try:
        apply_scaling_da(q_model, q_obs, {"A": 1, "B": 2}, {"A": tree_a, "B": tree_b})
    finally:
        sda_logger.removeHandler(caplog.handler)
    assert any("not disjoint" in rec.message for rec in caplog.records)


def test_theta_changes_correction() -> None:
    """theta controls the concavity of the upstream transfer."""
    idx = _series_index(1)
    q_model = pd.DataFrame({1: [10.0], 2: [5.0]}, index=idx)
    q_obs = pd.DataFrame({"A": [20.0]}, index=idx)  # dQ_o = 10
    tree = build_one_gage_tree(
        gage_fp=1,
        rconn={1: [2]},
        area_sqkm={1: 100.0, 2: 50.0},
        stop_segs=frozenset(),
        theta=0.77,
    )
    # theta = 1 (linear): dQ(2) = 10 * (50/100)^1 = 5.
    tree_lin = build_one_gage_tree(
        gage_fp=1,
        rconn={1: [2]},
        area_sqkm={1: 100.0, 2: 50.0},
        stop_segs=frozenset(),
        theta=1.0,
    )
    _, dq_lin = apply_scaling_da(q_model, q_obs, {"A": 1}, {"A": tree_lin})
    np.testing.assert_allclose(dq_lin[2].iloc[0], 5.0, rtol=1e-9)
    # theta = 0.77: dQ(2) = 10 * (0.5)^0.77 > 5 (concave, theta < 1).
    _, dq_default = apply_scaling_da(q_model, q_obs, {"A": 1}, {"A": tree})
    np.testing.assert_allclose(dq_default[2].iloc[0], 10.0 * 0.5**0.77, rtol=1e-9)
    assert dq_default[2].iloc[0] > dq_lin[2].iloc[0]


def test_clamps_negative_q_to_zero() -> None:
    """Q_model + dQ < 0 should clamp at 0, not propagate negative discharge."""
    idx = _series_index(1)
    # dQ_o = 0 - 10 = -10. On segment 2: dQ = -10*(50/100)^0.77 ~ -5.86,
    # so Q_corrected(2) = 3 - 5.86 < 0 -> clamp to 0.
    q_model = pd.DataFrame({1: [10.0], 2: [3.0]}, index=idx)
    q_obs = pd.DataFrame({"A": [0.0]}, index=idx)
    tree = build_one_gage_tree(
        gage_fp=1,
        rconn={1: [2]},
        area_sqkm={1: 100.0, 2: 50.0},
        stop_segs=frozenset(),
        theta=0.77,
    )
    q_corr, _dq = apply_scaling_da(q_model, q_obs, {"A": 1}, {"A": tree})
    assert q_corr[2].iloc[0] == 0.0
    assert q_corr[1].iloc[0] == 0.0


def test_missing_observation_skips_correction() -> None:
    """NaN at a timestep means no correction is applied for that gage at that step."""
    idx = _series_index(2)
    q_model = pd.DataFrame({1: [5.0, 8.0], 2: [3.0, 4.0]}, index=idx)
    q_obs = pd.DataFrame({"A": [10.0, np.nan]}, index=idx)
    tree = build_one_gage_tree(
        gage_fp=1,
        rconn={1: [2]},
        area_sqkm={1: 100.0, 2: 50.0},
        stop_segs=frozenset(),
        theta=0.77,
    )
    q_corr, _dq = apply_scaling_da(q_model, q_obs, {"A": 1}, {"A": tree})
    # First step: dQ_o = 5 -> applied. Second step: NaN -> no change.
    assert q_corr[1].iloc[0] == 10.0
    assert q_corr[1].iloc[1] == 8.0  # unchanged
    assert q_corr[2].iloc[1] == 4.0  # unchanged


def test_gage_to_fp_mismatch_skips_site() -> None:
    """If gage_to_fp[site] != tree.gage_fp, the site is skipped (no correction)."""
    idx = _series_index(1)
    q_model = pd.DataFrame({1: [5.0]}, index=idx)
    q_obs = pd.DataFrame({"A": [10.0]}, index=idx)
    tree = build_one_gage_tree(
        gage_fp=1,
        rconn={},
        area_sqkm={1: 100.0},
        stop_segs=frozenset(),
        theta=0.77,
    )
    # Mismatched mapping: site 'A' supposedly at fp 999, but tree was built for fp 1.
    q_corr, dq = apply_scaling_da(q_model, q_obs, {"A": 999}, {"A": tree})
    # Skipped -> no correction applied.
    assert q_corr[1].iloc[0] == 5.0
    assert dq[1].iloc[0] == 0.0


def test_missing_obs_column_skips_site() -> None:
    idx = _series_index(1)
    q_model = pd.DataFrame({1: [5.0]}, index=idx)
    q_obs = pd.DataFrame({"B": [10.0]}, index=idx)  # no 'A' column
    tree = build_one_gage_tree(
        gage_fp=1,
        rconn={},
        area_sqkm={1: 100.0},
        stop_segs=frozenset(),
        theta=0.77,
    )
    # Should not raise.
    q_corr, _dq = apply_scaling_da(q_model, q_obs, {"A": 1}, {"A": tree})
    # No correction -> Q unchanged.
    assert q_corr[1].iloc[0] == 5.0


def test_obs_decay_persists_last_dq() -> None:
    """With obs_decay_tau_min set, stale dQ persists as dQ_last * exp(-dt/tau)."""
    # 3 hourly steps; obs only at t=0. tau = 60 min -> at t=1 (dt=60), factor e^-1.
    idx = pd.date_range("2018-02-01", periods=3, freq="h", name="time")
    q_model = pd.DataFrame({1: [5.0, 5.0, 5.0], 2: [3.0, 3.0, 3.0]}, index=idx)
    q_obs = pd.DataFrame({"A": [15.0, np.nan, np.nan]}, index=idx)
    areas = {1: 100.0, 2: 60.0}
    theta = 0.77
    tree = build_one_gage_tree(
        gage_fp=1,
        rconn={1: [2]},
        area_sqkm=areas,
        stop_segs=frozenset(),
        theta=theta,
    )
    q_corr, dq = apply_scaling_da(
        q_model,
        q_obs,
        {"A": 1},
        {"A": tree},
        obs_decay_tau_min=60.0,
    )
    # dQ_o(0) = 15 - 5 = 10. Apply at gage -> q_corr[1, t=0] = 15.
    np.testing.assert_allclose(q_corr[1].iloc[0], 15.0)
    # Stale dt=60min -> factor exp(-60/60) = 1/e. q_corr = 5 + 10/e.
    np.testing.assert_allclose(q_corr[1].iloc[1], 5.0 + 10.0 / np.e, rtol=1e-9)
    # Stale dt=120min -> factor exp(-120/60) = 1/e^2. q_corr = 5 + 10/e^2.
    np.testing.assert_allclose(q_corr[1].iloc[2], 5.0 + 10.0 / (np.e**2), rtol=1e-9)
    # Upstream gets the decayed dQ_o area-scaled by (A_2/A_o)^theta.
    ratio = (areas[2] / areas[1]) ** theta
    np.testing.assert_allclose(dq[2].iloc[1], (10.0 / np.e) * ratio, rtol=1e-9)
    np.testing.assert_allclose(dq[2].iloc[2], (10.0 / (np.e**2)) * ratio, rtol=1e-9)


def test_obs_decay_resets_on_fresh_obs() -> None:
    """A fresh observation overrides decay; the new dQ_o is what gets propagated."""
    idx = pd.date_range("2018-02-01", periods=4, freq="h", name="time")
    q_model = pd.DataFrame({1: [10.0, 10.0, 10.0, 10.0]}, index=idx)
    # Obs at t=0 (dQ=5), gap at t=1, fresh obs at t=2 (dQ=-3), gap at t=3.
    q_obs = pd.DataFrame({"A": [15.0, np.nan, 7.0, np.nan]}, index=idx)
    tree = build_one_gage_tree(
        gage_fp=1,
        rconn={},
        area_sqkm={1: 100.0},
        stop_segs=frozenset(),
        theta=0.77,
    )
    q_corr, _dq = apply_scaling_da(
        q_model,
        q_obs,
        {"A": 1},
        {"A": tree},
        obs_decay_tau_min=60.0,
    )
    np.testing.assert_allclose(q_corr[1].iloc[0], 15.0)
    # t=1 decay from dQ=5.
    np.testing.assert_allclose(q_corr[1].iloc[1], 10.0 + 5.0 / np.e, rtol=1e-9)
    # t=2 fresh obs.
    np.testing.assert_allclose(q_corr[1].iloc[2], 7.0)
    # t=3 decay from new dQ = 7-10 = -3.
    np.testing.assert_allclose(q_corr[1].iloc[3], 10.0 + (-3.0) / np.e, rtol=1e-9)


def test_obs_decay_none_matches_zero_behavior() -> None:
    """obs_decay_tau_min=None must reproduce the legacy zero-during-gap behavior exactly."""
    idx = pd.date_range("2018-02-01", periods=3, freq="h", name="time")
    q_model = pd.DataFrame({1: [5.0, 5.0, 5.0]}, index=idx)
    q_obs = pd.DataFrame({"A": [15.0, np.nan, np.nan]}, index=idx)
    tree = build_one_gage_tree(
        gage_fp=1,
        rconn={},
        area_sqkm={1: 100.0},
        stop_segs=frozenset(),
        theta=0.77,
    )
    q_corr, _dq = apply_scaling_da(q_model, q_obs, {"A": 1}, {"A": tree})
    # t=0 obs applied; t=1,2 no correction -> q_corr = q_model.
    np.testing.assert_allclose(q_corr[1].iloc[0], 15.0)
    np.testing.assert_allclose(q_corr[1].iloc[1], 5.0)
    np.testing.assert_allclose(q_corr[1].iloc[2], 5.0)


def test_decay_state_carries_across_loops() -> None:
    """decay_state + time_ref continue the exponential persistence across calls."""
    ref = pd.Timestamp("2000-01-01 00:00")
    idx1 = pd.date_range("2000-01-01 00:00", periods=2, freq="h")
    idx2 = pd.date_range("2000-01-01 02:00", periods=2, freq="h")
    tree = build_one_gage_tree(1, {}, {1: 100.0}, frozenset(), theta=0.77)
    state: dict = {}
    qm1 = pd.DataFrame({1: [5.0, 5.0]}, index=idx1)
    qo1 = pd.DataFrame({"A": [15.0, np.nan]}, index=idx1)  # dQ=10 at t0
    qc1, _ = apply_scaling_da(
        qm1, qo1, {"A": 1}, {"A": tree}, obs_decay_tau_min=60.0, decay_state=state, time_ref=ref
    )
    np.testing.assert_allclose(qc1[1].iloc[0], 15.0, rtol=1e-9)
    np.testing.assert_allclose(qc1[1].iloc[1], 5.0 + 10.0 / np.e, rtol=1e-9)  # 60 min gap
    # loop 2: no fresh obs -> decay CONTINUES from loop-1 obs (120, 180 min gaps)
    qm2 = pd.DataFrame({1: [5.0, 5.0]}, index=idx2)
    qo2 = pd.DataFrame({"A": [np.nan, np.nan]}, index=idx2)
    qc2, _ = apply_scaling_da(
        qm2, qo2, {"A": 1}, {"A": tree}, obs_decay_tau_min=60.0, decay_state=state, time_ref=ref
    )
    np.testing.assert_allclose(qc2[1].iloc[0], 5.0 + 10.0 / np.e**2, rtol=1e-9)
    np.testing.assert_allclose(qc2[1].iloc[1], 5.0 + 10.0 / np.e**3, rtol=1e-9)


def test_no_decay_state_has_no_cross_call_memory() -> None:
    """Without decay_state (legacy), an all-gap call applies nothing."""
    idx = pd.date_range("2000-01-01 02:00", periods=2, freq="h")
    tree = build_one_gage_tree(1, {}, {1: 100.0}, frozenset(), theta=0.77)
    qm = pd.DataFrame({1: [5.0, 5.0]}, index=idx)
    qo = pd.DataFrame({"A": [np.nan, np.nan]}, index=idx)
    qc, _ = apply_scaling_da(qm, qo, {"A": 1}, {"A": tree}, obs_decay_tau_min=60.0)
    np.testing.assert_array_equal(qc[1].to_numpy(), [5.0, 5.0])
