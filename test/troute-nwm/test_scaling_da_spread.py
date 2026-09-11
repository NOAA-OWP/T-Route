"""Celerity-free lag: distance localization plus temporal spreading.

Both replace quantities that could not be estimated defensibly. The propagation
limit uses reach length, which the hydrofabric knows exactly. The lag is a
spread rather than a shift, because Muskingum-Cunge routes a diffusion wave and
one gage increment corresponds to a range of upstream times.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from nwm_routing.scaling_da_apply import ScalingDA

from troute.scaling_da import build_gage_trees_from_mappings


def _da(**kw):
    o = ScalingDA.__new__(ScalingDA)
    for k, v in kw.items():
        setattr(o, k, v)
    return o


def test_distance_accumulates_the_parent_chain_in_km():
    trees = build_gage_trees_from_mappings(
        {100: [101], 101: [102], 102: []}, {"G": 100},
        {100: 30.0, 101: 20.0, 102: 10.0}, theta_default=0.77,
    )
    da = _da(max_reach_km=200.0,
             _dx=pd.Series({100: 5000.0, 101: 3000.0, 102: 1000.0}, dtype=float))
    dist, bad = da._tree_distance_km(trees["G"])
    assert bad == 0
    of = dict(zip(map(int, trees["G"].seg_order), dist))
    assert of[100] == pytest.approx(0.0)
    assert of[101] == pytest.approx(5.0)          # crosses 100
    assert of[102] == pytest.approx(5.0 + 3.0)    # plus 101


def test_reach_without_length_is_held_outside_the_limit():
    """Distance 0 would keep a whole subtree inside the limit: fail closed."""
    trees = build_gage_trees_from_mappings(
        {100: [101], 101: []}, {"G": 100}, {100: 30.0, 101: 20.0}, theta_default=0.77
    )
    da = _da(max_reach_km=50.0, _dx=pd.Series({101: 1000.0}, dtype=float))
    dist, bad = da._tree_distance_km(trees["G"])
    assert bad == 1
    of = dict(zip(map(int, trees["G"].seg_order), dist))
    assert of[101] > 50.0


def test_spread_looks_forward_only_and_conserves_the_increment():
    """The window must average dQ_o AHEAD of each step, never behind it.

    An upstream segment at time t is corrected by what the gage reports once
    this water arrives, so an innovation at step 10 must reach steps at and
    before 10 and no step after: a later step's water has already gone past.
    """
    da = _da(innovation_spread_h=4.0)
    n = 21
    dq = np.zeros(n)
    dq[10] = 12.0
    out = da._smooth_innovation(dq, dt=3600.0)
    win = 5  # 4 h at dt = 1 h, inclusive of the step itself
    assert out.shape == dq.shape
    assert out.sum() == pytest.approx(dq.sum(), rel=1e-9)   # volume preserved
    assert out.max() < dq.max()                             # attenuated
    assert not (out[11:] > 0).any()                         # nothing after the spike
    assert (out[10 - win + 1:11] > 0).all()                 # the full window before it
    idx = np.arange(n)
    centroid = (out * idx).sum() / out.sum()
    assert centroid == pytest.approx(10.0 - (win - 1) / 2)


def test_zero_width_leaves_the_innovation_untouched():
    da = _da(innovation_spread_h=0.0)
    dq = np.array([1.0, 5.0, 2.0, 0.0])
    np.testing.assert_array_equal(da._smooth_innovation(dq, dt=3600.0), dq)


def test_spread_preserves_a_constant_innovation_exactly():
    """Edge padding matters: a shrinking window would sag at the ends."""
    da = _da(innovation_spread_h=6.0)
    dq = np.full(12, 3.0)
    np.testing.assert_allclose(da._smooth_innovation(dq, dt=3600.0), dq, rtol=1e-12)


def _chain_tree():
    """Gage 100 <- 101 <- 102 (areas 30/20/10)."""
    return build_gage_trees_from_mappings(
        {100: [101], 101: [102], 102: []}, {"G": 100},
        {100: 30.0, 101: 20.0, 102: 10.0}, theta_default=0.77,
    )["G"]


class TestTraceIsPartitionInvariant:
    """``max_loop_size`` is a memory knob and must not change discharge.

    The backward ck trace broke that two ways at once: it recomputed tau from
    every window's own ``cn`` field, and its walk stopped at the start of the
    record, so a LONGER window both moved tau and resolved MORE segments. Both
    are fixed by tracing once over a fixed span (``lag_window_h``) taken from
    the start of the first window, then caching.
    """

    def _da(self, lag_window_h=6.0, trees=None):
        da = ScalingDA.__new__(ScalingDA)
        da.travel_time_lag = True
        da.max_reach_km = 1e9
        da._dx = None
        da.innovation_spread_h = 0.0
        da.lag_window_h = lag_window_h
        # Every tree in the RUN is traced together, not only the ones spreading
        # in the current window, so the object needs its full tree set.
        da.trees = trees if trees is not None else {"G": _chain_tree()}
        return da

    def _lag(self, nt, cn, cp):
        trees = {"G": _chain_tree()}
        return self._da(trees=trees)._build_lag(
            trees, dt=3600.0, nt=nt, cn=cn[:nt], cn_colpos=cp
        )["G"]

    def test_shift_does_not_depend_on_window_length(self):
        cp = {100: 0, 101: 1, 102: 2}
        cn = np.full((24, 3), 0.5)   # 2 steps per reach over the fixed span
        cn[6:, :] = 0.25             # slower water only a longer window can see
        short = self._lag(6, cn, cp)[1]
        long_ = self._lag(24, cn, cp)[1]
        np.testing.assert_array_equal(short, long_)
        np.testing.assert_array_equal(short, [0, 2, 4])
        # 1-D: one lag per segment, which is what the COMPILED kernel takes;
        # any other shape is rejected at the apply boundary.
        assert short.ndim == 1

    def test_the_resolved_set_does_not_grow_with_the_window(self):
        """A segment the fixed span cannot resolve is dropped, not guessed at,
        however much history the window happens to hold."""
        cp = {100: 0, 101: 1, 102: 2}
        cn = np.full((24, 3), 0.1)   # 10 steps per reach: the 6-step span cannot cross one
        short = self._lag(6, cn, cp)[0]
        long_ = self._lag(24, cn, cp)[0]
        np.testing.assert_array_equal(short, long_)
        np.testing.assert_array_equal(short, [1.0, 0.0, 0.0])

    def test_a_tau_that_used_the_whole_record_is_unresolved(self):
        """A trace that only crossed its reach in the OLDEST sample available is
        a lower bound, not a measurement, and must not be applied as a shift."""
        da = self._da()
        cn = np.full((48, 3), 1.0 / 48.0)   # crosses exactly at the end of the record
        tau, counts = da._tree_tau_trace(
            _chain_tree(), 48, cn, {100: 0, 101: 1, 102: 2}
        )
        assert counts == {"inherited": 1, "no_cn": 0, "dry": 0, "short": 0,
                          "lower_bound": 1}
        assert np.isfinite(tau[0])
        assert not np.isfinite(tau[1:]).any()

    def test_tau_is_traced_once_and_cached(self):
        cp = {100: 0, 101: 1, 102: 2}
        trees = {"G": _chain_tree()}
        da = self._da(trees=trees)
        cn = np.full((6, 3), 0.5)
        first = da._build_lag(trees, dt=3600.0, nt=6, cn=cn, cn_colpos=cp)
        second = da._build_lag(
            trees, dt=3600.0, nt=6, cn=np.full((6, 3), 0.25), cn_colpos=cp
        )
        np.testing.assert_array_equal(first["G"][1], second["G"][1])

    def test_a_later_window_needs_no_courant_field_at_all(self):
        """Once every tree is traced the driver stops exporting `cn`, so the
        cached tau must still apply with none supplied."""
        cp = {100: 0, 101: 1, 102: 2}
        trees = {"G": _chain_tree()}
        da = self._da(trees=trees)
        cn = np.full((6, 3), 0.5)
        first = da._build_lag(trees, dt=3600.0, nt=6, cn=cn, cn_colpos=cp)
        assert da._trace_cached(trees)
        later = da._build_lag(trees, dt=3600.0, nt=6, cn=None, cn_colpos=None)
        np.testing.assert_array_equal(first["G"][1], later["G"][1])
        np.testing.assert_array_equal(later["G"][1], [0, 2, 4])

    def test_a_quiet_first_window_still_sets_the_span(self):
        """The span must be the run's opening, not the first window that happens
        to carry an observation.

        Every early return in ``apply_in_kernel`` is keyed on THIS window's
        observations, and which window first carries one moves with
        ``max_loop_size``. Tracing from the first window that has a Courant field
        instead makes the span partition-independent, so the trace has to be
        filled on a window that spreads nothing at all.
        """
        trees = {"G": _chain_tree()}
        da = self._da(trees=trees)
        nts = 6
        # cn/ck/X interleaved, cn = 0.5 -> 2 steps per reach; nudge all zero, so
        # every innovation-driven path returns early.
        cour = np.zeros((3, nts * 3), dtype=np.float32)
        cour[:, 0::3] = 0.5
        cour[:, 1::3] = 1.0
        arr = np.zeros((3, 4 * nts), dtype=np.float32)
        arr[:, 0::4] = 5.0
        arr[:, 2::4] = 1.0
        rr = [[np.array([100, 101, 102]), arr, cour, (np.array([100]), np.zeros(1),
               np.zeros(1)), 0, 0, 0, 0, np.zeros((3, nts), dtype=np.float32),
               np.zeros((1, nts + 1), dtype=np.float32)]]
        da.gage_seg = {"G": 100}
        da.min_flow = 1e-6
        da.apply_in_kernel(rr, nts=nts, dt=3600, t0="2000-01-01")
        assert da._trace_cached(trees)
        assert da._tau_span == 6

    def test_every_tree_is_traced_not_only_the_spreading_ones(self):
        """A gage that first fires in a later window must not read that window's
        span: which window it first fires in depends on max_loop_size."""
        cp = {100: 0, 101: 1, 102: 2}
        trees = {"G": _chain_tree(), "H": _chain_tree()}
        da = self._da(trees=trees)
        # Only G spreads this window; H has no innovation yet.
        da._build_lag({"G": trees["G"]}, dt=3600.0, nt=6,
                      cn=np.full((6, 3), 0.5), cn_colpos=cp)
        assert da._trace_cached(trees)


class TestTraceCheckpoint:
    """The trace is measured once over the run's opening span and reused, so it
    is result-determining state: a BMI checkpoint must carry it, and a load
    must restore it exactly or explicitly invalidate it -- never keep a stale
    one, never silently retrace under a changed identity."""

    def _traced(self, lag_window_h=6.0):
        da = TestTraceIsPartitionInvariant()._da(lag_window_h=lag_window_h)
        da._build_lag(da.trees, dt=3600.0, nt=6,
                      cn=np.full((6, 3), 0.5), cn_colpos={100: 0, 101: 1, 102: 2})
        return da

    def test_round_trip_restores_the_exact_trace(self):
        src = self._traced()
        ckpt = src.trace_checkpoint()
        dst = TestTraceIsPartitionInvariant()._da()
        dst.restore_trace_checkpoint(ckpt, dt=3600.0)
        assert dst._trace_cached(dst.trees)
        np.testing.assert_array_equal(
            dst._tau_cache[("G", 6)][0], src._tau_cache[("G", 6)][0]
        )

    def test_nothing_traced_serializes_as_none(self):
        da = TestTraceIsPartitionInvariant()._da()
        assert da.trace_checkpoint() is None

    def test_missing_checkpoint_clears_a_stale_trace(self):
        """Loading into an already-used model must drop that model's own trace;
        keeping it is the silent divergence this state entry exists to stop."""
        da = self._traced()
        da.restore_trace_checkpoint(None, dt=3600.0)
        assert not da._trace_cached(da.trees)

    def test_changed_identity_is_discarded_not_reinterpreted(self):
        ckpt = self._traced().trace_checkpoint()
        other_dt = TestTraceIsPartitionInvariant()._da()
        other_dt.restore_trace_checkpoint(ckpt, dt=300.0)   # tau is in steps OF A dt
        assert not other_dt._trace_cached(other_dt.trees)
        other_window = TestTraceIsPartitionInvariant()._da(lag_window_h=12.0)
        other_window.restore_trace_checkpoint(ckpt, dt=3600.0)
        assert not other_window._trace_cached(other_window.trees)

    def test_same_root_same_size_different_interior_is_rejected(self):
        """tau is positional over seg_order, so a tree with the same root and
        size but a different interior (hydrofabric revision, changed stop set)
        must not accept the old array: every tau would land on the wrong
        segment."""
        chain = build_gage_trees_from_mappings(
            {100: [101], 101: [102], 102: [103], 103: []}, {"G": 100},
            {100: 40.0, 101: 30.0, 102: 20.0, 103: 10.0}, theta_default=0.77,
        )
        forked = build_gage_trees_from_mappings(
            {100: [101], 101: [102, 103], 102: [], 103: []}, {"G": 100},
            {100: 40.0, 101: 30.0, 102: 20.0, 103: 10.0}, theta_default=0.77,
        )
        src = TestTraceIsPartitionInvariant()._da(trees=chain)
        src._build_lag(chain, dt=3600.0, nt=6, cn=np.full((6, 4), 0.5),
                       cn_colpos={100: 0, 101: 1, 102: 2, 103: 3})
        dst = TestTraceIsPartitionInvariant()._da(trees=forked)
        dst.restore_trace_checkpoint(src.trace_checkpoint(), dt=3600.0)
        assert not dst._trace_cached(dst.trees)


class TestTraceFollowsTheSolverNotThePhysics:
    """The solver clamps K at one timestep, so the routed wave crosses at most
    one segment per step while the exported cn stays unclamped physics; the
    trace must apply ``min(cn, 1)`` or under-estimate tau by cn wherever
    cn > 1 (verified on a synthetic chain: routed speed pins at dx/dt)."""

    def test_cn_above_one_still_costs_one_step_per_reach(self):
        cn = np.full((12, 3), 2.0)   # physics says half a step per reach
        tau, counts = ScalingDA._tree_tau_trace(
            _da(), _chain_tree(), 12, cn, {100: 0, 101: 1, 102: 2}
        )
        assert sum(counts.values()) == 0
        np.testing.assert_allclose(tau, [0.0, 1.0, 2.0])

    def test_cn_below_one_is_untouched_by_the_clamp(self):
        cn = np.full((12, 3), 0.5)
        tau, _ = ScalingDA._tree_tau_trace(
            _da(), _chain_tree(), 12, cn, {100: 0, 101: 1, 102: 2}
        )
        np.testing.assert_allclose(tau, [0.0, 2.0, 4.0])

    def test_a_dry_reach_is_counted_dry_not_short(self):
        """All-zero cn (a dry channel or reservoir pool) and a record that is
        merely too short call for different responses: a longer lag_window_h
        could resolve the second and can never resolve the first."""
        cn = np.zeros((12, 3))
        tau, counts = ScalingDA._tree_tau_trace(
            _da(), _chain_tree(), 12, cn, {100: 0, 101: 1, 102: 2}
        )
        assert counts["dry"] == 1        # the first hop saw no live sample
        assert counts["inherited"] == 1  # the second never got a chance
        assert counts["short"] == 0
        assert not np.isfinite(tau[1:]).any()

    def test_tau_charges_the_parent_reach_and_syncs_junction_siblings(self):
        """tau[j] must cross the PARENT's reach (own-reach charging is off by
        one), and junction siblings must share tau exactly or the flow-ratio
        split desynchronizes from the branch corrections. Per-reach cn values
        differ so own-reach charging would show (1 vs 10 steps)."""
        trees = build_gage_trees_from_mappings(
            {100: [101], 101: [102, 103], 102: [], 103: []}, {"G": 100},
            {100: 40.0, 101: 30.0, 102: 10.0, 103: 20.0}, theta_default=0.77,
        )
        nt = 24
        cn = np.empty((nt, 4))
        cn[:, 0] = 0.5    # reach 100: 2 steps to cross
        cn[:, 1] = 0.25   # reach 101: 4 steps
        cn[:, 2] = 1.0    # reach 102: would be 1 step if (wrongly) charged
        cn[:, 3] = 0.1    # reach 103: would be 10 steps if (wrongly) charged
        tau, counts = ScalingDA._tree_tau_trace(
            _da(), trees["G"], nt, cn, {100: 0, 101: 1, 102: 2, 103: 3}
        )
        of = dict(zip(map(int, trees["G"].seg_order), tau))
        assert sum(counts.values()) == 0
        assert of[101] == pytest.approx(2.0)
        assert of[102] == pytest.approx(6.0)
        assert of[103] == pytest.approx(6.0)

    def test_nan_samples_cover_no_distance(self):
        """min(nan, 1.0) is 1.0 in Python; the finite guard must run FIRST or
        a dead sample silently counts as a full reach crossing."""
        cn = np.full((6, 3), np.nan)
        cn[::2, :] = 0.5             # live samples only at even steps
        tau, counts = ScalingDA._tree_tau_trace(
            _da(), _chain_tree(), 6, cn, {100: 0, 101: 1, 102: 2}
        )
        # Walking back from step 5: nan, 0.5, nan, 0.5 -> four samples for the
        # first hop, two of them dead. That leaves two samples (one dead) for
        # the second hop: 0.5 of the reach covered when the record ran out, so
        # it is counted as short (a longer window could resolve it), not dry.
        assert tau[1] == pytest.approx(4.0)
        assert counts["short"] == 1
        assert sum(counts.values()) == 1
        assert not np.isfinite(tau[2])


