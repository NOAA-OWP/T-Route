"""Unit tests for the Stage A in-kernel downstream DA path.

Covers the two pieces that differ from the output-only path:
  * ``apply_scaling_da(dq_o_by_site=...)`` -- drive the upstream spread
    from a supplied per-gage delta (the kernel-recorded ``nudge``) instead of
    ``obs - Q_model(gage)``; must equal the obs-driven result when the delta
    matches.
  * ``ScalingDA.apply_in_kernel`` -- map the kernel ``nudge`` (``r[9]``)
    to gage segments, reconstruct the gage background (``Q_analyzed - nudge``),
    spread upstream, and scatter back leaving the gage segment a no-op (it was
    already overridden in the MC kernel).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from nwm_routing.scaling_da_apply import ScalingDA

from troute.routing.fast_reach.scaling_da import apply_scaling_da
from troute.scaling_da import build_gage_trees_from_mappings


def _bare(**attrs) -> ScalingDA:
    o = ScalingDA.__new__(ScalingDA)
    # These tests exercise the spread mechanics with run_results that carry no
    # Courant block; with the class-default lag ON that is now a hard error
    # (fail closed), so the lag is off unless a test asks for it.
    o.travel_time_lag = False
    for k, v in attrs.items():
        setattr(o, k, v)
    return o


def _linear_tree():
    """Gage at seg 100, linear upstream chain 100<-101<-102 (areas 30/20/10)."""
    rconn = {100: [101], 101: [102], 102: []}
    area = {100: 30.0, 101: 20.0, 102: 10.0}
    return build_gage_trees_from_mappings(rconn, {"G": 100}, area, theta_default=0.77)


def test_seed_untimed_rewrites_only_the_handoff_instant():
    """With the lag on, the hand-off window's FINAL timestep must carry the
    UNTIMED correction (that instant is all a forecast inherits, and the lagged
    read there decays past the analysis edge), while earlier timesteps keep the
    traced timing. A ramp innovation discriminates the two: untimed reads
    dq_o(t), lagged reads dq_o(t + tau)."""
    trees = _linear_tree()
    nts = 6
    arr = np.zeros((3, 4 * nts), dtype=np.float32)
    arr[:, 0::4] = 5.0
    cour = np.zeros((3, nts * 3), dtype=np.float32)
    cour[:, 0::3] = 0.5                      # tau: 101 -> 2 steps, 102 -> 4
    cour[:, 1::3] = 1.0
    nudge = np.zeros((1, nts + 1), dtype=np.float32)
    nudge[0, 1:] = np.arange(1.0, nts + 1)   # ramp 1..6
    rr = [[np.array([100, 101, 102]), arr, cour, (np.array([100]), np.zeros(1),
           np.zeros(1)), 0, 0, 0, 0, np.zeros((3, nts), dtype=np.float32),
           nudge]]
    o = _bare(trees=trees, gage_seg={"G": 100}, min_flow=1e-6,
              max_reach_km=1e9, innovation_spread_h=0.0, _dx=None,
              travel_time_lag=True, lag_window_h=6.0, da_decay_min=120.0)
    o.apply_in_kernel(rr, nts=nts, dt=3600, t0="2000-01-01", seed_untimed=True)
    f101 = (20.0 / 30.0) ** 0.77
    q101 = arr[1, 0::4]
    # Earlier step keeps the traced timing: at t=1 the lag (shift 2) reads
    # dq_o[3] = 4, not the untimed dq_o[1] = 2.
    assert q101[1] == pytest.approx(5.0 + 4.0 * f101, rel=1e-5)
    # The hand-off instant is untimed: dq_o[5] = 6 exactly, not the
    # edge-decayed lagged read.
    assert q101[-1] == pytest.approx(5.0 + 6.0 * f101, rel=1e-5)


def test_lag_with_no_courant_field_fails_closed():
    """travel_time_lag on + a kernel that exported no Courant block is a config
    the run cannot honor (e.g. an all-diffusive domain). Silently applying the
    correction at observation time would be a different estimator than
    configured, so it raises instead."""
    import pytest

    trees = _linear_tree()
    nts = 2
    arr = np.zeros((3, 4 * nts), dtype=np.float32)
    arr[:, 0::4] = 5.0
    nudge = np.zeros((1, nts + 1), dtype=np.float32)
    nudge[0, 1:] = 2.0
    rr = [[np.array([100, 101, 102]), arr, 0, (np.array([100]), np.zeros(1),
           np.zeros(1)), 0, 0, 0, 0, np.zeros((3, nts), dtype=np.float32),
           nudge]]
    o = _bare(trees=trees, gage_seg={"G": 100}, min_flow=1e-6,
              max_reach_km=200.0, innovation_spread_h=0.0, _dx=None,
              travel_time_lag=True, lag_window_h=48.0)
    with pytest.raises(RuntimeError, match="no Courant"):
        o.apply_in_kernel(rr, nts=nts, dt=3600, t0="2000-01-01")


def test_kernel_dq_o_override_matches_obs_path():
    """dq_o_by_site reproduces the obs-driven spread when delta == obs - Q_gage."""
    trees = _linear_tree()
    idx = pd.date_range("2000-01-01", periods=2, freq="h")
    q_model = pd.DataFrame({100: [10.0, 10.0], 101: [6.0, 6.0], 102: [4.0, 4.0]}, index=idx)
    gmap = {"G": 100}

    # obs-driven: obs=12 -> dq_o = 12-10 = 2 at the gage.
    q_obs = pd.DataFrame({"G": [12.0, 12.0]}, index=idx)
    q_a, _ = apply_scaling_da(q_model, q_obs, gmap, trees)

    # override-driven: supply dq_o = 2 directly, with background at the gage.
    q_bg = q_model.copy()
    q_b, _ = apply_scaling_da(
        q_bg,
        None,
        gmap,
        trees,
        dq_o_by_site={"G": np.array([2.0, 2.0])},
    )
    pd.testing.assert_frame_equal(q_a, q_b)
    # sanity: gage recovered the obs; upstream got an area-scaled FRACTION of dq_o
    # (101 = 6 + 2*(20/30)^0.77), i.e. above background but below background+dq_o.
    assert np.allclose(q_b[100].to_numpy(), 12.0)
    assert (q_b[101].to_numpy() > 6.0).all()
    assert (q_b[101].to_numpy() < 8.0).all()


def _run_results(nudge_vals):
    """One subnetwork: segs [100,101,102], gage 100 overridden to 12, nts=len(vals).

    r[1] q-columns are the ANALYZED (post-override) flow; r[3][0] the gage seg;
    r[9] the nudge (gages, nts+1) with col 0 = IC (0) and cols 1: the applied delta.
    """
    nudge_vals = np.asarray(nudge_vals, dtype=float)
    nts = nudge_vals.size
    ids = np.array([100, 101, 102])
    q = np.array([12.0, 6.0, 4.0])  # analyzed: gage already = obs
    arr = np.zeros((3, 4 * nts), dtype=np.float32)
    arr[:, 0::4] = np.tile(q[:, None], (1, nts))  # q at every timestep
    # depth, so the discharge-to-depth transform has a state to carry into
    arr[:, 2::4] = np.tile(np.array([2.0, 1.5, 1.0])[:, None], (1, nts))
    nudge = np.zeros((1, nts + 1), dtype=np.float32)
    nudge[0, 1:] = nudge_vals
    r = [
        ids,
        arr,
        0,
        (np.array([100]), np.zeros(1), np.zeros(1)),
        0,
        0,
        0,
        0,
        np.zeros((3, nts), dtype=np.float32),
        nudge,
    ]
    return [r]


def test_apply_in_kernel_spreads_upstream_gage_is_noop():
    o = _bare(
        trees=_linear_tree(),
        gage_seg={"G": 100},
        min_flow=1e-6,
        max_source_pbias=None,
        _loop_obs=None,
    )
    rr = _run_results([2.0, 2.0])
    before = rr[0][1][:, 0::4].copy()
    o.apply_in_kernel(rr, nts=2, dt=3600, t0="2000-01-01")
    after = rr[0][1][:, 0::4]
    # gage seg 100 (row 0) unchanged -- already corrected in-kernel.
    np.testing.assert_allclose(after[0], before[0])
    # upstream 101/102 (rows 1,2) got the area-scaled spread (increased).
    assert (after[1] > before[1]).all()
    assert (after[2] > before[2]).all()
    # background reconstruction: 101 delta == 2 * (20/30)^0.77.
    expected_101 = 6.0 + 2.0 * (20.0 / 30.0) ** 0.77
    np.testing.assert_allclose(after[1], expected_101, rtol=1e-5)


def test_apply_in_kernel_zero_nudge_is_noop():
    """obs == model -> nudge 0 -> strict no-op (idempotency)."""
    o = _bare(
        trees=_linear_tree(),
        gage_seg={"G": 100},
        min_flow=1e-6,
        max_source_pbias=None,
        _loop_obs=None,
    )
    rr = _run_results([0.0, 0.0])
    before = rr[0][1][:, 0::4].copy()
    o.apply_in_kernel(rr, nts=2, dt=3600, t0="2000-01-01")
    np.testing.assert_array_equal(rr[0][1][:, 0::4], before)


def _two_tree():
    """Two disjoint gages: G1 seg 100<-101, G2 seg 200<-201."""
    rconn = {100: [101], 101: [], 200: [201], 201: []}
    area = {100: 30.0, 101: 20.0, 200: 30.0, 201: 20.0}
    return build_gage_trees_from_mappings(rconn, {"G1": 100, "G2": 200}, area, theta_default=0.77)


def test_zero_own_innovation_still_spreads_the_halo():
    """A window with no innovation of its own still spreads, via the halo.

    The temporal spread averages across the window boundary, so the next
    window's innovation reaches timesteps inside this one. Inclusion is
    therefore decided from the smoothed, halo-extended series rather than from
    this window's raw values, which are all zero here.
    """
    o = _bare(
        trees=_linear_tree(),
        gage_seg={"G": 100},
        min_flow=1e-6,
        max_source_pbias=None,
        max_reach_km=200.0,
        innovation_spread_h=12.0,
        _loop_obs=None,
        _dx=pd.Series({100: 3600.0, 101: 3600.0, 102: 3600.0}, dtype=float),
    )
    rr = _run_results([0.0, 0.0])  # zero own innovation
    before = rr[0][1][:, 0::4].copy()
    o.apply_in_kernel(rr, nts=2, dt=3600, t0="2000-01-01",
                      halo={100: np.array([5.0, 5.0])})
    after = rr[0][1][:, 0::4]
    # The upstream segments are corrected even though this window's own
    # innovation is zero throughout.
    assert (after[1] > before[1]).all()
    assert (after[2] > before[2]).all()
    # The gage keeps the value the in-kernel override gave it.
    np.testing.assert_allclose(after[0], before[0], rtol=1e-6)


def test_spread_reaches_the_warmstate_columns():
    """State seeding depends on the corrected discharge landing in the exact
    columns ``new_q0`` reads: ``r[1][:, [-4, -4, -2, -1]]`` -> qu0/qd0/h0/ql0.

    Column ``-4`` of an ``(n_seg, 4*nts)`` array is ``4*(nts-1)``, which is in the
    ``0::4`` discharge stride, so the last timestep's corrected q is what carries
    over. This asserts that indexing claim rather than trusting it. Depth
    (``-2``) is now carried with it: it is state, and MC derives celerity and X
    from it, so seeding a corrected discharge against the uncorrected depth
    hands the next chunk the right flow on the wrong geometry.
    """
    o = _bare(
        trees=_linear_tree(),
        gage_seg={"G": 100},
        min_flow=1e-6,
        max_source_pbias=None,
        _loop_obs=None,
    )
    rr = _run_results([2.0, 2.0])
    o.apply_in_kernel(rr, nts=2, dt=3600, t0="2000-01-01")

    arr = rr[0][1]
    warmstate = arr[:, [-4, -4, -2, -1]]  # exactly what AbstractNetwork.new_q0 slices
    expected_101 = 6.0 + 2.0 * (20.0 / 30.0) ** 0.77
    # upstream reach 101 carries its CORRECTED discharge into qu0 and qd0. This is the
    # load-bearing assertion: it fails if -4 ever stops landing in the 0::4 stride.
    np.testing.assert_allclose(warmstate[1, 0], expected_101, rtol=1e-5)
    np.testing.assert_allclose(warmstate[1, 1], expected_101, rtol=1e-5)
    # depth (-2) is carried WITH the discharge, so the seeded state is internally
    # consistent: MC derives celerity and X from depth, and the previous
    # behaviour handed the next chunk a corrected flow on the uncorrected
    # geometry. The gage row is untouched by the spread, so its depth holds.
    np.testing.assert_allclose(warmstate[0, 2], 2.0, rtol=1e-6)
    expected_h101 = 1.5 * (expected_101 / 6.0) ** 0.6
    np.testing.assert_allclose(warmstate[1, 2], expected_h101, rtol=1e-5)


def test_seed_state_only_on_the_final_window():
    """The warmstate is seeded once per RUN, not once per window.

    Muskingum-Cunge holds no storage state, so a dQ written into q0 is a one-shot mass
    pulse rather than a persistent analysis increment: it transits the source gage and
    inflates its background, which debits the NEXT window's innovation (obs - background)
    by the amount we injected. Re-seeding every window therefore starves every tree of
    correction, including the sibling branches held-out gages sit on. Measured on the
    Ohio LOO fold, 24 h windows: held-out bias -23.8% with one window boundary vs -29.1%
    with eleven.

    This survives the removal of the arm switch. "Always prognostic" means the spread is
    always the state-seeding kind, NOT that it seeds on every window -- collapsing the
    two would reintroduce exactly the debit above.
    """
    from nwm_routing.scaling_da_apply import should_seed_state

    n = 4
    da = object()  # should_seed_state only needs "not None" now
    assert [should_seed_state(da, i, n) for i in range(n)] == [False, False, False, True]

    # No DA configured at all.
    assert not any(should_seed_state(None, i, n) for i in range(n))

    # Single-window run: that window IS the hand-off, so it seeds.
    assert should_seed_state(da, 0, 1)


def test_spread_runs_exactly_once_per_window():
    """The spread is applied once per window -- never twice, never zero.

    The two drivers pick the position with `seed_state` and run the output pass under
    `not seed_state`, so the two branches partition every window. A second call in the
    same window would spread on top of already-corrected interior flow, because
    apply_in_kernel reconstructs the background only at the tree root.
    """
    from nwm_routing.scaling_da_apply import should_seed_state

    n = 5
    da = object()
    for i in range(n):
        before = should_seed_state(da, i, n)
        after = not before
        assert before + after == 1, f"window {i}: applied {before + after} times"


def test_chunked_spread_matches_unchunked_through_apply_in_kernel():
    """Chunking the spread must not change the result, WITH a lag in play.

    The equivalent check on the spread helper alone did not catch a driver-level
    defect: the chunk loop sliced the lag arrays by time, which is meaningless
    for a per-segment lag and raised only once a window was long enough to
    chunk. Ohio at 48 h never chunks, so this has to be forced.
    """
    o_kwargs = dict(  # noqa: C408  (kwargs are threaded into _bare(**...) below)
        trees=_linear_tree(),
        gage_seg={"G": 100},
        min_flow=1e-6,
        max_reach_km=200.0,
        _dx=pd.Series({100: 3600.0, 101: 3600.0, 102: 3600.0}, dtype=float),
    )
    n = 8
    nudge = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]

    rr_full = _run_results(nudge)
    _bare(spread_chunk_timesteps=0, **o_kwargs).apply_in_kernel(
        rr_full, nts=n, dt=3600, t0="2000-01-01"
    )
    rr_chunked = _run_results(nudge)
    _bare(spread_chunk_timesteps=3, **o_kwargs).apply_in_kernel(
        rr_chunked, nts=n, dt=3600, t0="2000-01-01"
    )
    # The WHOLE result array, not just discharge. Comparing 0::4 alone hid that
    # the chunk loop wrote discharge only: depth is STATE (h0 seeds the next
    # window and MC derives celerity from it), so a chunked run handed the
    # forecast a different geometry while reporting identical flow.
    np.testing.assert_array_equal(rr_chunked[0][1], rr_full[0][1])
    # and the spread actually did something, or the comparison is vacuous
    assert (rr_full[0][1][1:, 0::4] != _run_results(nudge)[0][1][1:, 0::4]).any()
    assert (rr_full[0][1][1:, 2::4] != _run_results(nudge)[0][1][1:, 2::4]).any()


class TestResolveSpreadChunk:
    def _o(self, **attrs):
        return _bare(min_flow=1e-6, **attrs)

    def test_explicit_config_wins_including_zero(self, monkeypatch):
        monkeypatch.setenv("SCALING_DA_CHUNK", "7")
        assert self._o(spread_chunk_timesteps=12)._resolve_spread_chunk(288, 10**6) == 12
        assert self._o(spread_chunk_timesteps=0)._resolve_spread_chunk(288, 10**6) == 0

    def test_env_var_is_honored_when_config_unset(self, monkeypatch):
        monkeypatch.setenv("SCALING_DA_CHUNK", "7")
        assert self._o(spread_chunk_timesteps=None)._resolve_spread_chunk(288, 100) == 7

    def test_auto_chunks_only_a_large_window(self, monkeypatch):
        monkeypatch.delenv("SCALING_DA_CHUNK", raising=False)
        o = self._o(spread_chunk_timesteps=None)
        # Ohio scale: no chunking.
        assert o._resolve_spread_chunk(288, 11_327) == 0
        # CONUS scale: chunk to keep one transient under the element budget.
        chunk = o._resolve_spread_chunk(288, 1_100_000)
        assert 0 < chunk < 288
        assert chunk * 1_100_000 <= o._SPREAD_CHUNK_BUDGET_ELEMS



def test_smoothing_keeps_the_gage_observation_and_the_true_background():
    """With a nonzero spread the raw and smoothed innovations differ, and both
    the gage's own output and the upstream split have to stay correct.

    The gage keeps the observation the in-kernel override placed there, which is
    background + RAW nudge. The upstream share is computed against the true
    background, so it is the same fraction of the smoothed innovation that a
    zero-spread run would apply to an equal innovation -- not a smaller one
    scaled down by an inflated denominator.
    """
    o = _bare(
        trees=_linear_tree(),
        gage_seg={"G": 100},
        min_flow=1e-6,
        max_source_pbias=None,
        _loop_obs=None,
        innovation_spread_h=2.0,
    )
    nudge = [2.0, 2.0, 8.0, 2.0]
    rr = _run_results(nudge)
    o.apply_in_kernel(rr, nts=len(nudge), dt=3600, t0="2000-01-01")
    after = rr[0][1][:, 0::4]

    # The gage row is the analyzed flow the kernel wrote: unchanged.
    np.testing.assert_allclose(after[0], 12.0, rtol=1e-5)

    # Upstream: background 6.0 plus the SMOOTHED innovation times the area
    # factor. Reproduce the smoothing exactly rather than hardcoding it.
    smoothed = o._smooth_innovation(np.asarray(nudge, dtype=float), dt=3600.0)
    expected = 6.0 + smoothed * (20.0 / 30.0) ** 0.77
    np.testing.assert_allclose(after[1], expected, rtol=1e-5)


def test_spread_volume_holds_inside_the_series_but_not_at_its_edges():
    """The forward mean conserves the increment only in the interior.

    An innovation within the window of the series START has no earlier output
    steps to carry its mass and loses most of it; one at the very END is
    amplified by the persistence padding that keeps the final step equal to its
    own raw value (which is what the forecast hand-off snapshots). Pinned
    because both are load-bearing and neither is obvious.
    """
    o = _bare(innovation_spread_h=4.0)
    n, win = 21, 5
    # trailing factor is (win+1)/2, not win: the padded copies past the last
    # output step have nowhere to land.
    for pos, expected in ((0, 12.0 / win), (10, 12.0), (n - 1, 12.0 * (win + 1) / 2)):
        dq = np.zeros(n)
        dq[pos] = 12.0
        out = o._smooth_innovation(dq, dt=3600.0)
        assert out.sum() == pytest.approx(expected)
    # the property the hand-off depends on, for every width
    dq = np.array([1.0, 5.0, 2.0, 9.0, 4.0, 7.0])
    for width in (0.0, 6.0, 12.0, 24.0):
        o.innovation_spread_h = width
        assert o._smooth_innovation(dq, dt=3600.0)[-1] == pytest.approx(dq[-1])


def test_the_correction_is_carried_into_depth_not_just_discharge():
    """Depth is STATE: new_q0 snapshots h0 to seed the next chunk, and MC derives
    its celerity and X weighting from depth. A corrected discharge paired with
    the uncorrected depth seeds the forecast with the right flow on the wrong
    geometry.
    """
    o = _bare(
        trees=_linear_tree(),
        gage_seg={"G": 100},
        min_flow=1e-6,
        max_source_pbias=None,
        _loop_obs=None,
        innovation_spread_h=0.0,
        travel_time_lag=False,
    )
    rr = _run_results([2.0, 2.0])
    depth_before = rr[0][1][:, 2::4].copy()
    q_before = rr[0][1][:, 0::4].copy()
    o.apply_in_kernel(rr, nts=2, dt=3600, t0="2000-01-01")
    q_after = rr[0][1][:, 0::4]
    depth_after = rr[0][1][:, 2::4]

    # upstream rows gained discharge, so their depth must rise with it
    assert (q_after[1] > q_before[1]).all()
    assert (depth_after[1] > depth_before[1]).all()
    # and by the wide-channel exponent: h scales as the discharge ratio ^ 3/5
    expected = depth_before[1] * (q_after[1] / q_before[1]) ** 0.6
    np.testing.assert_allclose(depth_after[1], expected, rtol=1e-5)
    # the gage row is untouched by the spread, so its depth is too
    np.testing.assert_allclose(depth_after[0], depth_before[0])


def test_depth_is_left_alone_where_the_discharge_ratio_is_not_a_depth_signal():
    """A ratio taken against a dry or near-zero background is noise, and a huge
    ratio would let the wide-channel approximation dominate the correction it is
    meant to transport."""
    o = _bare(
        trees=_linear_tree(),
        gage_seg={"G": 100},
        min_flow=1e-6,
        max_source_pbias=None,
        _loop_obs=None,
        innovation_spread_h=0.0,
        travel_time_lag=False,
    )
    rr = _run_results([2.0, 2.0])
    rr[0][1][:, 2::4] = 0.0    # no usable depth anywhere
    o.apply_in_kernel(rr, nts=2, dt=3600, t0="2000-01-01")
    np.testing.assert_allclose(rr[0][1][:, 2::4], 0.0)


def _confluence_tree_with_stopped_lake():
    """Gage 100 <- confluence 101 <- {102 kept, 103 stopped (a lake)}."""
    rconn = {100: [101], 101: [102, 103], 102: [], 103: []}
    area = {100: 30.0, 101: 20.0, 102: 10.0}  # 103 is a stop, so it needs no area
    return build_gage_trees_from_mappings(
        rconn, {"G": 100}, area, waterbody_segs={103}, theta_default=0.77
    )


def test_unresolvable_tree_does_not_strip_the_gage_nudge():
    """A site the spread cannot serve must keep its analyzed value intact.

    apply_in_kernel reconstructs the background by subtracting the RAW nudge from
    the gage column before spreading, and the spread adds it back at the tree root.
    If the tree is rejected in between -- here because the stopped lake 103 has no
    column in this window's results, so its flow cannot enter the confluence
    denominator -- nothing adds it back, and _scatter_back would publish the gage
    with its own DA nudge silently removed.
    """
    trees = _confluence_tree_with_stopped_lake()
    assert list(trees["G"].pruned_segs) == [103]
    nts = 2
    # Results carry 100/101/102 only: the stopped lake is NOT a column here.
    arr = np.zeros((3, 4 * nts), dtype=np.float32)
    arr[:, 0::4] = 5.0
    arr[0, 0::4] = 12.0  # gage, analyzed (nudge already applied in the kernel)
    nudge = np.zeros((1, nts + 1), dtype=np.float32)
    nudge[0, 1:] = 2.0
    rr = [[np.array([100, 101, 102]), arr, 0, (np.array([100]), np.zeros(1),
           np.zeros(1)), 0, 0, 0, 0, np.zeros((3, nts), dtype=np.float32),
           nudge]]
    o = _bare(trees=trees, gage_seg={"G": 100}, min_flow=1e-6,
              max_reach_km=1e9, innovation_spread_h=0.0, _dx=None,
              da_decay_min=120.0)
    o.apply_in_kernel(rr, nts=nts, dt=3600, t0="2000-01-01")

    np.testing.assert_allclose(arr[0, 0::4], 12.0, rtol=1e-6,
                               err_msg="the gage lost its own nudge")
    np.testing.assert_allclose(arr[1, 0::4], 5.0, rtol=1e-6,
                               err_msg="confluence corrected despite a rejected tree")
    np.testing.assert_allclose(arr[2, 0::4], 5.0, rtol=1e-6,
                               err_msg="branch corrected despite a rejected tree")


def test_root_mismatch_does_not_strip_the_gage_nudge():
    """A tree rooted somewhere other than the crosswalk's segment is rejected early.

    Every id here HAS a column, so the positioning preflight passes; the refusal comes
    from the ``gage_fp != gage_to_fp[site]`` check inside apply_scaling_da. Settling it
    before the background subtraction is what keeps the gage coherent -- otherwise the
    nudge comes off and no spread puts it back. Filtering happens before the candidate
    set is built, so the chunked and seed_untimed branches inherit it.
    """
    trees = _linear_tree()  # rooted at 100
    assert trees["G"].gage_fp == 100
    nts = 2
    arr = np.zeros((3, 4 * nts), dtype=np.float32)
    arr[:, 0::4] = 5.0
    arr[1, 0::4] = 12.0  # segment 101, where the crosswalk wrongly places the gage
    nudge = np.zeros((1, nts + 1), dtype=np.float32)
    nudge[0, 1:] = 2.0
    rr = [[np.array([100, 101, 102]), arr, 0, (np.array([101]), np.zeros(1),
           np.zeros(1)), 0, 0, 0, 0, np.zeros((3, nts), dtype=np.float32),
           nudge]]
    # Crosswalk says 101; the tree says 100.
    o = _bare(trees=trees, gage_seg={"G": 101}, min_flow=1e-6,
              max_reach_km=1e9, innovation_spread_h=0.0, _dx=None,
              da_decay_min=120.0)
    o.apply_in_kernel(rr, nts=nts, dt=3600, t0="2000-01-01")

    np.testing.assert_allclose(arr[1, 0::4], 12.0, rtol=1e-6,
                               err_msg="the gage lost its own nudge")
    np.testing.assert_allclose(arr[0, 0::4], 5.0, rtol=1e-6,
                               err_msg="corrected despite a rejected tree")
    np.testing.assert_allclose(arr[2, 0::4], 5.0, rtol=1e-6,
                               err_msg="corrected despite a rejected tree")


def _spread_over(nts, nudge, t0, spread_h):
    """Run the in-kernel spread over one window and return the routed discharge."""
    arr = np.zeros((3, 4 * nts), dtype=np.float32)
    arr[:, 0::4] = 5.0
    nud = np.zeros((1, nts + 1), dtype=np.float32)
    nud[0, 1:] = nudge
    rr = [[np.array([100, 101, 102]), arr, 0, (np.array([100]), np.zeros(1),
           np.zeros(1)), 0, 0, 0, 0, np.zeros((3, nts), dtype=np.float32), nud]]
    o = _bare(trees=_linear_tree(), gage_seg={"G": 100}, min_flow=1e-6,
              max_reach_km=1e9, innovation_spread_h=spread_h, _dx=None,
              da_decay_min=120.0)
    o.apply_in_kernel(rr, nts=nts, dt=3600, t0=t0, seed_untimed=True)
    return arr[:, 0::4].copy()


_RAMP = np.array([1.0, 2.0, 3.0, 4.0])


def test_the_partition_does_not_reach_the_result_at_zero_span():
    """With no forward average and no lag the spread is output-only and applied
    per timestep, so splitting a window cannot change the answer.

    This is what lets _build_run_sets serve a short update instead of refusing it:
    the NWM Standard AnA is 3 forcing columns against a max_loop_size default of
    24, and without this the shipped operational config could not run the DA.
    """
    whole = _spread_over(4, _RAMP, "2000-01-01", 0.0)
    split = np.concatenate(
        [_spread_over(2, _RAMP[:2], "2000-01-01", 0.0),
         _spread_over(2, _RAMP[2:], "2000-01-01 02:00", 0.0)], axis=1
    )
    np.testing.assert_array_equal(
        whole, split,
        err_msg="the window partition changed the result at zero span, so the "
                "guard in _build_run_sets cannot be relaxed",
    )


def test_the_partition_does_reach_the_result_once_the_span_is_nonzero():
    """The other half of the contract: a forward average reads across the window
    boundary, so the partition IS part of the result and must stay guarded."""
    whole = _spread_over(4, _RAMP, "2000-01-01", 2.0)
    split = np.concatenate(
        [_spread_over(2, _RAMP[:2], "2000-01-01", 2.0),
         _spread_over(2, _RAMP[2:], "2000-01-01 02:00", 2.0)], axis=1
    )
    assert not np.allclose(whole, split), (
        "a nonzero innovation_spread_h must make the partition matter; if this "
        "passes, the hard error in _build_run_sets is guarding nothing"
    )
