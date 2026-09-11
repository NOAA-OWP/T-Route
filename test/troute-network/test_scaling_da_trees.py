"""Unit tests for the simple-scaling upstream-tree builder on synthetic topologies.

Ported from the standalone proof-of-concept. The Ohio-gpkg integration tests
are deferred to the stage that wires :func:`build_gage_trees` onto t-route's
real network topology; here we cover :func:`build_one_gage_tree` against
synthetic reverse-connectivity graphs.
"""

from __future__ import annotations

import pytest

from troute.scaling_da import build_one_gage_tree

# Per-segment areas (in km^2) used across the synthetic graphs.
AREAS_CHAIN = {1: 100.0, 2: 80.0, 3: 60.0, 4: 40.0, 5: 20.0}
AREAS_JUNC = {1: 100.0, 2: 100.0, 3: 60.0, 4: 40.0}


def test_single_segment_tree() -> None:
    tree = build_one_gage_tree(
        gage_fp=1,
        rconn={},
        area_sqkm={1: 100.0},
        stop_segs=frozenset(),
        theta=0.77,
    )
    assert tree.n_segments == 1
    assert tree.gage_fp == 1
    assert tree.gage_area_sqkm == 100.0
    assert tree.seg_order.tolist() == [1]
    assert tree.seg_areas_sqkm.tolist() == [100.0]
    assert tree.theta == 0.77


def test_linear_chain_tree() -> None:
    # 5 -> 4 -> 3 -> 2 -> 1 (gage)
    rconn = {1: [2], 2: [3], 3: [4], 4: [5]}
    tree = build_one_gage_tree(
        gage_fp=1,
        rconn=rconn,
        area_sqkm=AREAS_CHAIN,
        stop_segs=frozenset(),
        theta=0.77,
    )
    assert tree.n_segments == 5
    assert tree.seg_order.tolist() == [1, 2, 3, 4, 5]
    # Per-segment areas follow seg_order.
    assert tree.seg_areas_sqkm.tolist() == [100.0, 80.0, 60.0, 40.0, 20.0]


def test_single_junction_tree() -> None:
    # 3 and 4 merge at 2, then 2 -> 1 (gage)
    rconn = {1: [2], 2: [3, 4]}
    tree = build_one_gage_tree(
        gage_fp=1,
        rconn=rconn,
        area_sqkm=AREAS_JUNC,
        stop_segs=frozenset(),
        theta=0.77,
    )
    assert tree.n_segments == 4
    # BFS: gage 1, then 2, then its upstream pair 3 and 4.
    assert tree.seg_order.tolist() == [1, 2, 3, 4]


def test_two_junctions_in_series() -> None:
    # 5,6 -> 4 -> 3 (gage), and 7 -> 3
    rconn = {3: [4, 7], 4: [5, 6]}
    areas = {3: 100.0, 4: 60.0, 5: 30.0, 6: 30.0, 7: 40.0}
    tree = build_one_gage_tree(
        gage_fp=3,
        rconn=rconn,
        area_sqkm=areas,
        stop_segs=frozenset(),
        theta=0.5,
    )
    assert tree.n_segments == 5
    # BFS order: 3 first, then 4 and 7 (children of 3), then 5 and 6 (children of 4).
    assert tree.seg_order.tolist() == [3, 4, 7, 5, 6]


def test_junction_structure_on_chain() -> None:
    """A linear chain has no confluences: every step is an area-scaling step."""
    rconn = {1: [2], 2: [3], 3: [4], 4: [5]}
    tree = build_one_gage_tree(
        gage_fp=1, rconn=rconn, area_sqkm=AREAS_CHAIN, stop_segs=frozenset(), theta=0.77
    )
    assert tree.seg_parent_idx.tolist() == [-1, 0, 1, 2, 3]
    assert tree.step_is_junction.tolist() == [False, False, False, False, False]


def test_junction_structure_two_junctions() -> None:
    """Parent index and confluence flags on a two-junction topology."""
    # 5,6 -> 4 -> 3 (gage), and 7 -> 3.
    rconn = {3: [4, 7], 4: [5, 6]}
    areas = {3: 100.0, 4: 60.0, 5: 30.0, 6: 30.0, 7: 40.0}
    tree = build_one_gage_tree(
        gage_fp=3, rconn=rconn, area_sqkm=areas, stop_segs=frozenset(), theta=0.5
    )
    # seg_order = [3, 4, 7, 5, 6]; each parent points to the downstream index.
    assert tree.seg_parent_idx.tolist() == [-1, 0, 0, 1, 1]
    # Gage 3 (children 4, 7) and node 4 (children 5, 6) are both confluences, so
    # their children take the flow-ratio split; the gage's own entry is False.
    assert tree.step_is_junction.tolist() == [False, True, True, True, True]


def test_junction_with_stopped_branch_still_flagged() -> None:
    """A physical confluence is flagged even when one branch is pruned by the stop rule."""
    # 2 and 3 merge at the gage 1; branch 3 is stopped (e.g. an upstream gage).
    rconn = {1: [2, 3]}
    tree = build_one_gage_tree(
        gage_fp=1,
        rconn=rconn,
        area_sqkm={1: 100.0, 2: 60.0, 3: 40.0},
        stop_segs=frozenset({3}),
        theta=0.77,
    )
    assert tree.seg_order.tolist() == [1, 2]  # 3 pruned
    # Node 1 is still a physical confluence (rconn[1] = [2, 3]), so the surviving
    # branch 2 takes the flow-ratio split, not area-scaling.
    assert tree.step_is_junction.tolist() == [False, True]


def test_stop_at_upstream_gage() -> None:
    # 5 -> 4 -> 3 (gage1), with gage2 at fp 5
    rconn = {3: [4], 4: [5]}
    tree = build_one_gage_tree(
        gage_fp=3,
        rconn=rconn,
        area_sqkm={3: 100.0, 4: 80.0, 5: 50.0},
        stop_segs=frozenset({5}),
        theta=0.77,
    )
    assert tree.n_segments == 2
    assert tree.seg_order.tolist() == [3, 4]
    # 5 is the stop -> not in the tree.
    assert 5 not in tree.seg_order.tolist()


def test_stop_at_waterbody() -> None:
    # 4 -> 3 (gage), with 4 inside a waterbody
    rconn = {3: [4]}
    tree = build_one_gage_tree(
        gage_fp=3,
        rconn=rconn,
        area_sqkm={3: 100.0, 4: 80.0},
        stop_segs=frozenset({4}),
        theta=0.77,
    )
    assert tree.n_segments == 1
    assert tree.seg_order.tolist() == [3]


def test_gage_in_stop_segs_raises() -> None:
    with pytest.raises(ValueError, match="cannot be in stop_segs"):
        build_one_gage_tree(
            gage_fp=1,
            rconn={},
            area_sqkm={1: 100.0},
            stop_segs=frozenset({1}),
            theta=0.77,
        )


def test_bfs_visits_each_segment_once() -> None:
    """BFS in a tree must not double-visit segments even if the rconn entry duplicates them."""
    # Intentional duplicate upstream, defensive check.
    rconn = {1: [2, 2], 2: [3]}
    tree = build_one_gage_tree(
        gage_fp=1,
        rconn=rconn,
        area_sqkm={1: 1.0, 2: 1.0, 3: 1.0},
        stop_segs=frozenset(),
        theta=0.5,
    )
    # Should produce a normal chain, not two copies of 2.
    assert tree.seg_order.tolist() == [1, 2, 3]


def test_with_positions_maps_seg_order_correctly() -> None:
    tree = build_one_gage_tree(
        gage_fp=10,
        rconn={10: [20, 30]},
        area_sqkm={10: 5.0, 20: 3.0, 30: 2.0},
        stop_segs=frozenset(),
        theta=0.77,
    )
    fp_to_position = {10: 100, 20: 200, 30: 300, 99: 999}
    out = tree.with_positions(fp_to_position)
    assert out.seg_positions.tolist() == [100, 200, 300]
    # with_positions must preserve the junction structure (10 is a confluence).
    assert out.seg_parent_idx.tolist() == [-1, 0, 0]
    assert out.step_is_junction.tolist() == [False, True, True]


def test_tree_with_unmapped_reservoir_segment_is_dropped() -> None:
    """A tree that walks onto a segment with no drainage area must not be kept.

    On NHF the synthetic reservoir ids live in ``connections``/``reverse_network`` but
    never get a ``dataframe`` row, so their area lookup misses and ``build_one_gage_tree``
    fills NaN. The area-scaling step then multiplies NaN through the whole subtree, and
    once the spread seeds state, that NaN reaches the warmstate and routes.

    The Ohio benchmark subset has zero rows in its ``lakes`` layer, so no integration
    run can exercise this path; this is the regression test that can.
    """
    from troute.scaling_da import build_gage_trees_from_mappings

    # head(30) -> lake(20) -> gage(10). The lake carries no drainage area.
    rconn = {10: [20], 20: [30], 30: []}
    area = {10: 100.0, 30: 50.0}  # note: no key for the lake id 20

    # Without a waterbody stop set (what NHF used to supply), the tree walks onto the
    # lake and the guard must reject the whole tree rather than emit NaN.
    trees = build_gage_trees_from_mappings(rconn, {"G": 10}, area, waterbody_segs=frozenset())
    assert "G" not in trees

    # With the lake correctly in the stop set, the tree is kept and simply stops there.
    trees = build_gage_trees_from_mappings(rconn, {"G": 10}, area, waterbody_segs=frozenset({20}))
    assert trees["G"].seg_order.tolist() == [10]


class TestExplicitSetsFailClosed:
    """Explicit qlat_forcing_sets bypass the enlargement and the fold, so a
    first set too short for the DA span must be rejected before routing: the
    traced span is part of the result and must not depend on the partition."""

    def _sets(self, *nts):
        return [{"nts": n} for n in nts]

    def test_short_first_set_in_a_long_run_raises(self):
        import pytest

        from troute.AbstractNetwork import _require_span_covering_first_set

        with pytest.raises(ValueError, match="opening window"):
            _require_span_covering_first_set(self._sets(96, 480), 48.0, 300.0)

    def test_a_run_genuinely_shorter_than_the_span_passes(self):
        """The trace warns and caps for a short RUN; only a short PARTITION of a
        long run is an error."""
        from troute.AbstractNetwork import _require_span_covering_first_set

        _require_span_covering_first_set(self._sets(96, 96), 48.0, 300.0)

    def test_a_covering_first_set_passes(self):
        from troute.AbstractNetwork import _require_span_covering_first_set

        _require_span_covering_first_set(self._sets(576, 96), 48.0, 300.0)

    def test_no_scaling_da_is_a_no_op(self):
        from troute.AbstractNetwork import _require_span_covering_first_set

        _require_span_covering_first_set(self._sets(4, 480), 0.0, 300.0)


class TestForcingWindowCoversTheSpread:
    """The -V5 window sizer must read the spread width out of the raw config.

    The halo that feeds the forward innovation window is one forcing window
    deep, so a window shorter than innovation_spread_h leaves its own tail on
    persistence and the result starts depending on max_loop_size, which is a
    memory knob. The BMI driver enlarges for the same reason; this pins the
    -V5 half's input.
    """

    def test_off_when_the_scaling_da_is_not_enabled(self):
        from troute.AbstractNetwork import _scaling_da_spread_h

        assert _scaling_da_spread_h(None) == 0.0
        assert _scaling_da_spread_h({}) == 0.0
        assert _scaling_da_spread_h({"streamflow_da": {"streamflow_scaling": False}}) == 0.0

    def test_the_lag_span_and_the_spread_ADD(self):
        """The innovation is averaged forward over innovation_spread_h and the lag
        then reads that average at t + tau, so a window's tail needs raw
        innovation out to tau_max + spread: 48 + 12, not max(48, 12). At the
        defaults (lag off, spread 0) nothing is required; each mechanism adds
        its own horizon when enabled."""
        from troute.AbstractNetwork import _scaling_da_spread_h

        assert _scaling_da_spread_h({"streamflow_da": {"streamflow_scaling": True}}) == 0.0
        assert _scaling_da_spread_h({"streamflow_da": {
            "streamflow_scaling": True,
            "streamflow_scaling_parameters": {"travel_time_lag": True},
        }}) == 48.0
        assert _scaling_da_spread_h({"streamflow_da": {
            "streamflow_scaling": True,
            "streamflow_scaling_parameters": {"travel_time_lag": True,
                                              "innovation_spread_h": 12.0},
        }}) == 60.0

    def test_explicit_value_wins_including_zero(self):
        from troute.AbstractNetwork import _scaling_da_spread_h

        def cfg(v):
            return {"streamflow_da": {"streamflow_scaling": True,
                                      "streamflow_scaling_parameters": {
                                          "innovation_spread_h": v,
                                          "travel_time_lag": False}}}

        assert _scaling_da_spread_h(cfg(6.0)) == 6.0
        # zero must not fall back to the default: it is the no-halo case, where
        # nothing reads past a window boundary and no enlargement is wanted.
        assert _scaling_da_spread_h(cfg(0.0)) == 0.0

    def test_a_shorter_lag_span_can_still_be_the_binding_one(self):
        from troute.AbstractNetwork import _scaling_da_spread_h

        cfg = {"streamflow_da": {"streamflow_scaling": True,
                                 "streamflow_scaling_parameters": {
                                     "travel_time_lag": True,
                                     "innovation_spread_h": 0.0,
                                     "lag_window_h": 6.0}}}
        assert _scaling_da_spread_h(cfg) == 6.0
        cfg["streamflow_da"]["streamflow_scaling_parameters"]["innovation_spread_h"] = 4.0
        assert _scaling_da_spread_h(cfg) == 10.0


class TestShortFinalWindowIsFoldedIn:
    """A final remainder shorter than the travel-time span is not self-contained.

    The enlargement exempts the final window, which is right for the forward
    halo and wrong for the lag: measured on the Ohio subset with a 12 h lag, a
    4-file remainder moved the last four hours by 17.1 m3/s (backward) and
    12.9 m3/s (forward) against an evenly divided run.
    """

    def _widths(self, n_columns, window, span=12):
        from troute.window_plan import plan_windows

        return [stop - start for start, stop in plan_windows(n_columns, window, span)]

    def test_no_window_is_left_under_the_span(self):
        # 96 columns at a window of 46 would fill to [46, 46, 4]; the 4 is under the
        # 12 column span, so the split evens out instead of leaving it.
        widths = self._widths(96, 46)
        assert sum(widths) == 96
        assert min(widths) >= 12

    def test_windows_never_exceed_the_requested_width(self):
        """The fold this replaced merged the remainder into its neighbor, which put
        that window over the width the caller sized memory for."""
        widths = self._widths(96, 46)
        assert max(widths) <= 46

    def test_the_split_evens_out_rather_than_leaving_a_remainder(self):
        # 104 columns at a width of 46 needs 3 windows; they come out within one
        # column of each other instead of [46, 46, 12].
        assert self._widths(104, 46) == [35, 35, 34]

    def test_a_run_shorter_than_the_span_is_one_window(self):
        assert self._widths(4, 46) == [4]

    def test_an_unsplittable_run_goes_in_one_window(self):
        # 47 columns against a 24 column span: any two windows leave one at 23.
        assert self._widths(47, 24, span=24) == [47]


class TestThetaPerTree:
    """theta resolves per gage tree: per_tree, then by_vpu, then default.

    One tree carries one exponent (the closed form telescopes only for a
    constant), so a gage id is the finest key the method admits. The proposal
    fits 0.77 on a 0.3-21.2 km2 semi-humid watershed and says simple scaling
    stops holding above roughly 50 km2, with regional exponents from 0.2 to 0.9,
    so this is how an operator carries their own regional values.
    """

    class _Net:
        def __init__(self, gage_vpu=None):
            self.gage_vpu = gage_vpu or {}

    def _resolve(self, **kw):
        from troute.scaling_da.preprocess import _theta_by_site

        return _theta_by_site(
            self._Net(kw.pop("gage_vpu", {})),
            kw.pop("per_tree", {}),
            kw.pop("by_vpu", {}),
            kw.pop("default", 0.77),
            known_sites=kw.pop("known_sites", ()),
        )

    def test_per_tree_keys_are_gage_ids(self):
        got = self._resolve(per_tree={"03031500": 0.55}, known_sites={"03031500"})
        assert got == {"03031500": 0.55}

    def test_per_tree_wins_over_by_vpu(self):
        got = self._resolve(
            per_tree={"03031500": 0.30},
            by_vpu={"05": 0.85},
            gage_vpu={"03031500": "05", "03049800": "05"},
        )
        assert got["03031500"] == 0.30   # the specific key
        assert got["03049800"] == 0.85   # falls to its VPU

    def test_sites_named_nowhere_are_absent_so_the_default_applies(self):
        got = self._resolve(per_tree={"03031500": 0.4}, known_sites={"03031500"})
        assert "09999999" not in got

    def test_an_unmatched_per_tree_id_warns_rather_than_silently_doing_nothing(self, caplog):
        import logging as _logging

        with caplog.at_level(_logging.WARNING, logger="TROUTE"):
            self._resolve(per_tree={"BADID": 0.4}, known_sites={"03031500"})
        assert any("per_tree" in r.getMessage() for r in caplog.records)


def test_the_lastobs_harvest_runs_for_a_scaling_run() -> None:
    """Stale-obs decay has to survive a window boundary.

    The kernel re-seeds lastobs from each window's own t0 observation when the frame
    is empty, so without this harvest a gage that falls silent stops being corrected
    at whatever boundary max_loop_size lands on, and a chunked AnA cycle differs from
    an unchunked one. The scaling arm drives the same nudging override as legacy
    nudging, so the kernel already records what the harvest reads.
    """
    import numpy as np
    import pandas as pd
    from troute.DataAssimilation import NudgingDA

    class _DA(NudgingDA):
        def __init__(self, params):
            self._data_assimilation_parameters = params
            self._last_obs_df = pd.DataFrame()

    # r[3] is (gage segment ids, time since obs, last obs value), as the kernel emits.
    run_results = [[
        np.array([10]), np.zeros((1, 4)), 0,
        (np.array([10]), np.array([300.0]), np.array([7.5])),
        0, 0, 0, 0, np.zeros((1, 1)), np.zeros((1, 2)),
    ]]

    scaling = _DA({"streamflow_da": {"streamflow_scaling": True,
                                     "streamflow_nudging": False}})
    scaling.update_after_compute(run_results, 3600)
    assert not scaling._last_obs_df.empty, (
        "a scaling run must harvest lastobs, or decay resets at every window boundary"
    )

    off = _DA({"streamflow_da": {"streamflow_scaling": False,
                                 "streamflow_nudging": False}})
    off.update_after_compute(run_results, 3600)
    assert off._last_obs_df.empty, "no streamflow DA means nothing to harvest"
