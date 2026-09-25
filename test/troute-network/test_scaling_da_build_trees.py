"""Unit tests for the multi-gage tree builder over primitive topology mappings.

Exercises :func:`build_gage_trees_from_mappings` (the network-agnostic core of
the AbstractNetwork adapter) on synthetic reverse-connectivity graphs: the two
stop rules (other gages, waterbodies), tree disjointness, the held-out
``gage_stop_segs`` override, ``site_filter``, and the invalid-area skip.
"""

from __future__ import annotations

from troute.scaling_da import build_gage_trees_from_mappings

# Mainstem chain 5 -> 4 -> 3 -> 2 -> 1 (outlet); rconn = segment -> [upstream].
RCONN_CHAIN = {1: [2], 2: [3], 3: [4], 4: [5]}
AREAS_CHAIN = {1: 100.0, 2: 80.0, 3: 60.0, 4: 40.0, 5: 20.0}


def test_single_gage_spans_whole_chain() -> None:
    trees = build_gage_trees_from_mappings(
        RCONN_CHAIN, {"A": 1}, AREAS_CHAIN, theta_default=0.77
    )
    assert set(trees) == {"A"}
    assert trees["A"].seg_order.tolist() == [1, 2, 3, 4, 5]
    assert trees["A"].gage_area_sqkm == 100.0


def test_two_gages_on_chain_are_disjoint() -> None:
    """A downstream gage's tree stops at the upstream gage; the two are disjoint."""
    # Gage A at outlet (1), gage B upstream at 3.
    trees = build_gage_trees_from_mappings(
        RCONN_CHAIN, {"A": 1, "B": 3}, AREAS_CHAIN, theta_default=0.77
    )
    # A stops at B (segment 3): A covers 1, 2 only.
    assert trees["A"].seg_order.tolist() == [1, 2]
    # B covers itself and everything above it (no further gage upstream): 3, 4, 5.
    assert trees["B"].seg_order.tolist() == [3, 4, 5]
    # Disjoint: no segment appears in both trees.
    seg_a = set(trees["A"].seg_order.tolist())
    seg_b = set(trees["B"].seg_order.tolist())
    assert seg_a.isdisjoint(seg_b)


def test_stop_at_waterbody() -> None:
    """A waterbody-interior segment halts the upstream walk."""
    # Segment 3 is inside a waterbody.
    trees = build_gage_trees_from_mappings(
        RCONN_CHAIN, {"A": 1}, AREAS_CHAIN, waterbody_segs={3}, theta_default=0.77
    )
    # BFS from 1 -> 2 -> (3 is a stop) -> halt. Tree = [1, 2].
    assert trees["A"].seg_order.tolist() == [1, 2]


def test_gage_on_waterbody_still_built() -> None:
    """A gage sitting on a waterbody segment is still corrected at its own segment."""
    trees = build_gage_trees_from_mappings(
        RCONN_CHAIN, {"A": 3}, AREAS_CHAIN, waterbody_segs={3}, theta_default=0.77
    )
    # Gage 3's own segment is never a stop; it walks upstream past it.
    assert trees["A"].seg_order.tolist() == [3, 4, 5]


def test_gage_stop_segs_override_for_heldout() -> None:
    """With only DA-source gages as stops, a correction propagates through held-out gages."""
    # Gages at 1 (source) and 3 (held-out). Only 1 is a DA source / stop.
    trees = build_gage_trees_from_mappings(
        RCONN_CHAIN,
        {"A": 1, "B": 3},
        AREAS_CHAIN,
        theta_default=0.77,
        gage_stop_segs={1},
        site_filter=frozenset({"A"}),
    )
    # A's tree spans the whole chain, passing *through* held-out gage B at 3.
    assert set(trees) == {"A"}
    assert trees["A"].seg_order.tolist() == [1, 2, 3, 4, 5]


def test_site_filter_restricts_built_trees() -> None:
    trees = build_gage_trees_from_mappings(
        RCONN_CHAIN, {"A": 1, "B": 3}, AREAS_CHAIN, site_filter=frozenset({"B"})
    )
    assert set(trees) == {"B"}


def test_skip_gage_with_missing_area() -> None:
    """A gage whose segment has no/zero drainage area is skipped, not crashed."""
    # Gage C at 1 has area; gage D at 99 is absent from the area map (NaN).
    trees = build_gage_trees_from_mappings(
        {1: [2], 99: []},
        {"C": 1, "D": 99},
        {1: 100.0, 2: 50.0},  # 99 missing -> NaN area
        theta_default=0.77,
    )
    assert "C" in trees
    assert "D" not in trees


def test_colocated_gages_deduped_to_one_tree() -> None:
    """Two sites on the same segment build a single tree (first-seen wins).

    Co-located gages are real at CONUS scale; building identical trees for both
    would double-count in the kernel's per-tree sum, so the builder keeps one.
    """
    trees = build_gage_trees_from_mappings(RCONN_CHAIN, {"A": 1, "B": 1}, AREAS_CHAIN)
    assert set(trees) == {"A"}  # first-seen kept; B dropped (same segment)
    assert trees["A"].seg_order.tolist() == [1, 2, 3, 4, 5]


def test_junction_split_in_built_tree() -> None:
    """A confluence in the network is flagged for the flow-ratio split."""
    # 3 and 4 merge at 2, then 2 -> 1 (gage).
    rconn = {1: [2], 2: [3, 4]}
    areas = {1: 100.0, 2: 90.0, 3: 50.0, 4: 40.0}
    trees = build_gage_trees_from_mappings(rconn, {"A": 1}, areas)
    tree = trees["A"]
    assert tree.seg_order.tolist() == [1, 2, 3, 4]
    # Children of the confluence (2) take the flow-ratio split; the linear 1->2 does not.
    order = tree.seg_order.tolist()
    assert not bool(tree.step_is_junction[order.index(2)])  # 1->2 is linear
    assert bool(tree.step_is_junction[order.index(3)])  # 2->3 is a junction step
    assert bool(tree.step_is_junction[order.index(4)])  # 2->4 is a junction step


def test_disjoint_across_branched_multigage_network() -> None:
    """Every segment is claimed by at most one tree across a branched multi-gage net."""
    # Outlet 1; tributaries 2 (gaged at B) and 3; above 2 is 4, above 3 is 5.
    rconn = {1: [2, 3], 2: [4], 3: [5]}
    areas = {1: 100.0, 2: 60.0, 3: 40.0, 4: 30.0, 5: 20.0}
    trees = build_gage_trees_from_mappings(rconn, {"A": 1, "B": 2}, areas)
    seg_a = set(trees["A"].seg_order.tolist())
    seg_b = set(trees["B"].seg_order.tolist())
    # B (at 2) takes 2 and 4; A (at 1) takes 1, 3, 5 and stops at B's segment 2.
    assert seg_b == {2, 4}
    assert seg_a == {1, 3, 5}
    assert seg_a.isdisjoint(seg_b)
