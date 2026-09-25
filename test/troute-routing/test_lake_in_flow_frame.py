"""Waterbody nodes must be addressable columns of the routed flow frame.

The simple-scaling DA halts each gage tree at waterbodies and records the cut
branch as *pruned*: the branch gets no correction, but its modeled flow still has
to enter the confluence-split denominator so a surviving sibling receives only its
proportional share (Edge Case 1). That lookup runs through
``GageTree.with_positions``, which FAILS CLOSED on any pruned segment absent from
the flow frame -- it raises, and the caller drops the whole site, rather than
publish a correction whose denominator is known to be short.

The flow frame's columns are the ids the kernel returns: ``data_idx`` minus the
offnetwork-upstream rows ``fill_index_mask`` strips. Lake nodes are absent from
``param_df`` entirely, so they reach ``data_idx`` only because
``ExecutionPlan._build_compute_job`` reindexes ``river_df`` to add them back.

Two links are pinned here, because either one breaks the DA silently:

1. every waterbody is a returned id, exactly once, across every plan method
   (including when it is also another job's offnetwork upstream);
2. a gage tree that stops at a lake can resolve that lake against the resulting
   column map, so no pruned branch is dropped from the denominator -- and when it
   cannot, nothing is published for that gage.
"""
from __future__ import annotations

import logging
from collections import Counter

import pandas as pd
import pytest

from troute import nhd_network
from troute.routing.compute import (
    AssimilationData,
    ExecutionPlan,
    NetworkTopology,
    ReachData,
    WaterbodyData,
)
from troute.routing.fast_reach.scaling_da import apply_scaling_da
from troute.scaling_da.gage_tree import GageTree, build_one_gage_tree

_WB_COLS = ["LkArea", "LkMxE", "OrificeA", "OrificeC", "OrificeE",
            "WeirC", "WeirE", "WeirL", "ifd", "qd0", "h0"]
_REACH_COLS = ["dt", "bw", "tw", "twcc", "dx", "n", "ncc", "cs", "s0", "alt"]

# 10, 20 -> lake 30 -> 40 -> lake 45 -> 50 (tailwater)
_CONNECTIONS = {10: [30], 20: [30], 30: [40], 40: [45], 45: [50], 50: []}
_LAKES = (30, 45)
_CHANNELS = (10, 20, 40, 50)

# Gage at 50. Confluence 45 is fed by lake 30 (stopped -> pruned) and channel 40
# (kept), which is the only shape where a dropped pruned position changes a number.
_CONFLUENCE = {10: [30], 30: [45], 20: [40], 40: [45], 45: [50], 50: []}
_CONFLUENCE_LAKES = (30,)
_CONFLUENCE_CHANNELS = (10, 20, 40, 45, 50)


def _plan(
    connections: dict[int, list[int]],
    lakes: tuple[int, ...],
    channels: tuple[int, ...],
    method: str,
    target_size: int,
) -> ExecutionPlan:
    rconn = nhd_network.reverse_network(connections)
    independent = nhd_network.reachable_network(rconn)
    topology = NetworkTopology(
        connections=connections,
        reverse_connections=rconn,
        # Values are unused by the plan builders, which re-derive paths from
        # connections_by_tw; the KEYS are the tailwater list.
        paths_by_tailwater={tw: [] for tw in independent},
        connections_by_tw=independent,
    )
    empty = pd.DataFrame()
    return ExecutionPlan(
        method,
        topology,
        # param_df carries CHANNELS ONLY -- this is the whole point: nothing in
        # the reach frame knows the lakes exist.
        ReachData(pd.DataFrame(1.0, index=list(channels), columns=_REACH_COLS)),
        WaterbodyData(
            pd.DataFrame(0.0, index=list(lakes), columns=_WB_COLS),
            pd.DataFrame({"reservoir_type": [1] * len(lakes)}, index=list(lakes)),
        ),
        AssimilationData(*([empty] * 13)),
        target_size,
    )


def _returned_ids(plan: ExecutionPlan) -> Counter:
    """Ids the kernel hands back, per job, MIRRORING ``fill_index_mask``.

    ``compute_network_structured`` returns ``data_idx[fill_index_mask]``
    (mc_reach.pyx), and the only rows that mask clears are the job's offnetwork
    upstreams. This is a mirror of the kernel, not the kernel, so it has to be
    updated if that mask is ever taught to clear anything else.

    Multiplicity is preserved deliberately. ``river_df`` is built by concatenating
    the job's channel index with its lake ids, so an id living in BOTH the reach
    frame and the waterbody frame yields a duplicate label; deduplicating with a
    ``set`` would hide that, while ``_assemble_q_model`` keeps both columns and
    the ``fp_id -> position`` map then silently retains only one of them.
    """
    counts: Counter = Counter()
    for jobs in plan.batches.values():
        for job in jobs:
            offnetwork = set(job.offnetwork_upstreams)
            counts.update(i for i in job.river_df.index if i not in offnetwork)
    return counts


@pytest.mark.parametrize(
    ("method", "target_size"),
    [
        ("serial", 0),
        ("by-network", 0),
        ("bmi", 0),
        ("by-subnetwork-jit", 2),
        ("by-subnetwork-jit-clustered", 10_000),
        # Target size 2 forces the network across several jobs, so at least one
        # lake becomes another job's offnetwork upstream and is stripped THERE.
        ("by-subnetwork-jit-clustered", 2),
    ],
)
def test_waterbodies_are_returned_columns(method: str, target_size: int) -> None:
    counts = _returned_ids(_plan(_CONNECTIONS, _LAKES, _CHANNELS, method, target_size))
    missing = [lake for lake in _LAKES if lake not in counts]
    assert not missing, (
        f"waterbody {missing} absent from every job's returned ids, so the scaling "
        "DA cannot read its flow: with_positions raises and the whole gage is "
        "skipped"
    )
    duplicated = [seg for seg, n in counts.items() if n > 1]
    assert not duplicated, (
        f"{duplicated} returned more than once; q_model would carry duplicate "
        "columns and the fp_id -> position map keeps only one of them"
    )


def test_all_segments_returned_exactly_once() -> None:
    """The same guarantee for channels, so a regression cannot be read as
    lake-specific when it is really the offnetwork-strip logic."""
    counts = _returned_ids(
        _plan(_CONNECTIONS, _LAKES, _CHANNELS, "by-subnetwork-jit-clustered", 2)
    )
    assert sorted(counts) == sorted(_CHANNELS + _LAKES)
    assert all(n == 1 for n in counts.values()), counts


@pytest.mark.parametrize(
    ("method", "target_size"),
    [("serial", 0), ("by-subnetwork-jit-clustered", 2)],
)
def test_lake_stopped_branch_resolves_against_the_flow_frame(
    method: str, target_size: int
) -> None:
    """The seam itself: a tree stopped at a lake must place that lake's position.

    A pruned branch with no position is silently absent from ``bsum`` in the
    kernel, which hands the surviving sibling at that confluence the lake's share.
    """
    plan = _plan(_CONFLUENCE, _CONFLUENCE_LAKES, _CONFLUENCE_CHANNELS, method, target_size)
    fp_to_position = {int(seg): pos for pos, seg in enumerate(sorted(_returned_ids(plan)))}

    rconn = nhd_network.reverse_network(_CONFLUENCE)
    tree = build_one_gage_tree(
        50,
        rconn,
        # Channels only: a lake carries no drainage area, which is exactly why it
        # has to be a stop rather than a tree member.
        {seg: 10.0 * i for i, seg in enumerate(_CONFLUENCE_CHANNELS, start=1)},
        stop_segs=frozenset(_CONFLUENCE_LAKES),
        theta=0.77,
    )
    assert list(tree.pruned_segs) == [30], (
        f"expected lake 30 recorded as the pruned branch at confluence 45, got "
        f"{list(tree.pruned_segs)}"
    )
    placed = tree.with_positions(fp_to_position)
    assert placed.pruned_positions.size == placed.pruned_segs.size, (
        "a pruned lake branch has no flow-frame column; its flow cannot enter the "
        "confluence denominator and channel 40 would inherit the share Edge Case 1 "
        "leaves unallocated"
    )


def _confluence_tree() -> GageTree:
    """Gage at 50, confluence 45 fed by stopped lake 30 and surviving channel 40."""
    return build_one_gage_tree(
        50,
        nhd_network.reverse_network(_CONFLUENCE),
        {seg: 10.0 * i for i, seg in enumerate(_CONFLUENCE_CHANNELS, start=1)},
        stop_segs=frozenset(_CONFLUENCE_LAKES),
        theta=0.77,
    )


def test_unresolvable_pruned_branch_raises() -> None:
    """No column for the stopped lake means no tree, not a short denominator."""
    members_only = {seg: pos for pos, seg in enumerate(_CONFLUENCE_CHANNELS)}
    assert 30 not in members_only
    with pytest.raises(KeyError):
        _confluence_tree().with_positions(members_only)


def test_unresolvable_pruned_branch_publishes_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The contract that matters: the gage is skipped, not corrected from a short sum.

    Before this failed closed, ``with_positions`` warned and returned a partial tree,
    and channel 40 took the share the lake's own DA is supposed to own.
    """
    idx = pd.date_range("2020-01-01", periods=2, freq="1h")
    # Tree members only -- the stopped lake 30 has no column.
    q_model = pd.DataFrame(
        {seg: [10.0, 10.0] for seg in _CONFLUENCE_CHANNELS}, index=idx, dtype=float
    )
    q_obs = pd.DataFrame({"G": [20.0, 20.0]}, index=idx, dtype=float)

    with caplog.at_level(logging.WARNING, logger="TROUTE"):
        _, dq = apply_scaling_da(q_model, q_obs, {"G": 50}, {"G": _confluence_tree()})

    assert (dq.to_numpy() == 0.0).all(), (
        "a correction was published for a gage whose confluence denominator could "
        f"not be completed:\n{dq}"
    )
    assert any("not in q_model columns" in r.getMessage() for r in caplog.records), (
        f"the skip must be logged, got: {[r.getMessage() for r in caplog.records]}"
    )
