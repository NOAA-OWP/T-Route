"""Regression tests for reservoir reach splitting in the clustered/partitioned
execution plan.

Bug (fixed): ``ExecutionPlan._clean_compute_jobs`` selected a *junction-only*
reach split whenever waterbodies were present but no USGS streamflow data
assimilation was configured -- the common NHF case. That merged a reservoir
node with its downstream channel into a single multi-segment reach. The
Muskingum-Cunge kernel (``compute_network_structured``) models only the first
waterbody in a reach and treats a ``reach_type == 1`` reach as a lone reservoir
id, so a multi-segment "reservoir" reach crashed ``binary_find`` with
``element <channel> not found in [<lake>]``.

The fix makes that branch break reaches at waterbody inlets/outlets (matching the
treewise plan). These tests assert the invariant the kernel relies on: every
reach that contains a waterbody segment is a singleton.
"""
import pandas as pd
import pytest

import troute.nhd_network as nhd_network

# The regression lives in the ExecutionPlan compute refactor (PR #98); the
# current development compute path splits reaches at waterbodies correctly and
# does not define these classes. The module-level skip makes this test a
# forward guard: it activates automatically when the refactor lands and fails
# if the waterbody reach split is reintroduced incorrectly.
try:
    from troute.routing.compute import (
        AssimilationData,
        ExecutionPlan,
        NetworkTopology,
        WaterbodyData,
    )
except ImportError:
    pytest.skip(
        "ExecutionPlan compute refactor not present; this guard activates "
        "when it lands",
        allow_module_level=True,
    )

_WB_COLS = ["LkArea", "LkMxE", "OrificeA", "OrificeC", "OrificeE",
            "WeirC", "WeirE", "WeirL", "ifd", "qd0", "h0"]


def _assimilation_data(usgs_df=None):
    """An AssimilationData with everything empty except an optional usgs_df."""
    empty = pd.DataFrame()
    return AssimilationData(
        reservoir_usgs_df=empty, reservoir_usgs_param_df=empty,
        reservoir_usace_param_df=empty, reservoir_usace_df=empty,
        reservoir_usbr_df=empty, reservoir_usbr_param_df=empty,
        reservoir_rfc_df=empty, reservoir_rfc_param_df=empty,
        great_lakes_df=empty, great_lakes_param_df=empty,
        great_lakes_climatology_df=empty,
        usgs_df=pd.DataFrame() if usgs_df is None else usgs_df,
        lastobs_df=empty,
    )


def _waterbody_data(lake_ids):
    df = pd.DataFrame(0.0, index=list(lake_ids), columns=_WB_COLS)
    types = pd.DataFrame({"reservoir_type": [1] * len(lake_ids)},
                         index=list(lake_ids))
    return WaterbodyData(df, types)


def _decompose(connections, lake_ids, usgs_df=None):
    """Run ExecutionPlan._clean_compute_jobs on a single-partition network and
    return the flattened list of reaches it produced."""
    rconn = nhd_network.reverse_network(connections)
    nodes = set(connections)
    for downstreams in connections.values():
        nodes.update(downstreams)
    tailwaters = [n for n in nodes if not connections.get(n)]
    topology = NetworkTopology(connections, rconn, {}, {})
    waterbody_data = _waterbody_data(lake_ids)
    assimilation_data = _assimilation_data(usgs_df)

    # one partition covering the whole network, keyed by its tailwater
    partitions_by_level = {0: {tailwaters[0]: list(nodes)}}

    plan = ExecutionPlan.__new__(ExecutionPlan)  # bypass __init__; method is self-contained
    result = plan._clean_compute_jobs(
        partitions_by_level, topology, waterbody_data, assimilation_data
    )
    reaches = []
    for paths_by_tw in result.values():
        for paths in paths_by_tw.values():
            reaches.extend(list(r) for r in paths)
    return reaches


# channels 10, 20 -> lake 30 -> channel 40 -> 50 (tailwater)
_LAKE_NETWORK = {10: [30], 20: [30], 30: [40], 40: [50], 50: []}


def test_reservoir_reach_is_singleton_without_streamflow_da():
    """With waterbodies present and no USGS DA, the reservoir must be its own
    reach -- not merged with the downstream channel (the original crash)."""
    reaches = _decompose(_LAKE_NETWORK, lake_ids=[30])
    assert reaches, "no reaches produced"
    lake_reaches = [r for r in reaches if 30 in r]
    assert lake_reaches, "lake node 30 not found in any reach"
    for reach in lake_reaches:
        assert reach == [30], (
            f"reservoir reach must be the singleton [30], got {reach} -- the "
            "reach splitter merged the reservoir with its downstream channel, "
            "which makes compute_network_structured.binary_find raise."
        )


def test_reservoir_reach_is_singleton_with_streamflow_da():
    """The gages+waterbodies split path must also keep reservoirs singletons."""
    usgs_df = pd.DataFrame(0.0, index=[40], columns=[0])  # a non-empty gage frame
    reaches = _decompose(_LAKE_NETWORK, lake_ids=[30], usgs_df=usgs_df)
    lake_reaches = [r for r in reaches if 30 in r]
    assert lake_reaches, "lake node 30 not found in any reach"
    for reach in lake_reaches:
        assert reach == [30], f"reservoir reach must be [30], got {reach}"


def test_no_waterbody_network_still_decomposes():
    """The no-waterbody else branch must still split a simple chain at junctions."""
    # 10 -> 30 -> 40 -> 50, plus a tributary 20 -> 30 (junction at 30)
    reaches = _decompose(_LAKE_NETWORK, lake_ids=[])
    # every node should appear exactly once across the reaches
    flat = [seg for reach in reaches for seg in reach]
    assert sorted(flat) == [10, 20, 30, 40, 50]
    assert len(flat) == len(set(flat)), "a segment appeared in more than one reach"
