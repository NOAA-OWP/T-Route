"""Regression tests for collapsing a lake's whole flowpath subnetwork.

``_refactor_reservoirs`` used to absorb only the outlet ``virtual_fp_id`` the
``lakes`` layer declares, leaving every other flowpath crossing the polygon to
route as Muskingum-Cunge channel *inside the reservoir* -- ~7.5 of the 8.5
flowpaths a CONUS lake has. These pin the behavior now that ``lake_vfp_crosswalk``
supplies the real one-to-many association.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import pytest

from troute.nhf_preprocess import NHFPreprocessMixin, _lake_vfp_clusters


class _Net(NHFPreprocessMixin):
    """The slice of the network state ``_refactor_reservoirs`` touches."""

    def __init__(self, links, lakes):
        self._dataframe = links
        self._waterbody_df = lakes
        self._connections = None
        self._terminal_codes = set(links["downstream"]).difference(links.index)
        self.zero_nodes = []
        self.vfp_nex_ids = np.array(sorted(links.index), dtype=np.int64)
        self._fp_outlet_crosswalk = defaultdict(list)
        for node in links.index:
            self._fp_outlet_crosswalk[int(node)] = [int(links.loc[node, "fp_id"])]
        self._link_lake_crosswalk = None
        self.waterbody_connections = {}

    # AbstractNetwork provides these; reproduced here so the test needs no
    # geopackage, forcing set, or config.
    @property
    def dataframe(self):
        return self._dataframe

    @dataframe.setter
    def dataframe(self, val):
        self._dataframe = val

    @property
    def waterbody_dataframe(self):
        return self._waterbody_df

    @waterbody_dataframe.setter
    def waterbody_dataframe(self, val):
        self._waterbody_df = val

    @property
    def connections(self):
        if self._connections is None:
            self._connections = {
                int(node): ([int(dn)] if dn not in self._terminal_codes else [])
                for node, dn in self._dataframe["downstream"].items()
            }
        return self._connections


def _links(edges):
    """Build a link table from ``(up_node, downstream, vfp_id)`` triples."""
    up, dn, vfp = zip(*edges)
    return pd.DataFrame(
        {
            "downstream": np.array(dn, dtype=np.int64),
            "vfp_id": np.array(vfp, dtype=np.int64),
            "fp_id": np.array(vfp, dtype=np.int64),
            "dx": np.full(len(up), 100.0),
        },
        index=pd.Index(np.array(up, dtype=np.int64), name="up_node_id"),
    )


def _lakes(rows):
    """Build a waterbody table from ``{synthetic_id: (og_id, outlet_vfp)}``."""
    return pd.DataFrame(
        {
            "og_nhf_lake_id": [og for og, _ in rows.values()],
            "virtual_fp_id": [vfp for _, vfp in rows.values()],
            "LkArea": 1.0,
        },
        index=pd.Index(list(rows), name="nhf_lake_id"),
    )


def _crosswalk(pairs):
    return pd.DataFrame(pairs, columns=["nhf_lake_id", "virtual_fp_id"])


# A lake spanning three flowpaths: arms 10 and 11 join at node 3, outlet arm 12
# leaves at node 5. Links 8 and 9 feed the arms from outside; vfp 13 is the
# channel below. (up_node, downstream, vfp)
_SPAN_EDGES = [
    (1, 3, 10), (2, 3, 11), (3, 4, 12), (4, 5, 12),
    (5, 6, 13), (8, 1, 14), (9, 2, 15),
]
_SPAN_LAKES = _lakes({100: (7001, 12)})
_SPAN_CW = _crosswalk([(7001, 10), (7001, 11), (7001, 12)])


def test_whole_subnetwork_is_absorbed_and_both_inlets_rewired():
    net = _Net(_links(_SPAN_EDGES), _SPAN_LAKES.copy())
    net._refactor_reservoirs(_SPAN_CW)

    assert set(net.dataframe["vfp_id"]).isdisjoint({10, 11})
    assert {1, 2, 4}.isdisjoint(net.dataframe.index)
    assert list(net.waterbody_dataframe.index) == [100]  # not demoted to MC
    assert net.connections[100] == [5]
    # Both inlets feed the lake. The old single-inlet guard rejected this outright.
    assert net.connections[8] == [100]
    assert net.connections[9] == [100]
    # Every absorbed flowpath reports the lake's outflow.
    assert set(net._fp_outlet_crosswalk[100]) == {10, 11, 12}


def test_no_crosswalk_absorbs_only_the_declared_outlet_flowpath():
    """Pre-crosswalk behavior, still reachable for older geopackages."""
    net = _Net(_links(_SPAN_EDGES), _SPAN_LAKES.copy())
    net._refactor_reservoirs(None)

    # The arms keep routing as Muskingum-Cunge channels inside the lake.
    assert set(net.dataframe["vfp_id"]).issuperset({10, 11})
    assert net.connections[100] == [5]
    assert set(net._fp_outlet_crosswalk[100]) == {12}


def test_lakes_sharing_one_declared_outlet_chain_together():
    """Same outlet flowpath, so one absorbed set: they chain rather than each
    claiming it."""
    lakes = _lakes({100: (7001, 12), 101: (7002, 12)})
    cw = _crosswalk([(7001, 10), (7001, 12), (7002, 11), (7002, 12)])
    net = _Net(_links(_SPAN_EDGES), lakes)
    net._refactor_reservoirs(cw)

    assert list(net.waterbody_dataframe.index) == [100, 101]
    # 101 -> 100 -> 5: one chain on the shared subnetwork, not two claims on it.
    assert net.connections[100] == [5]
    assert net.connections[101] == [100]
    assert net.connections[8] == [101]


def test_serial_lakes_chain_through_topology_not_row_order():
    """101 is upstream of 100 on one stem. Each keeps its OWN outlet, so the chain
    comes from topology and reversing the table cannot reverse the reservoirs."""
    edges = [(1, 2, 10), (2, 3, 11), (3, 4, 12), (4, 5, 13)]
    cw = _crosswalk([(7001, 12), (7001, 11), (7002, 10)])

    def wire(rows):
        net = _Net(_links(edges), _lakes(rows))
        net._refactor_reservoirs(cw)
        return net

    for rows in (
        {100: (7001, 12), 101: (7002, 10)},
        {101: (7002, 10), 100: (7001, 12)},  # same lakes, reversed row order
    ):
        net = wire(rows)
        # 101 feeds 100, never the reverse.
        assert net.connections[100] == [4], rows
        assert net.connections[101] == [net.connections[101][0]], rows
        assert 100 not in net.connections[101], rows


def test_crosswalked_flowpath_below_the_outlet_is_not_absorbed():
    """vfp 20 clips the polygon and carries on downstream. Absorbing it would
    delete real channel, so the set is cut at the outlet and 20 survives."""
    cw = _crosswalk([(7001, 10), (7001, 11), (7001, 12), (7001, 20)])
    edges = [*_SPAN_EDGES, (20, 21, 20), (21, 22, 20)]
    net = _Net(_links(edges), _SPAN_LAKES.copy())
    net._refactor_reservoirs(cw)

    assert 20 in set(net.dataframe["vfp_id"])
    assert {20, 21}.issubset(net.dataframe.index)
    # ...and the arms inside the lake are still collapsed.
    assert set(net.dataframe["vfp_id"]).isdisjoint({10, 11})
    assert list(net.waterbody_dataframe.index) == [100]
    assert net.connections[100] == [5]


def test_parallel_lakes_keep_their_own_arms_instead_of_being_chained():
    """Separate arms merging below both. Chaining would push arm 10's inflow
    through the lake on arm 11, so each takes only what drains to its own outlet."""
    lakes = _lakes({100: (7001, 10), 101: (7002, 11)})
    cw = _crosswalk([(7001, 10), (7001, 12), (7002, 11), (7002, 12)])
    net = _Net(_links(_SPAN_EDGES), lakes)
    net._refactor_reservoirs(cw)

    assert list(net.waterbody_dataframe.index) == [100, 101]
    # Each lake sits on its own arm, discharging to the confluence, not in series.
    assert net.connections[100] == [3]
    assert net.connections[101] == [3]
    # The shared downstream flowpath is claimed by neither.
    assert 12 in set(net.dataframe["vfp_id"])


def test_lake_with_no_routed_links_is_demoted():
    lakes = _lakes({100: (7001, 99)})
    net = _Net(_links(_SPAN_EDGES), lakes)
    net._refactor_reservoirs(_crosswalk([(7001, 99)]))

    assert net.waterbody_dataframe.empty
    assert len(net.dataframe) == len(_SPAN_EDGES)


def test_lake_with_no_flowpath_at_all_is_left_alone_not_demoted():
    """Owns no flowpath, so nothing to absorb -- but it must stay in the waterbody
    set. Demoting it would unhook an fp_id-only Great Lake from its type-6 DA."""
    lakes = _lakes({100: (7001, 12), 101: (7002, None)})
    net = _Net(_links(_SPAN_EDGES), lakes)
    net._refactor_reservoirs(_SPAN_CW)

    assert list(net.waterbody_dataframe.index) == [100, 101]
    assert net.connections[100] == [5]


def test_a_lake_id_equal_to_a_flowpath_id_does_not_merge_unrelated_lakes():
    """Lake ids are bounded against routing node ids, not against virtual_fp_id, so
    an untagged union-find would merge lake 10 with flowpath 10 and cascade."""
    lakes = _lakes({10: (7001, 10), 11: (7002, 11)})
    cw = _crosswalk([(7001, 10), (7002, 11)])
    lake_cluster, _ = _lake_vfp_clusters(lakes, cw)

    assert lake_cluster[10] != lake_cluster[11]


@pytest.mark.parametrize("crosswalk", [None, pd.DataFrame()])
def test_clusters_degenerate_to_the_declared_outlet_without_a_crosswalk(crosswalk):
    lakes = _lakes({100: (7001, 12), 101: (7002, 12), 102: (7003, 10)})
    lake_cluster, vfp_cluster = _lake_vfp_clusters(lakes, crosswalk)

    # Lakes sharing a declared outlet merge; the third stays on its own.
    assert lake_cluster[100] == lake_cluster[101] != lake_cluster[102]
    assert set(vfp_cluster) == {10, 12}
