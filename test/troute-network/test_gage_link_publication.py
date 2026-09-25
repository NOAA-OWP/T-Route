"""A gaged flowpath must publish the link its observation is assimilated on.

A gage is placed at the outlet of its virtual flowpath, but output is published at
the outlet of the flowpath. Those differ whenever a flowpath holds several virtual
flowpaths in series (157 of 665 gages on VPU 01), and the published series was then
a routed distance below the value the DA had replaced.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from troute.NHF import NHF

# up_node_id -> fp_id. Flowpath 100 is split across three routing links, so its
# outlet (link 3, highest segment_order) is not where the gage sits (link 1).
_LINKS = pd.DataFrame(
    {"fp_id": [100, 100, 100, 200], "segment_order": [0, 1, 2, 0]},
    index=pd.Index([1, 2, 3, 4], name="up_node_id"),
)


def _network(gage_links: list[int]) -> NHF:
    net = NHF.__new__(NHF)
    net._dataframe = _LINKS
    net._gages = {"gages": {link: f"0{link}" for link in gage_links}}
    # what _build_fp_outlet_crosswalk produces: the flowpath outlet link
    net._fp_outlet_crosswalk = defaultdict(list, {3: [100], 4: [200]})
    return net


def test_gaged_flowpath_moves_to_the_assimilated_link() -> None:
    net = _network([1])
    net._publish_gage_link_for_gaged_flowpaths()
    assert net._fp_outlet_crosswalk[1] == [100], "fp 100 should publish at the gage link"
    assert 3 not in net._fp_outlet_crosswalk, "the old outlet entry should be gone"
    assert net._fp_outlet_crosswalk[4] == [200], "ungaged flowpaths are untouched"


def test_gage_already_at_the_outlet_is_left_alone() -> None:
    net = _network([3])
    before = {k: list(v) for k, v in net._fp_outlet_crosswalk.items()}
    net._publish_gage_link_for_gaged_flowpaths()
    assert {k: list(v) for k, v in net._fp_outlet_crosswalk.items()} == before


def test_merged_entries_are_not_split() -> None:
    """A link carrying several fps is a merged mapping; moving one changes the sum."""
    net = _network([1])
    net._fp_outlet_crosswalk = defaultdict(list, {3: [100, 999], 4: [200]})
    net._publish_gage_link_for_gaged_flowpaths()
    assert net._fp_outlet_crosswalk[3] == [100, 999]
    assert 1 not in net._fp_outlet_crosswalk


def test_no_gages_is_a_no_op() -> None:
    net = _network([])
    net._gages = {}
    before = {k: list(v) for k, v in net._fp_outlet_crosswalk.items()}
    net._publish_gage_link_for_gaged_flowpaths()
    assert {k: list(v) for k, v in net._fp_outlet_crosswalk.items()} == before
