"""The fingerprint must depend on the network, and on nothing else.

A hash that moves for an unchanged network refuses every restart that network wrote.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from troute.network_fingerprint import (
    FINGERPRINT_KEY,
    check_fingerprint,
    network_fingerprint,
)

_RECORD = "og_nhf_lake_id"


def _links(ids: list[int], fp_ids: list[int], orders: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"fp_id": fp_ids, "segment_order": orders},
        index=pd.Index(ids, name="up_node_id"),
    )


def _lakes(record_ids: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {_RECORD: record_ids},
        index=pd.Index(range(5001, 5001 + len(record_ids)), name="nhf_lake_id"),
    )


# Lake 5001 drains to link 1, lake 5002 to link 2.
BASE_EDGES = {3: [1], 1: [2], 2: [], 5001: [1], 5002: [2]}


def _fp(
    links: pd.DataFrame, lakes: pd.DataFrame, edges: dict[int, list[int]] = BASE_EDGES
) -> str:
    return network_fingerprint(links, lakes, _RECORD, edges)


BASE_LINKS = _links([3, 1, 2], [30, 10, 20], [0.0, 0.0, 1.0])
BASE_LAKES = _lakes([222, 111])
BASE = _fp(BASE_LINKS, BASE_LAKES)


def test_row_order_does_not_change_it():
    assert _fp(BASE_LINKS.sort_index(), BASE_LAKES) == BASE


def test_lake_order_does_not_change_it():
    assert _fp(BASE_LINKS, BASE_LAKES.sort_values(_RECORD)) == BASE


def test_column_dtype_does_not_change_it():
    """fp_id floats the moment any row is null, so the same network presents it as
    int64 or float64 depending on whether a subset kept a null row. Hashing the repr
    would refuse every restart that network wrote."""
    assert _fp(BASE_LINKS.astype({"fp_id": "float64"}), BASE_LAKES) == BASE
    assert _fp(BASE_LINKS.astype({"segment_order": "float64"}), BASE_LAKES) == BASE


def test_a_relabeled_link_changes_it():
    """The case the fingerprint exists for: the id set is identical, but an id now
    names a different segment, so presence and coverage checks both pass."""
    relabeled = _links([3, 1, 2], [10, 30, 20], [0.0, 0.0, 1.0])
    assert _fp(relabeled, BASE_LAKES) != BASE


def test_an_added_or_dropped_link_changes_it():
    assert _fp(_links([1, 2], [10, 20], [0.0, 1.0]), BASE_LAKES) != BASE


def test_a_dropped_lake_changes_it():
    """Flagging a dam run-of-river does exactly this, and renumbers every lake after
    it."""
    assert _fp(BASE_LINKS, _lakes([222])) != BASE


def test_a_null_identity_column_is_distinct_from_a_value():
    nulled = BASE_LINKS.astype({"fp_id": "float64"}).copy()
    nulled.loc[1, "fp_id"] = np.nan
    assert _fp(nulled, BASE_LAKES) != BASE


def test_an_empty_network_still_hashes():
    assert _fp(pd.DataFrame(), pd.DataFrame(), {}) == _fp(pd.DataFrame(), pd.DataFrame(), {})


def test_edge_order_does_not_change_it():
    shuffled = {node: list(reversed(dn)) for node, dn in reversed(BASE_EDGES.items())}
    assert _fp(BASE_LINKS, BASE_LAKES, shuffled) == BASE


def test_a_moved_reservoir_outlet_changes_it():
    """A lake's outlet is recorded in the routing graph alone: its absorbed links leave
    the link table, so no row there names where the lake drains."""
    moved = {**BASE_EDGES, 5001: [2]}
    assert _fp(BASE_LINKS, BASE_LAKES, moved) != BASE


def test_a_matching_stamp_passes():
    check_fingerprint(BASE, BASE, "a state file")


@pytest.mark.parametrize(
    ("saved", "match"),
    [(None, "carries no network_fingerprint"), ("other", "different network")],
)
def test_a_bad_stamp_is_refused(saved: str | None, match: str):
    with pytest.raises(ValueError, match=match):
        check_fingerprint(saved, BASE, "a state file")


def test_the_key_is_what_the_writers_use():
    assert FINGERPRINT_KEY == "network_fingerprint"


def test_a_shifted_lake_allocation_changes_it():
    """A lake demoted to a channel after allocation leaves its slot empty, and the same
    dam flagged run-of-river leaves before it, so one lake set holds different routed ids."""
    demoted = _lakes([999, 111, 222]).drop(index=5001)  # 111@5002, 222@5003
    flagged = _lakes([111, 222])  # 111@5001, 222@5002
    assert sorted(demoted[_RECORD]) == sorted(flagged[_RECORD])
    assert _fp(BASE_LINKS, demoted) != _fp(BASE_LINKS, flagged)


def test_a_rewired_edge_changes_it():
    """Two lakes can swap which flowpaths they own while every link id, fp_id and
    segment_order stays put: only the routing edges move. Each lake then sits where the
    other did, so its stored pool elevation belongs to a different water body."""
    rewired = BASE_LINKS.copy()
    rewired["downstream"] = [9, 8, 7]
    swapped = rewired.copy()
    swapped.loc[3, "downstream"] = 7
    swapped.loc[2, "downstream"] = 9
    assert _fp(rewired, BASE_LAKES) != _fp(swapped, BASE_LAKES)
