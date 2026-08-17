"""Choosing one gage per routing link, deterministically.

Only one observation can be assimilated per routing link, but the CONUS
hydrofabric puts 4096 USGS gages on 1729 shared virtual flowpaths. Nothing in the
gages layer separates them positionally: within every colliding group their
``segment_order`` and ``dn_virtual_nex_id`` are identical, so they cannot be placed
on distinct sub-links. One has to be chosen, and the choice must not depend on the
order rows happen to appear in the geopackage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from troute.nhf_preprocess import NHFPreprocessMixin


class _Preproc(NHFPreprocessMixin):
    """The mixin with no geopackage, so ranking falls back to status + site number."""

    supernetwork_parameters: dict = {}


@pytest.fixture
def preproc() -> _Preproc:
    return _Preproc()


def _gages(rows) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["site_no", "status", "virtual_fp_id", "dn_virtual_nex_id",
                 "up_node_id", "link_segment_order"],
    )


def test_active_gage_wins_over_discontinued(preproc):
    df = _gages([
        ("02000001", "USGS-discontinued", 10, 99, 500, 0),
        ("02000002", "USGS-active", 10, 99, 500, 0),
    ])
    out = preproc._one_link_per_gage(preproc._gage_selection_rank(df), "USGS")
    assert list(out["site_no"]) == ["02000002"]


def test_choice_is_stable_under_row_order(preproc):
    """The defect this guards: to_dict() kept whichever row came last, so a
    re-exported geopackage could silently change which gage is assimilated."""
    rows = [
        ("02000001", "USGS-discontinued", 10, 99, 500, 0),
        ("02000002", "USGS-active", 10, 99, 500, 0),
        ("02000003", "USGS-active", 10, 99, 500, 0),
    ]
    forward = preproc._one_link_per_gage(preproc._gage_selection_rank(_gages(rows)), "USGS")
    reverse = preproc._one_link_per_gage(
        preproc._gage_selection_rank(_gages(list(reversed(rows)))), "USGS"
    )
    assert list(forward["site_no"]) == list(reverse["site_no"])
    # tie broken by site number, so the lower one wins
    assert list(forward["site_no"]) == ["02000002"]


def test_gages_on_distinct_links_are_all_kept(preproc):
    df = _gages([
        ("02000001", "USGS-active", 10, 99, 500, 0),
        ("02000002", "USGS-active", 20, 88, 600, 0),
    ])
    out = preproc._one_link_per_gage(preproc._gage_selection_rank(df), "USGS")
    assert sorted(out["site_no"]) == ["02000001", "02000002"]


def test_outlet_link_is_chosen_within_a_flowpath(preproc):
    """A gage joins to every sub-link of its flowpath; the outlet is the one used."""
    df = _gages([
        ("02000001", "USGS-active", 10, 99, 500, 0),
        ("02000001", "USGS-active", 10, 99, 501, 1),
        ("02000001", "USGS-active", 10, 99, 502, 2),
    ])
    out = preproc._one_link_per_gage(preproc._gage_selection_rank(df), "USGS")
    assert list(out["up_node_id"]) == [502]


def test_unplaceable_gage_is_dropped_not_kept_as_null(preproc, caplog):
    df = _gages([
        ("02000001", "USGS-active", 10, 99, np.nan, np.nan),
        ("02000002", "USGS-active", 20, 88, 600, 0),
    ])
    out = preproc._one_link_per_gage(preproc._gage_selection_rank(df), "USGS")
    assert list(out["site_no"]) == ["02000002"]
    assert out["up_node_id"].dtype.kind == "i"
