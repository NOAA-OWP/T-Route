"""Regression tests for Great Lakes (reservoir_type 6) handling in NHF.

The Great Lakes carry no level-pool parameters (LkArea is NaN in NHF), so they
can only be modeled as type-6 reservoirs whose flows come from data
assimilation. ``troute.nhf_preprocess._great_lakes_for_da`` decides whether and
which Great Lakes re-enter the routable waterbody set:

  * with Great Lakes persistence DA enabled they must be carried at their
    ORIGINAL lake ids so ``compute.py`` can link climatology/observations to
    them by id, and so the ``reservoir_type = 6`` assignment in
    ``preprocess_waterbodies`` matches;
  * with it disabled they must be left out, because ``compute.py`` would
    otherwise demote the type-6 reservoirs to level pool and the kernel would
    crash on their missing parameters.

These tests pin that gating + anchoring decision (the root cause flagged in
review: pulling the Great Lakes out unconditionally severed the type-6 DA path).
"""
import numpy as np
import pandas as pd

from troute.nhf_preprocess import _great_lakes_for_da, GREAT_LAKES_IDS

# Mirrors the NHF 1.2.0 lakes layer: every Great Lake has NaN level-pool params;
# three have an fp_id, 4800007 has none (and is absent from the DA crosswalk).
_GL_FP_IDS = {
    4800002: 1_278_348_000_000_000.0,
    4800004: 1_287_683_000_000_000.0,
    4800006: 1_286_166_000_000_000.0,
    4800007: np.nan,
}


def _gl_df():
    idx = pd.Index(list(_GL_FP_IDS), name="lake_id")
    return pd.DataFrame(
        {
            "fp_id": [_GL_FP_IDS[i] for i in idx],
            "LkArea": np.nan,                 # the defining trait: no level-pool params
            "LkMxE": [180.5, 351.2, 170.0, 73.5],
        },
        index=idx,
    )


def _da(enabled):
    return {
        "reservoir_da": {
            "reservoir_persistence_da": {
                "reservoir_persistence_greatLake": enabled,
            }
        }
    }


def test_da_enabled_keeps_anchored_great_lakes_at_original_ids():
    """With GL DA on, the fp_id-bearing Great Lakes are returned at their
    ORIGINAL ids (so the type-6 DA link works), fp_id cast to int."""
    anchored, gl_da_enabled = _great_lakes_for_da(_gl_df(), _da(True))

    assert gl_da_enabled is True
    # 4800007 has no fp_id and cannot be anchored; the other three survive.
    assert sorted(anchored.index) == [4800002, 4800004, 4800006]
    # Original ids preserved (NOT synthetic-renamed) -> climatology can match.
    assert set(anchored.index).issubset(set(GREAT_LAKES_IDS))
    # fp_id integral and exact (ids are < 2^53), dtype int for the link join.
    assert anchored["fp_id"].dtype.kind == "i"
    assert anchored.loc[4800002, "fp_id"] == 1_278_348_000_000_000


def test_da_disabled_excludes_all_great_lakes():
    """With GL DA off, no Great Lake re-enters the waterbody set (else the
    kernel crashes on their missing level-pool params)."""
    anchored, gl_da_enabled = _great_lakes_for_da(_gl_df(), _da(False))
    assert gl_da_enabled is False
    assert anchored.empty


def test_missing_da_config_defaults_to_excluded():
    """A config without the greatLake key must default to excluded (safe)."""
    for cfg in ({}, {"reservoir_da": {}}, {"reservoir_da": {"reservoir_persistence_da": {}}}):
        anchored, gl_da_enabled = _great_lakes_for_da(_gl_df(), cfg)
        assert gl_da_enabled is False
        assert anchored.empty


def test_empty_gl_df_is_handled():
    """No Great Lakes present -> empty result, flag still reflects config."""
    empty = _gl_df().iloc[0:0]
    anchored, gl_da_enabled = _great_lakes_for_da(empty, _da(True))
    assert gl_da_enabled is True
    assert anchored.empty


def test_all_great_lakes_unanchored_returns_empty_but_da_enabled():
    """If every Great Lake lacks an fp_id, none are anchored, but the DA flag
    still reports enabled (so the caller still loads climatology)."""
    gl = _gl_df().copy()
    gl["fp_id"] = np.nan
    anchored, gl_da_enabled = _great_lakes_for_da(gl, _da(True))
    assert gl_da_enabled is True
    assert anchored.empty


# nhf_1.2.2 moved the Great Lakes off fp_id: the two lakes carrying real USGS
# gages (4800002 -> 04127885, 4800004 -> 04159130) have a null fp_id and are
# anchored through virtual_fp_id instead. Verified against the CONUS 1.2.2 lakes
# layer. Anchoring on fp_id dropped exactly those two, which silently severed the
# type-6 DA path for the lakes the feature exists to serve.
_GL_VFP_IDS = {
    4800002: 1_278_347_000_000_000.0,   # null fp_id in 1.2.2
    4800004: 1_276_841_000_000_000.0,   # null fp_id in 1.2.2
    4800006: 1_286_155_000_000_000.0,
    4800007: 1_287_248_000_000_000.0,
}


def _gl_df_122():
    idx = pd.Index(list(_GL_VFP_IDS), name="lake_id")
    return pd.DataFrame(
        {
            "fp_id": [np.nan, np.nan, 1_286_193_000_000_000.0, 1_287_248_000_000_000.0],
            "virtual_fp_id": [_GL_VFP_IDS[i] for i in idx],
            "LkArea": np.nan,
            "LkMxE": [180.5, 351.2, 170.0, 73.5],
        },
        index=idx,
    )


def test_gage_bearing_great_lakes_survive_when_fp_id_is_null():
    """The 1.2.2 shape: all four anchor through virtual_fp_id, including the two
    with a null fp_id that the gage crosswalk depends on."""
    anchored, enabled = _great_lakes_for_da(_gl_df_122(), _da(True))
    assert enabled is True
    assert set(anchored.index) == set(GREAT_LAKES_IDS)
    # the two gage-bearing lakes must be present despite having no fp_id
    assert 4800002 in anchored.index and 4800004 in anchored.index
    assert anchored["virtual_fp_id"].dtype.kind == "i"


def test_da_disabled_still_excludes_all_great_lakes_on_122():
    anchored, enabled = _great_lakes_for_da(_gl_df_122(), _da(False))
    assert enabled is False
    assert anchored.empty


def test_lake_anchored_only_by_fp_id_is_kept():
    """The mirror of the 1.2.2 case: valid fp_id, null virtual_fp_id.

    Anchoring decisions are made per row, so a lake that carries only one of the two
    ids is still anchorable and must be retained. Choosing one column for the whole
    frame would drop it.
    """
    df = _gl_df_122()
    df.loc[4800006, "virtual_fp_id"] = np.nan  # keeps a valid fp_id
    anchored, _ = _great_lakes_for_da(df, _da(True))
    assert 4800006 in anchored.index
    assert set(anchored.index) == set(GREAT_LAKES_IDS)


def test_lake_with_neither_anchor_is_dropped():
    df = _gl_df_122()
    df.loc[4800007, ["fp_id", "virtual_fp_id"]] = np.nan
    anchored, _ = _great_lakes_for_da(df, _da(True))
    assert 4800007 not in anchored.index
    assert len(anchored) == len(GREAT_LAKES_IDS) - 1
