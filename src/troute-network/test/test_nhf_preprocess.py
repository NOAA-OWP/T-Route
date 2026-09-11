"""Tests for ``troute.nhf_preprocess``.

Covers:
  * ``_validate_flowpaths_channel_params`` -- the load-time guard that
    rejects ``flowpaths`` containing non-finite (NaN/Inf) channel parameters
    which would otherwise propagate into NaN routing output via ``sqrt(s0)``
    in the Muskingum-Cunge kernel.
  * ``_validate_required_columns`` -- the load-time guard that rejects input
    geopackages missing columns the NHF build requires (e.g. ``segment_order``
    in ``reference_flowpaths``), replacing cryptic downstream ``KeyError``s.
"""

import numpy as np
import pandas as pd
import pytest

import troute.nhf_preprocess as nhf_preprocess
from troute.nhf_preprocess import (
    _BAD_FPID_PREVIEW_LIMIT,
    _FLOWPATHS_CHANNEL_COLS,
    LAYERS_TO_READ,
    _groupby_to_list_dict,
    _missing_requested_columns,
    _validate_flowpaths_channel_params,
    _validate_required_columns,
)


def _make_flowpaths(n=5, extra_cols=None):
    """Build a minimal valid flowpaths DataFrame with all channel cols finite."""
    data = {c: np.linspace(0.1, 1.0, n) for c in _FLOWPATHS_CHANNEL_COLS}
    data["fp_id"] = np.arange(1000, 1000 + n)
    if extra_cols:
        data.update(extra_cols)
    return pd.DataFrame(data)


# ----- no-op cases ----------------------------------------------------------

def test_validator_none_is_noop():
    # Must not raise on missing input.
    _validate_flowpaths_channel_params(None)


def test_validator_empty_df_is_noop():
    _validate_flowpaths_channel_params(pd.DataFrame())


def test_validator_no_channel_cols_is_noop():
    # If none of the kernel-critical columns are present, nothing to validate.
    df = pd.DataFrame({"fp_id": [1, 2, 3], "something_else": [10.0, 20.0, 30.0]})
    _validate_flowpaths_channel_params(df)


def test_validator_all_finite_passes():
    df = _make_flowpaths(n=10)
    _validate_flowpaths_channel_params(df)


# ----- raising cases --------------------------------------------------------

def test_validator_raises_on_nan_slope():
    df = _make_flowpaths(n=4)
    df.loc[1, "slope"] = np.nan
    with pytest.raises(ValueError) as excinfo:
        _validate_flowpaths_channel_params(df)
    msg = str(excinfo.value)
    assert "1 of 4 segments" in msg
    assert "slope" in msg
    assert "Muskingum-Cunge" in msg
    # The bad fp_id (1001) must appear in the preview.
    assert "1001" in msg


def test_validator_raises_on_positive_inf():
    df = _make_flowpaths(n=3)
    df.loc[2, "n"] = np.inf
    with pytest.raises(ValueError) as excinfo:
        _validate_flowpaths_channel_params(df)
    assert "'n': 1" in str(excinfo.value)


def test_validator_raises_on_negative_inf():
    df = _make_flowpaths(n=3)
    df.loc[0, "btmwdth"] = -np.inf
    with pytest.raises(ValueError) as excinfo:
        _validate_flowpaths_channel_params(df)
    assert "btmwdth" in str(excinfo.value)


def test_validator_reports_multiple_bad_columns():
    df = _make_flowpaths(n=4)
    df.loc[0, "slope"] = np.nan
    df.loc[1, "n"] = np.nan
    df.loc[2, "topwdth"] = np.inf
    with pytest.raises(ValueError) as excinfo:
        _validate_flowpaths_channel_params(df)
    msg = str(excinfo.value)
    # 3 distinct affected rows.
    assert "3 of 4 segments" in msg
    # Each bad column should be reported individually.
    assert "'slope': 1" in msg
    assert "'n': 1" in msg
    assert "'topwdth': 1" in msg


def test_validator_counts_one_row_with_multiple_bad_cols_as_one_segment():
    # A single row with NaNs in two columns counts as 1 segment (row-wise)
    # but contributes 1 to each affected-column count.
    df = _make_flowpaths(n=3)
    df.loc[1, "slope"] = np.nan
    df.loc[1, "n"] = np.nan
    with pytest.raises(ValueError) as excinfo:
        _validate_flowpaths_channel_params(df)
    msg = str(excinfo.value)
    assert "1 of 3 segments" in msg
    assert "'slope': 1" in msg
    assert "'n': 1" in msg


def test_validator_preview_capped_with_overflow_marker():
    # More than _BAD_FPID_PREVIEW_LIMIT bad fp_ids → preview is truncated
    # and the message includes an "(and N more)" marker.
    n = _BAD_FPID_PREVIEW_LIMIT + 5
    df = _make_flowpaths(n=n)
    df.loc[:, "slope"] = np.nan  # mark all rows bad
    with pytest.raises(ValueError) as excinfo:
        _validate_flowpaths_channel_params(df)
    msg = str(excinfo.value)
    assert f"and {n - _BAD_FPID_PREVIEW_LIMIT} more" in msg


def test_validator_handles_missing_fp_id_column():
    # If fp_id is absent (e.g. test fixture), validation should still raise
    # but without listing IDs.
    df = _make_flowpaths(n=3).drop(columns=["fp_id"])
    df.loc[0, "slope"] = np.nan
    with pytest.raises(ValueError) as excinfo:
        _validate_flowpaths_channel_params(df)
    msg = str(excinfo.value)
    assert "1 of 3 segments" in msg
    # Preview list should be empty (no fp_id column to draw from).
    assert "Affected fp_ids: []" in msg


def test_validator_ignores_non_channel_column_nans():
    # NaN in a column that is NOT a Muskingum-Cunge input should not trigger.
    df = _make_flowpaths(n=3, extra_cols={"unrelated_col": [1.0, np.nan, 3.0]})
    _validate_flowpaths_channel_params(df)


def test_validator_works_with_partial_channel_columns():
    # Real geopackages may not contain every entry in _FLOWPATHS_CHANNEL_COLS.
    # The validator should restrict to whichever subset is present.
    cols = ["fp_id", "slope", "n"]
    df = pd.DataFrame({
        "fp_id": [10, 20, 30],
        "slope": [0.1, np.nan, 0.3],
        "n": [0.03, 0.04, 0.05],
    })
    with pytest.raises(ValueError) as excinfo:
        _validate_flowpaths_channel_params(df)
    msg = str(excinfo.value)
    assert "'slope': 1" in msg
    assert "20" in msg


# ----- _validate_required_columns ------------------------------------------
# Guards that the input geopackage carries every column the NHF build needs.
# The reported production failure was a missing ``segment_order`` column in
# ``reference_flowpaths`` surfacing as ``KeyError: ['segment_order'] not in
# index`` deep inside discretization.


def _columns_for(name):
    """The explicit columns list a layer declares (or [] if columns=None)."""
    cols = next(c for n, c, _ in LAYERS_TO_READ if n == name)
    return list(cols) if isinstance(cols, list) else []


def _full_available():
    """{layer_name: set(columns)} where every validated layer carries all of
    its requested columns -- the metadata view _validate_required_columns
    derives from pyogrio.read_info on a complete hydrofabric."""
    return {name: set(cols) for name, cols, _ in LAYERS_TO_READ
            if isinstance(cols, list)}


def _validated_layer_names():
    return [name for name, cols, _ in LAYERS_TO_READ if isinstance(cols, list)]


# --- pure _missing_requested_columns logic ---------------------------------

def test_missing_requested_columns_complete_dataset_is_empty():
    assert _missing_requested_columns(_full_available()) == {}


def test_reference_flowpaths_spec_includes_segment_order():
    # Guard the spec itself: the layer that caused the production bug must
    # declare segment_order among its loaded columns.
    assert "segment_order" in _columns_for("reference_flowpaths")


def test_missing_requested_columns_missing_segment_order():
    avail = _full_available()
    avail["reference_flowpaths"].discard("segment_order")
    missing = _missing_requested_columns(avail)
    assert missing == {"reference_flowpaths": ["segment_order"]}


def test_missing_requested_columns_missing_flowpaths_col():
    avail = _full_available()
    avail["flowpaths"].discard("n")
    assert _missing_requested_columns(avail) == {"flowpaths": ["n"]}


def test_missing_requested_columns_aggregates_across_layers():
    avail = _full_available()
    avail["reference_flowpaths"].discard("segment_order")
    avail["virtual_flowpaths"].discard("dn_virtual_nex_id")
    missing = _missing_requested_columns(avail)
    assert missing["reference_flowpaths"] == ["segment_order"]
    assert missing["virtual_flowpaths"] == ["dn_virtual_nex_id"]


def test_missing_requested_columns_ignores_none_layers():
    # lakes/gages/hydrolocations/virtual_nexus declare columns=None and are
    # never validated, even if entirely absent from the available map.
    avail = _full_available()
    assert all(n not in avail for n in ("lakes", "gages", "hydrolocations",
                                        "virtual_nexus"))
    assert _missing_requested_columns(avail) == {}


def test_missing_requested_columns_absent_layer_reports_full_set():
    # A layer omitted from the available map (absent from the geopackage)
    # reports its entire requested set as missing.
    avail = _full_available()
    del avail["reference_flowpaths"]
    missing = _missing_requested_columns(avail)
    assert set(missing["reference_flowpaths"]) == set(
        _columns_for("reference_flowpaths"))


def test_missing_requested_columns_extra_columns_ignored():
    # Available columns beyond the requested set are fine.
    avail = _full_available()
    avail["reference_flowpaths"].add("some_extra_col")
    assert _missing_requested_columns(avail) == {}


# --- _validate_required_columns wrapper (read_info-driven, fails early) -----

def _patch_read_info(monkeypatch, fields_by_layer):
    """Make pyogrio.read_info return controlled field lists per layer (only
    called for layers in present_layers, so absent layers are never looked up)."""
    def fake_read_info(path, layer):
        return {"fields": list(fields_by_layer[layer])}
    monkeypatch.setattr(nhf_preprocess.pyogrio, "read_info", fake_read_info)


def test_validate_required_columns_passes_on_complete_metadata(monkeypatch):
    avail = _full_available()
    _patch_read_info(monkeypatch, avail)
    _validate_required_columns("dummy.gpkg", present_layers=set(avail))  # no raise


def test_validate_required_columns_raises_actionable_error(monkeypatch):
    avail = _full_available()
    avail["reference_flowpaths"].discard("segment_order")
    _patch_read_info(monkeypatch, avail)
    with pytest.raises(ValueError) as excinfo:
        _validate_required_columns("dummy.gpkg", present_layers=set(avail))
    msg = str(excinfo.value)
    assert "reference_flowpaths" in msg
    assert "segment_order" in msg
    assert "hydrofabric" in msg  # actionable about dataset versioning


def test_validate_required_columns_absent_layer_raises(monkeypatch):
    # reference_flowpaths absent from the geopackage (not in present_layers) ->
    # reported missing its full requested set; read_info is never called for it.
    avail = _full_available()
    _patch_read_info(monkeypatch, avail)
    present = set(avail) - {"reference_flowpaths"}
    with pytest.raises(ValueError) as excinfo:
        _validate_required_columns("dummy.gpkg", present_layers=present)
    assert "reference_flowpaths" in str(excinfo.value)


# ----- _groupby_to_list_dict (Step N2 helper) ------------------------------
# These tests exist to guarantee parity with the pandas idiom
#   df.groupby(key)[val].apply(list).to_dict()
# that this helper replaces. Anything pandas does (NaN-key handling, key
# unboxing, empty-frame behavior) the helper must match for the
# crosswalk_nex_flowpath_poi callers in nhf_preprocess to stay correct.


def _pandas_reference(df, key, val):
    """The legacy pandas idiom the helper replaces."""
    return df.groupby(key)[val].apply(list).to_dict()


def test_groupby_helper_matches_pandas_basic_int_keys():
    df = pd.DataFrame({"k": [10, 20, 10, 30, 20], "v": [1, 2, 3, 4, 5]})
    out = _groupby_to_list_dict(df, "k", "v")
    assert out == _pandas_reference(df, "k", "v")
    # And the keys are python ints, not numpy scalars.
    assert all(type(k) is int for k in out)


def test_groupby_helper_matches_pandas_with_nan_keys():
    # Pandas' default groupby drops NaN keys; the numpy helper must too.
    df = pd.DataFrame({
        "k": [10.0, np.nan, 10.0, 20.0, np.nan, 30.0],
        "v": [1, 2, 3, 4, 5, 6],
    })
    out = _groupby_to_list_dict(df, "k", "v")
    ref = _pandas_reference(df, "k", "v")
    assert out == ref
    # Explicitly: no NaN key in the output.
    assert not any(isinstance(k, float) and pd.isna(k) for k in out)


def test_groupby_helper_matches_pandas_all_nan_keys():
    df = pd.DataFrame({
        "k": [np.nan, np.nan, np.nan],
        "v": [1, 2, 3],
    })
    out = _groupby_to_list_dict(df, "k", "v")
    assert out == {}
    assert out == _pandas_reference(df, "k", "v")


def test_groupby_helper_matches_pandas_object_keys():
    df = pd.DataFrame({
        "k": ["a", "b", "a", "c"],
        "v": [1, 2, 3, 4],
    })
    out = _groupby_to_list_dict(df, "k", "v")
    assert out == _pandas_reference(df, "k", "v")


def test_groupby_helper_empty_frame():
    df = pd.DataFrame({"k": [], "v": []}, dtype=float)
    assert _groupby_to_list_dict(df, "k", "v") == {}
