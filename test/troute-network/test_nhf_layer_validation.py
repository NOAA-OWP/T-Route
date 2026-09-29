"""NHF geopackage validation must agree with what the build actually tolerates.

read_geo_file loads an ABSENT optional layer as an empty DataFrame, so a lake-free
domain (no ``lakes``, ``reservoir_da`` or ``lake_vfp_crosswalk``) must validate. A
PRESENT layer must carry every column the build requests from it.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from troute.nhf_preprocess import (
    LAYERS_TO_READ,
    LEVEL_POOL_PARAMS,
    OPTIONAL_LAYERS,
    REQUIRED_COLUMNS,
    RUN_OF_RIVER_FIELD,
    WATERBODY_DF_FIELDS,
    _missing_requested_columns,
    _normalize_run_of_river,
    read_geo_file,
)


def _full_fields() -> dict[str, set[str]]:
    """Every validated layer present, carrying exactly its requested columns.

    Full-load layers name no column list, so REQUIRED_COLUMNS is all they pin.
    """
    fields = {
        name: set(columns)
        for name, columns, _ in LAYERS_TO_READ
        if columns is not None
    }
    for name, required in REQUIRED_COLUMNS.items():
        fields.setdefault(name, set()).update(required)
    return fields


def test_complete_geopackage_validates_clean():
    assert _missing_requested_columns(_full_fields()) == {}


def test_absent_optional_layers_are_not_missing():
    """A lake-free domain has no reservoir_da/lakes layer; the build loads them as
    empty frames, so validation must not reject the file for lacking them."""
    fields = _full_fields()
    for layer in OPTIONAL_LAYERS:
        fields.pop(layer, None)
    assert _missing_requested_columns(fields) == {}


def test_absent_core_layer_is_still_an_error():
    fields = _full_fields()
    fields.pop("reference_flowpaths")
    missing = _missing_requested_columns(fields)
    assert "reference_flowpaths" in missing
    assert "segment_order" in missing["reference_flowpaths"]


def test_required_flowpath_columns_still_enforced():
    fields = _full_fields()
    fields["flowpaths"].discard("n")
    missing = _missing_requested_columns(fields)
    assert missing == {"flowpaths": ["n"]}


def test_present_optional_layer_is_column_checked():
    """Optional means the LAYER may be absent; a present one must still be sound."""
    fields = _full_fields()
    fields["reservoir_da"].discard("site_no")
    missing = _missing_requested_columns(fields)
    assert missing == {"reservoir_da": ["site_no"]}


def test_gages_missing_hy_id_is_rejected():
    """hy_id is the only key tying a gage to its hydrolocation; unchecked, a gages
    layer without it becomes a KeyError deep inside a pandas merge."""
    fields = _full_fields()
    fields["gages"].discard("hy_id")
    assert _missing_requested_columns(fields) == {"gages": ["hy_id"]}


def test_absent_gages_layer_is_still_tolerated():
    """Pinning a column must not make the LAYER mandatory -- read_geo_file loads an
    absent one as an empty frame and the build carries on."""
    fields = _full_fields()
    fields.pop("gages")
    assert _missing_requested_columns(fields) == {}


def test_a_lake_free_geopackage_reaches_the_headwater_forcing(tmp_path: Path):
    """Validation accepts a domain without lakes, reservoir_da or lake_vfp_crosswalk,
    and NHF.__init__ hands the lakes frame straight to _force_headwater_routing, which
    indexes its columns."""
    from troute.NHF import _force_headwater_routing

    path = tmp_path / "nhf.gpkg"
    for name, columns, _ in LAYERS_TO_READ:
        if columns is not None and name not in OPTIONAL_LAYERS:
            gpd.GeoDataFrame(
                {c: [1.0] for c in columns}, geometry=[Point(0.0, 0.0)], crs=4326
            ).to_file(path, layer=name, driver="GPKG")
    tables = read_geo_file({"geo_file_path": str(path)}, 1)

    assert tables["lakes"].empty
    assert RUN_OF_RIVER_FIELD in tables["lakes"].columns
    _force_headwater_routing(
        tables["virtual_flowpaths"], tables["reference_flowpaths"], tables["lakes"]
    )


# --------------------------------------------------- run-of-river tag


def test_run_of_river_is_not_a_level_pool_parameter():
    """Both completeness gates name LEVEL_POOL_PARAMS, the columns the kernel reads.
    A tag decides nothing about whether a lake can be routed."""
    assert RUN_OF_RIVER_FIELD not in LEVEL_POOL_PARAMS
    assert RUN_OF_RIVER_FIELD not in WATERBODY_DF_FIELDS


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Shapes a GeoPackage reader returns for a nullable INTEGER or BOOLEAN column,
        # depending on the reader: float64 with NaN, or the strings "True"/"False" and
        # None.
        (pd.Series([1.0, 0.0, float("nan")]), [True, False, False]),
        (pd.Series(["True", "False", None], dtype=object), [True, False, False]),
        # Shapes a caller can still hand over directly.
        (pd.Series([0, 1]), [False, True]),
        (pd.Series([None, 1], dtype=object), [False, True]),
        (pd.Series([True, False]), [True, False]),
        (pd.Series([True, pd.NA], dtype="boolean"), [True, False]),
        (pd.Series([1, pd.NA], dtype="Int64"), [True, False]),
        (pd.Series([float("nan"), float("nan")]), [False, False]),
    ],
)
def test_run_of_river_parses_to_bool(raw: pd.Series[object], expected: list[bool]):
    normalized = _normalize_run_of_river(pd.DataFrame({RUN_OF_RIVER_FIELD: raw}))
    assert normalized[RUN_OF_RIVER_FIELD].dtype == bool
    assert normalized[RUN_OF_RIVER_FIELD].tolist() == expected


@pytest.mark.parametrize("raw", [["yes", "no"], [0, 2]])
def test_unusable_run_of_river_values_are_rejected(raw: list[object]):
    """Guessing routes a real reservoir as channel, or leaves a low-head dam
    impounding water that is not there."""
    with pytest.raises(ValueError, match="must hold 0, 1 or NULL"):
        _normalize_run_of_river(pd.DataFrame({RUN_OF_RIVER_FIELD: raw}))
