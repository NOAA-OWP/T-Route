"""NHF geopackage validation must agree with what the build actually tolerates.

read_geo_file loads an ABSENT layer as an empty DataFrame ("used conditionally
downstream"), and the downstream code guards the two scaling-DA flowpath columns
individually. The up-front validator briefly disagreed with both tolerances: it
required every explicitly-listed layer to exist and every listed column to be
present, so a lake-free domain (no ``reservoir_da``) or a pre-1.1.4 hydrofabric
(no ``total_da_sqkm``/``vpu_id``) was rejected before routing started, for data
the run was never going to use.
"""

from __future__ import annotations

from troute.nhf_preprocess import (
    LAYERS_TO_READ,
    OPTIONAL_COLUMNS,
    OPTIONAL_LAYERS,
    REQUIRED_COLUMNS,
    _missing_requested_columns,
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


def test_scaling_only_flowpath_columns_are_optional():
    """A pre-1.1.4 hydrofabric lacks total_da_sqkm/vpu_id. With the scaling DA
    off nothing reads them; with it on, build_scaling_da_setup raises its own
    clear error about total_da_sqkm. Either way the file must pass ingest
    validation."""
    fields = _full_fields()
    fields["flowpaths"] -= OPTIONAL_COLUMNS["flowpaths"]
    assert _missing_requested_columns(fields) == {}


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
    """hy_id is the only key tying a gage to its hydrolocation. nhf 1.2.3 dropped
    it; unchecked that becomes a KeyError deep inside a pandas merge."""
    fields = _full_fields()
    fields["gages"].discard("hy_id")
    assert _missing_requested_columns(fields) == {"gages": ["hy_id"]}


def test_absent_gages_layer_is_still_tolerated():
    """Pinning a column must not make the LAYER mandatory -- read_geo_file loads an
    absent one as an empty frame and the build carries on."""
    fields = _full_fields()
    fields.pop("gages")
    assert _missing_requested_columns(fields) == {}
