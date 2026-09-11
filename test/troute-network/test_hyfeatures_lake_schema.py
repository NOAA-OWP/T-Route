"""The lakes/hydrolocations key columns move between hydrofabric vintages.

``read_geopkg`` keys off which columns are PRESENT rather than off a version number,
because the schema is not a monotonic progression. Verified against the domains in
``test/``:

===================  ==================================================  ==========
layer                v2.01                                               v2.2
===================  ==================================================  ==========
``lakes``            ``hl_link``, ``id``, ``hl_reference``; no            ``lake_id``
                     ``lake_id``                                         only
``hydrolocations``   no ``nex_id``                                       ``nex_id``
===================  ==================================================  ==========

So v2.01 is the vintage MISSING ``lake_id`` and ``nex_id``, and v2.2 reintroduced both.
Reading that backwards produces code that looks version-aware and silently mis-keys one
of them, which is what these cases pin down.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from troute.HYFeaturesNetwork import read_geopkg

REPO = Path(__file__).resolve().parents[2]

# (label, domain, does the lakes layer carry lake_id natively?)
DOMAINS = [
    ("v2.01", "test/LowerColorado_TX_v4/domain/LowerColorado_NGEN_v201.gpkg", False),
    ("v2.2", "test/LowerColorado_TX_HYFeatures_v22/domain/lower_colorado.gpkg", True),
    ("usbr", "test/USBR_persistence/domain/09112500.gpkg", True),
]

# read_geopkg does `.get("reservoir_persistence_da", False).get(...)`, so the nested
# dicts have to exist or it raises AttributeError on the bool.
COMPUTE = {
    "data_assimilation_parameters": {
        "streamflow_da": {},
        "reservoir_da": {"reservoir_persistence_da": {}, "reservoir_rfc_da": {}},
    }
}
WATERBODY = {"break_network_at_waterbodies": True}


@pytest.mark.parametrize(("label", "rel", "has_native_lake_id"), DOMAINS)
def test_lake_id_resolves_on_every_hydrofabric_vintage(
    label: str, rel: str, has_native_lake_id: bool
) -> None:
    path = REPO / rel
    if not path.exists():
        pytest.skip(f"{label} domain not present: {rel}")

    import geopandas as gpd

    raw = gpd.read_file(path, layer="lakes", rows=1)
    assert ("lake_id" in raw.columns) is has_native_lake_id, (
        f"{label} lakes layer changed shape; this test's premise needs rechecking"
    )

    _, lakes, _, _ = read_geopkg(str(path), COMPUTE, WATERBODY, 1)

    assert not lakes.empty, f"{label}: no lakes read"
    # Whichever column the vintage ships, the id is resolved and usable as an int key.
    assert "lake_id" in lakes.columns, f"{label}: lake_id not resolved"
    assert lakes["lake_id"].notna().all(), f"{label}: null lake_id"
    assert lakes["lake_id"].dtype.kind == "i", f"{label}: lake_id not integral"
    # Downstream lookups need "id"; on v2.01 it is native, on v2.2 it comes from the
    # hydrolocations merge. Either way it must survive without _x/_y suffixing.
    assert "id" in lakes.columns, f"{label}: id missing after the merge"
    dup = f"{label}: duplicate merge of an attribute the lakes layer already carried"
    assert "id_x" not in lakes.columns, dup
    assert "id_y" not in lakes.columns, dup


@pytest.mark.parametrize(("label", "rel", "_native"), DOMAINS)
def test_read_ngen_waterbody_df_handles_every_vintage(
    label: str, rel: str, _native: bool
) -> None:
    """Regression: the reader dropped the v2.01 column list unconditionally.

    ``df.drop(["id", "toid", "hl_id", "hl_reference", "hl_uri", "geometry"])`` plus a
    blind ``hl_link -> lake_id`` rename is exactly the v2.01 shape, so a v2.2 layer
    raised ``KeyError: "['id', 'toid', 'hl_id', 'hl_reference', 'hl_uri'] not found in
    axis"``. Both copies of this function (HYFeaturesNetwork and nhf_preprocess) are
    checked, because they had drifted into identical copies of the same bug.
    """
    path = REPO / rel
    if not path.exists():
        pytest.skip(f"{label} domain not present: {rel}")

    from troute.HYFeaturesNetwork import read_ngen_waterbody_df as hyfeatures_reader
    from troute.nhf_preprocess import read_ngen_waterbody_df as nhf_reader

    for reader in (hyfeatures_reader, nhf_reader):
        df = reader(str(path))
        assert not df.empty, f"{label}/{reader.__module__}: no lakes"
        assert df.index.name == "lake_id"
        assert df.index.dtype.kind == "i"
        # The non-parameter columns are dropped where present, never demanded.
        for dropped in ("id", "toid", "hl_id", "hl_reference", "hl_uri", "geometry"):
            assert dropped not in df.columns


@pytest.mark.parametrize(
    "compute_parameters",
    [
        pytest.param({}, id="no-da-block"),
        pytest.param({"data_assimilation_parameters": {}}, id="empty-da"),
        pytest.param({"data_assimilation_parameters": None}, id="null-da"),
        pytest.param(
            {"data_assimilation_parameters": {"reservoir_da": None}}, id="null-reservoir-da"
        ),
        pytest.param(
            {"data_assimilation_parameters": {"reservoir_da": {"reservoir_persistence_da": None}}},
            id="null-persistence-da",
        ),
    ],
)
def test_read_geopkg_survives_absent_or_null_da_blocks(
    compute_parameters: dict[str, object],
) -> None:
    """Regression: ``.get("reservoir_persistence_da", False).get(...)``.

    The middle level defaulted to ``False`` and was then called with ``.get``, so any
    config without that block died with "AttributeError: 'bool' object has no attribute
    'get'" before routing started. A present-but-null YAML key hits the same path,
    because ``.get(k, {})`` returns None when the key exists with a null value.
    """
    path = REPO / "test/LowerColorado_TX_v4/domain/LowerColorado_NGEN_v201.gpkg"
    if not path.exists():
        pytest.skip("v2.01 domain not present")

    flowpaths, _, _, _ = read_geopkg(
        str(path), compute_parameters, {"break_network_at_waterbodies": True}, 1
    )
    assert not flowpaths.empty
