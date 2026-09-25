"""The completeness drop must gate on level-pool parameters, nothing else.

It used to be a bare ``dropna()``, so every column the frame happened to carry
became a drop criterion: a null native ``lake_id`` (what the hydrofabric gives a
dam with no NHD 2.1 match) removed the lake from the reservoir set, and enabling
``lakeout_output`` added crs/lat/lon to the same gate.
"""

from __future__ import annotations

import pandas as pd
import pytest

from troute.nhf_preprocess import (
    LAKE_ID_FIELD,
    LEVEL_POOL_PARAMS,
    NATIVE_LAKE_ID_FIELD,
    _clean_waterbodies,
)
from troute.routing.compute import WATERBODY_INITIAL_CONDITIONS, WATERBODY_VIEW_COLS


def _lakes(**overrides: object) -> pd.DataFrame:
    """One routable lake, every level-pool parameter present and consistent."""
    row: dict[str, object] = {
        LAKE_ID_FIELD: 1_000_001,
        NATIVE_LAKE_ID_FIELD: "nid-4242",
        "fp_id": 55,
        "virtual_fp_id": 77,
        "ifd": 0.9,
        "LkArea": 1.0,
        "LkMxE": 30.0,
        "OrificeA": 1.0,
        "OrificeC": 0.1,
        "OrificeE": 10.0,
        "WeirC": 0.4,
        "WeirE": 20.0,
        "WeirL": 5.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_a_lake_with_no_native_lake_id_is_still_routable() -> None:
    """A dam with no NHD 2.1 match carries a null ``lake_id`` and full parameters."""
    clean, gl = _clean_waterbodies(_lakes(**{NATIVE_LAKE_ID_FIELD: None}), LAKE_ID_FIELD)
    assert len(clean) == 1
    assert gl.empty


def test_lakeout_columns_do_not_gate_the_reservoir_set() -> None:
    """Whether lakeout is configured must not change which lakes route."""
    clean, _ = _clean_waterbodies(
        _lakes(crs=4326, lat=float("nan"), lon=float("nan")), LAKE_ID_FIELD
    )
    assert len(clean) == 1


@pytest.mark.parametrize("param", LEVEL_POOL_PARAMS)
def test_a_missing_level_pool_parameter_still_drops_the_lake(param: str) -> None:
    """The gate must keep working for the parameters the kernel actually reads."""
    clean, _ = _clean_waterbodies(_lakes(**{param: float("nan")}), LAKE_ID_FIELD)
    assert clean.empty


def test_the_gate_covers_every_hydrofabric_column_the_kernel_reads() -> None:
    """The two lists are maintained independently and must agree.

    Parametrizing the cases above over LEVEL_POOL_PARAMS cannot catch a parameter
    left out of it; only comparing against the kernel-facing view can. A column the
    kernel reads but the gate omits is the silent failure: it reaches the level-pool
    kernel as NaN and produces plausible wrong numbers rather than an error.
    """
    from_kernel = set(WATERBODY_VIEW_COLS) - set(WATERBODY_INITIAL_CONDITIONS)
    assert set(LEVEL_POOL_PARAMS) == from_kernel
    assert len(LEVEL_POOL_PARAMS) == len(from_kernel)
