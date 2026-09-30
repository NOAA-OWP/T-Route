"""Great Lakes climatology follows its lake, not its row number.

The kernel locates a lake in the parameter array and reads the climatology at that
same position (``mc_reach.pyx``: ``np.where(gl_param_idx == lake)`` then
``gl_climatology[pos]``). The two frames were built in independent orders, so
reordering the parameter rows handed a lake another lake's climatology, with nothing
raised and a plausible number produced.
"""
from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_HARNESS = Path(__file__).parent / "test_rfc_persist_kernel.py"
pytestmark = pytest.mark.skipif(not _HARNESS.is_file(), reason="kernel harness absent")

_CLIMATOLOGY = {30: 111.0, 50: 222.0}


def _outflows(param_order: list[int]) -> dict[int, float]:
    """Route two Great Lakes with the parameter rows in the given order."""
    spec = importlib.util.spec_from_file_location("_gl_harness", _HARNESS)
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)

    from troute.routing.compute import compute_nhd_routing_v02

    signature = inspect.signature(compute_nhd_routing_v02)
    harness._CONNECTIONS = {10: [30], 30: [50], 50: [40], 40: []}
    reaches, waterbodies, _, qlats, q0 = harness._base_frames(1)
    waterbodies.loc[50] = waterbodies.loc[30]
    types = pd.DataFrame({"reservoir_type": [6, 6]}, index=[30, 50])
    params = pd.DataFrame({
        "lake_id": param_order,
        "previous_assimilated_outflows": [np.nan, np.nan],
        "previous_assimilated_time": [0.0, 0.0],
        "update_time": [0.0, 0.0],
    })
    # No observation, so each lake falls back to its own climatology.
    observations = pd.DataFrame(
        {"lake_id": [30, 50], "Discharge": [np.nan, np.nan], "time": [0, 0]}
    )
    climatology = pd.DataFrame(
        [[_CLIMATOLOGY[30]] * 12, [_CLIMATOLOGY[50]] * 12], index=[30, 50]
    )

    def with_great_lakes(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.arguments.update(
            wbody_conn={30: [30], 50: [50]},
            great_lakes_df=observations,
            great_lakes_param_df=params,
            great_lakes_climatology_df=climatology,
        )
        return compute_nhd_routing_v02(*bound.args, **bound.kwargs)

    harness.compute_nhd_routing_v02 = with_great_lakes
    results, _ = harness._route(
        harness._T0, 12, reaches, waterbodies, types, qlats, q0,
        pd.DataFrame(), pd.DataFrame(), [None, None, None],
    )
    return {
        int(lake): float(np.asarray(r[1])[i, 0])
        for r in results for i, lake in enumerate(r[0]) if lake in _CLIMATOLOGY
    }


def test_each_lake_gets_its_own_climatology() -> None:
    assert _outflows([30, 50]) == pytest.approx(_CLIMATOLOGY)


def test_reordering_the_parameter_rows_changes_nothing() -> None:
    """The defect: this order handed lake 30 lake 50's climatology."""
    assert _outflows([50, 30]) == pytest.approx(_CLIMATOLOGY)
