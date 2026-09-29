"""Run-of-river dams route as MC channel, unless RFC assimilates them.

The choice is made once, at network build, from the hydrofabric alone, since
``ExecutionPlan`` is reused for every forcing window. Precedence: da_type 4, then
run_of_river, then a regular dam.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from troute.nhf_preprocess import (
    GREAT_LAKES_IDS,
    LAKE_ID_FIELD,
    RUN_OF_RIVER_FIELD,
    pass_through_flags,
    route_run_of_river_as_channel,
)

_LEVEL_POOL_PARAMS = {
    "ifd": 0.9, "LkArea": 1.0, "LkMxE": 10.0, "OrificeA": 1.0, "OrificeC": 0.1,
    "OrificeE": 1.0, "WeirC": 0.4, "WeirE": 5.0, "WeirL": 10.0,
}


def _lakes(ids: list[int], flags: list[bool], native: list[int] | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            LAKE_ID_FIELD: ids,
            "lake_id": native if native is not None else ids,
            "fp_id": [i * 10 for i in ids],
            "virtual_fp_id": [i * 100 for i in ids],
            RUN_OF_RIVER_FIELD: flags,
            **{k: [v] * len(ids) for k, v in _LEVEL_POOL_PARAMS.items()},
        }
    )


def _reservoir_da(ids: list[int], da_types: list[int]) -> pd.DataFrame:
    return pd.DataFrame({LAKE_ID_FIELD: ids, "da_type": da_types})


def _kept(lakes: pd.DataFrame) -> list[int]:
    return lakes[LAKE_ID_FIELD].tolist()


def test_a_flagged_dam_leaves_the_reservoir_set():
    lakes = _lakes([1, 2, 3], [False, True, False])
    out = route_run_of_river_as_channel(lakes, _reservoir_da([1, 2, 3], [1, 1, 1]))
    assert _kept(out) == [1, 3]


def test_rfc_takes_precedence_over_the_flag():
    """A flagged dam the hydrofabric also marks da_type 4 stays a reservoir, so its
    forecast is still assimilated."""
    lakes = _lakes([1, 2], [True, True])
    out = route_run_of_river_as_channel(lakes, _reservoir_da([1, 2], [4, 1]))
    assert _kept(out) == [1]


def test_precedence_cannot_depend_on_per_window_data():
    """ExecutionPlan is built once and reused, so a forecast or observation frame must
    never reach this decision. Asserted on behavior: the same hydrofabric gives the
    same answer with no other input available."""
    lakes = _lakes([1, 2], [True, True])
    da = _reservoir_da([1, 2], [4, 1])
    assert _kept(route_run_of_river_as_channel(lakes, da)) == [1]
    assert _kept(route_run_of_river_as_channel(lakes.copy(), da.copy())) == [1]


def test_an_unflagged_dam_is_untouched():
    lakes = _lakes([1, 2], [False, False])
    out = route_run_of_river_as_channel(lakes, _reservoir_da([1, 2], [1, 4]))
    assert _kept(out) == [1, 2]


def test_a_great_lake_is_never_routed_as_channel():
    """The Great Lakes carry no level-pool parameters and exist in the routable set
    only as DA targets; pulling one out here would sever that."""
    gl = GREAT_LAKES_IDS[0]
    lakes = _lakes([1, 999], [True, True], native=[1, gl])
    out = route_run_of_river_as_channel(lakes, _reservoir_da([1, 999], [1, 1]))
    assert _kept(out) == [999]


def test_every_dam_flagged_empties_the_reservoir_set():
    """preprocess_waterbodies branches on an empty lakes frame to disable reservoir
    DA, so the all-flagged domain has to reach it as empty."""
    lakes = _lakes([1, 2], [True, True])
    out = route_run_of_river_as_channel(lakes, _reservoir_da([1, 2], [1, 1]))
    assert out.empty


@pytest.mark.parametrize("missing", ["OrificeA", "LkArea"])
def test_a_flagged_headwater_is_forced_without_level_pool_parameters(missing: str):
    """A flagged low-head dam may carry no level-pool parameters, and gated on them its
    headwater flowpath is neither forced nor absorbed, so the reach vanishes."""
    from troute.NHF import _force_headwater_routing

    lakes = _lakes([1], [True])
    lakes.loc[:, missing] = np.nan
    vfps = pd.DataFrame(
        {"virtual_fp_id": [100], "up_virtual_nex_id": [np.nan], "dn_virtual_nex_id": [5]}
    )
    refs = pd.DataFrame({"virtual_fp_id": [100], "div_id": [1], "fp_id": [10]})
    out_vfps, _, _, _ = _force_headwater_routing(vfps, refs, lakes)
    assert out_vfps.loc[0, "up_virtual_nex_id"] > 5, (
        "flagged headwater was not forced, so its flowpath will be dropped"
    )


def test_the_crosswalk_is_not_an_input():
    """A flagged dam's crosswalk rows need no filtering: _lake_vfp_clusters keeps only
    rows resolving to a surviving lake, so the signature must not take one."""
    import inspect

    params = set(inspect.signature(route_run_of_river_as_channel).parameters)
    assert params == {"lakes", "reservoir_da"}


def test_the_driver_calls_the_filter_with_the_signature_it_has():
    """Every NHF(...) construction is integration-marked, so an arity mismatch between
    NHF.__init__ and this function reaches no unit test. Bind the real signature to the
    real call site's arguments."""
    import inspect

    from troute import NHF as nhf_module

    src = inspect.getsource(nhf_module.NHF.__init__)
    assert "route_run_of_river_as_channel(waterbodies, reservoir_da)" in src, (
        "NHF.__init__ does not call the filter the way this test pins"
    )
    inspect.signature(route_run_of_river_as_channel).bind(
        lakes=pd.DataFrame(), reservoir_da=pd.DataFrame()
    )


def test_short_reach_protection_is_computed_before_the_filter():
    """Discretization allocates in-between node ids as one contiguous arange in row
    order, so unprotecting a flagged dam's flowpath shifts every later link id and
    silently relabels any state keyed by them."""
    import inspect

    from troute import NHF as nhf_module

    src = inspect.getsource(nhf_module.NHF.__init__)
    assert src.index("wb_fp_ids = ") < src.index("route_run_of_river_as_channel("), (
        "wb_fp_ids must come from the unfiltered lakes"
    )


def _da_rows(ids: list[int], da_types: list[int], sites: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {LAKE_ID_FIELD: ids, "da_type": da_types, "site_no": sites}
    )


def test_an_rfc_reservoir_dropped_from_the_waterbody_set_is_named():
    """A lake with no virtual_fp_id, no level-pool parameters or inconsistent
    elevations leaves the routable set and routes as MC channel. Its gage still
    reaches the crosswalk, so the forecast is read against a lake that is not there."""
    from troute.nhf_preprocess import unrouted_rfc_gages

    da = _da_rows([1, 2, 3], [4, 4, 1], ["CCKK2", "CHET1", "POND"])
    lost = unrouted_rfc_gages(da, {1})
    assert lost[LAKE_ID_FIELD].tolist() == [2]
    assert lost["site_no"].tolist() == ["CHET1"]


def test_a_dropped_level_pool_is_not_named():
    """Losing a level pool is routine: nothing was going to be assimilated into it."""
    from troute.nhf_preprocess import unrouted_rfc_gages

    da = _da_rows([1, 2], [1, 1], ["A", "B"])
    assert unrouted_rfc_gages(da, set()).empty


def test_every_rfc_reservoir_routed_names_nothing():
    from troute.nhf_preprocess import unrouted_rfc_gages

    da = _da_rows([1, 2], [4, 4], ["CCKK2", "CHET1"])
    assert unrouted_rfc_gages(da, {1, 2}).empty


def test_the_flagged_rfc_dam_kept_is_the_one_marked_to_pass_its_inflow():
    """Lakes 1 flagged RFC, 2 flagged level pool, 3 unflagged RFC, 4 unflagged."""
    lakes = _lakes([1, 2, 3, 4], [True, True, False, False])
    kept = route_run_of_river_as_channel(lakes, _reservoir_da([1, 2, 3, 4], [4, 1, 4, 1]))
    flags = pass_through_flags(kept)
    assert dict(zip(_kept(kept), flags.tolist())) == {1: 1.0, 3: 0.0, 4: 0.0}


def test_lakes_without_the_flag_column_pass_nothing():
    lakes = _lakes([1], [False]).drop(columns="run_of_river")
    assert pass_through_flags(lakes) == 0.0
