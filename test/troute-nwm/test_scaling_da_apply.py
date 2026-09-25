"""Unit tests for the in-run ScalingDA adapter (the up_node_id-space glue).

Exercises the pure methods directly (built via __new__ so no network/gpkg/store
is needed): the run_results q de-interleave and the scatter-back. The in-kernel
source-trust screen is covered in test_scaling_da_in_kernel.py; the kernel/tree
math is covered in test/troute-network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from nwm_routing.scaling_da_apply import ScalingDA


def _bare(**attrs) -> ScalingDA:
    o = ScalingDA.__new__(ScalingDA)
    for k, v in attrs.items():
        setattr(o, k, v)
    return o


def test_assemble_q_model_deinterleaves_q_columns():
    """q lives at flat columns 0::4 (q,v,d,ql per timestep); rest are dropped."""
    o = _bare()
    ids = np.array([10, 20])
    # seg10: q=[1,3]; seg20: q=[5,7]; v/d/ql filled with sentinels.
    arr = np.array([[1, -1, -1, -1, 3, -1, -1, -1],
                    [5, -2, -2, -2, 7, -2, -2, -2]], dtype=np.float32)
    qm = o._assemble_q_model([(ids, arr)], nts=2, dt=300, t0="2000-01-01")
    assert qm.shape == (2, 2)
    assert list(qm.columns) == [10, 20]
    np.testing.assert_array_equal(qm[10].to_numpy(), [1, 3])
    np.testing.assert_array_equal(qm[20].to_numpy(), [5, 7])


def test_scatter_back_writes_q_and_carries_it_into_depth():
    o = _bare()
    ids = np.array([10, 20])
    arr = np.zeros((2, 8), dtype=np.float32)
    idx = pd.date_range("2000-01-01", periods=2, freq="h")
    q_corr = pd.DataFrame({10: [1.5, 3.5], 20: [5.5, 7.5]}, index=idx)
    o._scatter_back([(ids, arr)], q_corr)
    # q columns (0::4) updated, per [seg, nts]
    np.testing.assert_array_almost_equal(arr[:, 0::4], [[1.5, 3.5], [5.5, 7.5]])
    # velocity and lateral inflow are never written
    assert (arr[:, 1::4] == 0).all()
    assert (arr[:, 3::4] == 0).all()
    # depth is CARRIED with the discharge, but only where there is a depth to
    # scale and a background to scale it against. Here the background q and the
    # depth are both zero, so there is no ratio to take and depth holds.
    assert (arr[:, 2::4] == 0).all()


def test_scatter_back_respects_per_subnetwork_id_order():
    """corr is indexed by id; each subnetwork's rows must be scattered by its own ids."""
    o = _bare()
    idx = pd.date_range("2000-01-01", periods=1, freq="h")
    q_corr = pd.DataFrame({10: [1.0], 20: [2.0], 30: [3.0]}, index=idx)
    # two subnetworks, ids in a different order than q_corr columns
    a = np.zeros((2, 4), dtype=np.float32)
    b = np.zeros((1, 4), dtype=np.float32)
    o._scatter_back([(np.array([30, 10]), a), (np.array([20]), b)], q_corr)
    assert a[0, 0] == 3.0
    assert a[1, 0] == 1.0
    assert b[0, 0] == 2.0  # seg20


class _HoldoutNetwork:
    """The minimum a full ScalingDA.__init__ touches, for the holdout guard."""

    def __init__(self):
        self.dataframe = pd.DataFrame(
            {
                "total_da_sqkm": {100: 30.0, 101: 20.0, 200: 15.0},
                "fp_id": {100: 1, 101: 1, 200: 2},
                "dx": {100: 1000.0, 101: 1000.0, 200: 1000.0},
            }
        )
        self.reverse_network = {100: [101], 101: [], 200: []}
        self.gages = {"gages": {100: "01000000", 200: "02000000"}}
        self.link_lake_crosswalk = None
        self.waterbody_dataframe = pd.DataFrame()
        self.gage_vpu = {}


def _holdout_init(tmp_path, holdout_lines):

    hf = tmp_path / "holdout.txt"
    hf.write_text("".join(f"{ln}\n" for ln in holdout_lines))
    base = tmp_path / "baseline"
    base.mkdir(exist_ok=True)
    return ScalingDA(
        _HoldoutNetwork(),
        {
            # synthetic path: no TimeSlice store needed for the guard under test
            "synthetic_obs_factor": 1.5,
            "synthetic_obs_baseline": str(base),
            "holdout_sites_file": str(hf),
        },
    )


def test_holdout_unknown_id_is_a_hard_error(tmp_path):
    """A holdout id matching nothing withholds nothing, while the log would claim it
    was held out -- so a scoring run would assimilate the very gages it says it
    excluded. Reject at construction, same as the missing-file case above it."""
    import pytest

    with pytest.raises(ValueError, match="match no gage"):
        _holdout_init(tmp_path, ["01000000", "99999999"])


def test_holdout_known_ids_are_withheld(tmp_path):
    da = _holdout_init(tmp_path, ["01000000"])
    assert "01000000" not in da._da_sites
    assert "02000000" in da._da_sites


class _LakeGageNetwork(_HoldoutNetwork):
    """Gage 200's own segment is a lake id (reservoir-routed)."""

    def __init__(self):
        super().__init__()
        self.waterbody_dataframe = pd.DataFrame(index=pd.Index([200]))


def test_gage_on_a_reservoir_segment_is_excluded_loudly(tmp_path, caplog):
    """A gage whose crosswalked segment the kernel routes as a reservoir cannot
    be scale-corrected (reservoir DA owns that segment). The exclusion is the
    production contract -- build_trees alone would happily keep it -- and it
    must be LOUD, because the run otherwise completes normally while silently
    assimilating nothing there."""
    import logging

    base = tmp_path / "baseline"
    base.mkdir(exist_ok=True)
    with caplog.at_level(logging.WARNING):
        da = ScalingDA(
            _LakeGageNetwork(),
            {"synthetic_obs_factor": 1.5, "synthetic_obs_baseline": str(base)},
        )
    assert "02000000" not in da._da_sites
    assert "01000000" in da._da_sites
    assert any(
        "reservoir-routed" in r.message and "02000000" in r.message
        for r in caplog.records
    )


class _NanAreaNetwork(_HoldoutNetwork):
    """One gage's upstream subtree carries a NaN drainage area, so its TREE is
    dropped -- but the gage itself is fine and must keep its downstream correction."""

    def __init__(self):
        super().__init__()
        # 200 <- 201 with NaN area: tree for 02000000 is dropped (NaN would propagate
        # into the corrected discharge). 100 <- 101 stays intact.
        self.dataframe = pd.DataFrame(
            {
                "total_da_sqkm": {100: 30.0, 101: 20.0, 200: 15.0, 201: float("nan")},
                "fp_id": {100: 1, 101: 1, 200: 2, 201: 2},
                "dx": {100: 1000.0, 101: 1000.0, 200: 1000.0, 201: 1000.0},
            }
        )
        self.reverse_network = {100: [101], 101: [], 200: [201], 201: []}


def test_dropped_tree_keeps_the_downstream_correction(tmp_path):
    """A tree dropped for spread-only reasons (NaN area in the subtree) must not
    remove the gage from the injection set: the at-gage insertion plus downstream
    routing needs no drainage area, and is exactly what legacy nudging would deliver.
    Only the upstream spread (which iterates self.trees) skips it."""
    base = tmp_path / "baseline"
    base.mkdir()
    da = ScalingDA(
        _NanAreaNetwork(),
        {"synthetic_obs_factor": 1.5, "synthetic_obs_baseline": str(base)},
    )
    assert "02000000" not in da.trees, "the NaN-area tree itself must still be dropped"
    assert "02000000" in da._da_sites, "the gage must stay in the injection set"
    assert da.gage_seg["02000000"] == 200
    assert "01000000" in da.trees
    assert "01000000" in da._da_sites


def test_prebuilt_network_setup_is_consumed(tmp_path):
    """NHF builds the static setup during network construction (NHF.__init__ ->
    build_scaling_da_setup); ScalingDA must consume that bundle, not rebuild it.
    The stub network here has no reverse_network and no total_da_sqkm, so a
    rebuild attempt would raise -- consuming the prebuilt bundle is the only way
    this constructs."""
    from troute.scaling_da import ScalingDASetup, build_gage_trees_from_mappings

    trees = build_gage_trees_from_mappings(
        {100: [101], 101: []}, {"01000000": 100}, {100: 30.0, 101: 20.0}
    )

    class _PrebuiltNetwork:
        # dx only: the default travel-time taper needs reach lengths, while a
        # rebuild would still raise on the missing total_da_sqkm.
        dataframe = pd.DataFrame({"dx": {100: 1000.0}})
        scaling_da_setup = ScalingDASetup(
            trees=trees,
            gage_seg={"01000000": 100},
            gage_fp={"01000000": 1},
            all_gage_seg={"01000000": 100},
            da_sites=["01000000"],
        )

    base = tmp_path / "baseline"
    base.mkdir()
    da = ScalingDA(
        _PrebuiltNetwork(),
        {"synthetic_obs_factor": 1.5, "synthetic_obs_baseline": str(base)},
    )
    assert da.gage_seg == {"01000000": 100}
    assert da._da_sites == ["01000000"]
    assert set(da.trees) == {"01000000"}


def test_remap_courant_keeps_the_x_diagnostic():
    """The courant frame's MultiIndex level values are ("cn", "ck", "X"); the NHF
    remap briefly filtered for lowercase "x", selected nothing, and the final
    reindex rebuilt every X column as all-NaN -- Courant output looked complete
    but carried no X anywhere on NHF."""
    from nwm_routing.output import remap_courant

    # Exactly the writer's construction: a FLAT index of (timestep, var) tuples
    # (output.py builds MultiIndex.from_product(...).to_flat_index(); a true
    # MultiIndex cannot survive the remap's explode/join).
    cols = pd.MultiIndex.from_product([range(2), ["cn", "ck", "X"]]).to_flat_index()
    courant = pd.DataFrame(
        [[0.5, 1.2, 0.3, 0.6, 1.3, 0.4]], index=pd.Index([101], name="link"), columns=cols
    )
    out = remap_courant(courant, {101: [7]})
    assert not out.isna().any().any()
    # .at, not .loc: the columns are TUPLE LABELS on a flat index, and .loc reads a
    # tuple key as MultiIndex coordinates.
    assert out.at[7, (0, "X")] == 0.3  # noqa: PD008
    assert out.at[7, (1, "X")] == 0.4  # noqa: PD008


def test_align_eloss_columns_matches_by_label_not_position():
    """ET loss must land at its own timestamps. The ET dataset is not sliced per
    forcing window, so a full-run loss frame read positionally applied the run's
    FIRST hours' ET in every window; and a loss series starting later than the
    window left-packed every value early. Both completed with plausible wrong
    discharge."""
    from troute.routing.compute import _align_eloss_columns

    win = pd.Index(["202105010200", "202105010300"])
    # full-run frame: window columns exist, at positions 2..3 -- positional reads
    # would have taken the first two columns instead.
    full_run = pd.DataFrame(
        [[9.0, 9.5, 1.0, 2.0]],
        index=[101],
        columns=pd.Index(["202105010000", "202105010100", *win]),
    )
    out = _align_eloss_columns(full_run, win)
    assert list(out.columns) == list(win)
    assert out.loc[101].tolist() == [1.0, 2.0]

    # loss series starting mid-window: the absent leading stamp is zero loss.
    late = pd.DataFrame([[2.0]], index=[101], columns=pd.Index([win[1]]))
    out = _align_eloss_columns(late, win)
    assert out.loc[101].tolist() == [0.0, 2.0]

    # disjoint labeling with real loss configured: refused, never silently zeroed.
    import pytest

    alien = pd.DataFrame([[3.0]], index=[101], columns=pd.Index(["2021-05-01 02:00"]))
    with pytest.raises(ValueError, match="share no label"):
        _align_eloss_columns(alien, win)
