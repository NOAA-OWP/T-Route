"""Tests for the wiring the scaling DA depends on but does not own.

Both contracts here were silently broken at some point and no test noticed, because
every existing scaling DA test exercises the kernel and the spread directly rather
than the path a real run takes.

  * The injected observations must land on links the execution plan actually split
    at. The plan takes its split points from ``network.gages``; if this module
    derives gage links by any other rule the override lands mid-reach and the
    innovation is measured against a segment the gage does not observe.
  * The frame the drivers hand to ``nwm_route`` must be the injected one. The BMI
    driver binds a window-sliced local before routing, so injecting into the
    ``DataAssimilation`` attribute after that point left the kernel routing the
    previous window's frame and the whole feature inert under ngen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from nwm_routing.scaling_da_apply import (
    ScalingDA,
    merge_injected_obs,
)

from troute.scaling_da.preprocess import resolve_gage_crosswalk


class _Network:
    """Minimal stand-in for the pieces resolve_gage_crosswalk reads."""

    def __init__(self, gages, fp_of):
        self.gages = {"gages": gages}
        self.dataframe = pd.DataFrame({"fp_id": pd.Series(fp_of, dtype="int64")})


def _bare(**attrs) -> ScalingDA:
    o = ScalingDA.__new__(ScalingDA)
    o.gage_fp = {}
    for k, v in attrs.items():
        setattr(o, k, v)
    return o


class TestCrosswalkMatchesPlanSplitPoints:
    """The DA must inject where the plan splits, which is network.gages."""

    def test_crosswalk_is_taken_from_the_network(self):
        # Link 202 is the gage's link per the network. A flowpath-outlet rule would
        # instead pick 203, the last link of fp 20, which is what used to happen and
        # what the plan never splits at.
        net = _Network(
            gages={202: "07374000"},
            fp_of={201: 20, 202: 20, 203: 20},
        )
        seg_of, _ = resolve_gage_crosswalk(net)
        assert seg_of == {"07374000": 202}

    def test_injected_links_are_a_subset_of_the_plan_split_set(self):
        """The property that actually matters, stated directly.

        The execution plan is handed ``set(network.gages['gages'])`` as its static
        split set. Every link the DA injects at has to be in it, or the override
        does not land on a reach boundary.
        """
        gages = {202: "07374000", 305: "07381490"}
        net = _Network(gages=gages, fp_of={201: 20, 202: 20, 203: 20, 305: 30})
        crosswalk, _ = resolve_gage_crosswalk(net)
        plan_split_set = set(gages)
        assert set(crosswalk.values()) <= plan_split_set

    def test_gage_fp_is_still_resolved_for_the_synthetic_baseline(self):
        """The frozen synthetic baseline is indexed by fp_id, not by routed link."""
        net = _Network(gages={202: "07374000"}, fp_of={201: 20, 202: 20})
        _, gage_fp = resolve_gage_crosswalk(net)
        assert gage_fp == {"07374000": 20}

    def test_no_network_gages_yields_no_crosswalk(self):
        assert resolve_gage_crosswalk(_Network(gages={}, fp_of={1: 1})) == ({}, {})


class TestMergeInjectedObs:
    """Injection overlays its own rows; it must not discard anyone else's."""

    @pytest.fixture
    def columns(self):
        return pd.date_range("2011-04-14", periods=3, freq="h")

    def test_other_rows_survive(self, columns):
        """The diversion gage's row has to still be there after injection.

        Replacing the frame wholesale left _resolve_diversion_da with nothing to
        find, so no water was transferred, and the mass balance check short
        circuits on the same lookup and reported nothing either.
        """
        injected = pd.DataFrame([[1.0, 2.0, 3.0]], index=pd.Index([100], name="link"),
                                columns=columns)
        existing = pd.DataFrame([[7.0, 8.0, 9.0]], index=pd.Index([999], name="link"),
                                columns=columns)
        out = merge_injected_obs(injected, existing)
        assert set(out.index) == {100, 999}
        np.testing.assert_allclose(out.loc[999].to_numpy(), [7.0, 8.0, 9.0])

    def test_injected_rows_win_on_conflict(self, columns):
        injected = pd.DataFrame([[1.0, 2.0, 3.0]], index=pd.Index([100], name="link"),
                                columns=columns)
        existing = pd.DataFrame([[9.0, 9.0, 9.0]], index=pd.Index([100], name="link"),
                                columns=columns)
        out = merge_injected_obs(injected, existing)
        assert len(out) == 1
        np.testing.assert_allclose(out.loc[100].to_numpy(), [1.0, 2.0, 3.0])

    def test_existing_rows_are_forced_onto_the_injected_column_grid(self, columns):
        """A column union would shift every gage's observations.

        The kernel reads this frame positionally as usgs_values[gage_i, timestep],
        so the merged frame must keep exactly the injected frame's columns.
        """
        injected = pd.DataFrame([[1.0, 2.0, 3.0]], index=pd.Index([100], name="link"),
                                columns=columns)
        # a different grid: offset by 30 minutes and one column longer
        other = pd.date_range("2011-04-14 00:30", periods=4, freq="h")
        existing = pd.DataFrame([[7.0, 8.0, 9.0, 10.0]],
                                index=pd.Index([999], name="link"), columns=other)
        out = merge_injected_obs(injected, existing)
        assert list(out.columns) == list(columns)
        np.testing.assert_allclose(out.loc[100].to_numpy(), [1.0, 2.0, 3.0])

    def test_a_row_emptied_by_the_reindex_is_reported(self, columns, caplog):
        """A foreign producer on a different grid loses every value in the reindex --
        for the diversion DA that reads as "no transfer", with no error anywhere. The
        merge cannot fix it (guessing an alignment would be worse), but it must say it
        happened."""
        import logging

        injected = pd.DataFrame([[1.0, 2.0, 3.0]], index=pd.Index([100], name="link"),
                                columns=columns)
        off_grid = pd.date_range("2011-04-14 00:30", periods=3, freq="h")
        existing = pd.DataFrame([[7.0, 8.0, 9.0]],
                                index=pd.Index([999], name="link"), columns=off_grid)
        with caplog.at_level(logging.WARNING):
            out = merge_injected_obs(injected, existing)
        assert out.loc[999].isna().all()
        assert "lost every value" in caplog.text

    def test_a_row_surviving_the_reindex_is_not_reported(self, columns, caplog):
        """Same grid: relabel only, no warning noise."""
        import logging

        injected = pd.DataFrame([[1.0, 2.0, 3.0]], index=pd.Index([100], name="link"),
                                columns=columns)
        existing = pd.DataFrame([[7.0, 8.0, 9.0]],
                                index=pd.Index([999], name="link"), columns=columns)
        with caplog.at_level(logging.WARNING):
            merge_injected_obs(injected, existing)
        assert "lost every value" not in caplog.text

    def test_empty_injection_preserves_the_existing_frame(self, columns):
        """When the scaling DA injects nothing, the diversion/nudging rows must
        survive. Returning the empty frame here silently disabled diversion DA on
        every loop of a run whose scaling DA had no usable gages."""
        existing = pd.DataFrame([[7.0, 8.0, 9.0]], index=pd.Index([999], name="link"),
                                columns=columns)
        out = merge_injected_obs(pd.DataFrame(), existing)
        pd.testing.assert_frame_equal(out, existing)

    def test_empty_existing_returns_the_injection(self, columns):
        injected = pd.DataFrame([[1.0, 2.0, 3.0]], index=pd.Index([100], name="link"),
                                columns=columns)
        pd.testing.assert_frame_equal(merge_injected_obs(injected, pd.DataFrame()), injected)
