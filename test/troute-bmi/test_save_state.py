"""Regression test for NGWPC-11208: subnetwork must be excluded from save state.

The subnetwork refactor changed ``Model._subnetwork`` from a serializable
list-of-dicts into an ``ExecutionPlan`` object. The old ``create_state`` ran
``list(self._subnetwork)``, which raises ``TypeError`` on an ExecutionPlan (no
``__iter__``). PR #112 drops the plan from save state entirely -- it is a
recomputable topology decomposition, rebuilt from the ``[None, None, None]``
sentinel on the first run after a load.

These tests drive ``create_state``/``load_state`` in isolation via
``Model.__new__`` so they need neither a config file nor a built network.
"""
from __future__ import annotations

import pickle

import pandas as pd
import pytest
from troute_nwm_bmi.troute_model import Model


class _ExecutionPlanLike:
    """Mimics ExecutionPlan: a plain object with no __iter__/__getitem__."""


class _NetworkStub:
    def __init__(self):
        self._q0 = pd.DataFrame({"q": [1.0, 2.0]})
        self._t0 = "2020-01-01_00:00:00"
        self.waterbody_updated = False

    def update_waterbody_water_elevation(self):
        self.waterbody_updated = True


class _DataAssimilationStub:
    def __init__(self):
        self._last_obs_df = pd.DataFrame({"discharge": [5.0]})
        self._reservoir_usgs_param_df = pd.DataFrame({"a": [1]})
        self._reservoir_usace_param_df = pd.DataFrame({"b": [2]})
        self._reservoir_usbr_param_df = pd.DataFrame({"e": [7]})
        self._reservoir_rfc_param_df = pd.DataFrame({"c": [3]})
        self._great_lakes_param_df = pd.DataFrame({"d": [4]})


class _ScalingDAStub:
    """Presence marker for load_state's branch; records the trace restore."""

    def __init__(self):
        self.restored_with = "never called"

    def trace_checkpoint(self):
        return None

    def restore_trace_checkpoint(self, ckpt, dt):
        self.restored_with = (ckpt, dt)


def _make_model(time, subnetwork):
    model = Model.__new__(Model)  # skip __init__ (no config/network build)
    model._time = time
    model._network = _NetworkStub()
    model._data_assimilation = _DataAssimilationStub()
    model._subnetwork = subnetwork
    return model


def test_execution_plan_is_not_iterable():
    """Anchors the root cause: the old list(self._subnetwork) call crashed."""
    with pytest.raises(TypeError):
        list(_ExecutionPlanLike())


def test_create_state_excludes_subnetwork_and_does_not_crash():
    # _subnetwork is a non-iterable plan, exactly the post-run reality that
    # broke the old code. create_state must ignore it.
    model = _make_model(3600.0, _ExecutionPlanLike())
    state = model.create_state()
    assert "subnetwork" not in state
    # must be picklable (this is what the BMI serialize path does)
    pickle.loads(pickle.dumps(state, pickle.HIGHEST_PROTOCOL))


def test_save_load_roundtrip_preserves_state_and_leaves_subnetwork_alone():
    src = _make_model(3600.0, _ExecutionPlanLike())
    # Mutate every saved frame away from the stub's defaults BEFORE saving. Without
    # this the destination stub is constructed with identical values, so the
    # assertions below pass whether or not load_state actually restored anything.
    da = src._data_assimilation
    da._last_obs_df = pd.DataFrame({"discharge": [99.0]})
    da._reservoir_usgs_param_df = pd.DataFrame({"a": [11]})
    da._reservoir_usace_param_df = pd.DataFrame({"b": [22]})
    da._reservoir_usbr_param_df = pd.DataFrame({"e": [77]})
    da._reservoir_rfc_param_df = pd.DataFrame({"c": [33]})
    da._great_lakes_param_df = pd.DataFrame({"d": [44]})
    src._network._q0 = pd.DataFrame({"q": [9.0, 8.0]})
    state = pickle.loads(pickle.dumps(src.create_state(), pickle.HIGHEST_PROTOCOL))

    # fresh model with the sentinel __init__ leaves behind
    dst = _make_model(0.0, [None, None, None])
    dst.load_state(state)

    assert dst._time == 3600.0
    pd.testing.assert_frame_equal(dst._network._q0, src._network._q0)
    assert dst._network._t0 == src._network._t0
    pd.testing.assert_frame_equal(
        dst._data_assimilation._last_obs_df, src._data_assimilation._last_obs_df
    )
    pd.testing.assert_frame_equal(
        dst._data_assimilation._great_lakes_param_df,
        src._data_assimilation._great_lakes_param_df,
    )
    # USBR persistence state must survive too, or a restarted run diverges from an
    # uninterrupted one at type-7 reservoirs.
    pd.testing.assert_frame_equal(
        dst._data_assimilation._reservoir_usbr_param_df,
        src._data_assimilation._reservoir_usbr_param_df,
    )
    # load_state applies waterbody elevation from restored q0...
    assert dst._network.waterbody_updated is True
    # ...and must NOT touch _subnetwork, so the next run rebuilds from the sentinel
    assert dst._subnetwork == [None, None, None]


def _spread_then_q0(spread: bool = True):
    """Run the DA spread (or not), then the REAL new_q0, and return the network.

    Mirrors the BMI loop's state-seeding ordering (spread, then snapshot). WHERE the
    spread is called relative to new_q0 lives in Model.run and is not reachable from
    here; what this exercises is the chain that carries a correction into state.
    """
    import numpy as np
    from nwm_routing.scaling_da_apply import ScalingDA

    from troute.AbstractNetwork import AbstractNetwork
    from troute.scaling_da import build_gage_trees_from_mappings

    trees = build_gage_trees_from_mappings(
        {100: [101], 101: [102], 102: []}, {"G": 100}, {100: 30.0, 101: 20.0, 102: 10.0}
    )
    da = ScalingDA.__new__(ScalingDA)
    da.trees, da.gage_seg = trees, {"G": 100}
    da.min_flow, da.max_source_pbias, da._loop_obs = 1e-6, None, None
    # These run_results carry no Courant block; with the class-default lag ON
    # that is a fail-closed error, and the lag is not what this file tests.
    da.travel_time_lag = False

    nts = 2
    arr = np.zeros((3, 4 * nts), dtype=np.float32)
    arr[:, 0::4] = np.tile(np.array([12.0, 6.0, 4.0])[:, None], (1, nts))
    nudge = np.zeros((1, nts + 1), dtype=np.float32)
    nudge[0, 1:] = 2.0
    run_results = [
        [np.array([100, 101, 102]), arr, 0, (np.array([100]), np.zeros(1), np.zeros(1)),
         0, 0, 0, 0, np.zeros((3, nts), dtype=np.float32), nudge]
    ]

    if spread:
        da.apply_in_kernel(run_results, nts=nts, dt=3600, t0="2000-01-01")
    network = _NetworkStub()
    # real slicing: r[1][:, [-4, -4, -2, -1]] -> qu0/qd0/h0/ql0
    AbstractNetwork.new_q0(network, run_results)
    return network


def test_upstream_correction_reaches_bmi_save_state():
    """The forecast-restart chain: spread -> new_q0 -> recorded -> create_state -> pickle.

    The state dict carries BOTH warmstates: "q0" is the cycling background (never
    seeded), "seeded_q0" the analysis hand-off with the upstream correction. A
    restarted forecast inherits the seeded one; a resumed analysis the cycling one.
    Which is which is load_state's decision -- create_state must faithfully persist
    both without mixing them.
    """
    expected_101 = 6.0 + 2.0 * (20.0 / 30.0) ** 0.77

    model = _make_model(3600.0, [None, None, None])
    background = _spread_then_q0(spread=False)
    model._network = background
    # Model.run records the corrected warmstate aside and restores the cycling one;
    # reproduce that end state here.
    model._seeded_q0 = _spread_then_q0(spread=True)._q0
    restored = pickle.loads(pickle.dumps(model.create_state(), pickle.HIGHEST_PROTOCOL))
    assert restored["seeded_q0"].loc[101, "qd0"] == pytest.approx(expected_101, rel=1e-5)
    # ...and the cycling background stays uncorrected, or a resumed analysis would
    # have its next innovation debited by the correction we injected ourselves.
    assert restored["q0"].loc[101, "qd0"] == pytest.approx(6.0, rel=1e-6)

    # no spread at all: nothing recorded, nothing seeded.
    off = _make_model(3600.0, [None, None, None])
    off._network = _spread_then_q0(spread=False)
    restored_off = pickle.loads(pickle.dumps(off.create_state(), pickle.HIGHEST_PROTOCOL))
    assert restored_off["q0"].loc[101, "qd0"] == pytest.approx(6.0, rel=1e-6)
    assert restored_off["seeded_q0"] is None


def test_load_state_installs_the_warmstate_the_run_will_need():
    """A forecast inherits the seeded state; a resumed analysis the cycling one.

    Persisting only the seeded state (the previous behavior) made checkpoint/resume
    diverge silently from an uninterrupted run: the resumed analysis started from a
    background that already contained the correction, so its next window's innovation
    was debited by the amount injected -- the exact feedback the cycling/seeded split
    exists to prevent.
    """
    cycling = pd.DataFrame({"qd0": [6.0]}, index=[101])
    seeded = pd.DataFrame({"qd0": [8.0]}, index=[101])
    state = {
        "time": 0.0, "q0": cycling, "seeded_q0": seeded, "t0": None,
        "last_obs": pd.DataFrame(), "usgs": pd.DataFrame(), "usace": pd.DataFrame(),
        "usbr": pd.DataFrame(), "rfc": pd.DataFrame(), "gl": pd.DataFrame(),
    }

    # Forecast: no scaling DA on the loading model -> seeded state installed.
    forecast = _make_model(0.0, [None, None, None])
    forecast.load_state(pickle.loads(pickle.dumps(state, pickle.HIGHEST_PROTOCOL)))
    assert forecast._network._q0.loc[101, "qd0"] == 8.0

    # Resumed analysis: scaling DA active -> cycling state installed.
    analysis = _make_model(0.0, [None, None, None])
    analysis._scaling_da = _ScalingDAStub()
    analysis.dt = 300
    analysis.load_state(pickle.loads(pickle.dumps(state, pickle.HIGHEST_PROTOCOL)))
    assert analysis._network._q0.loc[101, "qd0"] == 6.0

    # Pre-split state file (no seeded_q0 key): q0 as written, old behavior.
    legacy = {k: v for k, v in state.items() if k != "seeded_q0"}
    old = _make_model(0.0, [None, None, None])
    old.load_state(pickle.loads(pickle.dumps(legacy, pickle.HIGHEST_PROTOCOL)))
    assert old._network._q0.loc[101, "qd0"] == 6.0
    assert old._seeded_q0 is None

    # The loaded hand-off is RETAINED (superseded only by the next routed window),
    # so a checkpoint survives load -> create round-trips; see the test below.
    assert forecast._seeded_q0 is not None
    assert analysis._seeded_q0 is not None


def test_checkpoint_round_trip_preserves_the_forecast_handoff():
    """load_state -> create_state with no routing in between must reproduce the
    state, seeded_q0 included. Clearing the hand-off at load (the previous
    behavior) meant an orchestrator that re-serializes an analysis checkpoint --
    re-sharding, copying between stores -- silently wrote seeded_q0=None, and a
    forecast later branched from that COPY inherited the uncorrected cycling
    background instead of the DA hand-off. Same inputs, no error, wrong forecast.
    """
    cycling = pd.DataFrame({"qd0": [6.0]}, index=[101])
    seeded = pd.DataFrame({"qd0": [8.0]}, index=[101])
    state = {
        "time": 0.0, "q0": cycling, "seeded_q0": seeded, "t0": None,
        "last_obs": pd.DataFrame(), "usgs": pd.DataFrame(), "usace": pd.DataFrame(),
        "usbr": pd.DataFrame(), "rfc": pd.DataFrame(), "gl": pd.DataFrame(),
    }
    # The pass-through actor is an ANALYSIS model (scaling DA active): it installs
    # the cycling q0 for itself, but what it re-serializes must still carry the
    # hand-off for whoever loads the copy next.
    relay = _make_model(0.0, [None, None, None])
    relay._scaling_da = _ScalingDAStub()
    relay.dt = 300
    relay.load_state(pickle.loads(pickle.dumps(state, pickle.HIGHEST_PROTOCOL)))
    resaved = pickle.loads(pickle.dumps(relay.create_state(), pickle.HIGHEST_PROTOCOL))
    assert resaved["seeded_q0"] is not None
    assert resaved["seeded_q0"].loc[101, "qd0"] == 8.0
    assert resaved["q0"].loc[101, "qd0"] == 6.0

    forecast = _make_model(0.0, [None, None, None])
    forecast.load_state(resaved)
    assert forecast._network._q0.loc[101, "qd0"] == 8.0


def test_cycling_warmstate_never_accumulates_the_correction():
    """Repeated BMI updates must not feed the correction back into the background.

    BMI.update() calls Model.run() once per coupling update, so under ngen every window
    is the last window of its own call. An index-based "is this the final window" test
    is therefore inert there: it seeded on EVERY update, which is the case measured at
    -29.1% held-out bias against -23.8% for a single seeding boundary.

    The invariant that replaces it: the warmstate that CONTINUES the run is taken before
    the spread, and the corrected one is kept aside for create_state(). So N updates
    leave the cycling q0 identical to N updates with no DA at all, while the state a
    forecast restarts from still carries the correction (asserted in the test above).
    """
    import numpy as np
    from nwm_routing.scaling_da_apply import ScalingDA

    from troute.AbstractNetwork import AbstractNetwork
    from troute.scaling_da import build_gage_trees_from_mappings

    trees = build_gage_trees_from_mappings(
        {100: [101], 101: [102], 102: []}, {"G": 100}, {100: 30.0, 101: 20.0, 102: 10.0}
    )
    da = ScalingDA.__new__(ScalingDA)
    da.trees, da.gage_seg = trees, {"G": 100}
    da.min_flow, da.max_source_pbias, da._loop_obs = 1e-6, None, None
    # These run_results carry no Courant block; with the class-default lag ON
    # that is a fail-closed error, and the lag is not what this file tests.
    da.travel_time_lag = False

    def one_window():
        nts = 2
        arr = np.zeros((3, 4 * nts), dtype=np.float32)
        arr[:, 0::4] = np.tile(np.array([12.0, 6.0, 4.0])[:, None], (1, nts))
        nudge = np.zeros((1, nts + 1), dtype=np.float32)
        nudge[0, 1:] = 2.0
        return [[np.array([100, 101, 102]), arr, 0,
                 (np.array([100]), np.zeros(1), np.zeros(1)),
                 0, 0, 0, 0, np.zeros((3, nts), dtype=np.float32), nudge]]

    # The cycling warmstate is taken BEFORE the spread, every window.
    net = _NetworkStub()
    for _ in range(3):
        rr = one_window()
        AbstractNetwork.new_q0(net, rr)          # cycling q0: pre-spread
        cycling = net._q0.copy()
        da.apply_in_kernel(rr, nts=2, dt=3600, t0="2000-01-01")
        seeded = AbstractNetwork.new_q0(net, rr)  # hand-off q0: post-spread
        net._q0 = cycling                         # restore, as Model.run does

    # Three updates in, the cycling background is still the uncorrected 6.0 -- it never
    # accumulated. If the correction were seeded every update this would have drifted up.
    assert net._q0.loc[101, "qd0"] == pytest.approx(6.0, rel=1e-6)
    # ...while the hand-off warmstate carries it.
    assert seeded.loc[101, "qd0"] == pytest.approx(6.0 + 2.0 * (20.0 / 30.0) ** 0.77, rel=1e-5)
