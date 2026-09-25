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

import numpy as np
import pandas as pd
import pytest
from troute_nwm_bmi.troute_model import Model

from troute.DataAssimilation import NudgingDA, _DiversionRow


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
        self._diversion_seed_in = {}
        self._diversion_rows = {}
        self._diversion_applied = {}
        self._usgs_df = pd.DataFrame()
        self._data_assimilation_parameters = {}

    # The real methods, bound to this stub: they read only the attributes above.
    diversion_seed_at = NudgingDA.diversion_seed_at
    refill_diversion_rows = NudgingDA.refill_diversion_rows
    diversion_applied = NudgingDA.diversion_applied


class _ScalingDAStub:
    """Presence marker for load_state's branch; records the trace restore."""

    def __init__(self):
        self.restored_with = "never called"

    def trace_checkpoint(self):
        return None

    def restore_trace_checkpoint(self, ckpt, dt):
        self.restored_with = (ckpt, dt)


def _da_config(**enabled):
    """A config switching the named reservoir DA types on. Default: all of them.

    ``_restore_da_frame`` now asks the run's own config which types it assimilates,
    so a stub with no config would report every type OFF and the frames would never
    be restored. These tests are about runs that DO use them.
    """
    flags = {"usgs": True, "usace": True, "usbr": True, "rfc": True, "gl": True,
             "nudging": True}
    flags.update(enabled)
    return {
        "compute_parameters": {
            "data_assimilation_parameters": {
                # lastobs belongs to the streamflow arms, and is dropped when both are off.
                "streamflow_da": {"streamflow_nudging": flags["nudging"]},
                "reservoir_da": {
                    "reservoir_persistence_da": {
                        "reservoir_persistence_usgs": flags["usgs"],
                        "reservoir_persistence_usace": flags["usace"],
                        "reservoir_persistence_usbr": flags["usbr"],
                        "reservoir_persistence_greatLake": flags["gl"],
                    },
                    "reservoir_rfc_da": {"reservoir_rfc_forecasts": flags["rfc"]},
                }
            }
        }
    }


def _make_model(time, subnetwork, **enabled):
    model = Model.__new__(Model)  # skip __init__ (no config/network build)
    model._time = time
    model._config = _da_config(**enabled)
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


def test_load_state_does_not_erase_live_reservoir_da_params():
    """A state written with a DA type OFF must not blank a run that has it ON.

    That combination is a legitimate handoff -- warm up without assimilation, then
    forecast with it -- but installing the saved empty frame left the observation
    frame populated and its parameters missing, and the run died on
    ``KeyError: 'totalCounts'`` inside _prep_reservoir_da_dataframes.
    """
    model = _make_model(0.0, _ExecutionPlanLike())
    live_rfc = model._data_assimilation._reservoir_rfc_param_df
    assert not live_rfc.empty

    # A checkpoint from a run with every reservoir DA type off.
    no_da_state = {
        "time": 0.0,
        "q0": pd.DataFrame({"q": [1.0, 2.0]}),
        "seeded_q0": None,
        "t0": "2020-01-01_00:00:00",
        "last_obs": pd.DataFrame(),
        "usgs": pd.DataFrame(),
        "usace": pd.DataFrame(),
        "usbr": pd.DataFrame(),
        "rfc": pd.DataFrame(),
        "gl": pd.DataFrame(),
        "scaling_tau": None,
    }
    model.load_state(no_da_state)

    da = model._data_assimilation
    for label, frame in [
        # An empty lastobs frame drops streamflow DA to open loop on any window
        # with no observations, so it is the same hazard as the reservoir frames.
        ("last observations", da._last_obs_df),
        ("USGS", da._reservoir_usgs_param_df),
        ("USACE", da._reservoir_usace_param_df),
        ("USBR", da._reservoir_usbr_param_df),
        ("RFC", da._reservoir_rfc_param_df),
        ("Great Lakes", da._great_lakes_param_df),
    ]:
        assert not frame.empty, f"{label} parameters erased by a no-DA state file"
    pd.testing.assert_frame_equal(da._reservoir_rfc_param_df, live_rfc)


def test_load_state_still_installs_real_reservoir_da_params():
    """The guard must not block the ordinary case: a populated saved frame wins.

    RFC is the exception, and has its own test below: its parameters describe the
    observation grid this cycle built, so a checkpoint's copy cannot be installed
    over them.
    """
    model = _make_model(0.0, _ExecutionPlanLike())
    saved_rfc = pd.DataFrame({"totalCounts": [99]})
    state = {
        "time": 0.0,
        "q0": pd.DataFrame({"q": [1.0, 2.0]}),
        "seeded_q0": None,
        "t0": "2020-01-01_00:00:00",
        "last_obs": pd.DataFrame(),
        "usgs": pd.DataFrame({"a": [11]}),
        "usace": pd.DataFrame({"b": [22]}),
        "usbr": pd.DataFrame({"e": [77]}),
        "rfc": saved_rfc,
        "gl": pd.DataFrame({"d": [44]}),
        "scaling_tau": None,
    }
    live_rfc = model._data_assimilation._reservoir_rfc_param_df.copy()
    model.load_state(state)
    da = model._data_assimilation
    pd.testing.assert_frame_equal(da._reservoir_usgs_param_df, state["usgs"])
    pd.testing.assert_frame_equal(da._reservoir_usace_param_df, state["usace"])
    pd.testing.assert_frame_equal(da._great_lakes_param_df, state["gl"])
    # RFC keeps what this cycle selected.
    pd.testing.assert_frame_equal(da._reservoir_rfc_param_df, live_rfc)


def test_a_checkpoint_with_fewer_gages_does_not_shrink_the_roster(caplog):
    """Downstream the lastobs index picks the roster, so a narrow checkpoint would
    quietly stop assimilating the gages it never saw."""
    import logging

    model = _make_model(0.0, _ExecutionPlanLike())
    da = model._data_assimilation
    da._last_obs_df = pd.DataFrame(
        {"time_since_lastobs": [0.0, 0.0, 0.0], "lastobs_discharge": [1.0, 2.0, 3.0]},
        index=[30, 50, 70],
    )
    saved = pd.DataFrame(
        {"time_since_lastobs": [9.0, 9.0], "lastobs_discharge": [8.0, 8.0]},
        index=[50, 30],
    )
    state = {
        "time": 0.0, "q0": pd.DataFrame({"q": [1.0]}), "seeded_q0": None,
        "t0": "2020-01-01_00:00:00", "last_obs": saved,
        "usgs": pd.DataFrame(), "usace": pd.DataFrame(), "usbr": pd.DataFrame(),
        "rfc": pd.DataFrame(), "gl": pd.DataFrame(), "scaling_tau": None,
    }
    with caplog.at_level(logging.WARNING):
        model.load_state(state)
    out = model._data_assimilation._last_obs_df
    assert set(out.index) == {30, 50, 70}
    # The checkpoint's history is kept where it had any.
    assert out.loc[30, "lastobs_discharge"] == 8.0
    # The gage it never saw starts with no history, which is the kernel's own default.
    assert pd.isna(out.loc[70, "lastobs_discharge"])
    assert "carries no last observations for 1 gage(s)" in caplog.text


def test_load_state_keeps_this_cycles_rfc_selection():
    """A checkpoint's cursor indexes the grid its own cycle built, not this one's."""
    model = _make_model(0.0, _ExecutionPlanLike())
    live_rfc = model._data_assimilation._reservoir_rfc_param_df.copy()
    state = {
        "time": 0.0, "q0": pd.DataFrame({"q": [1.0]}), "seeded_q0": None,
        "t0": "2020-01-01_00:00:00", "last_obs": pd.DataFrame(),
        "usgs": pd.DataFrame(), "usace": pd.DataFrame(), "usbr": pd.DataFrame(),
        "rfc": pd.DataFrame({"timeseries_idx": [72], "totalCounts": [99]}),
        "gl": pd.DataFrame(), "scaling_tau": None,
    }
    model.load_state(state)
    pd.testing.assert_frame_equal(
        model._data_assimilation._reservoir_rfc_param_df, live_rfc
    )


def test_load_state_does_not_carry_the_rfc_horizon_deadline():
    """The horizon is measured from the forecast's issue, so every cycle derives it.

    Carrying a checkpoint's deadline would deny a newly adopted forecast its own
    allowance, and re-arm an old one that should have expired.
    """
    model = _make_model(0.0, _ExecutionPlanLike())
    da = model._data_assimilation
    fresh = pd.Timestamp("2020-01-12")
    da._reservoir_rfc_param_df = pd.DataFrame(
        {"timeseries_idx": [5], "persist_until": [fresh]}, index=[101]
    )
    state = {
        "time": 0.0, "q0": pd.DataFrame({"q": [1.0]}), "seeded_q0": None,
        "t0": "2020-01-01_00:00:00", "last_obs": pd.DataFrame(),
        "usgs": pd.DataFrame(), "usace": pd.DataFrame(), "usbr": pd.DataFrame(),
        "rfc": pd.DataFrame(
            {"timeseries_idx": [72], "persist_until": [pd.Timestamp("2020-01-08")]},
            index=[101],
        ),
        "gl": pd.DataFrame(), "scaling_tau": None,
    }
    model.load_state(state)
    out = model._data_assimilation._reservoir_rfc_param_df
    assert out.loc[101, "timeseries_idx"] == 5
    assert out.loc[101, "persist_until"] == fresh




def test_load_state_rejects_a_no_da_checkpoint_after_routing():
    """Keeping live DA state is only right while it still describes the checkpoint.

    Re-deserializing into a model that has already routed would otherwise rewind
    time and q0 while retaining DA state that has moved past it, so a retry would
    not reproduce the first attempt.
    """
    model = _make_model(3600.0, _ExecutionPlanLike())
    model._has_routed = True
    no_da_state = {
        "time": 0.0,
        "q0": pd.DataFrame({"q": [1.0, 2.0]}),
        "seeded_q0": None,
        "t0": "2020-01-01_00:00:00",
        "last_obs": pd.DataFrame(),
        "usgs": pd.DataFrame(),
        "usace": pd.DataFrame(),
        "usbr": pd.DataFrame(),
        "rfc": pd.DataFrame(),
        "gl": pd.DataFrame(),
        "scaling_tau": None,
    }
    with pytest.raises(ValueError, match="already routed"):
        model.load_state(no_da_state)


def test_load_state_into_a_fresh_model_is_still_allowed():
    """The workflow this exists for: warm up with DA off, forecast with it on."""
    model = _make_model(0.0, _ExecutionPlanLike())
    live_rfc = model._data_assimilation._reservoir_rfc_param_df
    model.load_state({
        "time": 0.0,
        "q0": pd.DataFrame({"q": [1.0, 2.0]}),
        "seeded_q0": None,
        "t0": "2020-01-01_00:00:00",
        "last_obs": pd.DataFrame(),
        "usgs": pd.DataFrame(),
        "usace": pd.DataFrame(),
        "usbr": pd.DataFrame(),
        "rfc": pd.DataFrame(),
        "gl": pd.DataFrame(),
        "scaling_tau": None,
    })
    pd.testing.assert_frame_equal(
        model._data_assimilation._reservoir_rfc_param_df, live_rfc
    )


def _no_da_state():
    return {
        "time": 21600.0,  # a warm-up checkpoint is never at t=0
        "q0": pd.DataFrame({"q": [1.0, 2.0]}),
        "seeded_q0": None,
        "t0": "2020-01-01_00:00:00",
        "last_obs": pd.DataFrame(),
        "usgs": pd.DataFrame(),
        "usace": pd.DataFrame(),
        "usbr": pd.DataFrame(),
        "rfc": pd.DataFrame(),
        "gl": pd.DataFrame(),
        "scaling_tau": None,
    }


def test_reloading_the_same_checkpoint_without_routing_is_allowed():
    """A harness that deserializes twice must not be told the model has routed.

    load_state installs the checkpoint's own nonzero time, so keying "has this
    model advanced" off self._time made the second identical load raise.
    """
    model = _make_model(0.0, _ExecutionPlanLike())
    live_rfc = model._data_assimilation._reservoir_rfc_param_df
    model.load_state(_no_da_state())
    model.load_state(_no_da_state())  # nothing routed in between
    pd.testing.assert_frame_equal(
        model._data_assimilation._reservoir_rfc_param_df, live_rfc
    )


def test_reset_time_does_not_disguise_a_routed_model():
    """reset_time zeroes the clock without rewinding DA state, so it must not make
    an advanced model look fresh to the next restore."""
    model = _make_model(3600.0, _ExecutionPlanLike())
    model._has_routed = True
    model._orig_t0 = "2020-01-01_00:00:00"
    model.reset_time()
    with pytest.raises(ValueError, match="already routed"):
        model.load_state(_no_da_state())


def _state_from(model):
    """What ``model`` would serialize, round-tripped through pickle like the BMI does."""
    return pickle.loads(pickle.dumps(model.create_state(), pickle.HIGHEST_PROTOCOL))


def test_a_disabled_run_does_not_launder_stale_reservoir_params():
    """RFC on -> off -> on must not carry the first run's parameters to the third.

    Emptiness cannot say whether a type is switched on: an enabled type with no
    observations yet is empty too. So a run with RFC DA OFF used to install the
    checkpoint's populated frame over its own empty one, never update it (it runs no
    RFC DA), and re-serialize it as if it had. The next run to switch RFC back on
    then inherited `file`, `timeseries_idx` and `update_time` values stale by however
    long RFC stayed off, and preferred them over the ones it had just built.
    """
    # Run A: RFC on, and it has assimilated.
    run_a = _make_model(3600.0, _ExecutionPlanLike())
    run_a._data_assimilation._reservoir_rfc_param_df = pd.DataFrame(
        {"totalCounts": [12], "timeseries_idx": [7], "update_time": [3600]}
    )
    state_a = _state_from(run_a)

    # Run B: RFC off. It must keep its own empty frame, not adopt A's.
    run_b = _make_model(0.0, _ExecutionPlanLike(), rfc=False)
    run_b._data_assimilation._reservoir_rfc_param_df = pd.DataFrame()
    run_b.load_state(state_a)
    assert run_b._data_assimilation._reservoir_rfc_param_df.empty, (
        "a run with RFC DA off adopted the checkpoint's RFC parameters"
    )

    # ...so what B writes carries no RFC state to launder.
    state_b = _state_from(run_b)
    assert state_b["rfc"].empty
    assert state_b["reservoir_da_enabled"]["rfc"] is False

    # Run C: RFC on again. It keeps the parameters it built rather than A's stale ones.
    run_c = _make_model(0.0, _ExecutionPlanLike())
    fresh = pd.DataFrame({"totalCounts": [3], "timeseries_idx": [0], "update_time": [0]})
    run_c._data_assimilation._reservoir_rfc_param_df = fresh
    run_c.load_state(state_b)
    pd.testing.assert_frame_equal(
        run_c._data_assimilation._reservoir_rfc_param_df, fresh
    )


def test_an_enabled_run_with_no_observations_yet_still_takes_the_checkpoint():
    """The symmetric case: empty live does NOT mean the type is off.

    A persistence type with DA on that has not seen an observation in this window has
    an empty live frame and legitimately needs the checkpoint's state. Distinguishing
    it from the disabled run above is exactly what the recorded flags are for. RFC is
    the exception and has its own test below: its parameters are derived per cycle.
    """
    run_a = _make_model(3600.0, _ExecutionPlanLike())
    saved = pd.DataFrame({"prev_persisted_outflow": [12.0], "update_time": [3600]})
    run_a._data_assimilation._reservoir_usgs_param_df = saved
    state = _state_from(run_a)
    assert state["reservoir_da_enabled"]["usgs"] is True

    run_b = _make_model(0.0, _ExecutionPlanLike())
    run_b._data_assimilation._reservoir_usgs_param_df = pd.DataFrame()  # nothing yet
    run_b.load_state(state)
    pd.testing.assert_frame_equal(
        run_b._data_assimilation._reservoir_usgs_param_df, saved
    )


def test_a_cycle_that_selected_no_rfc_forecast_does_not_adopt_the_checkpoints(caplog):
    """RFC parameters are derived from this cycle's files, so an empty live frame is
    an answer rather than a gap. Adopting the checkpoint's cursor would index a grid
    this cycle never built, and write it back out as if this run had produced it."""
    import logging

    run_a = _make_model(3600.0, _ExecutionPlanLike())
    run_a._data_assimilation._reservoir_rfc_param_df = pd.DataFrame(
        {"totalCounts": [12], "timeseries_idx": [48]}
    )
    state = _state_from(run_a)

    run_b = _make_model(0.0, _ExecutionPlanLike())
    run_b._data_assimilation._reservoir_rfc_param_df = pd.DataFrame()
    with caplog.at_level(logging.WARNING):
        run_b.load_state(state)
    assert run_b._data_assimilation._reservoir_rfc_param_df.empty
    assert "selected no RFC forecast" in caplog.text


def test_switching_a_type_on_mid_cycle_works_on_a_fresh_model():
    """The workflow the recorded flags exist for: cycle N runs RFC off, cycle N+1
    runs it on. ngen builds a model per cycle, so the restoring model is fresh and
    the checkpoint's deliberate emptiness costs it nothing."""
    off = _make_model(0.0, _ExecutionPlanLike(), rfc=False)
    off._data_assimilation._reservoir_rfc_param_df = pd.DataFrame()
    state = _state_from(off)
    assert state["reservoir_da_enabled"]["rfc"] is False

    fresh = _make_model(0.0, _ExecutionPlanLike())  # RFC on, nothing routed
    live = fresh._data_assimilation._reservoir_rfc_param_df
    fresh.load_state(state)  # must not raise
    pd.testing.assert_frame_equal(
        fresh._data_assimilation._reservoir_rfc_param_df, live
    )


def test_a_routed_model_still_refuses_a_checkpoint_with_no_frame():
    """A model that HAS routed would keep state the restored time does not describe,
    and the kernel reads update_time and timeseries_idx straight out of it. That the
    checkpoint recorded the type as off does not make the live frame any fresher."""
    off = _make_model(0.0, _ExecutionPlanLike(), rfc=False)
    off._data_assimilation._reservoir_rfc_param_df = pd.DataFrame()
    state = _state_from(off)

    routed = _make_model(3600.0, _ExecutionPlanLike())  # RFC on
    routed._has_routed = True
    with pytest.raises(ValueError, match="already routed"):
        routed.load_state(state)


def test_a_rejected_load_leaves_the_model_untouched():
    """A refused restore must not half-apply.

    load_state used to rewind the clock and clear _has_routed before any restore
    guard ran, so the first attempt raised, the model was left half-loaded, and an
    identical retry accepted exactly what had just been refused.
    """
    routed = _make_model(3600.0, _ExecutionPlanLike())
    routed._has_routed = True
    before_time = routed._time
    before_rfc = routed._data_assimilation._reservoir_rfc_param_df.copy()
    legacy = _no_da_state()
    legacy.pop("reservoir_da_enabled", None)

    for attempt in (1, 2):
        with pytest.raises(ValueError, match="already routed"):
            routed.load_state(legacy)
        assert routed._has_routed is True, f"attempt {attempt} cleared _has_routed"
        assert routed._time == before_time, f"attempt {attempt} moved the clock"
        pd.testing.assert_frame_equal(
            routed._data_assimilation._reservoir_rfc_param_df, before_rfc
        )


def test_a_legacy_checkpoint_still_rejects_a_routed_model():
    """Without the flags there is no way to tell deliberate from stale, so reject.

    Checkpoints written before the flags were recorded keep the old behavior.
    """
    routed = _make_model(3600.0, _ExecutionPlanLike())
    routed._has_routed = True
    legacy = _no_da_state()
    legacy.pop("reservoir_da_enabled", None)
    with pytest.raises(ValueError, match="already routed"):
        routed.load_state(legacy)


def test_a_nudging_off_run_does_not_launder_a_stale_lastobs():
    """lastobs launders exactly as the reservoir frames did.

    time_since_lastobs is a RELATIVE offset, so a nudging-off run that adopted the
    checkpoint's frame, never refreshed it, and re-serialized it as its own would
    hand the next nudging-on run day-old observations as if they were current.
    """
    on = _make_model(3600.0, _ExecutionPlanLike())
    on._data_assimilation._last_obs_df = pd.DataFrame({"discharge": [42.0]})
    state = _state_from(on)

    off = _make_model(0.0, _ExecutionPlanLike(), nudging=False)
    off._data_assimilation._last_obs_df = pd.DataFrame()
    off.load_state(state)
    assert off._data_assimilation._last_obs_df.empty, (
        "a run with streamflow nudging off adopted the checkpoint's lastobs"
    )
    assert _state_from(off)["last_obs"].empty


def test_the_enabled_flags_track_the_real_config_schema():
    """The flag names are hardcoded; a rename in troute-config would silently make
    every type read False, keep every live frame, and exit 0. Build the flags through
    the real pydantic model so a rename fails HERE rather than in a quiet run."""
    from troute.config.compute_parameters import DataAssimilationParameters

    dumped = DataAssimilationParameters(
        streamflow_da={"streamflow_nudging": True},
        reservoir_da={
            "reservoir_persistence_da": {
                "reservoir_persistence_usgs": True,
                "reservoir_persistence_usace": True,
                "reservoir_persistence_usbr": True,
                "reservoir_persistence_greatLake": True,
            },
            "reservoir_rfc_da": {"reservoir_rfc_forecasts": True},
        },
    ).model_dump()

    model = Model.__new__(Model)
    model._config = {"compute_parameters": {"data_assimilation_parameters": dumped}}
    assert model._reservoir_da_enabled() == {
        "usgs": True, "usace": True, "usbr": True, "rfc": True, "gl": True
    }
    assert model._owns_lastobs() is True


def test_a_scaling_run_keeps_its_checkpointed_lastobs():
    """Scaling harvests lastobs, so a restore must not treat the frame as unowned.

    Asking only about streamflow_nudging made a scaling-only run drop the
    checkpoint's lastobs and restart its decay history, which is exactly the
    cross-cycle continuity the harvest exists to provide.
    """
    model = _make_model(0.0, _ExecutionPlanLike(), nudging=False)
    model._config["compute_parameters"]["data_assimilation_parameters"][
        "streamflow_da"]["streamflow_scaling"] = True
    model._data_assimilation._last_obs_df = pd.DataFrame()  # nothing built yet
    saved = pd.DataFrame({"discharge": [42.0], "time_since_lastobs": [-1800.0]})

    state = _no_da_state()
    state["last_obs"] = saved
    model.load_state(state)

    pd.testing.assert_frame_equal(model._data_assimilation._last_obs_df, saved)


def test_a_wider_checkpoint_roster_is_trimmed_to_this_run():
    """A nudging checkpoint carries gages the scaling arm deliberately excludes.

    Scaling's roster is a strict subset of nudging's (holdouts, reservoir-routed and
    co-located gages are dropped), and the checkpoint records no producer mode.
    Installing it whole hands _prep_da_dataframes segments the current frames do not
    carry, which raises "not in index", or silently persists excluded gages.
    """
    model = _make_model(0.0, _ExecutionPlanLike())
    model.dt = 300
    model._scaling_da = _ScalingDAStub()
    model._scaling_da.gage_seg = {"A": 1}   # this run assimilates seg 1
    model._data_assimilation._last_obs_df = pd.DataFrame()

    state = _no_da_state()
    state["last_obs"] = pd.DataFrame({"discharge": [5.0, 9.0]}, index=[1, 2])
    model.load_state(state)

    got = model._data_assimilation._last_obs_df
    assert list(got.index) == [1], f"seg 2 is not in this run's roster: {list(got.index)}"


def test_a_matching_roster_is_left_alone():
    model = _make_model(0.0, _ExecutionPlanLike())
    model.dt = 300
    model._scaling_da = _ScalingDAStub()
    model._scaling_da.gage_seg = {"A": 1, "B": 2}
    model._data_assimilation._last_obs_df = pd.DataFrame()
    saved = pd.DataFrame({"discharge": [5.0, 9.0]}, index=[1, 2])
    state = _no_da_state(); state["last_obs"] = saved
    model.load_state(state)
    pd.testing.assert_frame_equal(model._data_assimilation._last_obs_df, saved)


# --- diversion seed: as of the checkpoint time out, re-filled on the way in ------------

_DIV_LINK = 1269985531956909
_DIV_T0 = pd.Timestamp("2011-05-01 00:00")


def _diversion_row(values: dict) -> _DiversionRow:
    grid = pd.date_range(_DIV_T0 - pd.Timedelta(hours=1), periods=25, freq="15min")
    raw = pd.Series(np.nan, index=grid, dtype=float)
    for when, q in values.items():
        raw[pd.Timestamp(when)] = q
    return _DiversionRow(raw, "07381482")


def test_create_state_takes_the_diversion_seed_as_of_the_checkpoint_time():
    """A BMI frame spans the whole horizon; the checkpoint must not see past t0."""
    model = _make_model(3600.0, [None, None, None])
    model._network._t0 = _DIV_T0
    model._data_assimilation._diversion_rows = {
        _DIV_LINK: _diversion_row({"2011-04-30 23:30": 3.0, "2011-05-01 02:00": 9.0})
    }
    seed = model.create_state()["diversion_seed"]
    assert seed == {_DIV_LINK: (pd.Timestamp("2011-04-30 23:30"), 3.0)}


def test_load_state_refills_the_diversion_row_from_the_restored_seed():
    """Restoring the seed alone changes nothing: the frame was filled at init."""
    state = _make_model(3600.0, [None, None, None]).create_state()
    state["diversion_seed"] = {_DIV_LINK: (pd.Timestamp("2011-04-30 12:00"), 4.0)}
    dst = _make_model(0.0, [None, None, None])
    da = dst._data_assimilation
    row = _diversion_row({})
    da._diversion_rows = {_DIV_LINK: row}
    da._usgs_df = row.raw.to_frame().T
    da._usgs_df.index = pd.Index([_DIV_LINK], dtype="int64")
    da._data_assimilation_parameters = {"diversion_da": {"diversion_persist_days": 11}}
    dst.load_state(pickle.loads(pickle.dumps(state, pickle.HIGHEST_PROTOCOL)))
    np.testing.assert_allclose(da._usgs_df.loc[_DIV_LINK].to_numpy(), 4.0)


def test_load_state_refills_on_the_frames_own_columns():
    """The frame was trimmed to t0 at init; the refill must not bring the raw row's
    lookback columns back, appended after the grid, which breaks the run's t0 slice."""
    state = _make_model(3600.0, [None, None, None]).create_state()
    state["diversion_seed"] = {_DIV_LINK: (pd.Timestamp("2011-04-30 12:00"), 4.0)}
    dst = _make_model(0.0, [None, None, None])
    da = dst._data_assimilation
    row = _diversion_row({})
    da._diversion_rows = {_DIV_LINK: row}
    da._usgs_df = row.raw.to_frame().T.loc[:, _DIV_T0:]
    da._usgs_df.index = pd.Index([_DIV_LINK], dtype="int64")
    trimmed = list(da._usgs_df.columns)
    da._data_assimilation_parameters = {"diversion_da": {"diversion_persist_days": 11}}
    dst.load_state(pickle.loads(pickle.dumps(state, pickle.HIGHEST_PROTOCOL)))
    assert list(da._usgs_df.columns) == trimmed
    assert da._usgs_df.columns.is_monotonic_increasing
    np.testing.assert_allclose(da._usgs_df.loc[_DIV_LINK].to_numpy(), 4.0)


def test_a_rollback_restores_the_earlier_checkpoints_seed():
    """Loading a later checkpoint, then an earlier one, must leave the earlier seed."""
    early = _make_model(3600.0, [None, None, None]).create_state()
    early["diversion_seed"] = {_DIV_LINK: (pd.Timestamp("2011-04-30 12:00"), 4.0)}
    late = _make_model(3600.0, [None, None, None]).create_state()
    late["diversion_seed"] = {_DIV_LINK: (pd.Timestamp("2011-05-01 12:00"), 9.0)}
    dst = _make_model(0.0, [None, None, None])
    da = dst._data_assimilation
    row = _diversion_row({})
    da._diversion_rows = {_DIV_LINK: row}
    da._usgs_df = row.raw.to_frame().T
    da._usgs_df.index = pd.Index([_DIV_LINK], dtype="int64")
    da._data_assimilation_parameters = {"diversion_da": {"diversion_persist_days": 11}}
    for state in (late, early):
        dst.load_state(pickle.loads(pickle.dumps(state, pickle.HIGHEST_PROTOCOL)))
    assert da._diversion_seed_in == early["diversion_seed"]
    np.testing.assert_allclose(da._usgs_df.loc[_DIV_LINK].to_numpy(), 4.0)


def test_the_applied_subtraction_rides_the_checkpoint():
    """The donor's last subtraction is state: the next window restores it into qdp,
    and a checkpoint that dropped it would seed from column 0 instead."""
    src = _make_model(3600.0, [None, None, None])
    src._data_assimilation._diversion_applied = {_DIV_LINK: 123.5}
    state = pickle.loads(pickle.dumps(src.create_state(), pickle.HIGHEST_PROTOCOL))
    assert state["diversion_applied"] == {_DIV_LINK: 123.5}
    dst = _make_model(0.0, [None, None, None])
    dst.load_state(state)
    assert dst._data_assimilation.diversion_applied == {_DIV_LINK: 123.5}


def test_a_newer_report_in_the_record_still_governs_the_fill():
    """The checkpoint's seed is restored as it is; a report the record holds after it
    governs the fill anyway, because the fill reads the raw row first."""
    state = _make_model(3600.0, [None, None, None]).create_state()
    state["diversion_seed"] = {_DIV_LINK: (pd.Timestamp("2011-04-30 12:00"), 4.0)}
    dst = _make_model(0.0, [None, None, None])
    da = dst._data_assimilation
    row = _diversion_row({"2011-04-30 23:30": 6.0})
    da._diversion_rows = {_DIV_LINK: row}
    da._usgs_df = row.raw.to_frame().T
    da._usgs_df.index = pd.Index([_DIV_LINK], dtype="int64")
    da._data_assimilation_parameters = {"diversion_da": {"diversion_persist_days": 11}}
    dst.load_state(pickle.loads(pickle.dumps(state, pickle.HIGHEST_PROTOCOL)))
    assert da._diversion_seed_in == state["diversion_seed"]
    filled = da._usgs_df.loc[_DIV_LINK]
    np.testing.assert_allclose(filled[:"2011-04-30 23:15"].to_numpy(), 4.0)
    np.testing.assert_allclose(filled["2011-04-30 23:30":].to_numpy(), 6.0)


def test_load_state_without_a_diversion_seed_leaves_the_row_as_built():
    """Older pickles carry no seed; nothing to hold, nothing changes."""
    state = _make_model(3600.0, [None, None, None]).create_state()
    del state["diversion_seed"]
    dst = _make_model(0.0, [None, None, None])
    da = dst._data_assimilation
    row = _diversion_row({})
    da._diversion_rows = {_DIV_LINK: row}
    da._usgs_df = row.raw.to_frame().T
    da._usgs_df.index = pd.Index([_DIV_LINK], dtype="int64")
    da._data_assimilation_parameters = {"diversion_da": {"diversion_persist_days": 11}}
    dst.load_state(pickle.loads(pickle.dumps(state, pickle.HIGHEST_PROTOCOL)))
    assert da._usgs_df.loc[_DIV_LINK].isna().all()

