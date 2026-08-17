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
