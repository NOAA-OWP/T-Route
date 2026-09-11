"""Tests for the BMI driver's run-set (window) construction.

The window partition is where every per-window DA operation lands: the assimilation
frame is rebuilt per window, the kernel's lastobs memory and the source trust screen
reset per window, and the prognostic upstream spread enters q0 once per window. The
driver used to derive the partition from available system memory alone, so the same
config produced different discharge depending on what else the machine was running.
These tests pin the repaired contract: the configured max_loop_size is the primary
control, exactly as in the -V5 driver, and the memory estimate is only a safety cap.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pandas as pd
import pytest
import troute_nwm_bmi.troute_model as tm


def _model(nts_cols=96, max_loop_size=None, dt=300, qts=12):
    """A Model with just enough state for _build_run_sets."""
    m = tm.Model.__new__(tm.Model)
    fp = {"qts_subdivisions": qts, "dt": dt, "nts": nts_cols * qts}
    if max_loop_size is not None:
        fp["max_loop_size"] = max_loop_size
    m._config = {"compute_parameters": {"forcing_parameters": fp}}
    m._network = SimpleNamespace(t0=datetime(2019, 5, 29, 0, 0))
    m.dt = dt  # plain attribute in __init__, not a property
    return m


def _qlats(n_cols, t0=datetime(2019, 5, 29, 0, 0)):
    cols = [
        (t0 + pd.Timedelta(hours=i)).strftime("%Y%m%d%H%M") for i in range(n_cols)
    ]
    return pd.DataFrame(0.5, index=pd.Index([101, 102], name="feature_id"), columns=cols)


@pytest.fixture
def plenty_of_memory(monkeypatch):
    """Memory never constrains: the partition must come from the config alone."""
    monkeypatch.setattr(
        tm.psutil, "virtual_memory", lambda: SimpleNamespace(available=2**62)
    )


class TestConfiguredWindowIsPrimary:
    def test_honors_max_loop_size(self, plenty_of_memory):
        m = _model(nts_cols=96, max_loop_size=24)
        sets = list(m._build_run_sets(_qlats(96)))
        assert [s["qlats"].shape[1] for s in sets] == [24, 24, 24, 24]
        # each window's t0 is read off its first column, hourly cadence apart
        t0s = [s["t0"] for s in sets]
        assert t0s == [datetime(2019, 5, 29, 0) + pd.Timedelta(hours=24 * i) for i in range(4)]

    def test_window_covering_the_run_yields_one_set(self, plenty_of_memory):
        m = _model(nts_cols=96, max_loop_size=200)
        sets = list(m._build_run_sets(_qlats(96)))
        assert len(sets) == 1
        assert sets[0]["qlats"].shape[1] == 96

    def test_no_config_and_no_pressure_yields_one_set(self, plenty_of_memory):
        m = _model(nts_cols=96)
        sets = list(m._build_run_sets(_qlats(96)))
        assert len(sets) == 1

    def test_nts_scales_by_qts_subdivisions(self, plenty_of_memory):
        m = _model(nts_cols=48, max_loop_size=24)
        sets = list(m._build_run_sets(_qlats(48)))
        assert [s["nts"] for s in sets] == [24 * 12, 24 * 12]


class TestMemoryIsOnlyACap:
    def test_partition_does_not_change_with_free_ram(self, monkeypatch):
        """The reproducibility contract: same config, different free RAM, same windows.

        With the old memory-driven partition these two scenarios produced different
        window splits and therefore different assimilated discharge.
        """
        splits = []
        for avail in (2**62, 2**34):  # plenty vs 16 GB
            monkeypatch.setattr(
                tm.psutil, "virtual_memory", lambda a=avail: SimpleNamespace(available=a)
            )
            m = _model(nts_cols=96, max_loop_size=24)
            splits.append([s["qlats"].shape[1] for s in list(m._build_run_sets(_qlats(96)))])
        assert splits[0] == splits[1] == [24, 24, 24, 24]

    def test_memory_cap_tightens_an_oversized_window(self, monkeypatch, caplog):
        """A configured window that cannot fit is capped, with a warning."""
        # available*0.9 sized so the memory loop is ~1/4 of the run
        m = _model(nts_cols=96, max_loop_size=96)
        required = 2 * 96 * 12 * 100  # the measured per-element factor
        monkeypatch.setattr(
            tm.psutil, "virtual_memory",
            lambda: SimpleNamespace(available=(required / 4) / 0.9),
        )
        with caplog.at_level("WARNING"):
            sets = list(m._build_run_sets(_qlats(96)))
        assert len(sets) > 1
        assert all(s["qlats"].shape[1] <= 24 for s in sets)
        assert "caps the run window" in caplog.text

    def test_memory_split_without_config_warns(self, monkeypatch, caplog):
        """No max_loop_size and real pressure: split, but say the results depend on it."""
        m = _model(nts_cols=96)
        required = 2 * 96 * 12 * 100  # the measured per-element factor
        monkeypatch.setattr(
            tm.psutil, "virtual_memory",
            lambda: SimpleNamespace(available=(required / 4) / 0.9),
        )
        with caplog.at_level("WARNING"):
            sets = list(m._build_run_sets(_qlats(96)))
        assert len(sets) > 1
        assert "max_loop_size" in caplog.text


class TestFinalTimestampComesFromThisWindow:
    """The DA file list is built from final_timestamp, so it must describe THIS
    update's forcing, never the configured full-run horizon.

    The single-window branch used t0 + config_nts*dt. Under ngen-style repeated
    updates t0 advances every call, so each update enumerated (and decoded)
    TimeSlices across the entire configured horizon: no CLI parity, O(horizon)
    observation I/O per update, and with a large interpolation_limit_min the
    reader's bidirectional interpolation could fill the current window's trailing
    gaps from FUTURE observations.
    """

    def test_single_window_uses_the_last_forcing_column(self, plenty_of_memory):
        # Config horizon is 96 h; this update supplies 24 h of forcing.
        m = _model(nts_cols=96)
        sets = list(m._build_run_sets(_qlats(24)))
        assert len(sets) == 1
        assert sets[0]["final_timestamp"] == datetime(2019, 5, 29, 23, 0)

    def test_single_window_never_reaches_the_config_horizon(self, plenty_of_memory):
        m = _model(nts_cols=96)
        sets = list(m._build_run_sets(_qlats(24)))
        config_horizon = datetime(2019, 5, 29, 0, 0) + pd.Timedelta(seconds=m.nts * m.dt)
        assert sets[0]["final_timestamp"] < config_horizon

    def test_single_and_split_branches_agree_on_the_convention(self, plenty_of_memory):
        # 48 columns as one window vs as two: the one-window final_timestamp must
        # equal the second split window's, both read off the last forcing column.
        whole = list(_model(nts_cols=48)._build_run_sets(_qlats(48)))
        split = list(_model(nts_cols=48, max_loop_size=24)._build_run_sets(_qlats(48)))
        assert whole[0]["final_timestamp"] == split[-1]["final_timestamp"]


class TestRamCapIsAnErrorUnderActiveAssimilation:
    """The scaling DA's window boundaries are part of the result (the in-kernel decay
    state resets at each one), so a RAM-derived partition means machine load changes
    discharge, silently, with exit 0. Without the DA the old warning behavior stands:
    the partition then only affects performance.
    """

    def _pressure(self, monkeypatch, nts_cols=96):
        required = 2 * nts_cols * 12 * 200
        monkeypatch.setattr(
            tm.psutil, "virtual_memory",
            lambda: SimpleNamespace(available=(required / 4) / 0.9),
        )

    def test_cap_below_configured_window_raises(self, monkeypatch):
        self._pressure(monkeypatch)
        m = _model(nts_cols=96, max_loop_size=96)
        m._scaling_da = object()
        with pytest.raises(MemoryError, match="part of the result"):
            list(m._build_run_sets(_qlats(96)))

    def test_memory_split_without_config_raises(self, monkeypatch):
        self._pressure(monkeypatch)
        m = _model(nts_cols=96)
        m._scaling_da = object()
        with pytest.raises(MemoryError, match="Set max_loop_size"):
            list(m._build_run_sets(_qlats(96)))

    # These two isolate the SPREAD half: the lag is off (also the default) and
    # the spread is set explicitly, since its default is 0.
    _NO_LAG = {"innovation_spread_h": 12.0, "travel_time_lag": False}

    def test_window_below_the_spread_window_is_enlarged(self, plenty_of_memory):
        # 8 columns at 1 h/col = 8 h, under the 12 h default innovation_spread_h:
        # the halo that feeds the forward window is one window deep, so a window
        # shorter than the spread leaves its own tail uncovered. Enlarging
        # max_loop_size cannot change results -- it is a memory knob -- but a
        # window below the spread would.
        m = _model(nts_cols=96, max_loop_size=8)
        m._scaling_da = SimpleNamespace(**self._NO_LAG)
        sets = list(m._build_run_sets(_qlats(96)))
        assert [s["qlats"].shape[1] for s in sets] == [12] * 8

    def test_window_covering_the_spread_window_is_untouched(self, plenty_of_memory):
        m = _model(nts_cols=96, max_loop_size=24)
        m._scaling_da = SimpleNamespace(**self._NO_LAG)
        sets = list(m._build_run_sets(_qlats(96)))
        assert [s["qlats"].shape[1] for s in sets] == [24] * 4

    def test_the_lag_span_and_spread_add_up_to_the_window(self, plenty_of_memory):
        """With the lag on, a window must cover the traced span AND the spread:
        the innovation is averaged forward over the spread and the lag reads that
        average at t + tau, so the tail needs both."""
        m = _model(nts_cols=96, max_loop_size=24)
        m._scaling_da = SimpleNamespace(innovation_spread_h=12.0,
                                        travel_time_lag=True, lag_window_h=48.0)
        sets = list(m._build_run_sets(_qlats(96)))
        assert [s["qlats"].shape[1] for s in sets] == [60, 36]

    def test_without_the_da_the_warning_behavior_stands(self, monkeypatch, caplog):
        self._pressure(monkeypatch)
        m = _model(nts_cols=96, max_loop_size=96)  # no _scaling_da attribute at all
        with caplog.at_level("WARNING"):
            sets = list(m._build_run_sets(_qlats(96)))
        assert len(sets) > 1
        assert "caps the run window" in caplog.text

    def test_a_short_update_is_not_reported_as_a_memory_problem(self, plenty_of_memory):
        """The cap has two causes and they need different fixes.

        With memory ample, mem_loop_size is just this update's forcing count, so
        an update shorter than the required span fails for a reason no bigger
        machine can fix. Reporting that as "free memory" sends an operator to
        the wrong place, which is exactly what happened when ngen-sized updates
        were first tried against the shipped span.
        """
        m = _model(nts_cols=6, max_loop_size=6)
        # The lag and spread are OFF by default now, so the scenario sets them
        # explicitly: a 48 h span against a 6-column update.
        m._scaling_da = SimpleNamespace(
            travel_time_lag=True, lag_window_h=48.0, innovation_spread_h=12.0
        )
        with pytest.raises(ValueError, match="Memory is NOT the limit"):
            list(m._build_run_sets(_qlats(6)))

    def test_a_genuine_memory_cap_still_reports_memory(self, monkeypatch):
        self._pressure(monkeypatch)
        m = _model(nts_cols=96, max_loop_size=96)
        m._scaling_da = object()
        with pytest.raises(MemoryError, match="part of the result"):
            list(m._build_run_sets(_qlats(96)))
