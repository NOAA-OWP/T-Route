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
from joblib import effective_n_jobs

from troute.window_plan import AUTO_WINDOW


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


def _required(cols, *, links=2, qts=12, courant=False, scaling=False, lag=False,
              pool=1):
    """The model's own budget for a test domain, so a fixture can sit either side of it.

    Mirrors _build_run_sets deliberately: these fixtures exercise the CAP LOGIC, and
    what the constants themselves should be is pinned separately against the measured
    sweeps in TestTheEstimateTracksBothMeasuredSweeps.
    """
    per_element = tm.per_element_bytes(courant, scaling, lag)
    overhead = tm.POOL_OVERHEAD if pool > 1 else 1.0
    return int(cols * (tm.per_column_bytes(scaling) + per_element * links * qts)
               * overhead)


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

    def test_no_config_and_no_pressure_yields_the_auto_window(self, plenty_of_memory):
        """No max_loop_size means automatic, and automatic is AUTO_WINDOW.

        Not the whole update: the memory estimate is a lower bound, so sizing up
        from it picks a window the machine may not hold. Memory only caps.
        """
        sets = list(_model(nts_cols=96)._build_run_sets(_qlats(96)))
        assert [rs["qlats"].shape[1] for rs in sets] == [AUTO_WINDOW] * 4

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
        required = _required(96)
        monkeypatch.setattr(
            tm.psutil, "virtual_memory",
            lambda: SimpleNamespace(available=(required / 4) / 0.9),
        )
        with caplog.at_level("WARNING"):
            sets = list(m._build_run_sets(_qlats(96)))
        assert len(sets) > 1
        assert all(s["qlats"].shape[1] <= 24 for s in sets)
        assert "caps the run window" in caplog.text

    def test_memory_split_without_config_is_reported(self, monkeypatch, caplog):
        """No max_loop_size and real pressure: split, and say so.

        Without a DA span the partition does not reach the result, so this is INFO
        rather than a warning: there is nothing for the operator to fix.
        """
        m = _model(nts_cols=96)
        required = _required(96)
        monkeypatch.setattr(
            tm.psutil, "virtual_memory",
            lambda: SimpleNamespace(available=(required / 4) / 0.9),
        )
        with caplog.at_level("INFO"):
            sets = list(m._build_run_sets(_qlats(96)))
        assert len(sets) > 1
        assert f"chose {sets[0]['qlats'].shape[1]} forcing timestep(s)" in caplog.text


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
        whole = list(_model(nts_cols=48, max_loop_size=48)._build_run_sets(_qlats(48)))
        split = list(_model(nts_cols=48, max_loop_size=24)._build_run_sets(_qlats(48)))
        assert whole[0]["final_timestamp"] == split[-1]["final_timestamp"]


class TestRamCapIsAnErrorUnderActiveAssimilation:
    """A scaling DA WITH A SPAN has window boundaries that are part of the result (the
    in-kernel decay state resets at each one), so a RAM-derived partition means machine
    load changes discharge, silently, with exit 0. Without the DA, or with a DA whose
    span is zero, the old warning behavior stands: the partition then only affects
    performance, which is measured in test/troute-nwm/test_scaling_da_in_kernel.py.
    """

    def _pressure(self, monkeypatch, nts_cols=96):
        required = _required(nts_cols, scaling=True)
        monkeypatch.setattr(
            tm.psutil, "virtual_memory",
            lambda: SimpleNamespace(available=(required / 4) / 0.9),
        )

    def test_cap_below_configured_window_raises(self, monkeypatch):
        self._pressure(monkeypatch)
        m = _model(nts_cols=96, max_loop_size=96)
        m._scaling_da = SimpleNamespace(  # a DA whose partition matters
            innovation_spread_h=12.0, travel_time_lag=False
        )
        with pytest.raises(MemoryError, match="machine load"):
            list(m._build_run_sets(_qlats(96)))

    def test_auto_sizes_rather_than_refusing(self, monkeypatch):
        """Auto is the default, so pressure alone must not stop a run.

        It picks a window that fits on a stricter budget than an explicit value gets.
        Refusing here made the operator set a number they had no way to compute.
        """
        self._pressure(monkeypatch)
        m = _model(nts_cols=96)                       # no max_loop_size -> auto
        m._scaling_da = SimpleNamespace(innovation_spread_h=0.0, travel_time_lag=False)
        sets = list(m._build_run_sets(_qlats(96)))
        assert len(sets) > 1

    def test_auto_still_refuses_when_the_span_cannot_fit(self, monkeypatch):
        """The one thing auto must not do is pick a window shorter than the DA span.

        A window below the span makes the partition reach the result, so when memory
        cannot hold the span there is nothing safe to choose. MemoryError specifically:
        accepting either exception hid which of the two diagnoses the operator gets.
        """
        self._pressure(monkeypatch)
        m = _model(nts_cols=96)                       # auto
        m._scaling_da = SimpleNamespace(innovation_spread_h=96.0, travel_time_lag=False)
        with pytest.raises(MemoryError):
            list(m._build_run_sets(_qlats(96)))

    def test_a_span_longer_than_the_update_is_not_blamed_on_memory(self, monkeypatch):
        """Freeing memory cannot make a 24-column update cover a 48-column span.

        The memory guard used to fire first here and send the operator chasing RAM.
        """
        self._pressure(monkeypatch, nts_cols=24)
        m = _model(nts_cols=24)
        m._scaling_da = SimpleNamespace(innovation_spread_h=48.0, travel_time_lag=False)
        with pytest.raises(ValueError, match="Memory is NOT the limit"):
            list(m._build_run_sets(_qlats(24)))

    def test_an_explicit_zero_is_auto_not_an_error(self, monkeypatch):
        """0 is the way to ask for auto, so it must behave as auto, not as a window."""
        self._pressure(monkeypatch)
        m = _model(nts_cols=96, max_loop_size=0)
        m._scaling_da = SimpleNamespace(  # a DA whose partition matters
            innovation_spread_h=12.0, travel_time_lag=False
        )
        sets = list(m._build_run_sets(_qlats(96)))
        # Auto under pressure still splits, but never below the 12 h span.
        assert len(sets) > 1
        assert all(rs["qlats"].shape[1] >= 12 for rs in sets[:-1])

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
        # 60 then a 36-column remainder, and 36 is under the span, so it folds into
        # the window before it rather than reading across a boundary it cannot supply.
        assert [s["qlats"].shape[1] for s in sets] == [96]

    def test_without_the_da_the_warning_behavior_stands(self, monkeypatch, caplog):
        self._pressure(monkeypatch)
        m = _model(nts_cols=96, max_loop_size=96)  # no _scaling_da attribute at all
        with caplog.at_level("WARNING"):
            sets = list(m._build_run_sets(_qlats(96)))
        assert len(sets) > 1
        assert "caps the run window" in caplog.text

    def test_a_zero_span_da_still_refuses_a_ram_split(self, monkeypatch):
        """Zero span exempts a SINGLE window, not a RAM-driven split.

        Multi-window equivalence now measures 0.0 (lastobs is harvested for scaling,
        and every window reads one run-spanning observation list), but a partition
        chosen by machine load is still not something to leave to chance, and the
        one-window case is the one that needs no measurement at all.
        """
        self._pressure(monkeypatch)
        m = _model(nts_cols=96, max_loop_size=96)
        m._scaling_da = SimpleNamespace(innovation_spread_h=0.0, travel_time_lag=False)
        with pytest.raises(MemoryError, match="machine load"):
            list(m._build_run_sets(_qlats(96)))

    def test_a_zero_span_da_serves_a_short_update(self, plenty_of_memory):
        """The case the exemption exists for: one window, nothing partitioned.

        Refusing here made the shipped NWM Standard AnA config unrunnable: 3 forcing
        columns against a max_loop_size of 24.
        """
        m = _model(nts_cols=3, max_loop_size=24)
        m._scaling_da = SimpleNamespace(innovation_spread_h=0.0, travel_time_lag=False)
        sets = list(m._build_run_sets(_qlats(3)))
        assert len(sets) == 1

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
        m._scaling_da = SimpleNamespace(  # a DA whose partition matters
            innovation_spread_h=12.0, travel_time_lag=False
        )
        with pytest.raises(MemoryError, match="machine load"):
            list(m._build_run_sets(_qlats(96)))


def test_the_pool_costs_a_constant_not_a_multiple(monkeypatch):
    """A worker pool adds a measured ~15%, not its own size.

    Workers route their own clusters and share the parent's pages: benchmark/RESULTS.md
    measures CONUS tree PSS at cpu_pool 8 as 27.8 GB against 24.6 GB main-process, and
    Tier A at parity. Treating the pool as a multiplier sized VPU01 into 2-column
    windows and refused a scaling run outright.
    """
    def windows(pool, headroom):
        m = _model(nts_cols=96, max_loop_size=96)
        m._config["compute_parameters"]["cpu_pool"] = pool
        one_worker = _required(96)
        monkeypatch.setattr(tm.psutil, "virtual_memory",
                            lambda: SimpleNamespace(available=one_worker * headroom / 0.9))
        return len(list(m._build_run_sets(_qlats(96))))

    # Budget between the serial requirement and the pooled one: serial fits, pooled
    # does not. A multiplier would have split the serial case too at 8x this budget.
    between = (1.0 + tm.POOL_OVERHEAD) / 2
    assert windows(1, between) == 1, "one worker fits its own requirement"
    assert windows(8, between) > 1, "the pool costs more than the main process alone"
    assert windows(8, tm.POOL_OVERHEAD * 1.01) == 1, "but only by the measured overhead"

    # joblib reads -1 as every core, so it is a POOL, not a fallback to serial.
    n_cpus = effective_n_jobs(-1)
    if n_cpus > 1:
        assert windows(-1, between) == windows(n_cpus, between), (
            f"cpu_pool -1 is {n_cpus} workers, not one"
        )


class TestAutoNeverSizesUpFromTheEstimate:
    """The estimate is calibrated on ONE domain (Tier A's sweep), so automatic sizing
    takes a fixed window and lets memory cap it. Deriving the window from the estimate
    instead would trust it far past what was measured -- and before calibration it read
    21x low, which picks the 8.8 GB single-window case on a small machine.
    """

    def test_comfortable_memory_gives_the_auto_window(self, plenty_of_memory):
        sets = list(_model(nts_cols=96)._build_run_sets(_qlats(96)))
        assert [rs["qlats"].shape[1] for rs in sets] == [AUTO_WINDOW] * 4

    def test_memory_still_caps_the_auto_window(self, monkeypatch):
        required = _required(96, scaling=True)
        monkeypatch.setattr(
            tm.psutil, "virtual_memory",
            lambda: SimpleNamespace(available=(required / 8) / 0.9),
        )
        m = _model(nts_cols=96)
        m._scaling_da = SimpleNamespace(innovation_spread_h=0.0, travel_time_lag=False)
        sets = list(m._build_run_sets(_qlats(96)))
        assert all(rs["qlats"].shape[1] < AUTO_WINDOW for rs in sets)

    def test_the_span_is_the_only_term_that_enlarges(self, plenty_of_memory):
        m = _model(nts_cols=96)
        m._scaling_da = SimpleNamespace(innovation_spread_h=48.0, travel_time_lag=False)
        sets = list(m._build_run_sets(_qlats(96)))
        assert [rs["qlats"].shape[1] for rs in sets] == [48, 48]


# Measured on ONE machine, both domains, cpu_pool 8, 8 forcing columns total, main
# process only (benchmark/results/{ohio,conus}_mls_sweep*.json). Window columns -> peak
# RSS in MB. Held here so the estimator is checked against numbers that do not come
# from the estimator.
SWEEPS = {
    ("ohio", False): (11327, {1: 620.7, 2: 733.9, 4: 876.6, 8: 1027.9}),
    ("conus", False): (1102154, {1: 10206.8, 2: 11087.4, 4: 11602.7, 8: 13292.0}),
    # Same sweep with streamflow_scaling on. CONUS has only two points: mls 8 drove
    # the measuring machine into swap, so its peak would have measured the swap.
    ("ohio", True): (11327, {1: 786.5, 2: 970.1, 4: 1294.5, 8: 1678.3}),
    ("conus", True): (1102154, {2: 14901.7, 4: 16211.5}),
}
QTS = 12


def _measured_slope(points):
    """Least-squares MB per forcing column. The intercept is the domain's baseline."""
    n = len(points)
    mx = sum(points) / n
    my = sum(points.values()) / n
    num = sum((w - mx) * (r - my) for w, r in points.items())
    return num / sum((w - mx) ** 2 for w in points)


class TestTheEstimateTracksBothMeasuredSweeps:
    """The tests whose budget does NOT come from per_element_bytes.

    Every other memory test derives its threshold from the helper it is testing, so all
    of them stay green however wrong the constants are. That is how the estimate came
    to read 21x under on one domain and 13x over on the other without a failure.
    """

    def test_the_model_matches_each_domain_and_arm(self):
        for (name, scaling), (links, points) in SWEEPS.items():
            predicted = (tm.per_column_bytes(scaling)
                         + tm.per_element_bytes(False, scaling) * links * QTS) / 1e6
            measured = _measured_slope(points)
            # Lower bound above the runtime's own 0.9 reserve: a model allowed to
            # read 15% low against a 10% reserve can still OOM. All four arms sit
            # between 1.00 and 1.02, so this costs nothing today.
            assert 0.95 <= predicted / measured <= 1.20, (
                f"{name} scaling={scaling}: model {predicted:.0f} MB/column against "
                f"{measured:.0f} measured ({predicted / measured:.2f}x)"
            )

    def test_one_per_element_constant_cannot_serve_both(self):
        """The measurement that forced the second term, kept as a regression guard.

        Read as a single per-element constant the two domains give 409 B and 31 B. Any
        future collapse back to one constant has to fail here, not in production on
        whichever domain was not fitted.
        """
        implied = {
            name: _measured_slope(points) * 1e6 / (links * QTS)
            for (name, scaling), (links, points) in SWEEPS.items() if not scaling
        }
        assert implied["ohio"] / implied["conus"] > 5, implied

    def test_the_per_column_term_dominates_a_small_domain(self):
        """Why fitting Ohio alone went wrong: at 11k links almost all of that 55 MB
        per column is the domain-independent term, not the per-element one."""
        links, _ = SWEEPS[("ohio", False)]
        per_element_share = (tm.per_element_bytes(False, False) * links * QTS
                             / (tm.per_column_bytes(False)
                                + tm.per_element_bytes(False, False) * links * QTS))
        assert per_element_share < 0.15

    def test_the_per_element_term_dominates_conus(self):
        links, _ = SWEEPS[("conus", False)]
        per_element_share = (tm.per_element_bytes(False, False) * links * QTS
                             / (tm.per_column_bytes(False)
                                + tm.per_element_bytes(False, False) * links * QTS))
        assert per_element_share > 0.80


class TestTheSplitStaysInsideTheCap:
    """Every emitted window must fit the memory cap AND cover the DA span.

    Filling to loop_size and folding a short remainder into its neighbor satisfied
    the span by breaking the cap: 47 columns at a cap of 23 came out as one window of
    47, and 100 columns at a cap of 25 ended [24, 24, 24, 28]. An even split cannot
    do that, because no window exceeds the ceiling of the average.
    """

    def _cap(self, monkeypatch, cols, divisions):
        required = _required(cols, scaling=True)
        monkeypatch.setattr(
            tm.psutil, "virtual_memory",
            lambda: SimpleNamespace(available=(required / divisions) / 0.9),
        )
        return cols / divisions

    def _model_with_span(self, cols, span_h):
        m = _model(nts_cols=cols)
        m._scaling_da = SimpleNamespace(innovation_spread_h=span_h, travel_time_lag=False)
        return m

    def test_no_window_exceeds_the_cap_or_falls_under_the_span(self, monkeypatch):
        cap = self._cap(monkeypatch, 100, 4)
        sets = list(self._model_with_span(100, 12.0)._build_run_sets(_qlats(100)))
        widths = [rs["qlats"].shape[1] for rs in sets]
        assert sum(widths) == 100
        assert max(widths) <= cap, f"{widths} exceeds the {cap} column cap"
        assert min(widths) >= 12, f"{widths} falls under the 12 column span"

    def test_an_unsplittable_update_beyond_the_cap_raises(self, monkeypatch):
        # 47 columns against a 24 column span: any two windows leave one at 23, and
        # the single window that would work is twice what memory allows.
        self._cap(monkeypatch, 47, 2)
        with pytest.raises(MemoryError, match="reduce the DA span"):
            list(self._model_with_span(47, 24.0)._build_run_sets(_qlats(47)))

    def test_an_unsplittable_update_within_the_cap_runs(self, plenty_of_memory):
        sets = list(self._model_with_span(47, 24.0)._build_run_sets(_qlats(47)))
        assert [rs["qlats"].shape[1] for rs in sets] == [47]


class TestTheScalingCostLandsWhereItWasMeasured:
    """The DA's cost is mostly PER COLUMN, and the code used to put none of it there.

    Modelled as +34 B/element and nothing per column, it missed the term that dominates
    (67.8 MB/column measured) and overstated the one that does not (13 B/element). On
    CONUS at mls=4 that is the difference between predicting 29 GB and the 16.2 GB the
    run actually used.
    """

    def test_scaling_adds_a_per_column_cost(self):
        added = tm.per_column_bytes(True) - tm.per_column_bytes(False)
        assert added / 1e6 == pytest.approx(67.8, abs=5)

    def test_scaling_adds_little_per_element(self):
        added = tm.per_element_bytes(False, True) - tm.per_element_bytes(False, False)
        assert added == pytest.approx(13, abs=4)

    def test_the_scaling_figure_does_not_carry_courant(self):
        """The sweep ran with travel_time_lag OFF, and the drivers turn Courant on for
        the lag's trace, so no swept run allocated a Courant block.

        Folding Courant into the DA figure under-read a lag-on run on the one window
        the lag makes widest, which is the direction that OOMs.
        """
        assert tm.per_element_bytes(True, True) > tm.per_element_bytes(False, True)

    def test_the_lag_adds_courant_and_its_trace(self):
        lag_only = tm.per_element_bytes(False, True, lag=True)
        no_lag = tm.per_element_bytes(False, True)
        assert lag_only - no_lag == (tm.COURANT_PER_ELEMENT_BYTES
                                     + tm.LAG_TRACE_PER_ELEMENT_BYTES)

    def test_the_lag_does_not_pay_for_courant_twice(self):
        """The lag is what turns Courant on, so asking for both is not two blocks."""
        assert (tm.per_element_bytes(True, True, lag=True)
                == tm.per_element_bytes(False, True, lag=True))

    def test_courant_alone_is_still_the_declared_width(self):
        added = tm.per_element_bytes(True, False) - tm.per_element_bytes(False, False)
        assert added == tm.COURANT_PER_ELEMENT_BYTES == 17

class TestAutoTakesWhatFits:
    """Auto chose no window, so a narrower one is simply what it would have chosen.

    Refusing when memory capped it below AUTO_WINDOW but still above the span made the
    default arm fail on a run it could have served: a 20 column window against a 12
    column span is valid, and auto was raising on it.
    """

    def _model(self, cols, span_h, divisions, monkeypatch):
        m = _model(nts_cols=cols)
        m._scaling_da = SimpleNamespace(innovation_spread_h=span_h, travel_time_lag=False)
        required = _required(cols, scaling=True)
        monkeypatch.setattr(
            tm.psutil, "virtual_memory",
            lambda: SimpleNamespace(available=(required / divisions) / 0.9),
        )
        return m

    def test_a_cap_above_the_span_is_used_not_refused(self, monkeypatch):
        sets = list(self._model(96, 12.0, 6, monkeypatch)._build_run_sets(_qlats(96)))
        widths = [rs["qlats"].shape[1] for rs in sets]
        assert len(widths) > 1
        assert min(widths) >= 12, widths
        assert max(widths) < AUTO_WINDOW, f"{widths}: memory should have narrowed it"

    def test_a_cap_below_the_span_still_refuses(self, monkeypatch):
        with pytest.raises(MemoryError):
            list(self._model(96, 48.0, 8, monkeypatch)._build_run_sets(_qlats(96)))

    def test_an_explicit_window_still_refuses_a_ram_split(self, monkeypatch):
        """Unchanged: pinning a value is a promise the partition is not machine load."""
        m = self._model(96, 12.0, 6, monkeypatch)
        m._config["compute_parameters"]["forcing_parameters"]["max_loop_size"] = 24
        with pytest.raises(MemoryError):
            list(m._build_run_sets(_qlats(96)))


class TestTheCapIsAByteBudgetNotTwoRoundings:
    """Every emitted window must fit the modeled byte budget, on ANY budget.

    Two ceilings rounded the wrong way: 10 columns against a budget of 3.4 gave a
    4-column window, 18% over. Only a non-integral budget exposes it, which is why
    every divisible case in this file passed.
    """

    def _budget(self, monkeypatch, cols, columns_affordable):
        column_bytes = _required(1)
        monkeypatch.setattr(
            tm.psutil, "virtual_memory",
            lambda: SimpleNamespace(
                available=(column_bytes * columns_affordable) / 0.9),
        )
        return column_bytes

    @pytest.mark.parametrize("affordable", [3.4, 2.6, 5.1, 7.9, 1.5])
    def test_no_window_costs_more_than_the_budget(self, monkeypatch, affordable):
        column_bytes = self._budget(monkeypatch, 10, affordable)
        # No _scaling_da: the budget above is the no-DA column cost, and the model must
        # be sized the same way or the comparison is against the wrong number.
        sets = list(_model(nts_cols=10)._build_run_sets(_qlats(10)))
        budget = column_bytes * affordable
        for rs in sets:
            cost = rs["qlats"].shape[1] * column_bytes
            assert cost <= budget, (
                f"a {rs['qlats'].shape[1]}-column window costs {cost / budget:.2f}x "
                f"the budget of {affordable} columns"
            )

    def test_an_exhausted_cgroup_raises_rather_than_dividing_by_zero(self, monkeypatch):
        """A cgroup at or over its limit leaves no budget at all.

        That went straight into ceil(required / available). Guessing "one column fits"
        would be a guess about a budget the kernel has already refused.
        """
        monkeypatch.setattr(tm, "job_memory_headroom", lambda *a, **k: (0, "cgroup"))
        with pytest.raises(MemoryError, match="no memory budget left"):
            list(_model(nts_cols=10)._build_run_sets(_qlats(10)))


class TestTheOutputCadenceReachesTheWindow:
    """stream_output_time enlarges the ROUTING window, on this driver too.

    One window writes one output part, so a window under the requested output span
    cannot fill one. The CLI enlarged for it and this driver did not, so the same
    config routed 48-column windows under -V5 and 24 here; under a nonzero DA span
    that is a difference in discharge, not just in file boundaries.
    """

    def _model_with_output(self, cols, hours):
        m = _model(nts_cols=cols)
        m._config["output_parameters"] = {
            "stream_output": {"stream_output_time": hours}
        }
        return m

    def test_a_longer_output_span_widens_the_window(self, plenty_of_memory):
        # 48 h of output against a 24-column preference: the window follows the output.
        sets = list(self._model_with_output(96, 48)._build_run_sets(_qlats(96)))
        assert [rs["qlats"].shape[1] for rs in sets] == [48, 48]

    def test_a_shorter_output_span_changes_nothing(self, plenty_of_memory):
        sets = list(self._model_with_output(96, 6)._build_run_sets(_qlats(96)))
        assert [rs["qlats"].shape[1] for rs in sets] == [AUTO_WINDOW] * 4

    def test_the_whole_run_sentinel_is_not_a_width(self, plenty_of_memory):
        """-1 means one file for the whole run, not a one-column window."""
        sets = list(self._model_with_output(96, -1)._build_run_sets(_qlats(96)))
        assert [rs["qlats"].shape[1] for rs in sets] == [AUTO_WINDOW] * 4


class TestTheEstimatorReadsTheLag:
    """The lag's arrays must reach the SIZING, not just per_element_bytes.

    Deleting the lag flag from the estimator left every direct-call test green, which
    would ship a model that under-reads exactly the runs whose widest window the lag
    forces. lag_window_h is zeroed here so the span stays 0 and only the memory term
    differs; otherwise the lag would widen the window for an unrelated reason.
    """

    def _model_at(self, lag, monkeypatch, columns_affordable=6.0):
        m = _model(nts_cols=48)
        m._scaling_da = SimpleNamespace(
            innovation_spread_h=0.0, travel_time_lag=lag, lag_window_h=0.0
        )
        column_bytes = _required(1, scaling=True, lag=lag)
        monkeypatch.setattr(
            tm.psutil, "virtual_memory",
            lambda: SimpleNamespace(available=(_required(1, scaling=True)
                                               * columns_affordable) / 0.9),
        )
        return m, column_bytes

    def test_the_lag_narrows_the_window_at_the_same_memory(self, monkeypatch):
        off, _ = self._model_at(False, monkeypatch)
        off_widths = [rs["qlats"].shape[1] for rs in off._build_run_sets(_qlats(48))]
        on, _ = self._model_at(True, monkeypatch)
        on_widths = [rs["qlats"].shape[1] for rs in on._build_run_sets(_qlats(48))]
        assert max(on_widths) < max(off_widths), (
            f"lag off gave {off_widths}, lag on {on_widths}: the lag's Courant block "
            "and trace did not reach the sizing"
        )

class TestTheWarningNamesTheRightCause:
    """Two different things narrow a window, and they need different warnings.

    The cadence text fired on a memory-capped run too, telling the operator discharge
    depended on update length while the next line said machine load. Only one of those
    is the reason, and under auto the machine-load one means the run is not
    reproducible, which is worth saying outright.
    """

    def test_a_short_update_is_blamed_on_the_update(self, plenty_of_memory, caplog):
        m = _model(nts_cols=8)
        m._scaling_da = SimpleNamespace(innovation_spread_h=1.0, travel_time_lag=False)
        with caplog.at_level("WARNING"):
            list(m._build_run_sets(_qlats(8)))
        assert "depends on the update length" in caplog.text
        assert "available memory, not the configuration" not in caplog.text

    def test_a_memory_cap_is_blamed_on_memory(self, monkeypatch, caplog):
        m = _model(nts_cols=96)
        m._scaling_da = SimpleNamespace(innovation_spread_h=1.0, travel_time_lag=False)
        monkeypatch.setattr(
            tm.psutil, "virtual_memory",
            lambda: SimpleNamespace(available=(_required(1, scaling=True) * 16) / 0.9),
        )
        with caplog.at_level("WARNING"):
            list(m._build_run_sets(_qlats(96)))
        assert "available memory, not the configuration" in caplog.text
        assert "depends on the update length" not in caplog.text
