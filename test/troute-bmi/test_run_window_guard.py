"""The run-window guard must refuse a short update only when the partition matters.

Under the scaling DA the window partition can be part of the result, so
``_build_run_sets`` refuses to shrink the configured ``max_loop_size`` silently.
That is only true when the DA has a TIME span: with ``innovation_spread_h`` at 0
and the lag off, the spread is output-only and per timestep, so a shorter window
is bit-identical (pinned in test/troute-nwm/test_scaling_da_in_kernel.py).

The case that forced this: the NWM Standard AnA is 3 hours, so 3 forcing columns,
against a much longer ``max_loop_size``. Refusing there made the operational
config shipped in test/operational_configurations/ unable to run the DA at all.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest
from troute_nwm_bmi.troute_model import Model

_COLUMNS = ["202001010100", "202001010200", "202001010300"]  # 3 hourly columns


class _NetworkStub:
    t0 = datetime(2020, 1, 1, 0, 0)


class _ScalingDAStub:
    def __init__(self, spread_h: float, lag: bool = False):
        self.innovation_spread_h = spread_h
        self.travel_time_lag = lag
        self.lag_window_h = 48.0


def _model(max_loop_size: int, scaling_da: _ScalingDAStub | None, cols: int = 3) -> Model:
    model = Model.__new__(Model)  # skip __init__ (no config/network build)
    model._config = {
        "compute_parameters": {
            "forcing_parameters": {
                "max_loop_size": max_loop_size,
                "qts_subdivisions": 12,
                "dt": 300,
                "nts": cols * 12,
            }
        }
    }
    model.dt = 300
    model._network = _NetworkStub()
    model._scaling_da = scaling_da
    return model


def _qlats(cols: int = 3) -> pd.DataFrame:
    times = [(datetime(2020, 1, 1, 1, 0) + pd.Timedelta(hours=i)).strftime("%Y%m%d%H%M")
             for i in range(cols)]
    return pd.DataFrame(1.0, index=[10, 20], columns=times)


def test_a_short_update_runs_when_the_da_has_no_span():
    """Standard AnA shape: 3 columns against a max_loop_size of 24."""
    runs = list(_model(24, _ScalingDAStub(0.0))._build_run_sets(_qlats()))
    assert len(runs) == 1, f"expected one window over 3 columns, got {len(runs)}"
    assert runs[0]["nts"] == 3 * 12  # routing timesteps, not columns


def test_a_short_update_is_reported_not_refused(caplog):
    """With a forward average the update length reaches the result, so an operator
    who configured a longer window is told what actually ran.

    Not an error: a window longer than the update cannot constrain memory either, so
    refusing stopped a run over a setting that was doing nothing.
    """
    model = _model(24, _ScalingDAStub(2.0))
    with caplog.at_level("WARNING"):
        runs = list(model._build_run_sets(_qlats()))
    assert [r["nts"] for r in runs] == [3 * 12]
    assert "operates over 3" in caplog.text


def test_a_span_that_fits_the_update_runs():
    """The span is only a problem when the update cannot cover it.

    3 columns at a window of 2 leaves a 1-column remainder, under the 2-column span,
    so it folds into the window before it instead of being routed as a window that
    reads across a boundary it cannot supply.
    """
    runs = list(_model(2, _ScalingDAStub(2.0))._build_run_sets(_qlats()))
    assert len(runs) == 1, f"expected the remainder folded in, got {len(runs)}"
    assert [r["nts"] for r in runs] == [3 * 12]


def test_no_assimilation_is_unaffected():
    """Without the DA a short update was always served; keep it that way."""
    runs = list(_model(24, None)._build_run_sets(_qlats()))
    assert len(runs) == 1


def test_the_update_da_run_spans_every_window():
    """The BMI driver must hand the scaling injection one run-spanning list.

    A per-window list makes the injected observations depend on max_loop_size,
    because the reader's gap fill is non-local. Pins the wiring, not just the helper.
    """
    model = _model(24, _ScalingDAStub(0.0))
    windows = [{"n": 1}, {"n": 2}, {"n": 3}]
    per_window = {
        1: {"usgs_timeslice_files": ["a.ncdf", "b.ncdf"]},
        2: {"usgs_timeslice_files": ["b.ncdf", "c.ncdf"]},
        3: {"usgs_timeslice_files": ["c.ncdf", "d.ncdf"]},
    }
    model._window_da_run = lambda run: per_window[run["n"]]  # type: ignore[assignment]

    spanning = model._update_da_run(windows)
    assert spanning["usgs_timeslice_files"] == ["a.ncdf", "b.ncdf", "c.ncdf", "d.ncdf"]


class TestAutoYieldsToTheUpdate:
    """An explicit window is a promise the DA will operate at that width, so it refuses
    to shrink. Automatic sizing promises nothing, so it must adapt to whatever the
    driver supplies rather than fail a run that is correct as a single window.
    """

    def test_auto_serves_an_update_that_covers_the_span(self):
        # 18 columns against a 12 h span: one window, span resolved, nothing to refuse.
        # Demanding the 24-column preference here failed a correct run.
        runs = list(_model(0, _ScalingDAStub(12.0), cols=18)._build_run_sets(_qlats(18)))
        assert [r["nts"] for r in runs] == [18 * 12]

    def test_auto_discloses_the_cadence_like_an_explicit_value(self, caplog):
        """The disclosed fact is equally true under auto, so it must be said there.

        Each update closes its own final window, so the same run split into 18 hour
        updates differs from one split into 96 hour updates near every boundary. The
        default arm must not be the quiet one.
        """
        with caplog.at_level("WARNING"):
            list(_model(0, _ScalingDAStub(12.0), cols=18)._build_run_sets(_qlats(18)))
        assert "automatic max_loop_size" in caplog.text
        assert "operates over 18" in caplog.text

    def test_auto_still_refuses_an_update_shorter_than_the_span(self):
        model = _model(0, _ScalingDAStub(12.0), cols=6)
        with pytest.raises(ValueError, match="span of 12"):
            list(model._build_run_sets(_qlats(6)))

    def test_auto_does_not_name_a_knob_nobody_set(self):
        model = _model(0, _ScalingDAStub(12.0), cols=6)
        with pytest.raises(ValueError) as excinfo:
            list(model._build_run_sets(_qlats(6)))
        assert "lower max_loop_size" not in str(excinfo.value)

    def test_an_explicit_window_runs_and_says_what_it_ran(self, caplog):
        """An explicit value longer than the update is inert, so it warns, not raises.

        The hard error is reserved for the wall auto has too: an update that cannot
        cover the span at all, which no choice of window escapes.
        """
        model = _model(24, _ScalingDAStub(12.0), cols=18)
        with caplog.at_level("WARNING"):
            runs = list(model._build_run_sets(_qlats(18)))
        assert [r["nts"] for r in runs] == [18 * 12]
        assert "max_loop_size is 24" in caplog.text

    def test_an_explicit_window_still_hits_the_span_wall(self):
        model = _model(24, _ScalingDAStub(12.0), cols=6)
        with pytest.raises(ValueError, match="span of 12"):
            list(model._build_run_sets(_qlats(6)))


class TestTheUpdateCadenceIsDisclosed:
    """Every window guard is about windows WITHIN one update. Nothing crosses between
    update() calls, so a chunked run's boundaries are result-bearing and unguarded.

    Not fixable here: carrying the halo would withhold an update's output until the
    next arrives. So it is said out loud, once, when the second update makes it true.
    """

    def _model_with_span(self, span_h, routed, warned=False):
        model = _model(0, _ScalingDAStub(span_h))
        model._has_routed = routed
        if warned:
            model._warned_cadence = True
        return model

    def test_the_first_update_is_silent(self, caplog):
        with caplog.at_level("WARNING"):
            self._model_with_span(12.0, routed=False)._warn_cross_update_cadence()
        assert "more than one update" not in caplog.text

    def test_the_second_update_says_so(self, caplog):
        with caplog.at_level("WARNING"):
            self._model_with_span(12.0, routed=True)._warn_cross_update_cadence()
        assert "more than one update" in caplog.text
        assert "span of 12" in caplog.text

    def test_it_is_said_once_not_every_update(self, caplog):
        model = self._model_with_span(12.0, routed=True)
        with caplog.at_level("WARNING"):
            for _ in range(5):
                model._warn_cross_update_cadence()
        assert caplog.text.count("more than one update") == 1

    def test_a_zero_span_run_is_silent(self, caplog):
        """At zero span the partition does not reach the result, so nor does the
        update cadence: multi-window equivalence measures 0.0."""
        with caplog.at_level("WARNING"):
            self._model_with_span(0.0, routed=True)._warn_cross_update_cadence()
        assert "more than one update" not in caplog.text

    def test_no_assimilation_is_silent(self, caplog):
        model = _model(0, None)
        model._has_routed = True
        with caplog.at_level("WARNING"):
            model._warn_cross_update_cadence()
        assert "more than one update" not in caplog.text


def test_run_actually_calls_the_cadence_disclosure():
    """Pins the CALL SITE, not just the method.

    Deleting the call from run() left every direct-call test green, which would have
    shipped a disclosure nobody ever sees. Read by source because exercising run()
    needs a built network.
    """
    import inspect

    from troute_nwm_bmi.troute_model import Model

    assert "_warn_cross_update_cadence()" in inspect.getsource(Model.run)
