"""One config must produce one window partition, on the CLI and under the BMI.

The two drivers each had their own chunking. The CLI filled to ``max_loop_size`` and
folded a short final remainder into its neighbor; the BMI spread the update evenly. On
the same config that is ``[24, 24, 24, 28]`` against ``[20, 20, 20, 20, 20]``, and the
folded window is wider than the memory the caller sized for.

Both now call the functions here, so the partition is shared by construction rather
than by two implementations agreeing.
"""

from __future__ import annotations

import inspect
from itertools import pairwise

import pytest

from troute.window_plan import AUTO_WINDOW, plan_windows, resolve_window


class TestResolveWindow:
    def test_zero_is_automatic(self):
        assert resolve_window(0) == AUTO_WINDOW
        assert resolve_window(None) == AUTO_WINDOW

    def test_an_explicit_value_is_kept(self):
        assert resolve_window(9) == 9

    def test_the_span_raises_either_form(self):
        assert resolve_window(0, 48) == 48
        assert resolve_window(9, 48) == 48

    def test_a_span_under_the_window_changes_nothing(self):
        assert resolve_window(24, 12) == 24

    def test_never_zero(self):
        assert resolve_window(0, 0) >= 1


class TestPlanWindows:
    @pytest.mark.parametrize(
        ("n", "window", "span"),
        [(100, 24, 12), (96, 24, 0), (47, 24, 24), (3, 24, 0), (1, 24, 0),
         (144, 24, 12), (25, 24, 12), (48, 48, 48), (7, 3, 3)],
    )
    def test_bounds_tile_the_run_without_gap_or_overlap(self, n, window, span):
        bounds = plan_windows(n, window, span)
        assert bounds[0][0] == 0
        assert bounds[-1][1] == n
        assert all(a[1] == b[0] for a, b in pairwise(bounds))

    @pytest.mark.parametrize(
        ("n", "window", "span"),
        [(100, 24, 12), (96, 24, 0), (144, 24, 12), (25, 24, 12), (7, 3, 0)],
    )
    def test_no_window_exceeds_the_requested_width(self, n, window, span):
        """The fold this replaced could double it: 47 columns at 24 became one of 47."""
        widths = [stop - start for start, stop in plan_windows(n, window, span)]
        assert max(widths) <= window

    @pytest.mark.parametrize(("n", "window", "span"), [(47, 24, 24), (30, 24, 20)])
    def test_the_one_case_that_may_exceed_it_is_the_single_window(self, n, window, span):
        """When no split holds every window at the span, one window is the only
        artifact-free answer, and it is necessarily wider than asked for.

        Callers with a memory ceiling must notice this: it is the single case where
        the partition is wider than what they sized for. Both drivers warn, and the
        BMI raises when the ceiling cannot hold it.
        """
        bounds = plan_windows(n, window, span)
        assert len(bounds) == 1
        assert bounds[0][1] - bounds[0][0] == n > window

    @pytest.mark.parametrize(
        ("n", "window", "span"),
        [(100, 24, 12), (144, 24, 12), (25, 24, 12), (47, 24, 24), (7, 3, 3)],
    )
    def test_no_window_falls_under_the_span(self, n, window, span):
        widths = [stop - start for start, stop in plan_windows(n, window, span)]
        assert min(widths) >= span or len(widths) == 1

    def test_an_empty_run_has_no_windows(self):
        assert plan_windows(0, 24, 12) == []

    def test_widths_differ_by_at_most_one(self):
        widths = [stop - start for start, stop in plan_windows(100, 24, 12)]
        assert max(widths) - min(widths) <= 1


def test_both_drivers_call_the_shared_rule():
    """A regression guard: neither driver may grow its own chunking again.

    Reading the source, not the behavior, because the alternative is a full network
    build on one side and a BMI model on the other.
    """
    from troute_nwm_bmi import troute_model

    from troute import AbstractNetwork

    cli = inspect.getsource(AbstractNetwork.AbstractNetwork.build_forcing_sets)
    bmi = inspect.getsource(troute_model.Model._build_run_sets)
    for name, src in (("-V5 build_forcing_sets", cli), ("BMI _build_run_sets", bmi)):
        assert "plan_windows(" in src, f"{name} no longer uses the shared split"
        assert "resolve_window(" in src, f"{name} no longer uses the shared resolution"


class TestTheOutputCadenceIsPartOfTheSharedRule:
    """One window writes one output part, so stream_output_time changes the ROUTING
    partition, not just the file layout.

    The CLI enlarged for it and the BMI did not. With stream_output_time 48 the same
    config routed 48-column windows under -V5 and 24 under the BMI, and under a
    nonzero DA span that is a difference in discharge, not just in file boundaries.
    """

    def test_the_output_cadence_can_enlarge_the_window(self):
        assert resolve_window(24, 0, 48) == 48

    def test_it_does_not_shrink_a_larger_window(self):
        assert resolve_window(72, 0, 48) == 72

    def test_the_span_and_the_cadence_both_apply(self):
        assert resolve_window(24, 60, 48) == 60
        assert resolve_window(24, 12, 48) == 48

    def test_zero_means_no_output_constraint(self):
        assert resolve_window(24, 0, 0) == 24


def test_both_drivers_apply_the_same_three_inputs():
    """A regression guard on the divergence itself.

    Reading the source because the alternative is a full network build on one side and
    a BMI model on the other; what it pins is that neither driver resolves a window
    from fewer inputs than the other.
    """
    from troute_nwm_bmi import troute_model

    from troute import AbstractNetwork


    cli = inspect.getsource(AbstractNetwork.AbstractNetwork.build_forcing_sets)
    bmi = inspect.getsource(troute_model.Model._build_run_sets)
    for name, src in (("-V5 build_forcing_sets", cli), ("BMI _build_run_sets", bmi)):
        assert "resolve_window(" in src, f"{name} no longer uses the shared resolution"
        assert "plan_windows(" in src, f"{name} no longer uses the shared split"


class TestGivingUpTheCapMinimally:
    """Yield the CAP by as little as possible rather than abandoning the split.

    Collapsing to one window turned a 2161-column run at span 60 into a single
    2161-column window, ~120 GB, where 36 windows of 60 and 61 hold the span. The
    regime is window == span, which every travel-time-lag run gets.
    """

    @pytest.mark.parametrize("n", [100, 2161, 121, 7, 1000])
    def test_the_split_survives_a_span_equal_to_the_window(self, n):
        span = 60 if n > 200 else 3
        bounds = plan_windows(n, span, span)
        widths = [b - a for a, b in bounds]
        assert sum(widths) == n
        assert min(widths) >= span, f"{n}: {widths} falls under the span"
        # The concession is MINIMAL: one more window would drop under the span, so
        # these are the narrowest windows that hold it.
        count = len(bounds)
        assert count == 1 or n // (count + 1) < span, (
            f"{n}: {count} windows of {min(widths)}..{max(widths)}, but "
            f"{count + 1} would still hold the span"
        )

    def test_one_column_of_run_length_does_not_flip_the_partition(self):
        a = [y - x for x, y in plan_windows(2160, 60, 60)]
        b = [y - x for x, y in plan_windows(2161, 60, 60)]
        assert abs(len(a) - len(b)) <= 1, (
            f"2160 -> {len(a)} windows but 2161 -> {len(b)}"
        )

    def test_a_genuinely_unsplittable_run_is_still_one_window(self):
        # 47 columns at span 24: two windows leave one at 23, so one window is the
        # only partition that holds, and it is the caller's job to afford it.
        assert plan_windows(47, 24, 24) == [(0, 47)]
