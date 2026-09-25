"""Unit tests for diversion data assimilation (Old River Control Structure).

The integration test in ``test/nhf/old_river`` needs a subset hydrofabric, NWM
retrospective forcing and USGS timeslices, so it skips wherever that data has not
been built. These tests cover the same machinery with no external data, so the
mechanism is guarded on every run.

What the scheme does, and what therefore has to hold:

The observed discharge at the diversion gage is SUBTRACTED from the donor flowpath
inside the routing kernel. Nothing adds it to the receiving river in code: the gage
sits on a headwater flowpath of the receiving system, so ordinary streamflow
nudging imposes the observed discharge there and the existing topology routes it
downstream. The gage's row is read from the TimeSlices and held for the persistence
horizon past the end of its record.
"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from troute import nhd_network
from troute.DataAssimilation import (
    NudgingDA,
    _DiversionRow,
    _diversion_seed_at,
    _fill_diversion_row,
    _last_report_before,
    _read_diversion_observations,
    _seed_from_record,
    new_diversion_applied,
)
from troute.routing.compute import RoutingResultsCollection, _resolve_diversion_da

DIVERSION_GAGE = "07381482"  # Old River Outflow Channel
DONOR_FP_ID = 1270479816524705  # Mississippi link at the control structure
GAGE_LINK = 1269985531956909  # routing link the gage resolves to


@pytest.fixture
def donor_and_gage() -> tuple[int, int]:
    """The (donor segment, gage link) pair the kernel map is built from."""
    return DONOR_FP_ID, GAGE_LINK


@pytest.fixture
def job_reaches(donor_and_gage) -> np.ndarray:
    """A compute job's sorted segment index that contains the donor segment."""
    donor, _ = donor_and_gage
    return np.sort(np.array([donor - 10, donor, donor + 10, donor + 20], dtype="int64"))


@pytest.fixture
def usgs_df_sub(donor_and_gage) -> pd.DataFrame:
    """Observations subset for a job, indexed by routing link like the real one."""
    _, gage_link = donor_and_gage
    return pd.DataFrame(
        [[100.0, 110.0, 120.0]],
        index=pd.Index([gage_link], dtype="int64"),
        columns=pd.date_range("2011-04-14", periods=3, freq="h"),
    )


class TestResolveDiversionDa:
    """Translating the global diversion map to job-local kernel indices."""

    def test_maps_donor_position_to_gage_row(self, donor_and_gage, job_reaches, usgs_df_sub):
        donor, gage_link = donor_and_gage
        kernel_map = _resolve_diversion_da({donor: gage_link}, job_reaches, usgs_df_sub)
        # key is the donor's POSITION in this job's index, value the gage's row
        expected_pos = int(np.searchsorted(job_reaches, donor))
        assert kernel_map == {expected_pos: 0}

    def test_empty_map_short_circuits(self, job_reaches, usgs_df_sub):
        assert _resolve_diversion_da({}, job_reaches, usgs_df_sub) == {}

    def test_donor_outside_job_is_skipped(self, donor_and_gage, usgs_df_sub):
        """A job that does not contain the donor must not divert anything.

        Guards the searchsorted lookup: without the identity check, a missing donor
        lands on the insertion point and would divert some unrelated segment.
        """
        donor, gage_link = donor_and_gage
        other_job = np.array([donor + 1000, donor + 2000], dtype="int64")
        assert _resolve_diversion_da({donor: gage_link}, other_job, usgs_df_sub) == {}

    def test_donor_present_but_gage_missing_warns(
        self, donor_and_gage, job_reaches, caplog
    ):
        """The donor is here but its observations are not, so no diversion is applied.

        This silently left the donor carrying water that should have been
        transferred; it must be reported.
        """
        donor, gage_link = donor_and_gage
        empty = pd.DataFrame(index=pd.Index([], dtype="int64"))
        with caplog.at_level(logging.WARNING):
            assert _resolve_diversion_da({donor: gage_link}, job_reaches, empty) == {}
        assert "gage link" in caplog.text


class TestFrameOnTheRoutingGrid:
    """The diversion's row is built on the window's routing grid."""

    @pytest.fixture
    def network(self):
        class _Network:
            t0 = pd.Timestamp("2011-06-01 00:00")
            _diversion_site_to_node = {DIVERSION_GAGE: GAGE_LINK}

        return _Network()

    @pytest.fixture
    def params(self) -> dict:
        return {"diversion_gage_crosswalk": {DONOR_FP_ID: DIVERSION_GAGE}}

    @pytest.mark.parametrize("dt,nts", [(300, 12), (60, 20), (900, 8)])
    def test_columns_follow_routing_timestep(self, network, params, dt, nts):
        """One column per routing timestep, not a fixed 5 minute grid.

        The kernel indexes this frame as ``usgs_values[gage_i, timestep]`` with the
        ROUTING step, so a hardcoded grid ran out early for a short dt and advanced
        too slowly for a long one.
        """
        out, _ = _fill_diversion_row(pd.DataFrame(), params, network, {"dt": dt, "nts": nts})
        assert out.shape[1] == nts + 1
        spacing = pd.Series(out.columns).diff().dropna().unique()
        assert list(spacing) == [pd.Timedelta(seconds=dt)]

    def test_hold_is_logged(self, network, params, caplog):
        """A held value is not an observation; a run must say how much it used."""
        idx = pd.date_range(network.t0, periods=4, freq=pd.Timedelta(seconds=300))
        existing = pd.DataFrame(
            [[7.0, np.nan, np.nan, np.nan]], index=pd.Index([GAGE_LINK], dtype="int64"),
            columns=idx,
        )
        with caplog.at_level(logging.WARNING):
            _fill_diversion_row(existing, params, network, {"dt": 300, "nts": 3})
        assert "held at the last observation" in caplog.text

    def test_gage_without_routing_link_is_reported_not_raised(self, network, params, caplog):
        """An unresolved gage used to raise KeyError mid-run."""
        network._diversion_site_to_node = {}
        with caplog.at_level(logging.WARNING):
            out, _ = _fill_diversion_row(pd.DataFrame(), params, network, {"dt": 300, "nts": 3})
        assert "no routing link" in caplog.text
        assert out.empty or GAGE_LINK not in out.index


class TestMassBalanceMonitor:
    """Reporting transferred water the donor could not supply.

    The transfer conserves mass by construction, since the same observed value is
    removed from the donor and imposed at the receiving headwater. The zero-flow
    clamp is the one exception: when the observed diversion exceeds the routed donor
    flow the donor is floored at zero and gives up less than the receiving river
    gains. Capping the transfer at the routed flow would cut across the parallel
    decomposition, so this is monitored rather than enforced and every occurrence
    has to be reported with the volume involved.
    """

    @staticmethod
    def _results(donor_flow):
        """A results collection with one donor reach carrying *donor_flow*."""
        n = len(donor_flow)
        flow = np.zeros((1, 4 * n), dtype="float32")
        flow[0, 0::4] = donor_flow

        class _R:
            ids = np.array([DONOR_FP_ID], dtype="int64")

        r = _R()
        r.flow = flow
        return [r]

    @staticmethod
    def _obs(values):
        """Observations on the kernel's grid: column 0 is t0, column j+1 feeds flow[j]."""
        return pd.DataFrame(
            [values],
            index=pd.Index([GAGE_LINK], dtype="int64"),
            columns=pd.date_range("2011-04-14", periods=len(values), freq="h"),
        )

    def test_clamped_donor_reports_the_unmatched_volume(self, caplog):
        from troute.routing.compute import _warn_diverted_mass_imbalance

        # Two timesteps clamped to zero while 100 and 200 m3/s were being diverted.
        # The observation frame carries the LEADING INITIAL-CONDITION COLUMN, which is the
        # convention the kernel reads: usgs_values[:, 0] seeds the state at t0
        # (mc_reach.pyx:461) and the diversion subtraction indexes usgs_values[:, timestep]
        # with timestep running from 1 (mc_reach.pyx:890). The routed flow has that slot
        # removed, so flow[k] pairs with column k+1. This test previously passed a
        # three-wide frame for three routed timesteps, which encoded the off-by-one it was
        # meant to guard and left the final timestep with no observation at all.
        with caplog.at_level(logging.WARNING):
            _warn_diverted_mass_imbalance(
                self._results([0.0, 50.0, 0.0]),
                {DONOR_FP_ID: GAGE_LINK},
                self._obs([0.0, 100.0, 10.0, 200.0]),
                dt=300,
            )
        assert "Mass is not conserved" in caplog.text
        # Both clamped timesteps, and the volume from THEIR columns: (100 + 200) * dt.
        assert "at 2 of 3 timestep(s)" in caplog.text
        assert f"{(100.0 + 200.0) * 300:.3g}" in caplog.text
        assert "2 of 3" in caplog.text
        # upper bound on created water: (100 + 200) m3/s over one 300 s step each
        assert "9e+04" in caplog.text or "90000" in caplog.text

    def test_observation_slice_matches_the_kernel(self, caplog):
        """flow[j] must be paired with observation column j + 1, not column j.

        The kernel runs timestep 1..nts subtracting usgs_values[gage_i, timestep] and
        returns flowveldepth[:, 1:], so column 0 (t0, the initial-condition slot) is
        never subtracted from anything. Pairing from column 0 shifted every clamp
        report one timestep early: here it would blame the second step, where nothing
        was requested, and miss the first, where 100 m3/s was taken from a dry donor.
        """
        from troute.routing.compute import _warn_diverted_mass_imbalance

        with caplog.at_level(logging.WARNING):
            _warn_diverted_mass_imbalance(
                self._results([0.0, 500.0]),
                {DONOR_FP_ID: GAGE_LINK},
                self._obs([np.nan, 100.0, np.nan]),
                dt=300,
            )
        assert "1 of 2" in caplog.text
        assert "3e+04" in caplog.text or "30000" in caplog.text

    def test_observations_shorter_than_the_run_are_skipped_not_truncated(self, caplog):
        """Too few columns to align: report it and skip, rather than compare a subset.

        A frame one column short could be an nts-wide series starting at t0 + dt as
        easily as a truncated nts+1 one, and the two need opposite offsets. Comparing
        over whatever overlaps would silently pick one and report a mass balance that
        may be attributed to the wrong timesteps.
        """
        from troute.routing.compute import _warn_diverted_mass_imbalance

        with caplog.at_level(logging.WARNING):
            _warn_diverted_mass_imbalance(
                self._results([0.0, 0.0, 0.0, 0.0]),
                {DONOR_FP_ID: GAGE_LINK},
                self._obs([np.nan, 100.0]),
                dt=300,
            )
        assert "cannot be aligned" in caplog.text
        assert "Mass is not conserved" not in caplog.text

    def test_dry_reach_with_nothing_to_divert_is_not_reported(self, caplog):
        """Zero flow is only a mass-balance problem if a transfer was requested.

        The previous check keyed on flow == 0 alone, so a genuinely dry donor with
        no observation was reported as though water had been created.
        """
        from troute.routing.compute import _warn_diverted_mass_imbalance

        with caplog.at_level(logging.WARNING):
            _warn_diverted_mass_imbalance(
                self._results([0.0, 0.0]),
                {DONOR_FP_ID: GAGE_LINK},
                self._obs([-1.0, np.nan, 0.0]),
                dt=300,
            )
        assert "Mass is not conserved" not in caplog.text

    def test_donor_that_supplied_the_transfer_is_silent(self, caplog):
        from troute.routing.compute import _warn_diverted_mass_imbalance

        with caplog.at_level(logging.WARNING):
            _warn_diverted_mass_imbalance(
                self._results([500.0, 400.0]),
                {DONOR_FP_ID: GAGE_LINK},
                self._obs([-1.0, 100.0, 100.0]),
                dt=300,
            )
        assert "Mass is not conserved" not in caplog.text


class TestDonorIsItsOwnReach:
    """The donor has to end a reach for the subtraction to reach the river.

    ``compute_reach_kernel`` routes a whole reach in one call, feeding each
    segment's outflow straight into the next as ``quc``; the diversion is
    subtracted afterwards, when the results are copied back out. A donor in the
    middle of a reach therefore hands its downstream neighbors the undiverted
    flow, and the water is only removed from the donor's own reported row. Worse,
    the next timestep is inconsistent rather than merely uncorrected: ``qup`` comes
    from the donor's reduced previous value while ``quc`` comes from its
    undiverted current one.

    ``compute_nhd_routing_v02`` therefore folds the donors into the plan's split
    set. These tests pin the property that makes that work.
    """

    # 5 -> 4 -> 3 -> 2 -> 1, a plain mainstem with no junctions.
    RCONN = {1: [2], 2: [3], 3: [4], 4: [5], 5: []}
    DONOR = 3

    def _reaches(self, split_nodes):
        path_func = partial(
            nhd_network.split_at_gages_and_junctions, split_nodes, self.RCONN
        )
        return nhd_network.dfs_decomposition(self.RCONN, path_func, source_nodes=[1])

    def test_unsplit_mainstem_is_one_reach(self):
        """Without a split the donor sits mid-reach, which is the broken case."""
        reaches = self._reaches(set())
        assert reaches == [[5, 4, 3, 2, 1]]
        assert reaches[0][-1] != self.DONOR

    def test_donor_becomes_a_single_segment_reach(self):
        """Split at the donor and it is its own reach, so it is that reach's tail.

        The kernel gathers a reach's inflow as flowveldepth[upstream_tail, timestep]
        at the current timestep, so once the donor is a tail the reduced value is
        exactly what the next reach downstream routes on.
        """
        reaches = self._reaches({self.DONOR})
        assert [self.DONOR] in reaches
        # and nothing downstream of it shares the reach
        for reach in reaches:
            if self.DONOR in reach:
                assert reach == [self.DONOR]

    def test_every_segment_still_routed_exactly_once(self):
        """Splitting must not drop or duplicate segments."""
        routed = [seg for reach in self._reaches({self.DONOR}) for seg in reach]
        assert sorted(routed) == [1, 2, 3, 4, 5]


class TestObservationGridAlignment:
    """The observation frame must sit on the kernel's timestep grid.

    ``build_da_sets`` reads timeslices from ``t0 - timeslice_lookback_hours`` so the
    interpolator has context before the run starts. Those padding columns used to
    survive into the frame the kernel indexes positionally, so with the default
    24 hour lookback every assimilated observation was read a day away from the
    step it was applied to.
    """

    T0 = pd.Timestamp("2023-04-02 00:00")
    DT = 300
    NTS = 4

    def _frame(self, start, periods):
        return pd.DataFrame(
            [np.arange(periods, dtype=float)],
            index=pd.Index([GAGE_LINK], dtype="int64"),
            columns=pd.date_range(start, periods=periods, freq=pd.Timedelta(seconds=self.DT)),
        )

    def _align(self, df):
        from troute.routing.compute import _align_obs_to_model_steps

        return _align_obs_to_model_steps(df, self.T0, self.DT, self.NTS)

    def test_lookback_padding_is_dropped(self):
        """Column 0 must land on t0, not on t0 minus the lookback."""
        # two steps of pad ahead of t0, then the run window
        padded = self._frame(self.T0 - pd.Timedelta(seconds=2 * self.DT), 2 + self.NTS + 1)
        out = self._align(padded)
        assert out.columns[0] == self.T0
        assert out.shape[1] == self.NTS + 1
        # the value the kernel now reads at model step 1 is the observation at t0+dt
        assert out.iloc[0, 1] == padded.loc[GAGE_LINK, self.T0 + pd.Timedelta(seconds=self.DT)]

    def test_already_aligned_frame_is_returned_untouched(self):
        aligned = self._frame(self.T0, self.NTS + 1)
        assert self._align(aligned) is aligned

    def test_short_window_is_padded_with_nan_not_truncated(self):
        """A DA window ending before the run does must not shorten the grid.

        The kernel sizes gage_maxtimestep from this frame and skips NaN, so missing
        steps have to be present and empty rather than absent.
        """
        out = self._align(self._frame(self.T0, 2))
        assert out.shape[1] == self.NTS + 1
        assert out.iloc[0, 2:].isna().all()

    def test_non_datetime_columns_of_the_right_width_pass_through(self):
        """The BMI array path can hand over positional columns already on the grid."""
        df = pd.DataFrame(
            [[1.0] * (self.NTS + 1)], index=pd.Index([GAGE_LINK], dtype="int64")
        )
        assert self._align(df) is df

    def test_positional_frame_of_the_wrong_width_is_refused(self):
        """Width is the only grid property checkable without timestamps.

        A positional frame used to pass through at ANY width, so the kernel read
        column j as t0 + j*dt regardless and assimilated observations at the wrong
        timesteps while still producing plausible discharge.
        """
        df = pd.DataFrame([[1.0, 2.0]], index=pd.Index([GAGE_LINK], dtype="int64"))
        with pytest.raises(ValueError, match="width is 2 where the kernel requires"):
            self._align(df)

    def test_empty_frame_passes_through(self):
        empty = pd.DataFrame()
        assert self._align(empty) is empty

    def test_window_that_misses_the_run_is_reported(self, caplog):
        """Silently assimilating nothing is the failure mode worth a warning."""
        stale = self._frame(self.T0 - pd.Timedelta(days=7), 3)
        with caplog.at_level(logging.WARNING):
            out = self._align(stale)
        assert out.notna().to_numpy().sum() == 0
        assert "no observation column lines up" in caplog.text

class TestMassImbalanceReportAlignment:
    """The clamp report must compare each routed flow against the RIGHT observation.

    ``usgs_df`` columns are the positional routing grid the kernel indexes as
    ``usgs_values[gage_i, timestep]``, and the kernel's timestep runs from 1 -- column 0 is
    the initial condition at ``t0``. The returned flow has that initial-condition slot
    removed, so ``flow[k]`` is routing timestep ``k+1`` and pairs with column ``k+1``.

    Slicing the observations from column 0 lines the two series up one timestep early. It
    does not change any routed result, but it mislabels which timesteps clamped and
    mis-sums the reported volume, which defeats the only purpose of the report.
    """

    DONOR, GAGE, DT = 101, 900, 300

    def _run(self, caplog, flow, obs):
        from troute.routing.compute import RoutingResults, _warn_diverted_mass_imbalance

        n = len(flow)
        arr = np.zeros((1, n * 4), dtype="float32")
        arr[0, 0::4] = flow
        results = [RoutingResults([np.array([self.DONOR]), arr])]
        usgs_df = pd.DataFrame([obs], index=pd.Index([self.GAGE], name="link"))
        with caplog.at_level(logging.WARNING):
            _warn_diverted_mass_imbalance(results, {self.DONOR: self.GAGE}, usgs_df, self.DT)
        return caplog.text

    def test_clamp_is_attributed_to_the_correct_timestep(self, caplog):
        """Only the timestep whose OWN observation asked for water may be reported.

        flow = [0, 5, 5]; the request sits in column 3, which is flow[2] and did NOT clamp.
        Reading from column 0 would pair flow[0]=0 with column 0's request and report a
        clamp that never happened.
        """
        text = self._run(caplog, flow=[0.0, 5.0, 5.0], obs=[7.0, 0.0, 0.0, 3.0])
        assert "clamped" not in text, "reported a clamp using the initial-condition column"

    def test_genuine_clamp_is_still_reported_with_the_right_volume(self, caplog):
        """flow[0]=0 pairs with column 1 (=4.0), so one clamp of 4.0 * dt must be reported."""
        text = self._run(caplog, flow=[0.0, 5.0, 5.0], obs=[99.0, 4.0, 0.0, 0.0])
        assert "clamped to zero flow at 1 of 3" in text
        assert f"{4.0 * self.DT:.3g}" in text

    def test_unalignable_observation_frame_is_skipped_not_guessed(self, caplog):
        """Too few columns to align: warn and skip rather than report something wrong."""
        text = self._run(caplog, flow=[0.0, 5.0, 5.0], obs=[4.0, 0.0])
        assert "cannot be aligned" in text
        assert "clamped to zero flow" not in text


class TestDiffusiveNudgingGate:
    """The diffusive DA switch must read the VALUE, not merely the key's presence.

    These are structural guard-rails, not behavioural tests: exercising
    compute_diffusive_routing end to end needs a full diffusive domain. They pin the
    two lines that made the bug, so a revert fails loudly.

    Paths are resolved from this file, never from the cwd -- the NHF integration cases
    chdir, so a relative path here passes alone and fails in the full suite.
    """

    REPO = Path(__file__).resolve().parents[2]

    def test_both_da_parameter_builders_always_set_the_key(self):
        """Which is why `in da_parameter_dict` was unconditionally true.

        nhd_network_utilities_v02 and DataAssimilation both do
        `da_parameter_dict["diffusive_streamflow_nudging"] = ...get(..., False)`, so
        the key exists even when the feature is off. Gating on presence therefore
        enabled diffusive DA on every diffusive run, and the frame it received had
        never been put on the kernel's timestep grid.
        """
        from troute.nhd_network_utilities_v02 import build_da_sets  # noqa: F401

        src = (self.REPO / "src/troute-network/troute/nhd_network_utilities_v02.py").read_text()
        assert 'da_parameter_dict["diffusive_streamflow_nudging"] = ' in src

        compute_src = (self.REPO / "src/troute-routing/troute/routing/compute.py").read_text()
        assert "if 'diffusive_streamflow_nudging' in da_parameter_dict:" not in compute_src
        assert "da_parameter_dict.get('diffusive_streamflow_nudging', False)" in compute_src

    def test_diffusive_frame_is_aligned_not_raw(self):
        """The diffusive branch must route its frame through the grid alignment."""
        compute_src = (self.REPO / "src/troute-routing/troute/routing/compute.py").read_text()
        assert (
            "diffusive_usgs_df = _align_obs_to_model_steps(usgs_df, t0, dt, nts)"
            in compute_src
        )


class TestHoldLastObservation:
    """Requirement 2.2.3.15 / use case 13.2: the last observation persists, flat.

    The fill works on the window's routing grid, causally: a cell takes the latest
    finite cell before it, the inherited seed covers cells before the row's first
    report, a cell with neither stays NaN, and a finite cell is never touched.
    """

    @pytest.fixture
    def network(self):
        class _Network:
            t0 = pd.Timestamp("2011-06-01 00:00")
            _diversion_site_to_node = {DIVERSION_GAGE: GAGE_LINK}

        return _Network()

    @pytest.fixture
    def params(self) -> dict:
        return {
            "diversion_gage_crosswalk": {DONOR_FP_ID: DIVERSION_GAGE},
            "diversion_persist_days": 11,
        }

    @staticmethod
    def _frame(network, values, other=None, freq="5min"):
        idx = pd.date_range(network.t0, periods=len(values), freq=freq)
        rows, index = [values], [GAGE_LINK]
        if other is not None:
            rows.append(other)
            index.append(GAGE_LINK + 1)
        return pd.DataFrame(rows, index=pd.Index(index, dtype="int64"), columns=idx, dtype=float)

    def test_row_is_extended_onto_the_routing_grid(self, network, params):
        """The reader's columns stop at the record; the window does not."""
        out, _ = _fill_diversion_row(
            self._frame(network, [7.0, 8.0, 9.0]), params, network, {"dt": 300}, nts=12
        )
        row = out.loc[GAGE_LINK]
        assert len(row) == 13
        np.testing.assert_allclose(row.iloc[3:].to_numpy(), 9.0)

    def test_hold_is_causal_across_an_interior_gap(self, network, params):
        out, _ = _fill_diversion_row(
            self._frame(network, [7.0, np.nan, np.nan, 9.0]), params, network,
            {"dt": 300}, nts=3,
        )
        np.testing.assert_allclose(out.loc[GAGE_LINK].to_numpy(), [7.0, 7.0, 7.0, 9.0])

    def test_seed_covers_cells_after_its_time_and_before_the_first_report(self, network, params):
        seed = {GAGE_LINK: (network.t0 + pd.Timedelta(minutes=10), 5.0)}
        out, _ = _fill_diversion_row(
            self._frame(network, [np.nan, np.nan, np.nan, np.nan, 8.0, np.nan]), params,
            network, {"dt": 300}, seed_in=seed, nts=5,
        )
        row = out.loc[GAGE_LINK].to_numpy()
        assert np.isnan(row[:2]).all(), "cells before the seed's time are not the seed's"
        np.testing.assert_allclose(row[2:], [5.0, 5.0, 8.0, 8.0])

    def test_newer_seed_outranks_a_stale_lookback_report(self, network, params):
        """A lookback report older than the seed must not hold past it."""
        seed = {GAGE_LINK: (network.t0 + pd.Timedelta(minutes=10), 9.0)}
        out, _ = _fill_diversion_row(
            self._frame(network, [1.0, np.nan, np.nan, np.nan, np.nan]), params, network,
            {"dt": 300}, seed_in=seed, nts=4,
        )
        np.testing.assert_allclose(out.loc[GAGE_LINK].to_numpy(), [1.0, 1.0, 9.0, 9.0, 9.0])

    def test_report_at_the_seeds_instant_wins(self, network, params):
        """A checkpoint's copy of an instant loses to the record's own cell there."""
        at = network.t0 + pd.Timedelta(minutes=5)
        seed = {GAGE_LINK: (at, 9.0)}
        out, rows = _fill_diversion_row(
            self._frame(network, [np.nan, 3.0, np.nan, np.nan, np.nan]), params, network,
            {"dt": 300}, seed_in=seed, nts=4,
        )
        np.testing.assert_allclose(out.loc[GAGE_LINK].to_numpy()[1:], 3.0)
        assert np.isnan(out.loc[GAGE_LINK].to_numpy()[0])
        assert _diversion_seed_at(rows, seed) == {GAGE_LINK: (at, 3.0)}

    def test_row_with_nothing_to_divert_is_reported(self, network, params, caplog):
        """A misnamed site or an absent station must not pass as a quiet run."""
        with caplog.at_level(logging.WARNING):
            _fill_diversion_row(
                self._frame(network, [np.nan] * 4), params, network, {"dt": 300}, nts=3
            )
        assert "no flow will be diverted" in caplog.text

    def test_hold_counts_cover_the_window_not_the_lookback(self, network, params, caplog):
        idx = pd.date_range(network.t0 - pd.Timedelta(minutes=10), periods=6, freq="5min")
        frame = pd.DataFrame([[1.0, 1.0, 1.0, np.nan, np.nan, np.nan]],
                             index=pd.Index([GAGE_LINK], dtype="int64"), columns=idx)
        with caplog.at_level(logging.WARNING):
            _fill_diversion_row(frame, params, network, {"dt": 300}, nts=3)
        assert "3 of 4 timesteps held" in caplog.text

    def test_seed_at_t0_fills_column_zero(self, network, params):
        """A restart at the last observation: its instant is column 0, and the kernel
        seeds the remembered subtraction from that column."""
        seed = {GAGE_LINK: (network.t0, 4.0)}
        out, _ = _fill_diversion_row(pd.DataFrame(), params, network, {"dt": 300},
                                     seed_in=seed, nts=2)
        np.testing.assert_allclose(out.loc[GAGE_LINK].to_numpy(), [4.0, 4.0, 4.0])

    def test_only_the_crosswalked_row_is_touched(self, network, params):
        out, _ = _fill_diversion_row(
            self._frame(network, [7.0, np.nan], other=[3.0, np.nan]), params, network,
            {"dt": 300}, nts=1,
        )
        assert np.isnan(out.loc[GAGE_LINK + 1].iloc[1])
        assert out.loc[GAGE_LINK].iloc[1] == 7.0

    def test_no_crosswalk_leaves_the_frame_alone(self, network, params):
        params["diversion_gage_crosswalk"] = {}
        frame = self._frame(network, [7.0, np.nan])
        out, rows = _fill_diversion_row(frame, params, network, {"dt": 300}, nts=1)
        assert out is frame
        assert rows == {}

    def test_hold_expires_after_the_horizon(self, network, params):
        """The same horizon idiom as the RFC DA: held for N days, then gone."""
        params["diversion_persist_days"] = 1
        out, _ = _fill_diversion_row(
            self._frame(network, [7.0] + [np.nan] * 3, freq="12h"), params, network,
            {"dt": 43200}, nts=3,  # 12 h steps: held at +12 h and +24 h, expired at +36 h
        )
        np.testing.assert_allclose(out.loc[GAGE_LINK].to_numpy()[:3], 7.0)
        assert np.isnan(out.loc[GAGE_LINK].to_numpy()[3])

    def test_zero_days_holds_nothing(self, network, params):
        params["diversion_persist_days"] = 0
        out, _ = _fill_diversion_row(
            self._frame(network, [7.0, np.nan, np.nan]), params, network, {"dt": 300}, nts=2
        )
        np.testing.assert_allclose(out.loc[GAGE_LINK].to_numpy()[0], 7.0)
        assert np.isnan(out.loc[GAGE_LINK].to_numpy()[1:]).all()

    def test_seed_at_decides_by_time_and_respects_tau(self, network, params):
        _, rows = _fill_diversion_row(
            self._frame(network, [np.nan, 6.0, np.nan]), params, network, {"dt": 300}, nts=2
        )
        report_time = network.t0 + pd.Timedelta(minutes=5)
        newer = {GAGE_LINK: (network.t0 + pd.Timedelta(hours=2), 1.0)}
        older = {GAGE_LINK: (network.t0 - pd.Timedelta(hours=2), 1.0)}
        assert _diversion_seed_at(rows, newer) == newer
        assert _diversion_seed_at(rows, older) == {GAGE_LINK: (report_time, 6.0)}
        assert _diversion_seed_at(rows, older, tau=network.t0) == older

    def test_partition_invariance_for_the_forecast_tail(self, network, params):
        """One window over 2N steps equals two windows of N with the seed handed over."""
        values = [7.0, 8.0, 9.0] + [np.nan] * 9  # record ends in the first half
        whole, _ = _fill_diversion_row(
            self._frame(network, values), params, network, {"dt": 300}, nts=11
        )
        first, rows = _fill_diversion_row(
            self._frame(network, values[:6]), params, network, {"dt": 300}, nts=5
        )

        class _Later:
            t0 = network.t0 + pd.Timedelta(minutes=30)
            _diversion_site_to_node = network._diversion_site_to_node

        second, _ = _fill_diversion_row(
            pd.DataFrame(), params, _Later(), {"dt": 300},
            seed_in=_diversion_seed_at(rows, {}), nts=5,
        )
        stitched = pd.concat([first.loc[GAGE_LINK].iloc[:6], second.loc[GAGE_LINK]])
        np.testing.assert_allclose(stitched.to_numpy(), whole.loc[GAGE_LINK].to_numpy())



def _kernel_result(ids, donors=(), applied=(), nts=2, with_diversion=True):
    """A kernel result tuple with empty DA elements and, optionally, element 11."""
    ids = np.asarray(ids, dtype=np.intp)
    n = len(ids)
    fl, ip = np.array([], dtype="float32"), np.array([], dtype=np.intp)
    r = [
        ids, np.zeros((n, nts * 4), dtype="float32"), 0,
        (ip, fl, fl), (ip, fl, fl, fl, fl), (ip, fl, fl, fl, fl), (ip, fl, fl, fl, fl),
        np.zeros((n, nts), dtype="float32"), (ip, fl, ip),
        np.zeros((0, nts + 1), dtype="float32"), (ip, fl, ip, ip),
    ]
    if with_diversion:
        r.append((np.asarray(donors, dtype=np.intp), np.asarray(applied, dtype="float32")))
    return tuple(r)


class TestDiversionStateInResults:
    """The applied amount rides the results as state keyed by donor, not as a series."""

    def test_harvest_keys_the_amount_by_donor_across_jobs(self):
        raw = [_kernel_result([20, 30], donors=[20], applied=[100.0]),
               _kernel_result([21], donors=[21], applied=[7.5])]
        assert RoutingResultsCollection(raw).diversion_applied() == {20: 100.0, 21: 7.5}
        assert new_diversion_applied(raw) == {20: 100.0, 21: 7.5}

    def test_a_result_without_the_element_carries_no_state(self):
        """The diffusive leg and the non-routing segments return eleven elements."""
        raw = [_kernel_result([20], with_diversion=False),
               _kernel_result([21], donors=[21], applied=[7.5])]
        assert RoutingResultsCollection(raw).diversion_applied() == {21: 7.5}
        assert new_diversion_applied(raw) == {21: 7.5}

    @pytest.mark.parametrize(("earlier", "later"), [
        ([], [20]), ([10], [10, 20]), ([10, 20], [20]),
    ])
    def test_appending_windows_keeps_the_later_windows_pairs(self, earlier, later):
        """Concatenating windows must not align the state to the earlier donor set."""
        a = RoutingResultsCollection([_kernel_result([10, 20, 40], donors=earlier,
                                                     applied=[1.0] * len(earlier))])
        b = RoutingResultsCollection([_kernel_result([10, 20, 40], donors=later,
                                                     applied=[5.0] * len(later))])
        assert a.append_timesteps(b).diversion_applied() == {d: 5.0 for d in later}


def _write_timeslice(folder, stamp: pd.Timestamp, values: dict[str, float], quality: int = 100,
                     content_stamp: pd.Timestamp | None = None):
    """One TimeSlice file in the reader's schema, named for its instant."""
    import netCDF4

    stamp_str = stamp.strftime("%Y-%m-%d_%H:%M:%S")
    content_str = (content_stamp or stamp).strftime("%Y-%m-%d_%H:%M:%S")
    sites = list(values)
    with netCDF4.Dataset(str(folder / f"{stamp_str}.15min.usgsTimeSlice.ncdf"), "w") as nc:
        nc.sliceTimeResolutionMinutes = "15"
        nc.createDimension("stationIdInd", len(sites))
        nc.createDimension("stationIdStringLength", 15)
        nc.createDimension("timeStringLength", 19)
        v = nc.createVariable("stationId", "S1", ("stationIdInd", "stationIdStringLength"))
        v[:] = np.array([list(s.ljust(15)) for s in sites], dtype="S1")
        v = nc.createVariable("time", "S1", ("stationIdInd", "timeStringLength"))
        v[:] = np.array([list(content_str.ljust(19)) for _ in sites], dtype="S1")
        v = nc.createVariable("discharge", "f4", ("stationIdInd",))
        v[:] = np.array([values[s] for s in sites], dtype="f4")
        v = nc.createVariable("discharge_quality", "i2", ("stationIdInd",))
        v[:] = np.full(len(sites), quality, dtype="i2")


class TestSeedFromTheRecord:
    """A fresh process holds the last report even when it predates the reader's window."""

    T0 = pd.Timestamp("2011-06-01 00:00")

    @pytest.fixture
    def folder(self, tmp_path):
        d = tmp_path / "usgs"
        d.mkdir()
        _write_timeslice(d, self.T0 - pd.Timedelta(days=13), {DIVERSION_GAGE: 1.0})
        _write_timeslice(d, self.T0 - pd.Timedelta(days=5), {DIVERSION_GAGE: 5.0})
        _write_timeslice(d, self.T0 - pd.Timedelta(days=4), {DIVERSION_GAGE: np.nan})
        _write_timeslice(d, self.T0 - pd.Timedelta(days=3), {DIVERSION_GAGE: 3.0}, quality=10)
        _write_timeslice(d, self.T0 + pd.Timedelta(hours=1), {DIVERSION_GAGE: 9.0})
        return d

    def test_the_newest_finite_report_within_the_horizon_wins(self, folder):
        found = _last_report_before(folder, DIVERSION_GAGE, self.T0, pd.Timedelta(days=11), 1)
        assert found == (self.T0 - pd.Timedelta(days=5), 5.0)

    def test_an_unreadable_file_is_skipped_not_fatal(self, folder, caplog):
        (folder / (self.T0 - pd.Timedelta(days=2)).strftime("%Y-%m-%d_%H:%M:%S")
         ).with_suffix(".15min.usgsTimeSlice.ncdf").write_bytes(b"not a netcdf")
        with caplog.at_level(logging.WARNING):
            found = _last_report_before(folder, DIVERSION_GAGE, self.T0, pd.Timedelta(days=11), 1)
        assert found == (self.T0 - pd.Timedelta(days=5), 5.0)
        assert "unreadable TimeSlice" in caplog.text

    def test_the_readers_validity_rules_apply(self, folder):
        """A newer report the reader would reject must not become the seed."""
        _write_timeslice(folder, self.T0 - pd.Timedelta(days=1), {DIVERSION_GAGE: -999.0})
        _write_timeslice(folder, self.T0 - pd.Timedelta(hours=12), {DIVERSION_GAGE: 0.0})
        found = _last_report_before(folder, DIVERSION_GAGE, self.T0, pd.Timedelta(days=11), 1)
        assert found == (self.T0 - pd.Timedelta(days=5), 5.0)

    def test_the_reports_own_timestamp_must_be_inside_the_window(self, folder):
        """A file named inside the window whose report is dated a month back is skipped."""
        _write_timeslice(folder, self.T0 - pd.Timedelta(hours=6), {DIVERSION_GAGE: 8.0},
                         content_stamp=self.T0 - pd.Timedelta(days=40))
        found = _last_report_before(folder, DIVERSION_GAGE, self.T0, pd.Timedelta(days=11), 1)
        assert found == (self.T0 - pd.Timedelta(days=5), 5.0)

    def test_a_report_beyond_the_horizon_is_not_found(self, folder, caplog):
        with caplog.at_level(logging.WARNING):
            found = _last_report_before(folder, DIVERSION_GAGE, self.T0, pd.Timedelta(days=4), 1)
        assert found is None
        # The staged history is named, so a folder too shallow for the horizon is visible.
        assert "oldest file in that span is 2011-05-28" in caplog.text

    def test_seed_only_a_gage_with_nothing_else(self, folder):
        class _Network:
            t0 = self.T0
            _diversion_site_to_node = {DIVERSION_GAGE: GAGE_LINK}

        params = {"diversion_gage_crosswalk": {DONOR_FP_ID: DIVERSION_GAGE},
                  "diversion_persist_days": 11}
        da_params = {"usgs_timeslices_folder": str(folder)}
        empty = pd.DataFrame()
        seeded = _seed_from_record(empty, params, da_params, _Network(), {})
        assert seeded == {GAGE_LINK: (self.T0 - pd.Timedelta(days=5), 5.0)}
        # A report at or before t0 in the frame, or a carried seed, means no scan.
        frame = pd.DataFrame([[7.0, np.nan]], index=pd.Index([GAGE_LINK], dtype="int64"),
                             columns=pd.date_range(self.T0, periods=2, freq="5min"))
        assert _seed_from_record(frame, params, da_params, _Network(), {}) == {}
        carried = {GAGE_LINK: (self.T0, 2.0)}
        assert _seed_from_record(empty, params, da_params, _Network(), carried) == carried
        assert _seed_from_record(empty, params, {}, _Network(), {}) == {}

    def test_a_report_after_t0_does_not_suppress_the_scan(self, folder):
        """A report later in the window is the hold's successor, not its source."""
        class _Network:
            t0 = self.T0
            _diversion_site_to_node = {DIVERSION_GAGE: GAGE_LINK}

        params = {"diversion_gage_crosswalk": {DONOR_FP_ID: DIVERSION_GAGE},
                  "diversion_persist_days": 11}
        da_params = {"usgs_timeslices_folder": str(folder)}
        frame = pd.DataFrame([[np.nan, np.nan, 7.0]], index=pd.Index([GAGE_LINK], dtype="int64"),
                             columns=pd.date_range(self.T0, periods=3, freq="h"))
        seeded = _seed_from_record(frame, params, da_params, _Network(), {})
        assert seeded == {GAGE_LINK: (self.T0 - pd.Timedelta(days=5), 5.0)}

    def test_a_zero_horizon_never_scans(self, folder, caplog):
        class _Network:
            t0 = self.T0
            _diversion_site_to_node = {DIVERSION_GAGE: GAGE_LINK}

        params = {"diversion_gage_crosswalk": {DONOR_FP_ID: DIVERSION_GAGE},
                  "diversion_persist_days": 0}
        with caplog.at_level(logging.WARNING):
            out = _seed_from_record(pd.DataFrame(), params, {"usgs_timeslices_folder": str(folder)},
                                    _Network(), {})
        assert out == {}
        assert caplog.text == ""

    def test_an_empty_scan_is_not_repeated_for_a_later_t0(self, tmp_path, caplog):
        """The CLI reseeds every window; a folder with nothing is scanned once."""
        folder = tmp_path / "empty"
        folder.mkdir()

        class _Network:
            t0 = self.T0
            _diversion_site_to_node = {DIVERSION_GAGE: GAGE_LINK}

        params = {"diversion_gage_crosswalk": {DONOR_FP_ID: DIVERSION_GAGE},
                  "diversion_persist_days": 11}
        da_params = {"usgs_timeslices_folder": str(folder)}
        cache: dict = {}
        with caplog.at_level(logging.WARNING):
            _seed_from_record(pd.DataFrame(), params, da_params, _Network(), {}, cache)
            _Network.t0 = self.T0 + pd.Timedelta(hours=2)
            _seed_from_record(pd.DataFrame(), params, da_params, _Network(), {}, cache)
        assert cache == {GAGE_LINK: self.T0}
        assert caplog.text.count("no report in the TimeSlices") == 1

    def test_the_deferred_scan_seeds_and_refills(self, folder):
        """The drivers call seed_from_record after any checkpoint restore."""
        from troute.DataAssimilation import NudgingDA, _DiversionRow

        class _Network:
            t0 = self.T0
            _diversion_site_to_node = {DIVERSION_GAGE: GAGE_LINK}

        class _Stub:
            seed_from_record = NudgingDA.seed_from_record
            refill_diversion_rows = NudgingDA.refill_diversion_rows

        grid = pd.date_range(self.T0, periods=4, freq="5min")
        raw = pd.Series(np.nan, index=grid, dtype=float)
        da = _Stub()
        da._data_assimilation_parameters = {
            "usgs_timeslices_folder": str(folder),
            "diversion_da": {"diversion_gage_crosswalk": {DONOR_FP_ID: DIVERSION_GAGE},
                             "diversion_persist_days": 11},
        }
        da._usgs_df = raw.to_frame().T
        da._usgs_df.index = pd.Index([GAGE_LINK], dtype="int64")
        da._diversion_rows = {GAGE_LINK: _DiversionRow(raw, DIVERSION_GAGE)}
        da._diversion_seed_in, da._diversion_scan_empty_before = {}, {}
        da.seed_from_record(_Network())
        assert da._diversion_seed_in == {GAGE_LINK: (self.T0 - pd.Timedelta(days=5), 5.0)}
        np.testing.assert_allclose(da._usgs_df.loc[GAGE_LINK].to_numpy(), 5.0)
        # Already seeded: a second call changes nothing and scans nothing.
        da.seed_from_record(_Network())
        assert da._diversion_scan_empty_before == {}

    def test_an_unreadable_window_read_holds_instead_of_raising(self, tmp_path, caplog):
        """A file removed between discovery and read must not end an hourly cycle."""
        class _Network:
            t0 = self.T0
            _diversion_site_to_node = {DIVERSION_GAGE: GAGE_LINK}

        params = {"diversion_gage_crosswalk": {DONOR_FP_ID: DIVERSION_GAGE}}
        da_params = {"usgs_timeslices_folder": str(tmp_path)}
        da_run = {"usgs_timeslice_files": ["2011-06-01_00:00:00.15min.usgsTimeSlice.ncdf"]}
        with caplog.at_level(logging.WARNING):
            out = _read_diversion_observations(pd.DataFrame(), params, da_params, _Network(),
                                               {"dt": 300, "cpu_pool": 1}, da_run)
        assert out.empty
        assert "could not be read" in caplog.text


class TestRollover:
    """Between CLI windows the seed is the last report at or before the next start."""

    def test_a_report_past_the_next_start_does_not_replace_the_seed(self):
        """The previous frame reaches past the next window's start; if that window's
        read then fails, its early hours must still hold the earlier report."""
        t0 = pd.Timestamp("2011-06-01 00:00")
        grid = pd.date_range(t0, periods=13, freq="15min")  # to 03:00
        raw = pd.Series(np.nan, index=grid, dtype=float)
        raw[t0] = 5.0
        raw[t0 + pd.Timedelta(hours=3)] = 9.0

        class _Network:
            t0 = pd.Timestamp("2011-06-01 02:00")  # the next window's start
            _diversion_site_to_node = {DIVERSION_GAGE: GAGE_LINK}

        class _DA:
            _data_assimilation_parameters = {
                "streamflow_da": {"streamflow_nudging": False},
                "diversion_da": {"diversion_gage_crosswalk": {DONOR_FP_ID: DIVERSION_GAGE}},
            }
            _run_parameters = {"dt": 900, "nts": 4}
            _diversion_seed_in = {}
            _diversion_scan_empty_before = {}
            diversion_seed_at = NudgingDA.diversion_seed_at
            update_for_next_loop = NudgingDA.update_for_next_loop

        da = _DA()
        da._usgs_df = raw.to_frame().T.set_axis(pd.Index([GAGE_LINK], dtype="int64"))
        da._diversion_rows = {GAGE_LINK: _DiversionRow(raw, DIVERSION_GAGE)}
        da.update_for_next_loop(_Network(), {})
        assert da._diversion_seed_in == {GAGE_LINK: (t0, 5.0)}
        np.testing.assert_allclose(da._usgs_df.loc[GAGE_LINK].to_numpy(), 5.0)


class TestHourlyCycles:
    """Standard AnA: two-hour windows cycling hourly, the seed carried between them."""

    def test_the_hold_expires_at_the_absolute_deadline_not_per_window(self):
        class _Network:
            t0 = pd.Timestamp("2011-06-01 00:00")
            _diversion_site_to_node = {DIVERSION_GAGE: GAGE_LINK}

        params = {"diversion_gage_crosswalk": {DONOR_FP_ID: DIVERSION_GAGE},
                  "diversion_persist_days": 11}
        report = (_Network.t0 - pd.Timedelta(days=10, hours=23), 4.0)
        deadline = report[0] + pd.Timedelta(days=11)
        seed = {GAGE_LINK: report}
        held_windows = 0
        for cycle in range(6):
            _Network.t0 = pd.Timestamp("2011-06-01 00:00") + pd.Timedelta(hours=cycle)
            out, rows = _fill_diversion_row(pd.DataFrame(), params, _Network, {"dt": 300},
                                            seed_in=seed, nts=24)
            row = out.loc[GAGE_LINK]
            expected = np.where(row.index <= deadline, 4.0, np.nan)
            np.testing.assert_array_equal(row.to_numpy(), expected)
            held_windows += int(row.notna().any())
            seed = _diversion_seed_at(rows, seed)
        assert held_windows == 2, "cycles 0 and 1 reach the deadline, later cycles do not"


class TestDiversionOnlyConfiguration:
    """A configuration with only the diversion block: the schema hands every omitted DA
    section over as None, and the DA objects must read them as empty."""

    @staticmethod
    def _params(tmp_path):
        from troute.config.compute_parameters import DataAssimilationParameters

        return DataAssimilationParameters(
            usgs_timeslices_folder=str(tmp_path),
            diversion_da={"diversion_gage_crosswalk": {DONOR_FP_ID: DIVERSION_GAGE}},
        ).model_dump()

    def test_persistence_da_initializes_with_the_sections_omitted(self, tmp_path):
        from troute.DataAssimilation import PersistenceDA

        params = self._params(tmp_path)
        assert params["reservoir_da"] is None and params["streamflow_da"] is None

        class _Network:
            t0 = pd.Timestamp("2011-06-01")
            _diversion_site_to_node = {}
            waterbody_dataframe = pd.DataFrame()

        da = PersistenceDA.__new__(PersistenceDA)
        da._data_assimilation_parameters = params
        da._run_parameters = {"dt": 300, "nts": 12}
        da._usgs_df = pd.DataFrame()  # what NudgingDA leaves for it in the real order
        PersistenceDA.__init__(da, _Network(), True, None, da_run={})
        assert da._reservoir_usgs_df.empty

    def test_the_rollover_runs_with_the_sections_omitted(self, tmp_path):
        params = self._params(tmp_path)

        class _Network:
            t0 = pd.Timestamp("2011-06-01 02:00")
            _diversion_site_to_node = {DIVERSION_GAGE: GAGE_LINK}

        class _DA:
            _data_assimilation_parameters = params
            _run_parameters = {"dt": 900, "nts": 4}
            _diversion_seed_in = {GAGE_LINK: (pd.Timestamp("2011-06-01 00:00"), 5.0)}
            _diversion_scan_empty_before = {}
            _diversion_rows = {}
            _usgs_df = pd.DataFrame()
            diversion_seed_at = NudgingDA.diversion_seed_at
            update_for_next_loop = NudgingDA.update_for_next_loop

        da = _DA()
        da.update_for_next_loop(_Network(), {})
        np.testing.assert_allclose(da._usgs_df.loc[GAGE_LINK].to_numpy(), 5.0)

    def test_the_persistence_rollover_runs_with_the_sections_omitted(self, tmp_path):
        from troute.DataAssimilation import PersistenceDA

        class _Network:
            t0 = pd.Timestamp("2011-06-01 02:00")
            waterbody_types_dataframe = pd.DataFrame()

        da = PersistenceDA.__new__(PersistenceDA)
        da._data_assimilation_parameters = self._params(tmp_path)
        da._run_parameters = {"dt": 900, "nts": 4}
        da._usgs_df = pd.DataFrame()
        PersistenceDA.update_for_next_loop(da, _Network(), {})
        assert da._usgs_df.empty


class TestCrosswalkOnAnotherDomain:
    """One configuration serves every domain; a domain without the structure routes on."""

    @staticmethod
    def _network(crosswalk):
        from troute.nhf_preprocess import NHFPreprocessMixin

        class _Net:
            data_assimilation_parameters = {"diversion_da": {"diversion_gage_crosswalk": crosswalk}}
            # Two flowpaths: 1 (links 100, 101) and 2 (link 200); one gage at link 200.
            _dataframe = pd.DataFrame({"fp_id": [1, 1, 2], "segment_order": [0, 1, 0]},
                                      index=pd.Index([100, 101, 200], dtype="int64"))
            _resolve_diversion_da = NHFPreprocessMixin._resolve_diversion_da

            def _gage_selection_rank(self, gages_join):
                return gages_join

            @staticmethod
            def _one_link_per_gage(sub, label, quiet=False):
                return sub

        return _Net()

    def test_a_flowpath_or_gage_outside_the_domain_is_skipped_with_a_warning(self, caplog):
        gages = pd.DataFrame({"site_no": [DIVERSION_GAGE], "up_node_id": [200]})
        net = self._network({1: DIVERSION_GAGE, 9: DIVERSION_GAGE, 2: "00000000"})
        with caplog.at_level(logging.WARNING):
            net._resolve_diversion_da(gages)
        assert net.diversion_da == {101: 200}
        assert "fp_id 9 is not in this domain" in caplog.text
        assert "gage 00000000 is not in this domain's gages" in caplog.text

    def test_a_gage_whose_donor_is_absent_is_not_a_diversion_gage(self):
        """The DA reads and holds every gage the site map names, so a gage whose donor
        is outside the domain must leave the map, or its flow would be imposed with
        nothing subtracted anywhere."""
        gages = pd.DataFrame({"site_no": [DIVERSION_GAGE, "11111111"], "up_node_id": [200, 101]})
        net = self._network({1: DIVERSION_GAGE, 9: "11111111"})
        net._resolve_diversion_da(gages)
        assert net.diversion_da == {101: 200}
        assert net._diversion_site_to_node == {DIVERSION_GAGE: 200}
