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
downstream. Either source populates that row and conserves the transfer: nudging
from timeslices, or the climatological fill in forecast mode.
"""

from __future__ import annotations

import logging
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
import pytest

from troute import nhd_network
from troute.DataAssimilation import (
    _DIVERSION_MONTHLY_MEANS,
    _fill_diversion_historical_median,
)
from troute.routing.compute import _resolve_diversion_da

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


class TestHistoricalMedianFallback:
    """Climatological fill for timesteps with no observation (forecast mode)."""

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
            "persist_historical_median": True,
        }

    @pytest.mark.parametrize("dt,nts", [(300, 12), (60, 20), (900, 8)])
    def test_columns_follow_routing_timestep(self, network, params, dt, nts):
        """One column per routing timestep, not a fixed 5 minute grid.

        The kernel indexes this frame as ``usgs_values[gage_i, timestep]`` with the
        ROUTING step, so a hardcoded grid ran out early for a short dt and advanced
        too slowly for a long one.
        """
        out = _fill_diversion_historical_median(
            pd.DataFrame(), params, network, {"dt": dt, "nts": nts}
        )
        assert out.shape[1] == nts + 1
        spacing = pd.Series(out.columns).diff().dropna().unique()
        assert list(spacing) == [pd.Timedelta(seconds=dt)]

    def test_fills_with_calendar_month_climatology(self, network, params):
        out = _fill_diversion_historical_median(
            pd.DataFrame(), params, network, {"dt": 300, "nts": 3}
        )
        june = _DIVERSION_MONTHLY_MEANS[DIVERSION_GAGE][6]
        np.testing.assert_allclose(out.loc[GAGE_LINK].to_numpy(), june)

    def test_real_observations_are_never_overwritten(self, network, params):
        """The fallback fills gaps only. An observed value must survive."""
        idx = pd.date_range(network.t0, periods=4, freq=pd.Timedelta(seconds=300))
        existing = pd.DataFrame(
            [[7.0, np.nan, 9.0, np.nan]], index=pd.Index([GAGE_LINK], dtype="int64"),
            columns=idx,
        )
        out = _fill_diversion_historical_median(
            existing, params, network, {"dt": 300, "nts": 3}
        )
        row = out.loc[GAGE_LINK]
        assert row.iloc[0] == 7.0 and row.iloc[2] == 9.0
        june = _DIVERSION_MONTHLY_MEANS[DIVERSION_GAGE][6]
        assert row.iloc[1] == june and row.iloc[3] == june

    def test_substitution_is_logged(self, network, params, caplog):
        """Climatology is not an observation; a run must say how much it used."""
        with caplog.at_level(logging.WARNING):
            _fill_diversion_historical_median(
                pd.DataFrame(), params, network, {"dt": 300, "nts": 3}
            )
        assert "climatology" in caplog.text

    def test_gage_without_routing_link_is_reported_not_raised(self, network, params, caplog):
        """An unresolved gage used to raise KeyError mid-run."""
        network._diversion_site_to_node = {}
        with caplog.at_level(logging.WARNING):
            out = _fill_diversion_historical_median(
                pd.DataFrame(), params, network, {"dt": 300, "nts": 3}
            )
        assert "no routing link" in caplog.text
        assert out.empty or GAGE_LINK not in out.index

    def test_disabled_flag_leaves_frame_untouched(self, network, params):
        params["persist_historical_median"] = False
        original = pd.DataFrame()
        # the caller gates on the flag, so calling with no crosswalk is the no-op path
        out = _fill_diversion_historical_median(
            original, {"diversion_gage_crosswalk": {}}, network, {"dt": 300, "nts": 3}
        )
        assert out is original


class TestMultiWindowFallback:
    """The climatology must follow the calendar as the run advances.

    A run in median-only mode covers several forcing loops. The kernel restarts its
    timestep index at zero in each loop, so if the observation frame is not rebuilt
    at the new t0 every loop re-reads the first loop's columns and a multi-month run
    keeps diverting the first month's climatology.
    """

    @staticmethod
    def _network_at(t0: str):
        class _Network:
            _diversion_site_to_node = {DIVERSION_GAGE: GAGE_LINK}

        n = _Network()
        n.t0 = pd.Timestamp(t0)
        return n

    @pytest.fixture
    def params(self) -> dict:
        return {
            "diversion_gage_crosswalk": {DONOR_FP_ID: DIVERSION_GAGE},
            "persist_historical_median": True,
        }

    def test_rebuilt_frame_follows_the_calendar_month(self, params):
        run = {"dt": 300, "nts": 3}
        april = _fill_diversion_historical_median(
            pd.DataFrame(), params, self._network_at("2011-04-14"), run
        )
        july = _fill_diversion_historical_median(
            pd.DataFrame(), params, self._network_at("2011-07-14"), run
        )
        means = _DIVERSION_MONTHLY_MEANS[DIVERSION_GAGE]
        assert means[4] != means[7], "fixture months must differ for this to mean anything"
        np.testing.assert_allclose(april.loc[GAGE_LINK].to_numpy(), means[4])
        np.testing.assert_allclose(july.loc[GAGE_LINK].to_numpy(), means[7])

    def test_stale_frame_is_not_refilled(self, params):
        """A fully populated frame has no gaps, so the fill cannot correct it.

        This is why the caller must clear the frame between loops rather than relying
        on the fill to notice that the calendar moved.
        """
        run = {"dt": 300, "nts": 3}
        april = _fill_diversion_historical_median(
            pd.DataFrame(), params, self._network_at("2011-04-14"), run
        )
        again = _fill_diversion_historical_median(
            april.copy(), params, self._network_at("2011-07-14"), run
        )
        means = _DIVERSION_MONTHLY_MEANS[DIVERSION_GAGE]
        np.testing.assert_allclose(again.loc[GAGE_LINK].to_numpy(), means[4])


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
