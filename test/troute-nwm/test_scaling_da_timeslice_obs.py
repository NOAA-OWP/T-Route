"""The scaling DA reads its observations from TimeSlice files.

It used to read a bespoke icechunk store, which made icechunk + zarr>=3 a runtime
dependency for one feature and gave the same quantity a second input path. These
tests pin the replacement at the seams that fail SILENTLY -- every one of them
produces a complete, plausible, exit-0 run that assimilated nothing:

  * the TimeSlice directory not being discovered at all (build_da_sets gated the
    file list on nudging, which the scaling DA does not require);
  * the observation frame not landing on the grid the kernel reads positionally;
  * a misconfigured directory degrading to a no-DA control instead of raising.

The fixture writes the on-disk schema directly, so a schema change breaks this
test independently of any external converter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from nwm_routing.scaling_da_apply import ScalingDA, timeslice_station_roster

from troute import hyfeature_network_utilities, nhd_network_utilities_v02
from troute.scaling_da.preprocess import resolve_obs_sites

# BOTH TimeSlice-set builders, because they are separate implementations of the same
# gate and the drivers do not agree on which one they call: the NHF (-V5) and BMI
# drivers -- the ones the scaling DA actually runs under -- use
# hyfeature_network_utilities, while the -V3/-V4 drivers use nhd_network_utilities_v02.
# This test previously imported the NHD module under the name ``hnu``, so it asserted
# the gate on the builder the scaling DA never calls; it passed green while the NHF
# path silently had no gate at all.
DA_SET_BUILDERS = [hyfeature_network_utilities, nhd_network_utilities_v02]

_ID_LEN = 15
_TIME_LEN = 19
_FMT = "%Y-%m-%d_%H:%M:%S"


def _write_timeslices(folder, sites, index, values):
    """Write one USGS TimeSlice per row of *values* (time x site)."""
    netCDF4 = pytest.importorskip("netCDF4")
    folder.mkdir(parents=True, exist_ok=True)
    ids = np.array([list(s.rjust(_ID_LEN)) for s in sites], dtype="S1")
    for t, row in zip(index, values):
        stamp = pd.Timestamp(t).strftime(_FMT)
        times = np.array([list(stamp.ljust(_TIME_LEN)) for _ in sites], dtype="S1")
        finite = np.isfinite(row)
        path = folder / f"{stamp}.15min.usgsTimeSlice.ncdf"
        with netCDF4.Dataset(str(path), "w", format="NETCDF4") as nc:
            nc.createDimension("stationIdInd", len(sites))
            nc.createDimension("stationIdStringLength", _ID_LEN)
            nc.createDimension("timeStringLength", _TIME_LEN)
            nc.createVariable(
                "stationId", "S1", ("stationIdInd", "stationIdStringLength")
            )[:] = ids
            nc.createVariable("time", "S1", ("stationIdInd", "timeStringLength"))[:] = times
            nc.createVariable("discharge", "f4", ("stationIdInd",))[:] = row.astype("f4")
            # 100 -> 1.0 after the reader's /100; the default qc_threshold of 1 masks
            # anything below, so 100 is the only value a good observation can carry.
            nc.createVariable("discharge_quality", "i2", ("stationIdInd",))[:] = np.where(
                finite, 100, 0
            ).astype("i2")
    return folder


@pytest.fixture
def obs_dir(tmp_path):
    """Two gages, hourly, one deliberate gap."""
    sites = ["01105933", "03010655"]
    index = pd.date_range("2000-01-01", periods=4, freq="h")
    values = np.array(
        [[10.0, 20.0], [12.0, 24.0], [np.nan, 28.0], [16.0, 32.0]], dtype=float
    )
    return _write_timeslices(tmp_path / "usgs_ts", sites, index, values), sites, index


def _reader(folder, sites_to_segs, cpu_pool=1):
    """A ScalingDA with only the observation-reading state populated."""
    o = ScalingDA.__new__(ScalingDA)
    o._ts_folder = folder
    o._qc_threshold = 1
    o._interpolation_limit = 59
    o._cpu_pool = cpu_pool
    o._obs_cache = None
    o._obs_sites = set(sites_to_segs)
    o.gage_seg = dict(sites_to_segs)
    o._da_sites = sorted(sites_to_segs)
    o.synthetic_factor = None
    o._loop_obs = None
    return o


class TestRoster:
    def test_roster_is_the_union_over_files(self, obs_dir):
        folder, sites, _ = obs_dir
        assert timeslice_station_roster(folder) == set(sites)

    def test_empty_directory_is_an_empty_roster(self, tmp_path):
        # Not an exception: __init__ turns this into the "nothing to assimilate"
        # error, with the directory named. A bare crash here would not say which
        # directory was wrong.
        assert timeslice_station_roster(tmp_path) == set()


class TestObservationsReachTheKernelGrid:
    def test_build_usgs_df_is_on_the_exact_grid_the_kernel_reads(self, obs_dir):
        """nts+1 columns at t0 + j*dt, indexed by routed segment.

        The kernel reads usgs_values[gage_i, timestep] positionally and ignores the
        timestamps, so a frame on any other grid assimilates each observation at the
        wrong model step rather than failing.
        """
        folder, sites, index = obs_dir
        da = _reader(folder, {sites[0]: 101, sites[1]: 202})
        t0, dt, nts = index[0], 3600.0, 3
        usgs = da.build_usgs_df(t0, dt, nts)

        assert list(usgs.index) == [101, 202]
        assert list(usgs.columns) == list(
            pd.date_range(t0, periods=nts + 1, freq=pd.Timedelta(seconds=dt))
        )
        # Column 0 is the t0 seed; values round-trip from the files unchanged.
        assert usgs.loc[202].to_numpy() == pytest.approx([20.0, 24.0, 28.0, 32.0])

    def test_hourly_observations_densify_to_a_sub_hourly_model_step(self, obs_dir):
        """The property that makes an hourly gage usable at dt=300.

        Without it the kernel sees an observation on the hour and NaN for the eleven
        five-minute steps in between, which is a different assimilation than every
        other t-route observation consumer performs on the same files.
        """
        folder, sites, index = obs_dir
        da = _reader(folder, {sites[1]: 202})
        usgs = da.build_usgs_df(index[0], 900.0, 4)  # 15-minute steps
        # 20.0 at t0 and 24.0 an hour later -> a quarter of the way is 21.0.
        assert usgs.loc[202].to_numpy()[1] == pytest.approx(21.0)

    def test_a_missing_hourly_observation_stays_missing(self, obs_dir):
        """Gaps are NOT invented. interpolation_limit_min (59) cannot bridge the
        120-minute hole a dropped hourly sample makes, so the value comes back NaN
        and the kernel persists/decays across it."""
        folder, sites, index = obs_dir
        da = _reader(folder, {sites[0]: 101})
        usgs = da.build_usgs_df(index[0], 3600.0, 3)
        assert np.isnan(usgs.loc[101].to_numpy()[2])

    def test_window_file_list_is_honored(self, obs_dir):
        """da_run carries the same per-window list the nudging path consumes."""
        folder, sites, index = obs_dir
        da = _reader(folder, {sites[1]: 202})
        only_first_two = [
            f"{pd.Timestamp(t).strftime(_FMT)}.15min.usgsTimeSlice.ncdf" for t in index[:2]
        ]
        usgs = da.build_usgs_df(
            index[0], 3600.0, 3, {"usgs_timeslice_files": only_first_two}
        )
        got = usgs.loc[202].to_numpy()
        assert got[:2] == pytest.approx([20.0, 24.0])
        assert np.isnan(got[3]), "a file outside the window list must not be read"


@pytest.mark.parametrize("builder", DA_SET_BUILDERS, ids=lambda m: m.__name__.split(".")[-1])
class TestDaSetsGate:
    """build_da_sets must enumerate USGS TimeSlices for a scaling-DA-only config.

    This is the regression that made the whole feature inert: the file list was gated
    on nudging or reservoir persistence, so a config with only scaling_da enabled got
    da_sets entries with no 'usgs_timeslice_files' key at all. ScalingDA then falls
    back to globbing the entire TimeSlice directory for every window, which is a
    different observation set than the nudging path gets from the same folder.

    Parametrized over BOTH builders on purpose. They are independent copies of this
    gate, and the fix reached only one of them the first time.
    """

    def _run_sets(self):
        return [{"final_timestamp": pd.Timestamp("2000-01-01 03:00:00")}]

    def test_scaling_da_alone_still_gets_a_file_list(self, builder, obs_dir):
        folder, _, _ = obs_dir
        da_sets = builder.build_da_sets(
            {
                "usgs_timeslices_folder": str(folder),
                "streamflow_da": {
                    "streamflow_nudging": False,
                    "streamflow_scaling": True,
                },
            },
            self._run_sets(),
            pd.Timestamp("2000-01-01"),
        )
        assert da_sets[0].get("usgs_timeslice_files"), (
            "streamflow_scaling enabled but no TimeSlice files were enumerated; the DA "
            "fall back to reading the whole directory every window"
        )

    def test_scaling_da_disabled_still_enumerates_nothing(self, builder, obs_dir):
        folder, _, _ = obs_dir
        da_sets = builder.build_da_sets(
            {
                "usgs_timeslices_folder": str(folder),
                "streamflow_da": {
                    "streamflow_nudging": False,
                    "streamflow_scaling": False,
                },
            },
            self._run_sets(),
            pd.Timestamp("2000-01-01"),
        )
        assert not da_sets[0].get("usgs_timeslice_files")

    def test_scaling_and_nudging_get_the_same_window_list(self, builder, obs_dir):
        """The head-to-head comparison this gate exists for.

        Two arms pointed at one directory must be handed the SAME per-window files.
        Before the gate, the scaling arm got no list and globbed the directory instead,
        so the two schemes were compared on different observation sets.
        """
        folder, _, _ = obs_dir
        common = {"usgs_timeslices_folder": str(folder)}
        scaling = builder.build_da_sets(
            {**common,
             "streamflow_da": {"streamflow_nudging": False, "streamflow_scaling": True}},
            self._run_sets(), pd.Timestamp("2000-01-01"),
        )
        nudging = builder.build_da_sets(
            {**common,
             "streamflow_da": {"streamflow_nudging": True, "streamflow_scaling": False}},
            self._run_sets(), pd.Timestamp("2000-01-01"),
        )
        assert scaling[0]["usgs_timeslice_files"] == nudging[0]["usgs_timeslice_files"]


class TestMisconfigurationIsLoud:
    def test_missing_directory_raises_rather_than_running_without_da(self, tmp_path):
        """The failure mode this replaces: a configured DA that quietly assimilated
        nothing and produced a complete run indistinguishable from success."""
        with pytest.raises(FileNotFoundError, match="not a directory"):
            resolve_obs_sites(
                {"01105933": 101},
                {"usgs_timeslices_folder": str(tmp_path / "nope")},
                1, synthetic=False,
            )

    def test_directory_with_no_timeslices_raises(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError, match="no readable"):
            resolve_obs_sites(
                {"01105933": 101},
                {"usgs_timeslices_folder": str(tmp_path / "empty")},
                1, synthetic=False,
            )


class TestSilentNoDataFailuresAreRejected:
    """Every case here used to complete normally while assimilating nothing."""

    def test_a_present_but_empty_window_list_is_honored(self, obs_dir):
        """build_da_sets selecting no file for a window must not widen to the
        directory. Falling back would assimilate observations the shared file-list
        gate deliberately excluded."""
        folder, sites, index = obs_dir
        da = _reader(folder, {sites[1]: 202})
        usgs = da.build_usgs_df(index[0], 3600.0, 3, {"usgs_timeslice_files": []})
        assert usgs.empty, "an empty window list must not fall back to the directory"

    def test_absent_key_still_falls_back_to_discovery(self, obs_dir):
        """A driver that never built da_sets keeps working."""
        folder, sites, index = obs_dir
        da = _reader(folder, {sites[1]: 202})
        assert not da.build_usgs_df(index[0], 3600.0, 3, {}).empty

    def test_no_domain_gage_in_the_roster_raises(self, obs_dir):
        """A TimeSlice directory for some other domain is not a no-DA run."""
        folder, _, _ = obs_dir
        with pytest.raises(ValueError, match=r"no source gages|none of the"):
            resolve_obs_sites(
                {"99999999": 101},  # not in the fixture's roster
                {"usgs_timeslices_folder": str(folder)},
                1, synthetic=False,
            )

    @pytest.mark.parametrize("dt", [30.0, 90.0])
    def test_a_dt_the_reader_cannot_express_raises(self, obs_dir, dt):
        """The shared reader resamples in whole minutes. dt=90 would resample onto
        60 s and reindex onto the 90 s kernel grid (every other step NaN); dt<60
        collapses to a "0min" frequency."""
        folder, sites, index = obs_dir
        da = _reader(folder, {sites[1]: 202})
        with pytest.raises(ValueError, match="whole minutes"):
            da.build_usgs_df(index[0], dt, 3)
