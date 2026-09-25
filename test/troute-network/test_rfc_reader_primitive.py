"""One reader for RFC TimeSeries files, and what it refuses.

The folder-based multi-gage path and the single-gage one both go through
``read_rfc_timeseries``, so they cannot derive the same quantities by different means
and agree only on hourly fixtures.

The file carries no time variable, so the axis is rebuilt from ``sliceStartTimeUTC`` and
``sliceTimeResolutionMinutes``. Everything downstream advances one array index per
cadence step, so a cadence taken from the wrong place assimilates at the wrong rate
without failing -- hence the guard.
"""
from __future__ import annotations

import shutil
from datetime import datetime, timedelta
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import pytest

from troute.DataAssimilation import _read_timeseries_files, assemble_rfc_dataframes
from troute.routing.fast_reach.reservoir_RFC_da import read_rfc_timeseries

_FIXTURE = (
    Path(__file__).parents[1] / "BMI" / "rfc_timeseries"
    / "2021-10-22_12.60min.KNFC1.RFCTimeSeries.ncdf"
)
_T0 = datetime(2021, 10, 21, 12)


def _window(t0: datetime, hours: int = 28) -> list[str]:
    out, d = [], t0
    while d <= t0 + timedelta(hours=hours):
        out.append(d.strftime("%Y-%m-%d_%H"))
        d += timedelta(hours=1)
    return out


def test_record_matches_the_file():
    r = read_rfc_timeseries(str(_FIXTURE))
    assert r.station_id == "KNFC1"
    assert r.timestep_seconds == 3600
    assert len(r.discharges) == len(r.datetimes) == r.total_counts
    # observedCounts is the count of history before the issue time, so it is also the
    # index OF the issue time. Both readers' index arithmetic rests on this.
    assert r.datetimes.get_loc(pd.Timestamp("2021-10-22 12:00:00")) == r.observed_counts


def test_cadence_disagreement_is_refused(tmp_path):
    dst = tmp_path / _FIXTURE.name
    shutil.copy(_FIXTURE, dst)
    with netCDF4.Dataset(str(dst), "a") as ds:
        ds.sliceTimeResolutionMinutes = "30"
    with pytest.raises(ValueError, match="disagrees about its own cadence"):
        read_rfc_timeseries(str(dst))


def test_selection_skips_a_newer_file_that_misses_t0(tmp_path):
    """Newest-first, but coverage-gated.

    The newest issue has the latest slice start, so it is the one most likely not to
    reach back to t0. Taking it unconditionally passed over an older file that did
    cover t0, and then failed the run.
    """
    shutil.copy(_FIXTURE, tmp_path / "2021-10-21_12.60min.KNFC1.RFCTimeSeries.ncdf")
    newest = tmp_path / "2021-10-22_12.60min.KNFC1.RFCTimeSeries.ncdf"
    shutil.copy(_FIXTURE, newest)
    with netCDF4.Dataset(str(newest), "a") as ds:
        ds.sliceStartTimeUTC = "2021-10-22_00:00:00"  # begins after t0
    out = _read_timeseries_files(
        str(tmp_path), _window(_T0), _T0, _T0 + timedelta(days=11)
    )
    assert out["file"].iloc[0] == "2021-10-21_12.60min.KNFC1.RFCTimeSeries.ncdf"


def test_no_candidate_covering_t0_is_fatal(tmp_path):
    dst = tmp_path / "2021-10-22_12.60min.KNFC1.RFCTimeSeries.ncdf"
    shutil.copy(_FIXTURE, dst)
    with netCDF4.Dataset(str(dst), "a") as ds:
        ds.sliceStartTimeUTC = "2021-10-22_00:00:00"
    with pytest.raises(ValueError, match="cover the simulation start"):
        _read_timeseries_files(str(tmp_path), _window(_T0), _T0, _T0 + timedelta(days=11))


@pytest.mark.parametrize(("routing_period", "expected"), [(300, True), (7200, False)])
def test_validation_uses_the_run_timestep(routing_period, expected):
    """The reader validated with a hardcoded 300 s.

    ``_validate_RFC_data`` refuses a routing period longer than an hour, because the
    kernel advances one forecast index per DA step. Passing a literal 300 meant that
    check could never fire on this path, so a run with dt > 3600 enabled RFC DA and
    then applied the forecast at the wrong rate.
    """
    out = _read_timeseries_files(
        str(_FIXTURE.parent), _window(_T0), _T0, _T0 + timedelta(days=11),
        routing_period=routing_period,
    )
    assert bool(out["use_rfc"].iloc[0]) is expected


def test_cadence_must_agree_exactly(tmp_path):
    """Floored minutes let a near-miss through.

    ``timeSteps`` of 3659 s floors to 60 minutes and would pass as hourly while
    drifting almost a minute per forecast sample, and a filename with no cadence
    token removes one leg of the three-way check.
    """
    dst = tmp_path / _FIXTURE.name
    shutil.copy(_FIXTURE, dst)
    with netCDF4.Dataset(str(dst), "a") as ds:
        ds.variables["timeSteps"][:] = 3659
    with pytest.raises(ValueError, match="disagrees about its own cadence"):
        read_rfc_timeseries(str(dst))

    untagged = tmp_path / "noCadenceToken.ncdf"
    shutil.copy(_FIXTURE, untagged)
    with pytest.raises(ValueError, match="no cadence token"):
        read_rfc_timeseries(str(untagged))


@pytest.mark.parametrize(
    ("spoil", "expected_use_rfc"), [("history", True), ("window", False)]
)
def test_only_the_forecast_window_is_validated(tmp_path, spoil, expected_use_rfc):
    """A ruined observed history is not a bad forecast.

    Whether there is anything to assimilate is a question about the forecast, so the
    missing values behind t0 must not answer it. An ABSURD value behind t0 is a
    different matter and has its own test: the kernel can walk back into one.
    """
    dst = tmp_path / _FIXTURE.name
    shutil.copy(_FIXTURE, dst)
    # t0 sits at index 24; the fixture's slice starts 24 h before it.
    spoiled = slice(0, 24) if spoil == "history" else slice(24, None)
    with netCDF4.Dataset(str(dst), "a") as ds:
        values = ds.variables["discharges"][:]
        values[0, spoiled] = -999.0
        ds.variables["discharges"][:] = values
    out = _read_timeseries_files(
        str(tmp_path), _window(_T0), _T0, _T0 + timedelta(days=11)
    )
    assert bool(out["use_rfc"].iloc[0]) is expected_use_rfc


def test_timesteps_without_a_units_attribute_is_seconds(tmp_path):
    """A file carrying no units on timeSteps still means seconds.

    Decoding it as a timedelta moves the units into ``.encoding``, leaving ``.attrs``
    empty and turning 3600 s into 3600 ns, which is why the cadence check has to read
    the variable raw.
    """
    dst = tmp_path / _FIXTURE.name
    shutil.copy(_FIXTURE, dst)
    with netCDF4.Dataset(str(dst), "a") as ds:
        ds.variables["timeSteps"].delncattr("units")
    assert read_rfc_timeseries(str(dst)).timestep_seconds == 3600


@pytest.mark.parametrize(
    ("units", "raw"), [("seconds", 3600), ("minutes", 60), ("hours", 1)]
)
def test_every_timesteps_unit_in_the_wild_resolves(tmp_path, units, raw):
    """The producer writes "seconds"; the NHF generator writes "hours" with a value of 1."""
    dst = tmp_path / _FIXTURE.name
    shutil.copy(_FIXTURE, dst)
    with netCDF4.Dataset(str(dst), "a") as ds:
        ds.variables["timeSteps"].units = units
        ds.variables["timeSteps"][:] = raw
    assert read_rfc_timeseries(str(dst)).timestep_seconds == 3600


def test_unrecognized_timesteps_units_are_refused(tmp_path):
    dst = tmp_path / _FIXTURE.name
    shutil.copy(_FIXTURE, dst)
    with netCDF4.Dataset(str(dst), "a") as ds:
        ds.variables["timeSteps"].units = "fortnights"
    with pytest.raises(ValueError, match="does not recognize"):
        read_rfc_timeseries(str(dst))


def test_only_the_newest_series_is_read(tmp_path):
    """A file accumulates issue times; newest_forecast names the current one.

    The generator's ``mergeOld`` appends a series rather than replacing the file. Reading
    every series as one array gave nseries x forecastInd values on a time axis that many
    steps long, and mixed superseded forecasts into the current one.
    """
    dst = tmp_path / _FIXTURE.name
    with netCDF4.Dataset(str(_FIXTURE)) as src:
        n = len(src.dimensions["forecastInd"])
        base = np.asarray(src.variables["discharges"][:]).ravel()
    with netCDF4.Dataset(str(dst), "w") as ds:
        ds.createDimension("nseries", 2)
        ds.createDimension("forecastInd", n)
        ds.createDimension("stationIdStrLen", 5)
        ds.createVariable("discharges", "f4", ("nseries", "forecastInd"))[:] = [base, base * 10]
        ds.createVariable("synthetic_values", "i1", ("nseries", "forecastInd"))[:] = 0
        for name, value in (("totalCounts", n), ("observedCounts", 48), ("forecastCounts", n - 48)):
            ds.createVariable(name, "i2", ("nseries",))[:] = [value, value]
        steps = ds.createVariable("timeSteps", "i4", ("nseries",))
        steps.units = "seconds"
        steps[:] = [3600, 3600]
        ds.createVariable("stationId", "S1", ("stationIdStrLen",))[:] = np.array(
            list("KNFC1"), dtype="S1"
        )
        # Per series, like the discharges: the horizon is measured from this.
        ds.createDimension("timeStrLen", 19)
        issued = ds.createVariable("issueTimeUTC", "S1", ("nseries", "timeStrLen"))
        issued[:] = np.array(
            [list("2021-10-20_12:00:00"), list("2021-10-21_12:00:00")], dtype="S1"
        )
        ds.sliceStartTimeUTC = "2021-10-20_12:00:00"
        ds.sliceTimeResolutionMinutes = "60"
        ds.newest_forecast = "1"

    record = read_rfc_timeseries(str(dst))
    assert len(record.discharges) == n
    assert record.discharges[0] == pytest.approx(base[0] * 10)


def test_mixed_cadences_in_one_assembly_are_refused():
    """One cadence per union grid.

    The pivot puts every gage on the union of their datetimes while the kernel advances
    one column per da_timestep, so an hourly gage sharing a grid with a 15 min one has
    its samples four columns apart and steps onto padding instead. Nothing downstream
    can detect that: the pads read as missing and the reservoir quietly drops to level
    pool.
    """
    t0 = pd.Timestamp("2021-10-21 12:00")
    frames = []
    for gage, stamps, cadence in (
        ("HOURL", pd.date_range(t0 - pd.Timedelta(hours=1), periods=4, freq="h"), 3600),
        ("QUART", pd.date_range(t0 - pd.Timedelta(hours=1), periods=13, freq="15min"), 900),
    ):
        frames.append(pd.DataFrame({
            "stationId": gage,
            "discharges": np.arange(len(stamps), dtype=float) + 10,
            "Datetime": stamps,
            "totalCounts": len(stamps),
            "timeseries_idx": list(stamps).index(t0),
            "file": "f",
            "use_rfc": True,
            "da_timestep": cadence,
            "issue_time": t0,
        }))
    crosswalk = pd.DataFrame(
        {"rfc_gage_id": ["HOURL", "QUART"], "rfc_lake_id": [1, 2]}
    ).set_index("rfc_lake_id")
    with pytest.raises(ValueError, match="mix cadences"):
        assemble_rfc_dataframes(
            pd.concat(frames, ignore_index=True), crosswalk, t0,
            {"reservoir_rfc_forecast_persist_days": 11},
        )


def test_an_absurd_discharge_behind_t0_disqualifies_the_forecast(tmp_path, caplog):
    """The backtrack reads backward from the cursor, so history is reachable.

    When the sample at the cursor is unusable ``reservoir_RFC_da`` walks down through
    the earlier samples, which means a value the forward-window check never saw can
    still land on the reservoir.
    """
    import logging

    from troute.routing.fast_reach.reservoir_RFC_da import ABSURD_DISCHARGE_CMS

    dst = tmp_path / _FIXTURE.name
    shutil.copy(_FIXTURE, dst)
    with netCDF4.Dataset(str(dst), "a") as ds:
        values = ds.variables["discharges"][:]
        values[0, 5] = ABSURD_DISCHARGE_CMS + 10_000  # well behind t0 at index 24
        ds.variables["discharges"][:] = values
    with caplog.at_level(logging.WARNING):
        out = _read_timeseries_files(
            str(tmp_path), _window(_T0), _T0, _T0 + timedelta(days=11)
        )
    assert bool(out["use_rfc"].iloc[0]) is False
    assert "before t0" in caplog.text
