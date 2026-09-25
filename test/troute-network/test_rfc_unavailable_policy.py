"""What happens when a forecast a reservoir needs is missing or does not cover the run.

Four places can reach that state, and one policy governs all of them, so a gage with
no file and a gage with a stale file are treated alike. One late gage must not break an
operational chain of runs.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pandas as pd
import pytest

from troute.DataAssimilation import _read_timeseries_files, assemble_rfc_dataframes

_REAL = Path(__file__).parent.parent / "BMI" / "rfc_timeseries"
_FIXTURE = next(_REAL.glob("*.RFCTimeSeries.ncdf"), None)
# Only the folder-reading tests need a real file; the assembly tests are synthetic.
needs_fixture = pytest.mark.skipif(_FIXTURE is None, reason="no RFC fixture available")
_FALLBACK_T0 = pd.Timestamp("2021-10-21 12:00:00")

_PARAMS = {"reservoir_rfc_forecast_persist_days": 11}
_LEVEL_POOL = {**_PARAMS, "reservoir_rfc_forecasts_unavailable_action": "level_pool"}


def _t0() -> pd.Timestamp:
    if _FIXTURE is None:
        return _FALLBACK_T0
    return pd.Timestamp(_FIXTURE.name.split(".")[0].replace("_", " ") + ":00:00")


def _dates(t0: pd.Timestamp) -> list[str]:
    return [(t0 - pd.Timedelta(hours=h)).strftime("%Y-%m-%d_%H") for h in range(28)]


def _crosswalk(gages: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {"rfc_gage_id": gages, "rfc_lake_id": list(range(101, 101 + len(gages)))}
    ).set_index("rfc_lake_id")


def _crosswalk_nhf(gages: list[str]) -> pd.DataFrame:
    """The shape NHF's preprocess_data_assimilation actually builds.

    It ends in ``.reset_index()``, so the lake id is a column and the index is a
    RangeIndex. Fixtures that only ever used the indexed shape could not see that the
    gageless lookup was reading row positions as lake ids.
    """
    return pd.DataFrame(
        {"rfc_gage_id": gages, "rfc_lake_id": list(range(101, 101 + len(gages)))}
    )


# --------------------------------------------------------------- no file in the window

@needs_fixture
def test_an_empty_folder_is_fatal_by_default(tmp_path):
    t0 = _t0()
    with pytest.raises(FileNotFoundError, match="no RFC timeseries file"):
        _read_timeseries_files(str(tmp_path), _dates(t0), t0, t0 + pd.Timedelta(days=11))


@needs_fixture
def test_an_empty_folder_degrades_under_level_pool(tmp_path, caplog):
    t0 = _t0()
    with caplog.at_level(logging.WARNING):
        got = _read_timeseries_files(
            str(tmp_path), _dates(t0), t0, t0 + pd.Timedelta(days=11),
            unavailable_action="level_pool",
        )
    assert got.empty
    assert "Running level pool there instead" in caplog.text


# ------------------------------------------------------- a file that does not cover t0

def _stale_folder(tmp_path: Path) -> tuple[Path, pd.Timestamp]:
    """The fixture relabeled to a stamp inside the window but a span that misses t0."""
    t0 = _t0() + pd.Timedelta(days=400)
    stamp = t0.strftime("%Y-%m-%d_%H")
    shutil.copy(_FIXTURE, tmp_path / f"{stamp}.60min.STALE1.RFCTimeSeries.ncdf")
    return tmp_path, t0


@needs_fixture
def test_a_forecast_that_misses_t0_is_fatal_by_default(tmp_path):
    d, t0 = _stale_folder(tmp_path)
    with pytest.raises(ValueError, match="cover the simulation start"):
        _read_timeseries_files(str(d), _dates(t0), t0, t0 + pd.Timedelta(days=11))


@needs_fixture
def test_a_forecast_that_misses_t0_degrades_under_level_pool(tmp_path, caplog):
    d, t0 = _stale_folder(tmp_path)
    with caplog.at_level(logging.WARNING):
        got = _read_timeseries_files(
            str(d), _dates(t0), t0, t0 + pd.Timedelta(days=11),
            unavailable_action="level_pool",
        )
    assert got.empty
    assert "STALE1" in caplog.text


def _stale_copy(dst: Path, stamp: pd.Timestamp) -> None:
    """The fixture, re-dated inside the window but with a span nowhere near t0."""
    import netCDF4

    shutil.copy(_FIXTURE, dst)
    with netCDF4.Dataset(str(dst), "a") as ds:
        ds.sliceStartTimeUTC = "2019-01-01_00:00:00"


@needs_fixture
def test_one_stale_gage_does_not_take_down_a_good_one(tmp_path, caplog):
    """The failure that breaks an hourly chain: most gages fine, one late."""
    t0 = _t0()
    shutil.copy(_FIXTURE, tmp_path / _FIXTURE.name)
    stamp = t0 - pd.Timedelta(hours=2)
    _stale_copy(tmp_path / f"{stamp:%Y-%m-%d_%H}.60min.STALE1.RFCTimeSeries.ncdf", stamp)
    good = _FIXTURE.name.split(".")[2]

    # Default policy: the one late gage ends the run.
    with pytest.raises(ValueError, match="STALE1"):
        _read_timeseries_files(str(tmp_path), _dates(t0), t0, t0 + pd.Timedelta(days=11))

    with caplog.at_level(logging.WARNING):
        got = _read_timeseries_files(
            str(tmp_path), _dates(t0), t0, t0 + pd.Timedelta(days=11),
            unavailable_action="level_pool",
        )
    assert set(got["stationId"]) == {good}
    assert "STALE1" in caplog.text


# ------------------------------------------------------------------- assembly failures

def test_no_observations_at_all_is_fatal_by_default():
    with pytest.raises(ValueError, match="no RFC timeseries observations"):
        assemble_rfc_dataframes(pd.DataFrame(), _crosswalk(["A1"]), _t0(), _PARAMS)


def test_no_observations_at_all_degrades_under_level_pool(caplog):
    with caplog.at_level(logging.WARNING):
        obs, par = assemble_rfc_dataframes(
            pd.DataFrame(), _crosswalk(["A1"]), _t0(), _LEVEL_POOL
        )
    assert obs.empty
    assert par.empty


def _frame(gage: str, t0: pd.Timestamp) -> pd.DataFrame:
    dates = pd.date_range(t0, periods=4, freq="h")
    return pd.DataFrame({
        "stationId": gage, "discharges": [1.0, 2.0, 3.0, 4.0], "Datetime": dates,
        "totalCounts": 4, "timeseries_idx": 0, "file": "f", "use_rfc": True,
        "da_timestep": 3600, "issue_time": t0,
    })


def test_observations_that_miss_t0_are_fatal_by_default():
    t0 = _t0()
    with pytest.raises(ValueError, match="do not cover the simulation start"):
        assemble_rfc_dataframes(
            _frame("A1", t0 + pd.Timedelta(days=3)), _crosswalk(["A1"]), t0, _PARAMS
        )


def test_observations_that_miss_t0_degrade_under_level_pool():
    t0 = _t0()
    obs, par = assemble_rfc_dataframes(
        _frame("A1", t0 + pd.Timedelta(days=3)), _crosswalk(["A1"]), t0, _LEVEL_POOL
    )
    assert obs.empty
    assert par.empty


def test_a_gage_with_no_usable_forecast_follows_the_policy():
    """Availability failures are one policy, whether they hit one gage or all of them."""
    t0 = _t0()
    with pytest.raises(ValueError, match="have a gage but no usable forecast"):
        assemble_rfc_dataframes(
            _frame("A1", t0), _crosswalk(["A1", "B2"]), t0, _PARAMS
        )


def test_a_gage_with_no_usable_forecast_degrades_under_level_pool(caplog):
    t0 = _t0()
    with caplog.at_level(logging.WARNING):
        _, par = assemble_rfc_dataframes(
            _frame("A1", t0), _crosswalk(["A1", "B2"]), t0, _LEVEL_POOL
        )
    assert not par.loc[102, "use_rfc"]
    assert "have a gage but no usable forecast" in caplog.text


def test_a_lake_the_hydrofabric_gives_no_gage_is_reported_not_fatal(caplog):
    """`site_no` is None on some domains. No forecast can ever exist for those lakes,
    so this is not an availability failure and must not end the run."""
    t0 = _t0()
    crosswalk = pd.DataFrame(
        {"rfc_gage_id": ["A1", None], "rfc_lake_id": [101, 102]}
    ).set_index("rfc_lake_id")
    with caplog.at_level(logging.WARNING):
        _, par = assemble_rfc_dataframes(_frame("A1", t0), crosswalk, t0, _PARAMS)
    assert not par.loc[102, "use_rfc"]
    assert "no gage in the hydrofabric" in caplog.text


# ------------------------------------------------------------------ the summary line

_WITH_WINDOW = {
    **_PARAMS,
    "reservoir_rfc_forecasts_lookback_hours": 28,
    "reservoir_rfc_forecasts_offset_hours": 0,
}


def _summaries(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records
            if r.levelno == logging.INFO and "reservoir RFC DA:" in r.getMessage()]


def test_the_run_reports_what_it_selected(caplog):
    """One greppable line per run: an operator does not read every warning."""
    t0 = _t0()
    with caplog.at_level(logging.INFO):
        assemble_rfc_dataframes(_frame("A1", t0), _crosswalk(["A1"]), t0, _WITH_WINDOW)
    lines = _summaries(caplog)
    assert len(lines) == 1
    assert "1 of 1 reservoir(s) have a forecast selected, 0 run level pool" in lines[0]
    assert "Searched" in lines[0]


def test_the_summary_counts_the_fallbacks(caplog):
    t0 = _t0()
    with caplog.at_level(logging.INFO):
        assemble_rfc_dataframes(
            _frame("A1", t0), _crosswalk(["A1", "B2"]), t0,
            {**_WITH_WINDOW, "reservoir_rfc_forecasts_unavailable_action": "level_pool"},
        )
    assert "1 of 2 reservoir(s) have a forecast selected, 1 run level pool" in caplog.text


def test_the_summary_reports_the_age_of_what_it_selected(caplog):
    """A domain whose issues are all near the lookback edge is one bad hour from none."""
    t0 = _t0()
    frame = _frame("A1", t0)
    frame["issue_time"] = t0 - pd.Timedelta(hours=26)
    with caplog.at_level(logging.INFO):
        assemble_rfc_dataframes(frame, _crosswalk(["A1"]), t0, _WITH_WINDOW)
    assert "26 to 26 h old" in caplog.text


def test_selecting_nothing_names_the_window_not_just_the_absence(caplog):
    """Every forecast rejected by validation still leaves a countable roster."""
    t0 = _t0()
    frame = _frame("A1", t0)
    frame["use_rfc"] = False
    with caplog.at_level(logging.INFO):
        assemble_rfc_dataframes(frame, _crosswalk(["A1"]), t0, _WITH_WINDOW)
    line = _summaries(caplog)[0]
    assert "no forecast selected for any of 1 reservoir(s)" in line
    assert "Searched" in line


def test_an_empty_window_still_reports_the_span_it_searched(caplog):
    """The case the line exists for: nothing dated inside the lookback at all.

    That path returns early, so the summary has to be emitted there too rather than
    only where a selection succeeded.
    """
    t0 = _t0()
    with caplog.at_level(logging.INFO):
        assemble_rfc_dataframes(
            pd.DataFrame(), _crosswalk(["A1", "B2"]), t0,
            {**_WITH_WINDOW, "reservoir_rfc_forecasts_unavailable_action": "level_pool"},
        )
    line = _summaries(caplog)[0]
    assert "no forecast selected for any of 2 reservoir(s)" in line
    assert "Searched" in line


def test_observations_that_miss_t0_also_report(caplog):
    t0 = _t0()
    with caplog.at_level(logging.INFO):
        assemble_rfc_dataframes(
            _frame("A1", t0 + pd.Timedelta(days=3)), _crosswalk(["A1"]), t0,
            {**_WITH_WINDOW, "reservoir_rfc_forecasts_unavailable_action": "level_pool"},
        )
    assert "no forecast selected for any of 1 reservoir(s)" in _summaries(caplog)[0]


def test_a_forecast_issued_after_t0_reports_a_negative_age(caplog):
    """Positive-offset windows reach past t0, so the age is signed."""
    t0 = _t0()
    frame = _frame("A1", t0)
    frame["issue_time"] = t0 + pd.Timedelta(hours=3)
    with caplog.at_level(logging.INFO):
        assemble_rfc_dataframes(frame, _crosswalk(["A1"]), t0,
                                {**_WITH_WINDOW, "reservoir_rfc_forecasts_offset_hours": 28})
    assert "-3 to -3 h old" in _summaries(caplog)[0]


def test_an_empty_window_is_reported_once(caplog):
    """The reader names the folder and window; the summary counts the reservoirs.

    Saying it a third time at assembly tells an operator nothing new.
    """
    t0 = _t0()
    with caplog.at_level(logging.INFO):
        obs, _ = assemble_rfc_dataframes(
            pd.DataFrame(), _crosswalk(["A1"]), t0,
            {**_WITH_WINDOW, "reservoir_rfc_forecasts_unavailable_action": "level_pool"},
        )
    assert obs.empty
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == []
    assert len(_summaries(caplog)) == 1


def test_an_empty_frame_is_still_fatal_by_default():
    """Suppressing the duplicate must not soften the policy."""
    with pytest.raises(ValueError, match="no RFC timeseries observations"):
        assemble_rfc_dataframes(pd.DataFrame(), _crosswalk(["A1"]), _t0(), _WITH_WINDOW)


# ------------------------------------------- the crosswalk shape the builders produce


@pytest.mark.parametrize("build", [_crosswalk, _crosswalk_nhf], ids=["indexed", "nhf"])
def test_a_gageless_lake_is_reported_whichever_crosswalk_shape(build, caplog):
    """NHF hands this over with the lake id in a column; the policy must not depend
    on which builder produced the crosswalk."""
    t0 = _t0()
    crosswalk = build(["A1", None])
    with caplog.at_level(logging.WARNING):
        _, par = assemble_rfc_dataframes(_frame("A1", t0), crosswalk, t0, _PARAMS)
    assert not par.loc[102, "use_rfc"]
    assert "no gage in the hydrofabric" in caplog.text
    # The gageless lake is NOT an availability failure, so it must not be reported
    # as one -- under the default action that would have ended the run.
    assert "have a gage but no usable forecast" not in caplog.text
