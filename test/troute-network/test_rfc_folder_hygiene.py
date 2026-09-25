"""Only files the RFC ingestion writes are read; anything else in the folder is ignored.

Names are split on ``.`` into a fixed five fields, so a name with a different number of
parts reaches pandas as a ragged row rather than a readable error.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pandas as pd
import pytest

from troute.DataAssimilation import _read_timeseries_files
from troute.routing.fast_reach.reservoir_RFC_da import _RFC_FILENAME

_REAL = Path(__file__).parent.parent / "BMI" / "rfc_timeseries"
_FIXTURE = next(_REAL.glob("*.RFCTimeSeries.ncdf"), None)
pytestmark = pytest.mark.skipif(_FIXTURE is None, reason="no RFC fixture available")


def _folder(tmp_path: Path, extra: list[str]) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    shutil.copy(_FIXTURE, tmp_path / _FIXTURE.name)
    for name in extra:
        (tmp_path / name).touch()
    return tmp_path


def _dates(t0: pd.Timestamp) -> list[str]:
    return [(t0 - pd.Timedelta(hours=h)).strftime("%Y-%m-%d_%H") for h in range(28)]


def _t0() -> pd.Timestamp:
    return pd.Timestamp(_FIXTURE.name.split(".")[0].replace("_", " ") + ":00:00")


@pytest.mark.parametrize(
    "extra",
    [
        pytest.param([], id="forecast_only"),
        pytest.param(["README.txt"], id="fewer_dots"),
        pytest.param([_FIXTURE.name + ".gz" if _FIXTURE else "x.gz"], id="more_dots"),
        pytest.param(["PARTIAL"], id="no_extension"),
        pytest.param(["retrieval.log", "notes.md.bak"], id="several"),
    ],
)
def test_stray_files_do_not_break_the_read(tmp_path, extra):
    t0 = _t0()
    got = _read_timeseries_files(
        str(_folder(tmp_path, extra)), _dates(t0), t0, t0 + pd.Timedelta(days=11)
    )
    assert not got.empty
    assert set(got["stationId"]) == {_FIXTURE.name.split(".")[2]}


def test_ignoring_a_stray_file_is_reported(tmp_path, caplog):
    t0 = _t0()
    with caplog.at_level(logging.WARNING):
        _read_timeseries_files(
            str(_folder(tmp_path, ["README.txt"])), _dates(t0), t0,
            t0 + pd.Timedelta(days=11),
        )
    assert "ignoring 1 file(s)" in caplog.text
    assert "README.txt" in caplog.text


def test_a_folder_of_only_stray_files_names_the_real_problem(tmp_path):
    """Not a ragged-DataFrame error: the run has no forecasts, and should say so."""
    (tmp_path / "README.txt").touch()
    t0 = _t0()
    with pytest.raises(FileNotFoundError, match="no RFC timeseries file"):
        _read_timeseries_files(str(tmp_path), _dates(t0), t0, t0 + pd.Timedelta(days=11))


def test_stray_files_do_not_change_a_single_value(tmp_path):
    """The filter must be invisible to the result, not merely non-fatal."""
    t0, horizon = _t0(), _t0() + pd.Timedelta(days=11)
    clean = _read_timeseries_files(str(_folder(tmp_path / "a", [])), _dates(t0), t0, horizon)
    strays = _read_timeseries_files(
        str(_folder(tmp_path / "b", ["README.txt", _FIXTURE.name + ".gz", "PARTIAL"])),
        _dates(t0), t0, horizon,
    )
    pd.testing.assert_frame_equal(clean, strays)


def test_one_warning_per_read_however_many_are_ignored(tmp_path, caplog):
    t0 = _t0()
    with caplog.at_level(logging.WARNING):
        _read_timeseries_files(
            str(_folder(tmp_path, ["a.log", "b.log", "c.log"])), _dates(t0), t0,
            t0 + pd.Timedelta(days=11),
        )
    ignoring = [r for r in caplog.records if "ignoring" in r.getMessage()]
    assert len(ignoring) == 1
    assert "ignoring 3 file(s)" in ignoring[0].getMessage()


def test_a_directory_named_like_a_forecast_is_ignored(tmp_path):
    """The name alone does not settle it; only a file can be read."""
    d = _folder(tmp_path, [])
    (d / "2023-01-01_00.60min.FAKE1.RFCTimeSeries.ncdf").mkdir()
    t0 = _t0()
    got = _read_timeseries_files(str(d), _dates(t0), t0, t0 + pd.Timedelta(days=11))
    assert set(got["stationId"]) == {_FIXTURE.name.split(".")[2]}


@pytest.mark.parametrize("cadence", ["05", "60", "180", "1440"])
def test_every_cadence_width_the_generator_can_write_is_accepted(cadence):
    """`zfill(2)` is a minimum width, so 3 and 4 digit cadences are legitimate."""
    assert _RFC_FILENAME.fullmatch(f"2023-12-10_12.{cadence}min.BNGM1.RFCTimeSeries.ncdf")


@pytest.mark.parametrize(
    "name",
    [
        "2023-12-10_12.60min.BNGM1.RFCTimeSeries.ncdf.gz",
        "2023-12-10_12.60min.BNGM1.RFCTimeSeries.ncdf\n",
        "2023-12-10.60min.BNGM1.RFCTimeSeries.ncdf",
        "2023-12-10_12.60.BNGM1.RFCTimeSeries.ncdf",
        "2023-12-10_12.60min.BN.GM1.RFCTimeSeries.ncdf",
        "README.txt",
    ],
)
def test_names_the_ingestion_does_not_write_are_rejected(name):
    assert _RFC_FILENAME.fullmatch(name) is None
