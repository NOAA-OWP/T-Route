"""The RFC selection window, in the shapes production actually runs.

The window is ``[t0 + offset - lookback, t0 + offset]`` matched against the issue stamp
in the filename, so ``offset`` decides whether it reaches forward of t0 at all and
``lookback`` decides how far back it searches for an issue.

Extended AnA runs ``lookback 28 / offset 0`` (issue 7962). A window narrower than the
interval between issues finds nothing at most cycle times, which is a configuration
problem rather than a code one, so it has to fail in a way that says so.
"""
from __future__ import annotations

import shutil
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

from troute.DataAssimilation import _read_timeseries_files

_REAL = Path(__file__).parent.parent / "BMI" / "rfc_timeseries"
_FIXTURE = next(_REAL.glob("*.RFCTimeSeries.ncdf"), None)
pytestmark = pytest.mark.skipif(_FIXTURE is None, reason="no RFC fixture available")


def _window(t0: pd.Timestamp, lookback: int, offset: int) -> list[str]:
    """The hourly stamps DataAssimilation builds from these two settings."""
    end = t0 + timedelta(hours=offset)
    start = end - timedelta(hours=lookback)
    stamps, d = [], start
    while d <= end:
        stamps.append(d.strftime("%Y-%m-%d_%H"))
        d += timedelta(hours=1)
    return stamps


def _issue(name: str) -> pd.Timestamp:
    return pd.Timestamp(name.split(".", 1)[0].replace("_", " ") + ":00:00")


def _folder(tmp_path: Path) -> Path:
    shutil.copy(_FIXTURE, tmp_path / _FIXTURE.name)
    return tmp_path


def test_the_window_is_backward_when_the_offset_is_zero() -> None:
    """offset 0 is what Extended AnA runs: nothing after t0 is eligible."""
    t0 = pd.Timestamp("2026-09-08 11:00")
    stamps = _window(t0, lookback=28, offset=0)
    assert stamps[0] == "2026-09-07_07"
    assert stamps[-1] == "2026-09-08_11"


def test_a_positive_offset_reaches_past_t0() -> None:
    """The shipped test configs use 28/28, which selects issues dated after t0."""
    t0 = pd.Timestamp("2026-09-08 11:00")
    stamps = _window(t0, lookback=28, offset=28)
    assert stamps[0] == "2026-09-08_11"
    assert stamps[-1] == "2026-09-09_15"


def test_the_extended_ana_window_selects_the_issue_in_hand(tmp_path) -> None:
    """lookback 28 / offset 0, the configuration confirmed in issue 7962."""
    t0 = _issue(_FIXTURE.name)
    got = _read_timeseries_files(
        str(_folder(tmp_path)), _window(t0, 28, 0), t0, t0 + timedelta(days=11)
    )
    assert set(got["stationId"]) == {_FIXTURE.name.split(".")[2]}


def test_a_window_narrower_than_the_issue_interval_names_the_window(tmp_path) -> None:
    """Issues arrive every 6 to 24 hours, so a 3 hour window usually finds none.

    That is a configuration problem, and the message has to carry the window bounds so
    it reads as one rather than as missing data.
    """
    t0 = _issue(_FIXTURE.name) + timedelta(hours=18)
    with pytest.raises(FileNotFoundError, match="lookback window"):
        _read_timeseries_files(
            str(_folder(tmp_path)), _window(t0, 3, 0), t0, t0 + timedelta(days=11)
        )


def test_a_narrow_window_degrades_when_the_policy_allows(tmp_path) -> None:
    """An operational cycle keeps running on level pool rather than breaking a chain."""
    t0 = _issue(_FIXTURE.name) + timedelta(hours=18)
    got = _read_timeseries_files(
        str(_folder(tmp_path)), _window(t0, 3, 0), t0, t0 + timedelta(days=11),
        unavailable_action="level_pool",
    )
    assert got.empty
