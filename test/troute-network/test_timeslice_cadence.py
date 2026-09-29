"""Cadence of the TimeSlice directory, measured rather than assumed.

``build_da_sets`` hardcoded 15 minutes and a ``pad_hours * 4`` pad, so any other
cadence had most of its enumerated stack counted as missing files. The two
independent implementations must reach the same helper, hence the parametrization.
"""
from datetime import datetime, timedelta

import pytest

import troute.hyfeature_network_utilities as hnu
import troute.nhd_network_utilities_v02 as nnu

_MODULES = pytest.mark.parametrize("module", [hnu, nnu], ids=["hyfeature", "nhd"])


def _write_slices(folder, start, step, count, skip=()):
    """Create empty TimeSlice files at *step* spacing, omitting indices in *skip*."""
    for i in range(count):
        if i in skip:
            continue
        stamp = (start + i * step).strftime("%Y-%m-%d_%H:%M:%S")
        (folder / f"{stamp}.15min.usgsTimeSlice.ncdf").touch()


def test_both_builders_share_one_helper():
    """The two build_da_sets copies must not drift apart on cadence."""
    assert nnu._timeslice_cadence is hnu._timeslice_cadence
    assert nnu._timeslice_lead_out is hnu._timeslice_lead_out


@_MODULES
@pytest.mark.parametrize(
    "step",
    [timedelta(minutes=15), timedelta(hours=1), timedelta(minutes=5)],
    ids=["15min", "hourly", "5min"],
)
def test_cadence_is_measured(tmp_path, module, step):
    _write_slices(tmp_path, datetime(2020, 1, 1), step, 6)
    assert module._timeslice_cadence(tmp_path) == step


@_MODULES
def test_a_gap_does_not_move_the_cadence(tmp_path, module):
    """The most common gap wins, so one absent file cannot double the answer."""
    _write_slices(tmp_path, datetime(2020, 1, 1), timedelta(hours=1), 8, skip=(3,))
    assert module._timeslice_cadence(tmp_path) == timedelta(hours=1)


@_MODULES
@pytest.mark.parametrize("count", [0, 1], ids=["empty", "single-file"])
def test_falls_back_when_unmeasurable(tmp_path, module, count):
    _write_slices(tmp_path, datetime(2020, 1, 1), timedelta(hours=1), count)
    assert module._timeslice_cadence(tmp_path) == timedelta(minutes=15)


@_MODULES
@pytest.mark.parametrize("folder", [None, "", "does/not/exist"])
def test_falls_back_without_a_directory(tmp_path, module, folder):
    target = tmp_path / folder if folder else folder
    assert module._timeslice_cadence(target) == timedelta(minutes=15)


@_MODULES
def test_unparseable_names_are_skipped(tmp_path, module):
    _write_slices(tmp_path, datetime(2020, 1, 1), timedelta(hours=1), 4)
    (tmp_path / "not-a-stamp.15min.usgsTimeSlice.ncdf").touch()
    assert module._timeslice_cadence(tmp_path) == timedelta(hours=1)


@_MODULES
@pytest.mark.parametrize(
    ("cadence", "expected"),
    [
        # An hour of lead-out at any cadence at or below an hour: four files at the
        # 15 minute cadence this used to hardcode, twelve at 5 minutes, one at hourly.
        (timedelta(minutes=15), timedelta(hours=1)),
        (timedelta(minutes=5), timedelta(hours=1)),
        (timedelta(hours=1), timedelta(hours=1)),
        # Coarser than the lead-out: one file, never zero, or the last model timestep
        # has no observation after it to interpolate against.
        (timedelta(hours=6), timedelta(hours=6)),
        (timedelta(days=1), timedelta(days=1)),
    ],
    ids=["15min", "5min", "hourly", "6-hourly", "daily"],
)
def test_lead_out_tracks_cadence(module, cadence, expected):
    assert module._timeslice_lead_out(cadence) == expected


@_MODULES
def test_lead_out_matches_the_old_hardcoded_value_at_15_minutes(module):
    """The historical `dt_timeslice * 4` was an hour only because dt was 15 minutes."""
    fifteen = timedelta(minutes=15)
    assert module._timeslice_lead_out(fifteen) == fifteen * 4


@_MODULES
def test_lead_out_is_never_shorter_than_one_file(module, tmp_path):
    for cadence in (timedelta(minutes=1), timedelta(hours=3), timedelta(days=7)):
        assert module._timeslice_lead_out(cadence) >= cadence
