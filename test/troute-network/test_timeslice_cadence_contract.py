"""Both timeslice products declare a cadence, and it has to be one the run can use.

The retrieval writes only cadences that divide 60 with no remainder
(``RFCHelper.makeAllTimeSeries``), which is what puts a sample on every hour and lets an
hourly run line up with the product. Nothing downstream reads the declared cadence, so a
file that is not what it claims is read at the wrong rate with no error.
"""
from __future__ import annotations

import numpy as np
import pytest

from troute.nhd_io import _check_timeslice_cadence

_NAME = "2023-12-15_00:00:00.{}min.usgsTimeSlice.ncdf"


@pytest.mark.parametrize("minutes", [1, 5, 15, 30, 60])
def test_a_cadence_that_divides_60_is_accepted(minutes: int) -> None:
    _check_timeslice_cadence(_NAME.format(minutes), minutes)


@pytest.mark.parametrize("minutes", [7, 45, 90, 180])
def test_a_cadence_that_does_not_divide_60_is_refused(minutes: int) -> None:
    with pytest.raises(ValueError, match="does not divide 60"):
        _check_timeslice_cadence(_NAME.format(minutes), minutes)


def test_a_file_that_disagrees_with_its_own_name_is_refused() -> None:
    with pytest.raises(ValueError, match="disagrees about its own cadence"):
        _check_timeslice_cadence(_NAME.format(15), 5)


def test_a_name_with_no_cadence_token_is_refused() -> None:
    with pytest.raises(ValueError, match="no cadence token"):
        _check_timeslice_cadence("usgsTimeSlice.ncdf", 15)


@pytest.mark.parametrize("declared", [15.9, "15.5", 0.25])
def test_a_fractional_cadence_is_refused(declared) -> None:
    """int() would truncate 15.9 to 15, which then agrees with a 15min filename."""
    with pytest.raises(ValueError, match="whole minutes"):
        _check_timeslice_cadence(_NAME.format(15), declared)


@pytest.mark.parametrize("declared", [15, "15", 15.0])
def test_a_whole_cadence_is_accepted_however_it_is_typed(declared) -> None:
    """netCDF attributes come back as str or as a number depending on the writer."""
    _check_timeslice_cadence(_NAME.format(15), declared)


def test_a_file_predating_the_attribute_is_left_alone() -> None:
    """Rejecting these would break archives for metadata that was never read before."""
    _check_timeslice_cadence("anything at all", None)


def test_the_rfc_reader_refuses_a_cadence_that_does_not_divide_60(tmp_path) -> None:
    """Same rule on the forecast side, where the cadence is already cross-checked."""
    import shutil
    from pathlib import Path

    import netCDF4

    from troute.routing.fast_reach.reservoir_RFC_da import read_rfc_timeseries

    src = next((Path(__file__).parent.parent / "BMI" / "rfc_timeseries").glob("*.ncdf"), None)
    if src is None:
        pytest.skip("no RFC fixture available")
    dst = tmp_path / src.name.replace(".60min.", ".90min.")
    shutil.copy(src, dst)
    with netCDF4.Dataset(str(dst), "a") as ds:
        ds.sliceTimeResolutionMinutes = "90"
        ds.variables["timeSteps"][:] = np.full(
            ds.variables["timeSteps"].shape, 90 * 60, dtype="i4"
        )
    with pytest.raises(ValueError, match="does not divide 60"):
        read_rfc_timeseries(dst)
