"""A reservoir's zero release is an observation.

A hydropower dam reports zero every hour its turbines are off. Read as missing, as for a
stream gage, those hours leave the persistence DA holding the last nonzero release.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import netCDF4
import numpy as np
import pandas as pd
import pytest

from troute import nhd_io
from troute.DataAssimilation import _create_reservoir_df

_T0 = pd.Timestamp("2026-09-01 00:00")
_SITE = "LD12_Ozark"
_LAKE = 7
# Generating, then off for three hours, then generating again.
_RELEASES = [300.0, 0.0, 0.0, 0.0, 400.0]


def _write(folder: Path, family: str) -> list[str]:
    """Hourly TimeSlices for one station, named the way the retrieval names them."""
    names = []
    for hour, q in enumerate(_RELEASES):
        stamp = (_T0 + pd.Timedelta(hours=hour)).strftime("%Y-%m-%d_%H:%M:%S")
        name = f"{stamp}.60min.{family}TimeSlice.ncdf"
        with netCDF4.Dataset(folder / name, "w") as nc:
            nc.sliceTimeResolutionMinutes = "60"
            nc.createDimension("stationIdInd", 1)
            nc.createDimension("stationIdStringLength", 15)
            nc.createDimension("timeStringLength", 19)
            ids = nc.createVariable("stationId", "S1", ("stationIdInd", "stationIdStringLength"))
            ids[:] = np.array([list(_SITE.rjust(15))], dtype="S1")
            times = nc.createVariable("time", "S1", ("stationIdInd", "timeStringLength"))
            times[:] = np.array([list(stamp)], dtype="S1")
            nc.createVariable("discharge", "f4", ("stationIdInd",))[:] = [q]
            nc.createVariable("discharge_quality", "i2", ("stationIdInd",))[:] = [100]
        names.append(name)
    return names


@pytest.mark.parametrize("zero_is_missing", [True, False])
def test_the_reader_drops_zero_only_when_asked(tmp_path: Path, zero_is_missing: bool) -> None:
    files = [tmp_path / n for n in _write(tmp_path, "usace")]
    crosswalk = pd.DataFrame({"gage": [_SITE], "lake": [_LAKE]})
    obs = nhd_io.get_obs_from_timeslices(
        crosswalk, "gage", "lake", files, 1, 59, 3600, _T0, 1,
        zero_is_missing=zero_is_missing,
    ).loc[_LAKE]
    # Two hours from either nonzero report, beyond the 59 minute interpolation limit.
    at_two = obs[_T0 + pd.Timedelta(hours=2)]
    if zero_is_missing:
        assert np.isnan(at_two)
    else:
        assert at_two == 0.0


def test_reservoir_da_keeps_a_zero_release(tmp_path: Path) -> None:
    names = _write(tmp_path, "usace")
    reservoir_df, _ = _create_reservoir_df(
        {"usace_timeslices_folder": str(tmp_path)}, {}, {}, {"cpu_pool": 1},
        SimpleNamespace(t0=_T0), {"usace_timeslice_files": names},
        pd.DataFrame({"usace_gage_id": [_SITE], "usace_lake_id": [_LAKE]}), "usace",
    )
    closed = reservoir_df.loc[_LAKE, _T0 + pd.Timedelta(hours=2)]
    assert closed == 0.0, f"the closed hour reads {closed}, the reservoir DA would release it"


def _bmi_forcing_reader():
    """model_DAforcing sits beside the BMI entry points in src/, outside any package."""
    src = str(Path(__file__).resolve().parents[2] / "src")
    sys.path.insert(0, src)
    try:
        import model_DAforcing
    finally:
        sys.path.remove(src)
    return model_DAforcing._read_timeslice_files


@pytest.mark.parametrize("zero_is_missing", [True, False])
def test_the_bmi_forcing_reader_drops_zero_only_when_asked(
        tmp_path: Path, zero_is_missing: bool) -> None:
    names = _write(tmp_path, "usace")
    obs = _bmi_forcing_reader()(str(tmp_path), [n[:19] for n in names], 1, 3600,
                                zero_is_missing=zero_is_missing)
    at_two = obs.loc[_SITE, _T0 + pd.Timedelta(hours=2)]
    if zero_is_missing:
        assert np.isnan(at_two)
    else:
        assert at_two == 0.0
