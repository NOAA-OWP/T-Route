"""Observations resample onto the model's own timestep, whatever dt is.

The kernel reads column ``j`` at ``t0 + j*dt``, so the resample target has to be dt
exactly. Building it from truncated minutes put a dt that is not a whole number of
minutes onto the wrong grid, and any dt under 60 s onto no grid at all. Interpolating
on a fixed minute grid then sampling the target left the off-minute points NaN.

These drive ``_interpolate_one`` itself rather than restating its expression, so a
change to the production helper is what the assertions see.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from troute.nhd_io import _interpolate_one

_LIMIT = 59  # minutes, comfortably longer than the 15 min gap below


def _observations() -> pd.DataFrame:
    """One gage reporting every 15 minutes, the cadence the retrieval writes."""
    index = pd.date_range("2023-12-15 00:00", periods=5, freq="15min")
    return pd.DataFrame({"gage": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=index)


@pytest.mark.parametrize("dt", [30, 90, 300, 900, 3600])
def test_every_step_on_the_grid_gets_a_value(dt: int) -> None:
    out = _interpolate_one(_observations(), _LIMIT, f"{dt}s")
    assert not np.isnan(out).any(), f"dt={dt} left {int(np.isnan(out).sum())} gaps"


@pytest.mark.parametrize("dt", [30, 90, 300])
def test_the_values_are_the_interpolation_not_padding(dt: int) -> None:
    """A grid full of the same repeated value would also pass the NaN check."""
    out = _interpolate_one(_observations(), _LIMIT, f"{dt}s").ravel()
    assert out[0] == pytest.approx(1.0)
    assert out[-1] == pytest.approx(5.0)
    assert (np.diff(out) > 0).all()


def test_the_usual_dt_is_unchanged_by_the_switch_to_seconds() -> None:
    """300 s and 5 min are the same grid, so production results do not move."""
    obs = _observations()
    assert np.array_equal(
        _interpolate_one(obs, _LIMIT, "300s"), _interpolate_one(obs, _LIMIT, "5min")
    )


def test_a_gap_longer_than_the_limit_still_stays_missing() -> None:
    """The finer grid must not quietly extend the interpolation limit."""
    index = pd.DatetimeIndex(["2023-12-15 00:00", "2023-12-15 06:00"])
    obs = pd.DataFrame({"gage": [1.0, 2.0]}, index=index)
    out = _interpolate_one(obs, 5, "300s")
    assert np.isnan(out).any()
