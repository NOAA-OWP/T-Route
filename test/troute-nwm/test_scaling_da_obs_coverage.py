"""TimeSlices that miss the simulation period must be fatal, not silent.

``scaling_da/preprocess.py`` states the contract: "All failure modes raise: a
misconfigured observation source must not degrade into a silent no-DA run." Its
guards check the station ROSTER, though, never temporal coverage. Observations whose
stations match but whose timestamps miss the run reindexed to an all-NaN frame of the
full width. That frame is not empty, so nothing downstream rejected it and
``_align_obs_to_model_steps`` saw columns already equal to the grid and returned
early without even its no-overlap warning. The run completed having assimilated
nothing.

Making it fatal was wrong: a forecast leg legitimately runs past the newest
observation, which is the same disjoint span a stale directory produces. Warn, as
nudging does for the identical condition, and let the kernel persist and decay.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest
from nwm_routing.scaling_da_apply import ScalingDA

_TIMESLICES = Path(__file__).parents[1] / "LowerColorado_TX" / "usgs_TimeSlice"
_SITES = ["08117995", "08120500"]
# The window these fixture files actually cover.
_COVERED_T0 = pd.Timestamp("2021-08-23 00:00:00")


def _reader() -> ScalingDA:
    """A ScalingDA carrying only what the observation path reads."""
    da = object.__new__(ScalingDA)
    da._da_sites = list(_SITES)
    da._obs_sites = set(_SITES)
    da._ts_folder = _TIMESLICES
    da._obs_cache = None
    da._qc_threshold = 1
    da._interpolation_limit = 59
    da._cpu_pool = 1
    da.synthetic_factor = None
    da.gage_seg = {"08117995": 101, "08120500": 202}
    return da


def _da_run() -> dict[str, list[str]]:
    files = sorted(_TIMESLICES.glob("*.usgsTimeSlice.ncdf"))[:8]
    return {"usgs_timeslice_files": [f.name for f in files]}


@pytest.mark.skipif(not _TIMESLICES.is_dir(), reason="LowerColorado TimeSlices absent")
def test_observations_missing_the_window_warn_and_carry_on(caplog):
    """A forecast leg starts after the newest observation. That must not be fatal."""
    with caplog.at_level(logging.WARNING, logger="TROUTE"):
        out = _reader().build_usgs_df(pd.Timestamp("2024-01-01"), 300.0, 12, _da_run())
    assert not out.notna().to_numpy().any()
    assert "do not overlap the simulation window" in caplog.text


@pytest.mark.skipif(not _TIMESLICES.is_dir(), reason="LowerColorado TimeSlices absent")
def test_covering_observations_are_assimilated():
    out = _reader().build_usgs_df(_COVERED_T0, 300.0, 12, _da_run())
    assert out.shape == (2, 13)
    assert out.notna().to_numpy().any()
