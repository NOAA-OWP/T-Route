"""A gage with history but nothing reported this window stays in the job.

The nudging roster comes from the lastobs index, so a gage carried in from a
checkpoint can outlive the observations for any single window. That is what the lastobs
decay is for, so the observation rows are reindexed rather than selected by label.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from troute.routing.compute import _prep_da_dataframes


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    times = pd.date_range("2021-10-21 12:00", periods=3, freq="5min")
    usgs = pd.DataFrame(np.arange(3.0).reshape(1, 3), index=[70], columns=times)
    lastobs = pd.DataFrame(
        {"time_since_lastobs": [0.0, 0.0], "lastobs_discharge": [1.0, 2.0],
         "gage": ["a", "b"]},
        index=[30, 70],
    )
    return usgs, lastobs


def test_a_gage_with_no_observation_this_window_does_not_raise() -> None:
    usgs, lastobs = _frames()
    out = _prep_da_dataframes(usgs, lastobs, pd.Index([30, 70]))
    assert out is not None


def test_a_diversion_gage_joins_the_roster_without_a_lastobs_row() -> None:
    """A lastobs file that predates the diversion must not silence its gage."""
    usgs, lastobs = _frames()
    lastobs = lastobs.drop(index=70)
    usgs_sub, lastobs_sub, positions = _prep_da_dataframes(
        usgs, lastobs, pd.Index([30, 70]), include_segments=(70,)
    )
    assert list(lastobs_sub.index) == [30, 70]
    assert lastobs_sub.loc[70].isna().all()
    assert not usgs_sub.loc[70].isna().any()
    assert list(positions) == [0, 1]


def test_the_unreported_gage_keeps_a_row_of_missing_observations() -> None:
    usgs, lastobs = _frames()
    usgs_df_sub = _prep_da_dataframes(usgs, lastobs, pd.Index([30, 70]))[0]
    assert 30 in usgs_df_sub.index
    assert usgs_df_sub.loc[30].isna().all()
    # The gage that did report keeps its values.
    assert not usgs_df_sub.loc[70].isna().any()
