"""Assembling last-observation state must not depend on how the network was split.

``new_lastobs`` concatenates one frame per compute job. A job holding no gages
contributes an empty frame, and pandas 3 stops ignoring those when it works out the
result dtype, so the column types would start depending on the partitioning. It also
warns about it today. Drop the empty frames instead, and keep the all-empty case
returning the same columns rather than raising out of ``pd.concat``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from troute.DataAssimilation import new_lastobs

_COLUMNS = ["time_since_lastobs", "lastobs_discharge"]


def _job(ids: list[int], since: list[float], discharge: list[float]):
    """One run_results entry, of which only element 3 carries last-obs state."""
    return (
        None, None, None,
        (np.array(ids), np.array(since, dtype=float), np.array(discharge, dtype=float)),
    )


def test_jobs_without_gages_are_dropped():
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        out = new_lastobs([_job([], [], []), _job([10, 11], [300.0, 600.0], [1.5, 2.5])], 60.0)
    assert list(out.columns) == _COLUMNS
    assert out.index.tolist() == [10, 11]
    # time_increment is removed from every retained row.
    assert out["time_since_lastobs"].tolist() == [240.0, 540.0]
    assert out["lastobs_discharge"].tolist() == [1.5, 2.5]


def test_no_gages_anywhere_returns_the_same_columns():
    """pd.concat on an all-empty list raises; the callers expect a frame."""
    out = new_lastobs([_job([], [], []), _job([], [], [])], 60.0)
    assert out.empty
    assert list(out.columns) == _COLUMNS


def test_column_dtypes_do_not_depend_on_the_partitioning():
    populated = _job([10], [300.0], [1.5])
    with_empty = new_lastobs([_job([], [], []), populated], 60.0)
    without_empty = new_lastobs([populated], 60.0)
    pd.testing.assert_frame_equal(with_empty, without_empty)
