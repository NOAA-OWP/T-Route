"""The legacy BMI publishes a depth too, and must not convert it twice.

Its two output paths rebuild the frame from the same ``_run_results``, back to back,
and its ``pd.concat(..., copy=False)`` has no column drop to sever the alias to the
kernel array. Converting in place would subtract the orifice invert twice.
"""
import numpy as np
import pandas as pd
import pytest

tm = pytest.importorskip(
    "troute_model", reason="legacy root-level BMI module is not on the path"
)

_LAKE, _CHANNEL, _NTS = 900, 10, 2
_INVERT = 260.0


@pytest.fixture
def results_and_waterbodies():
    """One compute job: a lake carrying elevation in its depth slots, and a channel."""
    arr = np.array(
        [
            [5.0, 0.0, 261.5, 5.0, 0.0, 262.0],   # lake: ELEVATION in the d slots
            [3.0, 1.2, 0.75, 3.0, 1.2, 0.80],     # channel: a real depth
        ],
        dtype="float32",
    )
    ids = np.array([_LAKE, _CHANNEL])
    inflow = np.zeros((2, _NTS), dtype="float32")
    results = [(ids, arr, 0, None, None, None, None, inflow)]
    waterbodies = pd.DataFrame({"OrificeE": [_INVERT]}, index=[_LAKE])
    return results, waterbodies, arr


def _depths(frame, seg):
    return [float(v) for c, v in frame.loc[seg].items() if c[1] == "d"]


def test_create_output_dataframes_publishes_stage(results_and_waterbodies):
    results, waterbodies, _ = results_and_waterbodies
    fvd, _ = tm._create_output_dataframes(results, _NTS, waterbodies)
    assert _depths(fvd, _LAKE) == pytest.approx([1.5, 2.0])
    assert _depths(fvd, _CHANNEL) == pytest.approx([0.75, 0.80], rel=1e-6)


def test_lakeout_keeps_the_elevation(results_and_waterbodies):
    results, waterbodies, _ = results_and_waterbodies
    _, lakeout = tm._create_output_dataframes(results, _NTS, waterbodies)
    assert [float(x) for x in lakeout.loc[_LAKE].values[-_NTS:]] == pytest.approx([261.5, 262.0])


def test_retrieve_last_output_publishes_stage(results_and_waterbodies):
    results, waterbodies, _ = results_and_waterbodies
    _, _, d_channel, _, _, d_lakeout = tm._retrieve_last_output(results, _NTS, waterbodies)
    assert float(d_channel.loc[_LAKE]) == pytest.approx(2.0)
    assert float(d_channel.loc[_CHANNEL]) == pytest.approx(0.80, rel=1e-6)
    # the lakeout view of the same row stays an elevation
    assert float(d_lakeout.loc[_LAKE]) == pytest.approx(262.0)


def test_neither_path_mutates_the_kernel_array(results_and_waterbodies):
    results, waterbodies, arr = results_and_waterbodies
    before = arr.copy()
    tm._create_output_dataframes(results, _NTS, waterbodies)
    tm._retrieve_last_output(results, _NTS, waterbodies)
    np.testing.assert_array_equal(arr, before)


def test_running_both_paths_converts_once_not_twice(results_and_waterbodies):
    """The regression: the second path must not re-subtract the invert."""
    results, waterbodies, _ = results_and_waterbodies
    fvd, _ = tm._create_output_dataframes(results, _NTS, waterbodies)
    _, _, d_channel, _, _, _ = tm._retrieve_last_output(results, _NTS, waterbodies)
    assert _depths(fvd, _LAKE)[-1] == pytest.approx(2.0)
    assert float(d_channel.loc[_LAKE]) == pytest.approx(2.0)
