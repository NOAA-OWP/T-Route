"""The RFC DA index must be bounded by the array it indexes, not by the file count.

``total_counts`` reaches the kernel rebased off the RFC file's own value, while
``time_series`` is the pivoted observation frame, truncated at the persist horizon
and padded onto whatever grid the other stations in the domain imply. Nothing tied
the two together, so a forecast that ended before the run did walked the index off
the end of the memoryview and the whole joblib worker died with
``IndexError: Out of bounds on buffer access (axis 0)``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from troute.DataAssimilation import assemble_rfc_dataframes
from troute.routing.fast_reach.reservoir_RFC_da import reservoir_RFC_da

_KWARGS = {
    "routing_period": 300.0,
    "DA_time_step": 300,
    "rfc_forecast_persist_seconds": 11 * 86400,
    "reservoir_type": 4,
    "inflow": 1.0,
    "water_elevation": 10.0,
    "levelpool_outflow": 1.0,
    "levelpool_water_elevation": 10.0,
    "lake_area": 1.0e6,
    "max_water_elevation": 20.0,
    "rfc_file": "f",
}


def _run(time_series, total_counts, idx, steps):
    """Step the DA the way the kernel does, carrying idx/update_time forward."""
    update_time = 0.0
    for step in range(1, steps + 1):
        _, _, update_time, idx, *_ = reservoir_RFC_da(
            True, time_series, idx, total_counts, current_time=300.0 * step,
            update_time=update_time, **_KWARGS
        )
    return idx


def test_index_holds_at_the_last_observation():
    ts = np.arange(10, dtype="float32")
    # total_counts equal to the array width must not let the index reach 10.
    assert _run(ts, 10, 8, 5) == 9


def test_index_is_bounded_by_a_truncated_array():
    # The pivoted frame is shorter than the file's own count.
    ts = np.arange(4, dtype="float32")
    assert _run(ts, 20, 0, 10) == 3


def test_empty_forecast_falls_back_to_level_pool():
    ts = np.zeros(0, dtype="float32")
    outflow, _, _, _, res_type, assimilated, _ = reservoir_RFC_da(
        True, ts, 0, 0, current_time=300.0, update_time=0.0, **_KWARGS
    )
    assert outflow == _KWARGS["levelpool_outflow"]
    assert res_type == 1
    assert assimilated == -9999.0


def test_the_last_assembled_observation_is_still_assimilated():
    """The bound must not eat the forecast tail.

    ``assemble_rfc_dataframes`` hands the kernel a total_counts that is already the
    inclusive index of the last observation in the pivoted frame, so treating it as a
    count stops one observation early. No synthetic ``total_counts == len`` case can
    show that, hence going through the assembler.
    """
    t0 = pd.Timestamp("2021-10-21 12:00:00")
    discharges = [10.0, 20.0, 30.0, 40.0]
    rfc_df = pd.DataFrame(
        {
            "stationId": "KNFC1",
            "discharges": discharges,
            "Datetime": pd.date_range(t0 - pd.Timedelta(hours=1), periods=4, freq="h"),
            "totalCounts": 4,
            "timeseries_idx": 1,
            "file": "f",
            "use_rfc": True,
            "da_timestep": 3600,
            "issue_time": t0,
        }
    )
    crosswalk = pd.DataFrame(
        {"rfc_gage_id": ["KNFC1"], "rfc_lake_id": [17609317]}
    ).set_index("rfc_lake_id")
    obs, params = assemble_rfc_dataframes(
        rfc_df, crosswalk, t0, {"reservoir_rfc_forecast_persist_days": 11}
    )
    series = obs.to_numpy(dtype="float32")[0]
    idx = int(params["timeseries_idx"].iloc[0])
    counts = int(params["totalCounts"].iloc[0])

    # Step far enough that the index would run off the end if it were unbounded.
    update_time, outflow = 0.0, None
    for step in range(1, 40):
        outflow, _, update_time, idx, *_ = reservoir_RFC_da(
            True, series, idx, counts, current_time=300.0 * step,
            update_time=update_time, **_KWARGS
        )
    assert idx == len(discharges) - 1
    assert outflow == discharges[-1]


def test_standalone_setup_without_a_file_falls_back_to_level_pool():
    """The not-found branch has to define everything the unconditional return hands back.

    ``lookback_hours`` was bound only when the backward search hit, and the four
    timeseries values only when the file opened, so a reservoir whose RFC file was
    missing raised ``UnboundLocalError`` out of setup. Past that, the 99999/1
    sentinels meant to fail validation are not iterable, so validation raised too.
    """
    from troute.routing.fast_reach.reservoir_RFC_da import preprocess_RFC_data

    use_rfc, series, idx, update_time, step_seconds, counts, _, _ = preprocess_RFC_data(
        "2021-10-21_12:00:00", 28, "NOSUCHGAGE", "test/BMI/rfc_timeseries/", 17609317, 300
    )
    assert use_rfc is False
    outflow, _, _, _, res_type, *_ = reservoir_RFC_da(
        use_rfc, series, idx, counts, current_time=300.0, update_time=update_time,
        **{**_KWARGS, "DA_time_step": step_seconds}
    )
    assert outflow == _KWARGS["levelpool_outflow"]
    assert res_type == 1


def test_validation_rejects_unusable_and_absurd_discharges():
    """Both checks were no-ops; restored with the asymmetry the mechanisms imply.

    ``any(time_series) < 0`` compares a BOOL to zero, so neither this nor the 90,000 cms
    check ever rejected anything. Missing values are checked with ALL, not ANY: the
    generator's mergeOld pads a short series out to the length of the one it merges
    with, using the file's own missingValue, so isolated -999s come out of the normal
    operational pipeline, and reservoir_RFC_da already walks back past one. Nothing
    downstream compensates for an absurd value, so one of those is still enough.
    """
    from troute.routing.fast_reach.reservoir_RFC_da import _validate_RFC_data

    def check(values):
        arr = np.asarray(values, dtype=float)
        return _validate_RFC_data(1, arr, np.zeros(arr.size), "", "", 300, False)

    assert check([5.0, 7.0]) is True
    assert check([5.0, -999.0, 7.0]) is True     # padded gap, backtrack handles it
    assert check([-999.0, -999.0]) is False      # nothing usable in the window
    assert check([5.0, 1.0e6]) is False          # absurd, nothing compensates
    # NaN fails every comparison, so it passed both checks and the consumer's backtrack.
    assert check([5.0, float("nan"), 7.0]) is True
    assert check([float("nan"), float("nan")]) is False
    assert check([float("nan"), -999.0]) is False


@pytest.mark.parametrize(
    ("routing_period", "cadence", "expected"),
    [
        (300, 3600, True),    # ngen's own shape: dt=300, qts_subdivisions=12
        (1800, 3600, True),
        (3600, 3600, True),   # keeps pace, and both drivers now seed alike
        (7200, 3600, False),
        (300, 900, True),
        (1800, 900, False),   # sub-hourly cadence, so a hardcoded 3600 would miss it
    ],
)
def test_routing_period_must_be_shorter_than_the_forecast_cadence(
    routing_period, cadence, expected
):
    """The bound is the file's cadence, not a literal hour.

    ``reservoir_RFC_da`` advances the forecast at most one index per routing step, so
    above the cadence it reads the forecast slowed by routing_period/cadence and never
    recovers. At the cadence it keeps pace, and both drivers seed the index alike, so it
    is allowed. The check compared against a hardcoded 3600, right only for hourly
    files: a 15 min forecast routed at 1800 s ran at half rate and passed validation.

    Unreachable from ngen, which subdivides its hourly forcing into dt=300 with
    qts_subdivisions=12, so dt is a fraction of the cadence by construction.
    """
    from troute.routing.fast_reach.reservoir_RFC_da import _validate_RFC_data

    good = np.array([5.0, 7.0])
    assert _validate_RFC_data(
        1, good, np.zeros(2), "", "", routing_period, False, da_time_step=cadence
    ) is expected


@pytest.mark.parametrize(
    ("series", "idx", "outflow", "res_type"),
    [
        ([3.0, float("nan"), 7.0], 0, _KWARGS["levelpool_outflow"], 1),
        ([3.0, 4.0, float("nan")], 1, 4.0, 4),   # walks back to the last good sample
        ([3.0, 4.0, -999.0], 1, 4.0, 4),
        ([float("nan")] * 3, 0, _KWARGS["levelpool_outflow"], 1),
    ],
)
def test_non_finite_samples_never_reach_the_reservoir(series, idx, outflow, res_type):
    """A NaN fill must reach neither the outflow nor the water elevation.

    Every comparison against NaN is false, so validation and the consumer's
    ``outflow < 0`` backtrack both need explicit tests. The elevation is formed before the
    backtrack runs, so repairing the outflow alone left it NaN.
    """
    out, elevation, _, _, dynamic_type, *_ = reservoir_RFC_da(
        True, np.asarray(series, dtype="float32"), idx, len(series) - 1,
        current_time=3600.0, update_time=0.0, **_KWARGS
    )
    assert float(out) == pytest.approx(outflow)
    assert np.isfinite(float(elevation))
    assert dynamic_type == res_type
