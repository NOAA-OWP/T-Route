"""An observation frame without its parameter frame must name its own cause.

``_prep_reservoir_da_dataframes`` gates each reservoir DA block on the OBSERVATION
frame and then reads the PARAMETER frame. The two are built together by
``DataAssimilation``, so a populated observation frame with empty parameters means
the parameters were replaced afterwards -- in practice by restoring a BMI state file
written with that DA type turned off. Unguarded, the run died deep inside the compute
package on ``KeyError: 'totalCounts'``, which names a column and not the mistake.
"""
from __future__ import annotations

import pandas as pd
import pytest

from troute.routing.compute import _prep_reservoir_da_dataframes

_EMPTY = pd.DataFrame()
_T0 = pd.Timestamp("2020-01-01")


def _prep(**frames):
    """Run the prep with every frame empty except the ones named."""
    order = [
        "reservoir_usgs_df", "reservoir_usgs_param_df",
        "reservoir_usace_df", "reservoir_usace_param_df",
        "reservoir_usbr_df", "reservoir_usbr_param_df",
        "reservoir_rfc_df", "reservoir_rfc_param_df",
        "great_lakes_df", "great_lakes_param_df", "great_lakes_climatology_df",
    ]
    args = [frames.get(name, _EMPTY) for name in order]
    return _prep_reservoir_da_dataframes(
        *args, frames["waterbody_types_df_sub"], _T0, False
    )


@pytest.mark.parametrize(
    ("label", "reservoir_type", "obs_key"),
    [
        ("USGS", 2, "reservoir_usgs_df"),
        ("USACE", 3, "reservoir_usace_df"),
        ("USBR", 7, "reservoir_usbr_df"),
        ("RFC", 4, "reservoir_rfc_df"),
    ],
)
def test_observations_without_parameters_raise_with_the_cause(
    label: str, reservoir_type: int, obs_key: str
) -> None:
    obs = pd.DataFrame({_T0: [5.0]}, index=[1001])
    with pytest.raises(ValueError, match=f"{label} reservoir DA has observations"):
        _prep(
            waterbody_types_df_sub=pd.DataFrame(
                {"reservoir_type": [reservoir_type]}, index=[1001]
            ),
            **{obs_key: obs},
        )


def test_great_lakes_observations_without_parameters_raise() -> None:
    """The GL block keys on a lake_id column rather than the index, so it needs its
    own case."""
    with pytest.raises(ValueError, match="Great Lakes reservoir DA has observations"):
        _prep(
            waterbody_types_df_sub=pd.DataFrame({"reservoir_type": [6]}, index=[1001]),
            great_lakes_df=pd.DataFrame({"lake_id": [1001], "q": [5.0]}),
            great_lakes_climatology_df=pd.DataFrame({"q": [5.0]}, index=[1001]),
        )


def test_a_job_with_no_waterbodies_of_that_type_does_not_raise() -> None:
    """An observation-less window demotes reservoirs to levelpool on the CACHED plan,
    so a later window sees observations, a non-empty types frame, and no rows of that
    type. The lookups are then .loc[[]], which an empty-with-columns frame serves."""
    out = _prep(
        # Types frame is non-empty but carries no type-2 row: all demoted.
        waterbody_types_df_sub=pd.DataFrame({"reservoir_type": [1]}, index=[1001]),
        reservoir_usgs_df=pd.DataFrame({_T0: [5.0]}, index=[1001]),
        reservoir_usgs_param_df=pd.DataFrame(
            columns=["update_time", "prev_persisted_outflow",
                     "persistence_update_time", "persistence_index"]
        ),
    )
    assert out


def test_no_observations_is_still_a_clean_no_op() -> None:
    """The guard must not fire on the ordinary DA-off run, where BOTH are empty."""
    out = _prep(
        waterbody_types_df_sub=pd.DataFrame({"reservoir_type": [4]}, index=[1001])
    )
    assert out  # a full tuple of empties, no raise
