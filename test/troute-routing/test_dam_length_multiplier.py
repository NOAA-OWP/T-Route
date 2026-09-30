"""The level pool overtops across `WeirL * dam_length_multiplier`, read per lake.

The reservoir wrappers take the multiplier at position 11 of the parameter row. A lake
with no crest length of its own gets NWM's 10.
"""

from __future__ import annotations

from array import array

import numpy as np
import pandas as pd
import pytest

from troute.network.reservoirs.levelpool.levelpool import MC_Levelpool
from troute.routing.compute import (
    NWM_DAM_LENGTH_MULTIPLIER,
    WATERBODY_VIEW_COLS,
    WaterbodyData,
)

# A pool so large that one step without inflow barely moves it, held 0.5 m above LkMxE.
AREA_KM2 = 1.0e4
LKMXE, WEIRE, ORIFICEE = 100.0, 99.0, 90.0
WEIRC, WEIRL, ORIFICEC, ORIFICEA = 0.4, 10.0, 0.1, 1.0
START = 100.5


def _outflow(multiplier: float) -> float:
    args = [
        AREA_KM2, LKMXE, ORIFICEA, ORIFICEC, ORIFICEE,
        WEIRC, WEIRE, WEIRL, 0.9, 0.0, START, multiplier,
    ]
    pool = MC_Levelpool(0, 1, array("l", []), args, 1)
    outflow, _ = pool.run(0.0, 0.0, 300)
    return float(outflow)


@pytest.mark.parametrize("multiplier", [1.0, 10.0, 35.0])
def test_the_overtopping_crest_is_weir_length_times_the_multiplier(multiplier: float) -> None:
    orifice = ORIFICEC * ORIFICEA * np.sqrt(2.0 * 9.81 * (START - ORIFICEE))
    weir = WEIRC * WEIRL * (LKMXE - WEIRE) ** 1.5
    overtop = WEIRC * (WEIRL * multiplier) * (START - LKMXE) ** 1.5
    assert _outflow(multiplier) == pytest.approx(orifice + weir + overtop, rel=1e-4)


def test_a_network_without_crest_lengths_gets_nwm_ten() -> None:
    lakes = pd.DataFrame(
        {col: [1.0, 2.0] for col in WATERBODY_VIEW_COLS if col != "dam_length_multiplier"},
        index=pd.Index([7, 8], name="lake_id"),
    )
    types = pd.DataFrame({"reservoir_type": [1, 1]}, index=lakes.index)
    view, _ = WaterbodyData(lakes, types).generate_view([7, 8])
    assert list(view.columns) == list(WATERBODY_VIEW_COLS)
    assert view["dam_length_multiplier"].tolist() == [NWM_DAM_LENGTH_MULTIPLIER] * 2
