"""A reservoir's elevation must reach the next window at the kernel's precision.

The kernel carries it in double and returns it as element 12 of each result. The
flowveldepth copy is float32, which near 100 m cannot hold a change below 7.6e-6 m, so a
large lake handed over through it loses its storage change at every window boundary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from troute.NHF import NHF

LAKE = 5001
ELEVATION = 100.0000123456789  # not representable in float32


def _result(elevations: tuple[np.ndarray, np.ndarray] | None) -> tuple[object, ...]:
    ids = np.array([11, LAKE], dtype=np.intp)
    flow = np.zeros((2, 8), dtype=np.float32)  # two timesteps of q, v, d, ql
    flow[1, -2] = np.float32(ELEVATION)
    raw: list[object] = [ids, flow, *([None] * 10)]
    if elevations is not None:
        raw.append(elevations)
    return tuple(raw)


def _network() -> NHF:
    net = NHF.__new__(NHF)
    net._waterbody_df = pd.DataFrame(
        {"qd0": [0.0], "h0": [99.0]}, index=pd.Index([LAKE], name="nhf_lake_id")
    )
    return net


def test_the_next_window_starts_from_the_double_elevation() -> None:
    net = _network()
    state = (np.array([LAKE], dtype=np.intp), np.array([ELEVATION], dtype=np.float64))
    net.new_q0([_result(state)])
    net.update_waterbody_water_elevation()
    assert net._waterbody_df.loc[LAKE, "h0"] == ELEVATION


def test_a_result_without_the_element_keeps_the_float32_copy() -> None:
    """The diffusive leg returns no reservoir state."""
    net = _network()
    q0 = net.new_q0([_result(None)])
    assert q0.loc[LAKE, "h0"] == float(np.float32(ELEVATION))
