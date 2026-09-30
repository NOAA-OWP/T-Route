"""Reservoir rows publish a depth, not a water-surface elevation.

The kernel keeps a reservoir's elevation in the depth slot, since that slot carries
reservoir state between run sets. LAKEOUT publishes it as ``water_sfc_elev``; every
other writer labels it ``depth``, so a lake read as a channel hundreds of meters deep.
``_convert_waterbody_depth`` rewrites those rows to stage above the orifice invert.
"""
import numpy as np
import pandas as pd
import pytest

from nwm_routing.output import _convert_waterbody_depth

_LAKE, _CHANNEL = 900, 10


def _fvd(rows: dict[int, list[float]], nts: int = 2) -> pd.DataFrame:
    """Frame with the real (timestep, variable) column layout: q, v, d per timestep."""
    cols = pd.MultiIndex.from_product([range(nts), ["q", "v", "d"]]).to_flat_index()
    return pd.DataFrame(rows, index=cols).T.astype("float32")


def _wb(orifice_e: float | None = 100.0, ids=(_LAKE,)) -> pd.DataFrame:
    data = {} if orifice_e is None else {"OrificeE": [orifice_e] * len(ids)}
    return pd.DataFrame(data, index=list(ids))


def _depths(frame: pd.DataFrame, seg: int) -> list[float]:
    return [float(v) for c, v in frame.loc[seg].items() if c[1] == "d"]


def test_lake_depth_becomes_stage_above_the_invert():
    fvd = _fvd({_LAKE: [5.0, 0.0, 261.5, 5.0, 0.0, 262.0]})
    _convert_waterbody_depth(fvd, _wb(260.0))
    assert _depths(fvd, _LAKE) == pytest.approx([1.5, 2.0])


def test_channel_rows_are_untouched():
    fvd = _fvd({_LAKE: [5.0, 0.0, 261.5, 5.0, 0.0, 262.0],
                _CHANNEL: [3.0, 1.2, 0.75, 3.0, 1.2, 0.8]})
    _convert_waterbody_depth(fvd, _wb(260.0))
    assert _depths(fvd, _CHANNEL) == pytest.approx([0.75, 0.8])


def test_flow_and_velocity_are_untouched():
    """Only the depth slot is overloaded; q and v are the lake's real output."""
    fvd = _fvd({_LAKE: [5.0, 0.0, 261.5, 6.0, 0.0, 262.0]})
    _convert_waterbody_depth(fvd, _wb(260.0))
    row = fvd.loc[_LAKE]
    assert [float(v) for c, v in row.items() if c[1] == "q"] == pytest.approx([5.0, 6.0])
    assert [float(v) for c, v in row.items() if c[1] == "v"] == pytest.approx([0.0, 0.0])


def test_below_the_invert_clamps_to_zero(caplog):
    """Reservoir DA can drive the elevation below the invert; that is not a depth."""
    fvd = _fvd({_LAKE: [0.0, 0.0, 228.6, 0.0, 0.0, 261.5]})
    with caplog.at_level("WARNING"):
        _convert_waterbody_depth(fvd, _wb(260.0))
    assert _depths(fvd, _LAKE) == pytest.approx([0.0, 1.5])
    assert "below their orifice invert" in caplog.text


def test_a_lake_at_its_invert_reports_zero():
    """The reported VPU 01 example: h0 == OrificeE, published as its own elevation."""
    fvd = _fvd({_LAKE: [0.02, 0.0, 99.235, 0.02, 0.0, 99.235]})
    _convert_waterbody_depth(fvd, _wb(99.235))
    assert _depths(fvd, _LAKE) == pytest.approx([0.0, 0.0])


@pytest.mark.parametrize(
    ("waterbodies", "reason"),
    [
        (pd.DataFrame(), "no waterbodies"),
        (pd.DataFrame(index=[_LAKE]), "no OrificeE column"),
        (_wb(260.0, ids=(999,)), "no waterbody row is in the frame"),
    ],
)
def test_no_op_cases(waterbodies, reason):
    fvd = _fvd({_LAKE: [5.0, 0.0, 261.5, 5.0, 0.0, 262.0]})
    before = fvd.copy()
    _convert_waterbody_depth(fvd, waterbodies)
    pd.testing.assert_frame_equal(fvd, before), reason
