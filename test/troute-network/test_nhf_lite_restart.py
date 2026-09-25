"""A lite channel restart reaches the kernel in the kernel's column order, keyed either by
routing link (the file the driver writes) or by flowpath (a hot start)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

from troute import nhd_io
from troute.NHF import NHF, Q0_COLUMNS

if TYPE_CHECKING:
    from pathlib import Path

LINKS = [11, 12, 21]  # two links on flowpath 1, one on flowpath 2
FP_OF_LINK = {11: 1, 12: 1, 21: 2}
T0 = pd.Timestamp("2011-04-19")


def _network() -> NHF:
    """A bare NHF with only the state the restart reader touches, as the sibling tests build it."""
    net = NHF.__new__(NHF)
    net._dataframe = pd.DataFrame(  # pyright: ignore[reportPrivateUsage]
        {"fp_id": [FP_OF_LINK[link] for link in LINKS]},
        index=pd.Index(LINKS, name="up_node_id"),
    )
    net._waterbody_df = pd.DataFrame(  # pyright: ignore[reportPrivateUsage]
        index=pd.Index([], name="lake_id"),
    )
    net.break_points = {"break_network_at_waterbodies": False}
    # Set by __init__, which this bare instance skips; read by the flowpath branch.
    net.div_reverse_lookup = {}  # pyright: ignore[reportAttributeAccessIssue]
    return net


def _read(path: Path) -> pd.DataFrame:
    net = _network()
    net.restart_parameters = {"lite_channel_restart_file": str(path)}
    net.initial_warmstate_preprocess(from_files=True, value_dict={})
    return net.q0


def _write_link_restart(tmp_path: Path, links: list[int]) -> Path:
    q0 = pd.DataFrame(
        {"qu0": [1.0 + k for k in range(len(links))], "qd0": 2.0, "h0": 0.3, "ql0": 0.04},
        index=pd.Index(links),
        dtype="float32",
    )
    nhd_io.write_lite_restart(
        q0, pd.DataFrame(columns=["qd0", "h0"]), T0,
        {"lite_restart_output_directory": str(tmp_path)},
    )
    return tmp_path / f"channel_restart_{T0:%Y%m%d%H%M}"


def test_a_driver_written_restart_round_trips_by_link(tmp_path: Path) -> None:
    written = pd.read_pickle(_write_link_restart(tmp_path, LINKS)).drop(columns="time")
    out = _read(tmp_path / f"channel_restart_{T0:%Y%m%d%H%M}")
    assert list(out.columns) == list(Q0_COLUMNS)
    pd.testing.assert_frame_equal(out, written.reindex(index=out.index), check_names=False)


def test_a_link_the_restart_lacks_starts_cold(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    out = _read(_write_link_restart(tmp_path, LINKS[:2]))
    assert out.loc[21].to_numpy().tolist() == [0.0, 0.0, 0.0, 0.0]
    assert out.loc[11, "qd0"] == pytest.approx(2.0)
    assert "covers 2 of 3 routing links" in caplog.text


def test_a_restart_with_links_beyond_this_network_is_not_reported_as_short(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Coverage is counted over the network's links, not the file's rows: a file from
    a larger domain covers every link here and says nothing."""
    out = _read(_write_link_restart(tmp_path, [*LINKS, 999]))
    assert list(out.index) == LINKS
    assert "routing links" not in caplog.text


def test_a_flowpath_restart_is_broadcast_in_kernel_order(tmp_path: Path) -> None:
    """The hot-start layout: one row per flowpath, columns in its own order.

    The kernel reads the row by position, so the order is pinned literally and every
    slot carries a distinct value.
    """
    hot = pd.DataFrame({
        "feature_id": [1, 2], "qd0": [12.0, 22.0], "h0": [0.5, 0.7],
        "qu0": [11.0, 21.0], "ql0": [0.1, 0.2], "time": T0,
    })
    hot.to_pickle(tmp_path / "restart.pkl")
    out = _read(tmp_path / "restart.pkl")
    assert list(out.columns) == ["qu0", "qd0", "h0", "ql0"]
    assert out.loc[12].to_numpy().tolist() == pytest.approx([11.0, 12.0, 0.5, 0.1])
    assert out.loc[21].to_numpy().tolist() == pytest.approx([21.0, 22.0, 0.7, 0.2])


def test_a_restart_with_a_duplicated_link_is_refused(tmp_path: Path) -> None:
    path = _write_link_restart(tmp_path, [11, 11, 12, 21])
    with pytest.raises(ValueError, match="duplicate link"):
        _read(path)


def test_a_restart_keyed_by_neither_is_refused(tmp_path: Path) -> None:
    stray = pd.DataFrame(
        {"qu0": [1.0], "qd0": [1.0], "h0": [0.1], "ql0": [0.0], "time": T0},
        index=pd.Index([999]),
    )
    stray.to_pickle(tmp_path / "stray.pkl")
    with pytest.raises(ValueError, match="keyed by neither"):
        _read(tmp_path / "stray.pkl")
