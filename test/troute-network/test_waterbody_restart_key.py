"""The lite waterbody restart is keyed by the hydrofabric lake id.

NHF waterbody ids are positional (``nhf_preprocess.preprocess_waterbodies``), so an inner
join on them loads each lake's state into its neighbor once the lake set changes. The id
sits in the index since the BMI forcing module reshapes the pickle against two columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from troute import nhd_io
from troute.AbstractNetwork import AbstractNetwork
from troute.nhf_preprocess import LAKE_ID_FIELD, RECORD_LAKE_ID_FIELD

# Far above any routing node id, matching how preprocess_waterbodies allocates.
_MAX_DF_ID = 1_000_000_000_000_000
_REAL_IDS = [111, 222, 333]


class _Stub:
    """Minimal stand-in carrying only what the re-index method reads."""

    def __init__(self, waterbody_dataframe: pd.DataFrame) -> None:
        self.waterbody_dataframe = waterbody_dataframe


def _nhf_waterbodies(real_ids: list[int]) -> pd.DataFrame:
    """An NHF waterbody frame: synthetic index, original ids in a record column."""
    df = pd.DataFrame({RECORD_LAKE_ID_FIELD: real_ids})
    df.index = pd.Index(np.arange(len(real_ids)) + _MAX_DF_ID, name=LAKE_ID_FIELD)
    return df


def _reindex(waterbodies: pd.DataFrame, restart: pd.DataFrame) -> pd.DataFrame:
    return AbstractNetwork._reindex_waterbody_restart(
        _Stub(waterbodies), restart, "restart.pkl"
    )


def _written_restart(waterbodies: pd.DataFrame, qd0, h0, tmp_path) -> pd.DataFrame:
    """Round-trip through the real writer and reader."""
    states = waterbodies.assign(qd0=qd0, h0=h0)
    nhd_io.write_lite_restart(
        pd.DataFrame({"qu0": [0.0], "qd0": [0.0], "h0": [0.0]}),
        states,
        pd.Timestamp("2026-01-01"),
        {"lite_restart_output_directory": str(tmp_path)},
    )
    written = next(tmp_path.glob("waterbody_restart_*"))
    restart, _ = nhd_io.read_lite_restart(written)
    return restart


def test_record_field_matches_nhf_preprocess():
    """nhd_io names the record column by literal; pin it to its owner."""
    assert nhd_io.WBODY_RECORD_ID_FIELD == RECORD_LAKE_ID_FIELD


def test_restart_is_indexed_by_hydrofabric_lake_id(tmp_path):
    restart = _written_restart(
        _nhf_waterbodies(_REAL_IDS), [10.0, 20.0, 30.0], [1.0, 2.0, 3.0], tmp_path
    )
    assert restart.index.name == nhd_io.WBODY_RESTART_ID_FIELD
    assert restart.index.tolist() == _REAL_IDS


def test_restart_keeps_its_two_column_shape(tmp_path):
    """The BMI forcing module reshapes this pickle against exactly two columns."""
    restart = _written_restart(
        _nhf_waterbodies(_REAL_IDS), [10.0, 20.0, 30.0], [1.0, 2.0, 3.0], tmp_path
    )
    assert restart.columns.tolist() == ["qd0", "h0"]


def test_shifted_lake_set_keeps_each_lake_its_own_state(tmp_path):
    """Lake 111 is dropped, so 222 and 333 renumber down one; merging on the synthetic
    index would give 222 the state written for 111."""
    restart = _written_restart(
        _nhf_waterbodies(_REAL_IDS), [10.0, 20.0, 30.0], [1.0, 2.0, 3.0], tmp_path
    )
    survivors = _nhf_waterbodies([222, 333])

    reindexed = _reindex(survivors, restart)
    merged = survivors.merge(reindexed, on=LAKE_ID_FIELD)

    by_real_id = merged.set_index(RECORD_LAKE_ID_FIELD)
    assert by_real_id.loc[222, "qd0"] == 20.0
    assert by_real_id.loc[222, "h0"] == 2.0
    assert by_real_id.loc[333, "qd0"] == 30.0
    assert by_real_id.loc[333, "h0"] == 3.0


def test_lakes_absent_from_the_restart_are_ignored(tmp_path):
    """Extra lakes in the restart are harmless; they are not this network's."""
    restart = _written_restart(
        _nhf_waterbodies(_REAL_IDS), [10.0, 20.0, 30.0], [1.0, 2.0, 3.0], tmp_path
    )
    # Same ids, but the network now routes only the last two.
    reindexed = _reindex(_nhf_waterbodies([222, 333]), restart)
    assert len(reindexed) == 2


def test_restart_missing_a_routed_lake_names_the_changed_lake_set(tmp_path):
    """An inner join would drop the lake from routing with no warm state. Some lakes
    matched, so the cause is a changed lake set."""
    restart = _written_restart(
        _nhf_waterbodies([111, 222]), [10.0, 20.0], [1.0, 2.0], tmp_path
    )
    with pytest.raises(ValueError, match="written for a different lake set") as err:
        _reindex(_nhf_waterbodies(_REAL_IDS), restart)
    assert "missing 1 of this network's 3" in str(err.value)


def test_a_restart_matching_no_lake_names_the_keying(tmp_path):
    """model_DAforcing's writer names its index lake_id while keying it by
    synthetic ids, so it passes the name gate and matches nothing. That is a different
    fault from a changed lake set and must not be reported as one."""
    restart = _written_restart(
        _nhf_waterbodies([777, 888, 999]), [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], tmp_path
    )
    with pytest.raises(ValueError, match="no lake matched at all") as err:
        _reindex(_nhf_waterbodies(_REAL_IDS), restart)
    assert "different lake set" not in str(err.value)


def test_legacy_restart_keyed_by_synthetic_ids_is_refused():
    """A file indexed by nhf_lake_id holds synthetic ids, which cannot be verified."""
    legacy = pd.DataFrame(
        {"qd0": [10.0, 20.0, 30.0], "h0": [1.0, 2.0, 3.0]},
        index=pd.Index(np.arange(3) + _MAX_DF_ID, name=LAKE_ID_FIELD),
    )
    with pytest.raises(ValueError, match="indexed by 'nhf_lake_id', not 'lake_id'"):
        _reindex(_nhf_waterbodies(_REAL_IDS), legacy)


def test_networks_without_synthetic_ids_are_untouched():
    """NHD / HYFeatures key waterbodies by their own ids; nothing to remap."""
    plain = pd.DataFrame({"LkArea": [1.0, 2.0]}, index=pd.Index([5, 6], name="lake_id"))
    legacy = pd.DataFrame(
        {"qd0": [1.0, 2.0], "h0": [3.0, 4.0]}, index=pd.Index([5, 6], name="lake_id")
    )
    pd.testing.assert_frame_equal(_reindex(plain, legacy), legacy)


def test_non_synthetic_network_round_trips(tmp_path):
    """A restart this writer produced must still load on those networks."""
    plain = pd.DataFrame({"LkArea": [1.0, 2.0]}, index=pd.Index([5, 6], name="lake_id"))
    restart = _written_restart(plain, [1.0, 2.0], [3.0, 4.0], tmp_path)

    reindexed = _reindex(plain, restart)
    assert reindexed.index.tolist() == [5, 6]
    assert reindexed["qd0"].tolist() == [1.0, 2.0]


def test_a_wrf_hydro_waterbody_restart_is_refused_on_a_synthetic_index():
    """That reader keys its states by the crosswalk's own lake ids, and the merge
    downstream joins them onto this network's index by numeric equality. On a synthetic
    index nothing there names the network that wrote it, so a match is a coincidence."""

    class _Net:
        break_points = {"break_network_at_waterbodies": True}
        restart_parameters = {
            "wrf_hydro_waterbody_restart_file": "HYDRO_RST.2020-08-26_00:00_DOMAIN1",
            "wrf_hydro_waterbody_ID_crosswalk_file": "x.nc",
            "wrf_hydro_waterbody_crosswalk_filter_file": "y.nc",
        }
        waterbody_dataframe = _nhf_waterbodies(_REAL_IDS)

    with pytest.raises(ValueError, match="allocated by position"):
        AbstractNetwork.initial_warmstate_preprocess(_Net(), True, None)
