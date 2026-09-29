"""The service weir and the overtopping crest come from the NID dam geometry.

The weir is the NID spillway width where NID reports one, else the hydrofabric's WeirL,
else a tenth of the NID crest. The overtopping crest is the NID crest where known and
NWM's 10 weir lengths elsewhere.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import pytest

from troute.nhf_preprocess import (
    DAM_CREST_LENGTH_FIELD,
    SPILLWAY_WIDTH_FIELD,
    NHFPreprocessMixin,
    _validate_nid_geometry,
    overtopping_geometry,
)
from troute.routing.compute import NWM_DAM_LENGTH_MULTIPLIER

NAN = float("nan")


@pytest.mark.parametrize(
    ("weir_length", "crest", "spillway", "weir", "overtop_crest"),
    [
        # both known: weir over the spillway, overtopping across the whole crest
        (183.0, 201.0, 12.8, 12.8, 201.0),
        # crest only: the hydrofabric's weir, the whole dam, stays
        (183.0, 183.0, NAN, 183.0, 183.0),
        # crest only and no hydrofabric weir: a tenth of the crest
        (NAN, 200.0, NAN, 20.0, 200.0),
        # spillway only: NWM's crest of 10 weir lengths around the real spillway
        (10.0, NAN, 12.8, 12.8, 128.0),
        # neither: the hydrofabric's weir and NWM's 10
        (10.0, NAN, NAN, 10.0, 100.0),
        # a crest shorter than its spillway is kept as reported
        (12.8, 9.0, 12.8, 12.8, 9.0),
    ],
)
def test_weir_and_overtopping_crest(
    weir_length: float, crest: float, spillway: float, weir: float, overtop_crest: float
) -> None:
    got_weir, multiplier = overtopping_geometry(
        pd.Series([weir_length]), pd.Series([crest]), pd.Series([spillway])
    )
    assert got_weir.iloc[0] == pytest.approx(weir)
    assert got_weir.iloc[0] * multiplier.iloc[0] == pytest.approx(overtop_crest)


def test_a_lake_with_no_geometry_keeps_nwm_ten() -> None:
    """A Great Lake carries neither NID length nor a weir; it keeps NaN and NWM's 10."""
    weir, multiplier = overtopping_geometry(
        pd.Series([NAN]), pd.Series([NAN]), pd.Series([NAN])
    )
    assert np.isnan(weir.iloc[0])
    assert multiplier.iloc[0] == NWM_DAM_LENGTH_MULTIPLIER


def _lakes(crest: list[object], spillway: list[object]) -> pd.DataFrame:
    return pd.DataFrame({DAM_CREST_LENGTH_FIELD: crest, SPILLWAY_WIDTH_FIELD: spillway})


def test_null_and_positive_lengths_pass() -> None:
    _validate_nid_geometry(_lakes([None, 201.0], [12.8, None]))


@pytest.mark.parametrize(
    ("crest", "spillway"),
    [([0.0], [12.8]), ([201.0], [-1.0]), ([float("inf")], [12.8]), (["long"], [12.8])],
)
def test_broken_lengths_are_rejected(crest: list[object], spillway: list[object]) -> None:
    with pytest.raises(ValueError, match="NULL or a positive length in meters"):
        _validate_nid_geometry(_lakes(crest, spillway))


class _Net(NHFPreprocessMixin):
    """The slice of network state ``preprocess_waterbodies`` touches, on one lake."""

    def __init__(self) -> None:
        # (up_node, downstream, vfp): the lake's arms 10 and 11 join its outlet vfp 12.
        edges = [(1, 3, 10), (2, 3, 11), (3, 4, 12), (4, 5, 12), (5, 6, 13)]
        up, dn, vfp = zip(*edges, strict=True)
        self._dataframe = pd.DataFrame(
            {"downstream": list(dn), "vfp_id": list(vfp), "fp_id": list(vfp), "dx": 100.0},
            index=pd.Index(list(up), name="up_node_id"),
        )
        self._waterbody_df = pd.DataFrame()
        self._connections: dict[int, list[int]] | None = None
        self._terminal_codes = set(self._dataframe["downstream"]) - set(self._dataframe.index)
        self.zero_nodes: list[int] = []
        self.vfp_nex_ids = np.array(sorted(self._dataframe.index), dtype=np.int64)
        self._fp_outlet_crosswalk: defaultdict[int, list[int]] = defaultdict(list)
        for n, v in zip(up, vfp, strict=True):
            self._fp_outlet_crosswalk[int(n)] = [int(v)]
        self._link_lake_crosswalk = None
        self.waterbody_connections: dict[int, int] = {}
        self.output_parameters: dict[str, object] = {}
        self.data_assimilation_parameters: dict[str, object] = {}

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._dataframe

    @dataframe.setter
    def dataframe(self, val: pd.DataFrame) -> None:
        self._dataframe = val

    @property
    def waterbody_dataframe(self) -> pd.DataFrame:
        return self._waterbody_df

    @waterbody_dataframe.setter
    def waterbody_dataframe(self, val: pd.DataFrame) -> None:
        self._waterbody_df = val

    @property
    def connections(self) -> dict[int, list[int]]:
        if self._connections is None:
            self._connections = {
                int(n): ([int(d)] if d not in self._terminal_codes else [])
                for n, d in self._dataframe["downstream"].items()
            }
        return self._connections


@pytest.mark.parametrize(
    ("crest", "spillway", "weir", "multiplier"),
    [(200.0, 12.0, 12.0, 200.0 / 12.0), (200.0, NAN, 20.0, 10.0), (NAN, 12.0, 12.0, 10.0)],
)
def test_a_lake_without_weir_length_is_kept_when_nid_supplies_one(
    crest: float, spillway: float, weir: float, multiplier: float
) -> None:
    """The completeness gate judges the derived WeirL, so NID geometry rescues a lake whose
    hydrofabric WeirL is NULL."""
    lakes = pd.DataFrame(
        {
            "nhf_lake_id": [7001], "lake_id": ["900"], "fp_id": [12.0], "virtual_fp_id": [12.0],
            "ifd": [0.9], "LkArea": [1.0], "LkMxE": [105.0], "OrificeA": [1.0],
            "OrificeC": [0.1], "OrificeE": [100.0], "WeirC": [0.4], "WeirE": [104.0],
            "WeirL": [NAN], DAM_CREST_LENGTH_FIELD: [crest], SPILLWAY_WIDTH_FIELD: [spillway],
        }
    )
    crosswalk = pd.DataFrame({"nhf_lake_id": [7001, 7001, 7001], "virtual_fp_id": [10, 11, 12]})
    net = _Net()
    net.preprocess_waterbodies(lakes, crosswalk)

    kept = net.waterbody_dataframe
    assert len(kept) == 1
    assert kept["WeirL"].iloc[0] == pytest.approx(weir)
    assert kept["dam_length_multiplier"].iloc[0] == pytest.approx(multiplier)
