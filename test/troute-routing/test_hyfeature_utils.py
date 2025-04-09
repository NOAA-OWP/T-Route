import pytest
from typing import Dict, Any, List
from pathlib import Path
import pandas as pd
import os
from troute.HYFeaturesNetwork import HYFeaturesNetwork
from test import temporarily_change_dir


def test_build_forcing_sets(
    hyfeatures_test_network: Dict[str, Any],
    hyfeatures_network_object: HYFeaturesNetwork,
    hyfeature_qlat_data: List[Dict[str, Any]],
) -> None:
    path = hyfeatures_test_network["path"]

    with temporarily_change_dir(path):
        run_sets = hyfeatures_network_object.build_forcing_sets()

    assert run_sets == hyfeature_qlat_data


def test_assemble_forcings(
    hyfeatures_test_network: Dict[str, Any],
    hyfeatures_network_object: HYFeaturesNetwork,
    hyfeature_qlat_data: List[Dict[str, Any]],
    q_lateral_hy_features: pd.DataFrame,
) -> None:
    path = hyfeatures_test_network["path"]

    with temporarily_change_dir(path):
        hyfeatures_network_object.assemble_forcings(
            hyfeature_qlat_data[0],
        )

    pd.testing.assert_frame_equal(
        q_lateral_hy_features,
        hyfeatures_network_object._qlateral,
        check_dtype=False,
        check_exact=False,
        rtol=1e-5,
    )
