from typing import Any, Dict

import os
import pytest
from pathlib import Path
import pandas as pd
from troute.HYFeaturesNetwork import HYFeaturesNetwork
from test import temporarily_change_dir


def test_network(
    hyfeatures_test_network: Dict[str, Any], hyfeature_network: pd.DataFrame
):
    cwd = hyfeatures_test_network["cwd"]
    path = hyfeatures_test_network["path"]
    supernetwork_parameters = hyfeatures_test_network["supernetwork_parameters"]
    waterbody_parameters = hyfeatures_test_network["waterbody_parameters"]
    data_assimilation_parameters = hyfeatures_test_network[
        "data_assimilation_parameters"
    ]
    restart_parameters = hyfeatures_test_network["restart_parameters"]
    compute_parameters = hyfeatures_test_network["compute_parameters"]
    forcing_parameters = hyfeatures_test_network["forcing_parameters"]
    hybrid_parameters = hyfeatures_test_network["hybrid_parameters"]
    preprocessing_parameters = hyfeatures_test_network["preprocessing_parameters"]
    output_parameters = hyfeatures_test_network["output_parameters"]

    with temporarily_change_dir(path):
        network = HYFeaturesNetwork(
            supernetwork_parameters,
            waterbody_parameters,
            data_assimilation_parameters,
            restart_parameters,
            compute_parameters,
            forcing_parameters,
            hybrid_parameters,
            preprocessing_parameters,
            output_parameters,
        )
    pd.testing.assert_frame_equal(
        hyfeature_network.reindex(
            sorted(hyfeature_network.columns), axis=1
        ).reset_index(drop=True),
        network._dataframe.reindex(
            sorted(network._dataframe.columns), axis=1
        ).reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        rtol=1e-5,
    )
