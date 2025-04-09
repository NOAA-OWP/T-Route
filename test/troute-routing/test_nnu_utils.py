import os
from pathlib import Path
from typing import Any, Dict, List

import pytest
import troute.nhd_network_utilities_v02 as nnu
from test import temporarily_change_dir


def test_build_nhd_forcing_sets(
    nhd_test_network: Dict[str, Any],
    warmstart_nhd_test: Dict[str, Any],
    nhd_qlat_data: Dict[str, Any],
) -> None:
    """Test the creation of forcing file sets for NHD network simulation.

    Parameters
    ----------
    nhd_test_network : Dict[str, Any]
        Dictionary containing config network settings
    warmstart_nhd_test : Dict[str, Any]
        Dictionary containing warmstart test data for the nhd network
    nhd_qlat_data : Dict[str, Any]
        Dictionary containing expected lateral flow data paths
    """
    path = nhd_test_network["path"]
    forcing_parameters = nhd_test_network["forcing_parameters"]

    t0 = warmstart_nhd_test["t0"]

    with temporarily_change_dir(path):
        run_sets = nnu.build_forcing_sets(forcing_parameters, t0)

    assert run_sets[0]["qlat_files"] == nhd_qlat_data["qlat_files"]
    assert run_sets[0]["nts"] == nhd_qlat_data["nts"]
    assert run_sets[0]["final_timestamp"] == nhd_qlat_data["final_timestamp"]


def test_da_sets(
    nhd_test_network: Dict[str, Any],
    warmstart_nhd_test: Dict[str, Any],
    nhd_qlat_data: Dict[str, Any],
    da_test_data: List[Dict[str, str]]
):
    run_sets = [nhd_qlat_data]
    t0 = warmstart_nhd_test["t0"]
    data_assimilation_parameters = nhd_test_network["data_assimilation_parameters"]
    
    with temporarily_change_dir(nhd_test_network["path"]):
        da_sets = nnu.build_da_sets(data_assimilation_parameters, run_sets, t0)

    assert da_sets == da_test_data


def test_parity_sets(
    nhd_test_network: Dict[str, Any],
    nhd_qlat_data: Dict[str, Any],
    nhd_validation_files: Dict[str, Any],
):
    run_sets = [nhd_qlat_data]
    parity_parameters = nhd_test_network["parity_parameters"]
    parity_sets = nnu.build_parity_sets(parity_parameters, run_sets)
    assert (
        parity_sets[0]["validation_files"] == nhd_validation_files["validation_files"]
    )
