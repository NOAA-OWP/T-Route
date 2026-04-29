from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import pytest
import troute.nhd_network as nhd_network
from test import temporarily_change_dir
from nwm_routing.__main__ import _handle_args_v03
from nwm_routing.input import _input_handler_v04


@pytest.fixture
def reservoir_ids() -> List[int]:
    """
    Provides a list of test reservoir IDs.

    Returns
    -------
    List[int]
        List containing reservoir IDs [401, 402, 403] used in network testing
    """
    return [401, 402, 403]


@pytest.fixture
def network_clean() -> List[List[int]]:
    """
    Provides a clean network configuration for testing.

    Each sublist represents a network node with format:
    [node_id, distance, downstream_id, waterbody_id]

    Returns
    -------
    List[List[int]]
        List of network nodes where each node contains:
        - Index 0: Node ID
        - Index 1: Distance/length value
        - Index 2: Downstream node ID (-999 for terminal nodes)
        - Index 3: Associated waterbody ID (0 for no waterbody)
    """
    return [
        [0, 456, -999, 0],
        [1, 178, 4, 0],
        [2, 394, 0, 0],
        [3, 301, 2, 0],
        [4, 798, 0, 403],
        [5, 679, 4, 403],
        [6, 523, 0, 0],
        [7, 815, 2, 0],
        [8, 841, -999, 0],
        [9, 514, 8, 0],
        [10, 458, 9, 0],
        [11, 832, 10, 0],
        [12, 543, 11, 0],
        [13, 240, 12, 0],
        [14, 548, 13, 0],
        [15, 920, 14, 0],
        [16, 920, 15, 401],
        [17, 514, 16, 401],
        [18, 458, 17, 0],
        [180, 458, 17, 0],
        [181, 458, 180, 0],
        [19, 832, 18, 0],
        [20, 543, 19, 0],
        [21, 240, 16, 401],
        [22, 548, 21, 0],
        [23, 920, 22, 0],
        [24, 240, 23, 0],
        [25, 548, 12, 0],
        [26, 920, 25, 402],
        [27, 920, 26, 402],
        [28, 920, 27, 0],
        [2800, 920, 2700, 0],
    ]


@pytest.fixture
def network_circulars(network_clean: List[List[int]]) -> List[List[int]]:
    """
    Extends the clean network by adding circular references for testing.

    Parameters
    ----------
    network_clean : List[List[int]]
        Base network configuration

    Returns
    -------
    List[List[int]]
        Extended network including circular references with same structure as network_clean
        plus additional circular path test cases
    """
    return [
        [50, 178, 51, 0],
        [51, 178, 50, 0],
        [60, 178, 61, 0],
        [61, 178, 62, 0],
        [62, 178, 60, 0],
        [70, 178, 71, 0],
        [71, 178, 72, 0],
        [72, 178, 73, 0],
        [73, 178, 70, 0],
        [80, 178, 81, 0],
        [81, 178, 82, 0],
        [82, 178, 83, 0],
        [83, 178, 84, 0],
        [84, 178, 80, 0],
    ] + network_clean


@pytest.fixture
def test_columns() -> Dict[str, int]:
    """
    Defines column name to index mapping for network data.

    Returns
    -------
    Dict[str, int]
        Mapping of column names to their corresponding indices:
        - key: Node ID column index
        - dx: Distance/length column index
        - downstream: Downstream node ID column index
        - waterbody: Waterbody ID column index
    """
    return {
        "key": 0,
        "dx": 1,
        "downstream": 2,
        "waterbody": 3,
    }


@pytest.fixture
def reverse_test_columns() -> Dict[int, str]:
    """
    Provides reverse mapping of test_columns for index to name lookup.

    Returns
    -------
    Dict[int, str]
        Mapping of column indices to their corresponding names
    """
    return {0: "key", 1: "dx", 2: "downstream", 3: "waterbody"}


@pytest.fixture
def expected_connections() -> Dict[int, List[int]]:
    """
    Defines expected network connectivity patterns.

    Returns
    -------
    Dict[int, List[int]]
        Mapping of node IDs to lists of their upstream connecting nodes
    """
    return {
        0: [],
        1: [4],
        2: [0],
        3: [2],
        4: [0],
        5: [4],
        6: [0],
        7: [2],
        8: [],
        9: [8],
        10: [9],
        11: [10],
        12: [11],
        13: [12],
        14: [13],
        15: [14],
        16: [15],
        17: [16],
        18: [17],
        180: [17],
        181: [180],
        19: [18],
        20: [19],
        21: [16],
        22: [21],
        23: [22],
        24: [23],
        25: [12],
        26: [25],
        27: [26],
        28: [27],
        2800: [],
    }


@pytest.fixture
def expected_rconn() -> Dict[int, List[int]]:
    """
    Defines expected reverse network connections.

    Returns
    -------
    Dict[int, List[int]]
        Mapping of node IDs to lists of their downstream connecting nodes
    """
    return {
        0: [2, 4, 6],
        1: [],
        4: [1, 5],
        2: [3, 7],
        3: [],
        5: [],
        6: [],
        7: [],
        8: [9],
        9: [10],
        10: [11],
        11: [12],
        12: [13, 25],
        13: [14],
        14: [15],
        15: [16],
        16: [17, 21],
        17: [18, 180],
        18: [19],
        180: [181],
        181: [],
        19: [20],
        20: [],
        21: [22],
        22: [23],
        23: [24],
        24: [],
        25: [26],
        26: [27],
        27: [28],
        28: [],
        2800: [],
    }


@pytest.fixture
def expected_wbody_connections():
    """
    Defines expected waterbody to node associations.

    Returns
    -------
    Dict[int, int]
        Mapping of node IDs to their associated waterbody IDs
    """
    return {4: 403, 5: 403, 16: 401, 17: 401, 21: 401, 26: 402, 27: 402}


@pytest.fixture
def test_param_df(
    network_clean: List[List[int]], test_columns: Dict[str, int]
) -> pd.DataFrame:
    """
    Creates a pandas DataFrame from network data with proper column naming.

    Parameters
    ----------
    network_clean : List[List[int]]
        Clean network configuration data
    test_columns : Dict[str, int]
        Column name to index mapping

    Returns
    -------
    pd.DataFrame
        DataFrame containing network configuration with proper column names and indexing
    """
    df = pd.DataFrame(network_clean)
    df = df.rename(columns=nhd_network.reverse_dict(test_columns))
    df = df.set_index("key")
    return df


@pytest.fixture
def test_terminal_code() -> int:
    """
    Provides the terminal node code used in network configuration.

    Returns
    -------
    int
        Code (-999) indicating terminal nodes in the network
    """
    return -999


@pytest.fixture
def test_waterbody_null_code() -> int:
    """
    Provides the null waterbody code used in network configuration.

    Returns
    -------
    int
        Code (0) indicating no associated waterbody for a node
    """
    return 0
