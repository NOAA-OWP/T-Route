from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
import yaml
from _pytest.fixtures import FixtureRequest


def find_config_files() -> List[Path]:
    """Finds all `.yaml` configuration files within specified directories

    Returns
    -------
    List[Path]
        A list of Path objects pointing to each valid configuration
    """
    test_dir = Path(__file__).parents[3] / "test"  # Searching for the t-route/test dir
    target_dirs = ["LowerColorado_TX", "LowerColorado_TX_v4", "LowerColorado_TX_HYFeatures_v22", "unit_test_hyfeature"]
    files = []
    for dir_name in target_dirs:
        files.extend(list((test_dir / dir_name).glob("*.yaml")))
    return files


@pytest.fixture(params=find_config_files())
def config_data(request: FixtureRequest) -> Tuple[Path, Dict[str, Any]]:
    """A fixture for loading yaml files into python dictionary mappings

    Parameters
    ----------
    request : FixtureRequest
        The pytest request object, containing the current parameter value

    Returns
    -------
    Tuple[Path, Dict[str, Any]]
        A tuple containing the path to the YAML file and the loaded data as a dictionary
    """
    data = yaml.load(request.param.read_text(), Loader=yaml.Loader)
    return request.param, data
