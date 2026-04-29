import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml
from troute_rnr import nwps
from troute_rnr.settings import Settings


@pytest.fixture
def mock_rfc_inputs() -> nwps.ProcessedData:
    """A data structure to mock processed data from the HPDT2 RFC point from May 5, 2025

    Return
    ------
    nwps.ProcessedData
        A pydantic object containing input data
    """
    json_path = Path.cwd() / "test/mock_data/HPDT2.json"
    with open(json_path) as f:
        data_dict = json.load(f)

    processed_data = nwps.ProcessedData.model_validate(data_dict)
    processed_data.reach.latest_observation = None
    print(processed_data.reach.latest_observation)
    return processed_data


@pytest.fixture
def mock_restart_file() -> pd.DataFrame:
    """A mock restart file to test against

    Return
    ------
    pd.Dataframe
        A restart dataframe
    """
    restart_path = Path.cwd() / "test/mock_data/HPDT2_2025-05-05_21:00.pkl"
    df = pd.read_pickle(restart_path)
    return df


@pytest.fixture
def mock_settings() -> Settings:
    """A data structure to mock settings from the settings config

    Return
    ------
    Settings
        The settings object
    """
    settings = Settings()
    settings.tmp_config = Path.cwd() / "test/mock_data/tmp_config.yaml"
    settings.tmp_geopackage = Path.cwd() / "test/mock_data/HPDT2.gpkg"
    settings.tmp_flow_files_path = Path.cwd() / "test/mock_data/tmp_flows"
    settings.tmp_flow_files_path.mkdir(exist_ok=True)
    settings.restart_path = Path.cwd() / "test/mock_data/tmp_restart_flow/"
    settings.restart_path.mkdir(exist_ok=True)
    return settings


@pytest.fixture
def mock_config() -> dict[str, Any]:
    """Reads a YAML configuration file.

    Loads configuration settings from a YAML file located at:
    mock_data/HPDT2_config.yaml

    Return
    dict[str, Any]
        The correct config for this forecast
    """
    config_path = Path.cwd() / "test/mock_data/HPDT2_config.yaml"
    with open(config_path) as f:
        config_data = yaml.safe_load(f)
    return config_data


@pytest.fixture
def mock_flows() -> pd.DataFrame:
    """Reads the first dataframe from the tmp_flows dir for HPDT2

    Return
    ------
    pd.DataFrame
        The T-Route flows dataframe
    """
    flows_path = Path.cwd() / "test/mock_data/flows/HPDT2.CHRTOUT_DOMAIN1.csv"
    df = pd.read_csv(flows_path)
    return df
