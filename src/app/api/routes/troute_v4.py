import json
import os
from datetime import datetime
from enum import Enum   
from pathlib import Path
from typing import Annotated

import yaml
from fastapi import APIRouter, Depends, HTTPException, Query
from nwm_routing.__main__ import main_v04 as t_route
from pydantic import conint
from troute.config import Config

from app.api.services.initialization import (
    create_initial_start_file,
    create_params,
    edit_yaml,
)
from app.api.services.utils import update_test_paths_with_prefix
from app.core import get_settings
from app.core.settings import Settings
from app.schemas import HttpStatusCode, TestStatus, TRouteStatus

router = APIRouter()


class HydrofabricVersion(str, Enum):
    v22 = "v2.2"
    v20_1 = "v20.1"


@router.get("/", response_model=TRouteStatus)
async def get_gauge_data(
    settings: Annotated[Settings, Depends(get_settings)],
    lid: str,
    feature_id: str,
    hy_id: str,
    initial_start: float,
    start_time: datetime,
    num_forecast_days: conint(ge=1, le=30),
    hydrofabric_version: HydrofabricVersion = Query(
        default=HydrofabricVersion.v20_1,
        description="Version of the hydrofabric to use. Defaults to v20.1",
    ),
) -> TRouteStatus:
    """An API call for running T-Route within the context of replace and route

    Parameters:
    ----------
    lid: str
        The Location of the RFC Point
    feature_id: str
        The COMID associated with the LID
    hy_id: str
        The HY_ID associated with the LID
    initial_start: float
        The initial start for T-Route
    start_time: str
        The start time for the forecast
    num_forecast_days: int
        The number of days in the forecast

    Returns:
    -------
    TRouteOutput
        A successful T-Route run
    """
    version = hydrofabric_version.value
    base_config = Path(str(settings.base_config).format(version))
    params = create_params(
        lid, feature_id, hy_id, initial_start, start_time, num_forecast_days, version, settings
    )
    restart_file = create_initial_start_file(params, settings)
    yaml_file_path = edit_yaml(base_config, params, restart_file)
    try:
        t_route(["-f", yaml_file_path.__str__()])
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

    yaml_file_path.unlink()

    return TRouteStatus(
        status_code=HttpStatusCode.CREATED,
        message="T-Route run successfully",
        lid=lid,
        feature_id=feature_id,
    )


@router.post("/tests/LowerColorado", response_model=TestStatus)
async def run_lower_colorado_tests(
    settings: Annotated[Settings, Depends(get_settings)],
    config_path: str = "test/LowerColorado_TX_v4/test_AnA_V4_HYFeature.yaml",
) -> TRouteStatus:
    """An API call for running the LowerColorado T-Route test using a predefined config file

    Parameters
    ---------
    config_path: str
        Path to the YAML configuration file for the test

    Returns
    -------
    TRouteStatus
        The status of the T-Route run
    """
    base = "/t-route"
    path_to_test_dir = Path(f"{base}/{config_path}").parent
    yaml_path = Path(f"{base}/{config_path}")

    os.chdir(path_to_test_dir)

    with open(yaml_path) as custom_file:
        data = yaml.load(custom_file, Loader=yaml.SafeLoader)

    # # Updating paths to work in docker
    # data = update_test_paths_with_prefix(
    #     data, path_to_test_dir, settings.lower_colorado_paths_to_update
    # )

    troute_configuration = Config.with_strict_mode(**data)

    tmp_yaml = path_to_test_dir / "tmp.yaml"

    dict_ = json.loads(troute_configuration.json())

    # converting timeslice back to string (Weird pydantic 1.10 workaround)
    dict_["compute_parameters"]["restart_parameters"]["start_datetime"] = data[
        "compute_parameters"
    ]["restart_parameters"]["start_datetime"]

    with open(tmp_yaml, "w") as file:
        yaml.dump(dict_, file)

    try:
        t_route(["-f", tmp_yaml.__str__()])
        return TestStatus(
            status_code=HttpStatusCode.CREATED,
            message="T-Route run successfully using defined configuration",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )
    finally:
        if tmp_yaml.exists():
            tmp_yaml.unlink()
