from pathlib import Path
from typing import List

from pydantic import BaseSettings


class Settings(BaseSettings):
    """
    Configuration settings for the application.

    This class uses Pydantic's BaseSettings to manage any variables,
    allowing for easy integration with environment variables and
    configuration files.

    Attributes
    ----------
    api_v1_str : str
        The base API string.
    base_config: str
        The path to the base_config file that we build custom config files from
    project_name : str
        The project's name for the OPENAPI spec
    qlat_input_path: str
        The path to the docker folders where rfc_channel_forcings live in the shared volume
    restart_path: str
        The path to where restart files will be saved to in the shared volume
    restart_file: str
        The regex string for finding restart files
    geofile_path: str
        The path to the docker folders where the geopackage is located in the shared volume
    paths_to_update: str
        The entries in a config that have relative paths. Used for changig in services/utils.py
    """

    api_v1_str: str = "/api/v1"
    base_config: Path = "/t-route/src/app/core/{}/base_config.yaml"
    project_name: str = "T-Route"
    qlat_input_path: str = "/t-route/data/rfc_channel_forcings/{}/{}"
    restart_path: str = "/t-route/data/troute_restart/{}/"
    restart_file: str = "HYDRO_RST_{}_DOMAIN1"
    geofile_path: str = "/t-route/data/rfc_geopackage_data/{}/{}/downstream.gpkg"
    lower_colorado_paths_to_update: List[List[str]] = [
        ["network_topology_parameters", "supernetwork_parameters", "geo_file_path"],
        ["compute_parameters", "hybrid_parameters", "diffusive_domain"],
        ["compute_parameters", "hybrid_parameters", "topobathy_domain"],
        ["compute_parameters", "hybrid_parameters", "coastal_boundary_domain"],
        ["compute_parameters", "forcing_parameters", "qlat_input_folder"],
        ["compute_parameters", "data_assimilation_parameters", "usace_timeslices_folder"],
        ["compute_parameters", "data_assimilation_parameters", "usgs_timeslices_folder"],
        ["compute_parameters", "data_assimilation_parameters", "canada_timeslices_folder"],
        ["compute_parameters", "data_assimilation_parameters", "LakeOntario_outflow"],
        ["compute_parameters", "data_assimilation_parameters", "reservoir_da", "reservoir_rfc_da", "reservoir_rfc_forecasts_time_series_path"]
    ]
    lower_colorado_paths_to_update: List[List[str]] = [
        ["network_topology_parameters", "supernetwork_parameters", "geo_file_path"],
        ["compute_parameters", "hybrid_parameters", "diffusive_domain"],
        ["compute_parameters", "hybrid_parameters", "topobathy_domain"],
        ["compute_parameters", "hybrid_parameters", "coastal_boundary_domain"],
        ["compute_parameters", "forcing_parameters", "qlat_input_folder"],
        ["compute_parameters", "data_assimilation_parameters", "usace_timeslices_folder"],
        ["compute_parameters", "data_assimilation_parameters", "usgs_timeslices_folder"],
        ["compute_parameters", "data_assimilation_parameters", "canada_timeslices_folder"],
        ["compute_parameters", "data_assimilation_parameters", "LakeOntario_outflow"],
        ["compute_parameters", "data_assimilation_parameters", "reservoir_da", "reservoir_rfc_da", "reservoir_rfc_forecasts_time_series_path"]
    ]
