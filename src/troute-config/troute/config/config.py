import os

from pydantic import model_validator, Field, BaseModel
from pathlib import Path

from typing import Any, Dict, Optional
from typing_extensions import Self

from .logging_parameters import LoggingParameters
from .network_topology_parameters import NetworkTopologyParameters
from .compute_parameters import ComputeParameters
from .output_parameters import OutputParameters
from .bmi_parameters import BMIParameters
from ._utils import use_strict


class Config(BaseModel):
    log_parameters: LoggingParameters = Field(default_factory=LoggingParameters)
    # TODO: not sure if default is None or {}. see nhd_io.read_config_file ~:100
    network_topology_parameters: Optional[NetworkTopologyParameters] = None
    # TODO: default appears to be {}. see nhd_io.read_config_file ~:138
    compute_parameters: ComputeParameters = Field(default_factory=dict)
    # TODO: default appears to be {}. see nhd_io.read_config_file ~:141
    output_parameters: OutputParameters = Field(default_factory=dict)
    bmi_parameters: Optional[BMIParameters] = None

    # TODO: `reservoir_data_assimilation_parameters` missing from `v3_doc.yaml` and only included
    # test/BMI/bmi_reservoir_example.yaml

    @classmethod
    def with_strict_mode(cls, **data: Dict[str, Any]) -> Self:
        """
        Create a `Config` instance that verifies existence of file and directory field types. If any
        do not exist, a `pydantic.ValidationError` is raised.

        Returns
        -------
        Self
            `Config` instance
        """
        with use_strict():
            return cls(**data)
    

    ##########################################################################################
    # Validate that, given certain configuration inputs, other inputs are provided.
    ##########################################################################################

    @model_validator(mode='before')
    @classmethod
    def check_levelpool_filepath(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Verify that the level pool parameter file is provided IF waterbodies are being routed."""
        try:
            network_type = values['network_topology_parameters'].supernetwork_parameters.network_type
            waterbody_parameters = values['network_topology_parameters'].waterbody_parameters
            if waterbody_parameters:
                simulate_waterbodies = waterbody_parameters.break_network_at_waterbodies
                levelpool = waterbody_parameters.level_pool

                if simulate_waterbodies and network_type=='NHDNetwork':
                    assert levelpool, 'Waterbody simulation is enabled for NHDNetwork, but level pool parameters are missing.'
                    levelpool_file = levelpool.level_pool_waterbody_parameter_file_path
                    assert levelpool_file, 'Waterbody simulation is enabled for NHDNetwork, but no level pool parameter file is provided.'
        except (KeyError, AttributeError, TypeError):
            # Skipping validation if the data does not exist as skip_on_failure is not supported in Pydantic >v2.0
            pass
        
        return values


    @model_validator(mode='before')
    @classmethod
    def check_diffusive_domain(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Verify that the diffusive domain file is provided IF diffusive routing is enabled."""
        try:
            hybrid_parameters = values['compute_parameters'].hybrid_parameters
            if hybrid_parameters:
                run_hybrid = hybrid_parameters.run_hybrid_routing
                if run_hybrid:
                    assert hybrid_parameters.diffusive_domain, 'Diffusive routing is enabled, but diffusive domain file is missing.'
        except (KeyError, AttributeError, TypeError):
            # Skipping validation if the data does not exist as skip_on_failure is not supported in Pydantic >v2.0
            pass

        return values


    @model_validator(mode='before')
    @classmethod
    def check_topobathy_domain(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Verify that a topobathy domain file is provided IF using natural cross-sections has been enabled."""
        try:
            hybrid_parameters = values['compute_parameters'].hybrid_parameters
            if hybrid_parameters:
                use_natl_xsections = hybrid_parameters.use_natl_xsections
                if use_natl_xsections:
                    assert hybrid_parameters.topobathy_domain, 'Use natural cross-sections is enabled, but topobathy domain file is missing.'
        except (KeyError, AttributeError, TypeError):
            # Skipping validation if the data does not exist as skip_on_failure is not supported in Pydantic >v2.0
            pass

        return values


    @model_validator(mode='before')
    @classmethod
    def check_refactored(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Verify that the refactored domain and topobathy files are provided IF refactored network is enabled."""
        try:
            hybrid_parameters = values['compute_parameters'].hybrid_parameters
            if hybrid_parameters:
                run_refactored_network = hybrid_parameters.run_refactored_network
                if run_refactored_network:
                    assert hybrid_parameters.refactored_domain, 'Run refactored network is enabled, but refactored domain file is missing.'
                    assert hybrid_parameters.refactored_topobathy_domain, 'Run refactored network is enabled, but refactored topobathy domain file is missing.'
        except (KeyError, AttributeError, TypeError):
            # Skipping validation if the data does not exist as skip_on_failure is not supported in Pydantic >v2.0
            pass

        return values


    @model_validator(mode='before')
    @classmethod
    def check_coastal_domain(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Verify that a coastal boundary domain file is provided IF diffusive routing is enabled and coastal forcing files are provided."""
        try:
            hybrid_parameters = values['compute_parameters'].hybrid_parameters
            forcing_parameters = values['compute_parameters'].forcing_parameters
            if hybrid_parameters:
                coastal_boundary_input_file = forcing_parameters.coastal_boundary_input_file
                if coastal_boundary_input_file:
                    assert hybrid_parameters.coastal_boundary_domain, 'Coastal boundary forcing files provided, but coastal boundary domain file is missing.'
        except (KeyError, AttributeError, TypeError):
            # Skipping validation if the data does not exist as skip_on_failure is not supported in Pydantic >v2.0
            pass

        return values


    @model_validator(mode='before')
    @classmethod
    def check_gage_segID_crosswalk_file(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Verify that a gage-segment cross-walk file is provided IF streamflow nudging is enabled on a NHD network."""
        try:
            da_parameters = values['compute_parameters'].data_assimilation_parameters
            if da_parameters:
                streamflow_DA = da_parameters.streamflow_da
                if streamflow_DA:
                    streamflow_nudging = streamflow_DA.streamflow_nudging
                    network_type = values['network_topology_parameters'].supernetwork_parameters.network_type
                    if streamflow_nudging and network_type=='NHDNetwork':
                        assert streamflow_DA.gage_segID_crosswalk_file, 'Streamflow nudging is enabled on NHDNetwork, but gage_segID_crosswalk_file is missing.'
        except (KeyError, AttributeError, TypeError):
            # Skipping validation if the data does not exist as skip_on_failure is not supported in Pydantic >v2.0
            pass

        return values


    @model_validator(mode='before')
    @classmethod
    def check_rfc_parameters(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Verify that RFC parameter file (IF on NHD network) and timeseries directory path are provided IF RFC reservoir DA is enabled."""
        try:
            da_parameters = values['compute_parameters'].data_assimilation_parameters
            if da_parameters:
                reservoir_da = da_parameters.reservoir_da
                if reservoir_da:
                    reservoir_rfc_da = reservoir_da.reservoir_rfc_da
                    reservoir_rfc_forecasts = False
                    if reservoir_rfc_da and reservoir_rfc_da.reservoir_rfc_forecasts:
                        reservoir_rfc_forecasts = reservoir_rfc_da.reservoir_rfc_forecasts
                        reservoir_rfc_forecasts_time_series_path = reservoir_rfc_da.reservoir_rfc_forecasts_time_series_path
                    if reservoir_rfc_forecasts:
                        error_message = ''
                        network_type = values['network_topology_parameters'].supernetwork_parameters.network_type
                        reservoir_parameter_file = reservoir_da.reservoir_parameter_file
                        if not reservoir_parameter_file and network_type=='NHDNetwork':
                            error_message += ' Reservoir_parameter_file is missing (and network type is NHDNetwork).'
                        if not reservoir_rfc_forecasts_time_series_path:
                            error_message  += ' RFC timeseries path is missing.'
                        else:
                            if not os.path.exists(reservoir_rfc_forecasts_time_series_path):
                                error_message += ' reservoir_rfc_forecasts_time_series_path provided does not exist. '                    
                        assert not error_message, 'RFC forecast is enabled, but:' + error_message
        except (KeyError, AttributeError, TypeError):
            # Skipping validation if the data does not exist as skip_on_failure is not supported in Pydantic >v2.0
            pass

        return values
    
    @model_validator(mode='before')
    @classmethod
    def check_usgs_reservoir_da_parameters(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Verify that USGS parameter file (IF on NHD network) and timeslice directory path are provided IF USGS reservoir DA is enabled."""
        try:
            da_parameters = values['compute_parameters'].data_assimilation_parameters
            if da_parameters:
                reservoir_da = da_parameters.reservoir_da
                if reservoir_da:
                    reservoir_persistence_da = reservoir_da.reservoir_persistence_da
                    reservoir_persistence_usgs = False
                    if reservoir_persistence_da:
                        reservoir_persistence_usgs = reservoir_persistence_da.reservoir_persistence_usgs
                        usgs_timeslices_folder = values['compute_parameters'].data_assimilation_parameters.usgs_timeslices_folder
                    if reservoir_persistence_usgs:
                        error_message = ''
                        network_type = values['network_topology_parameters'].supernetwork_parameters.network_type
                        reservoir_parameter_file = reservoir_da.reservoir_parameter_file
                        if not reservoir_parameter_file and network_type=='NHDNetwork':
                            error_message += ' Reservoir_parameter_file is missing (and network type is NHDNetwork).'
                        if not usgs_timeslices_folder:
                            error_message  += ' USGS_timeslices_folder is missing.'
                        else:
                            if not os.path.exists(usgs_timeslices_folder):
                                error_message += ' USGS_timeslices_folder path provided does not exist. '
                        assert not error_message, 'USGS reservoir DA is enabled, but:' + error_message
        except (KeyError, AttributeError, TypeError):
            # Skipping validation if the data does not exist as skip_on_failure is not supported in Pydantic >v2.0
            pass

        return values


    @model_validator(mode='before')
    @classmethod
    def check_usace_reservoir_da_parameters(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Verify that USACE parameter file (IF on NHD network) and timeslice directory path are provided IF USACE reservoir DA is enabled."""
        try:
            da_parameters = values['compute_parameters'].data_assimilation_parameters
            if da_parameters:
                reservoir_da = da_parameters.reservoir_da
                if reservoir_da:
                    reservoir_persistence_da = reservoir_da.reservoir_persistence_da
                    reservoir_persistence_usace = False
                    if reservoir_persistence_da:
                        reservoir_persistence_usace = reservoir_persistence_da.reservoir_persistence_usace
                        usace_timeslices_folder = values['compute_parameters'].data_assimilation_parameters.usace_timeslices_folder
                    if reservoir_persistence_usace:
                        error_message = ''
                        network_type = values['network_topology_parameters'].supernetwork_parameters.network_type
                        reservoir_parameter_file = reservoir_da.reservoir_parameter_file
                        if not reservoir_parameter_file and network_type=='NHDNetwork':
                            error_message += ' Reservoir_parameter_file is missing (and network type is NHDNetwork).'
                        if not usace_timeslices_folder:
                            error_message  += ' USACE_timeslices_folder is missing.'
                        else:
                            if not os.path.exists(usace_timeslices_folder):
                                error_message += ' USACE_timeslices_folder path provided does not exist. '                    
                        assert not error_message, 'USACE reservoir DA is enabled, but:' + error_message
        except (KeyError, AttributeError, TypeError):
            # Skipping validation if the data does not exist as skip_on_failure is not supported in Pydantic >v2.0
            pass

        return values


    @model_validator(mode='before')
    @classmethod
    def check_qlat_inputs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Verify that a forcing directory is provided. TODO: Add bypass for this check once t-route is connected to BMI."""
        try:
            forcing_parameters = values['compute_parameters'].forcing_parameters
            if forcing_parameters:
                qlat_forcing_sets = forcing_parameters.qlat_forcing_sets
                qlat_input_folder = forcing_parameters.qlat_input_folder
                if not qlat_forcing_sets:
                    assert qlat_input_folder, 'No qlat_input_folder is specified in the forcing_parameters'
        except (KeyError, AttributeError, TypeError):
            # Skipping validation if the data does not exist as skip_on_failure is not supported in Pydantic >v2.0
            pass

        return values


    @model_validator(mode='before')
    @classmethod
    def check_wrf_hydro_restart_files(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Verify that auxilary WRF-Hydro files exist IF a WRF-Hydro restart file is being used."""
        try:
            restart_parameters = values['compute_parameters'].restart_parameters
            if restart_parameters:
                wrf_hydro_channel_restart_file = restart_parameters.wrf_hydro_channel_restart_file
                wrf_hydro_waterbody_restart_file = restart_parameters.wrf_hydro_waterbody_restart_file
                if wrf_hydro_channel_restart_file:
                    assert restart_parameters.wrf_hydro_channel_ID_crosswalk_file, 'WRF-Hydro channel restart file provided, but wrf_hydro_channel_ID_crosswalk_file file is missing.'
                if wrf_hydro_waterbody_restart_file:
                    error_message = ''
                    if not restart_parameters.wrf_hydro_waterbody_ID_crosswalk_file:
                        error_message += ' wrf_hydro_channel_ID_crosswalk_file is missing.'
                    if not restart_parameters.wrf_hydro_waterbody_crosswalk_filter_file:
                        error_message += ' wrf_hydro_waterbody_crosswalk_filter_file is missing.'
                    assert not error_message, 'WRF-Hydro waterbody_restart file is provided, but:' + error_message
        except (KeyError, AttributeError, TypeError):
            # Skipping validation if the data does not exist as skip_on_failure is not supported in Pydantic >v2.0
            pass

        return values


    @model_validator(mode='before')
    @classmethod
    def check_start_datetime(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Verify that a start datetime is provided."""
        try:
            restart_parameters = values['compute_parameters'].restart_parameters
            if restart_parameters:
                wrf_hydro_channel_restart_file = restart_parameters.wrf_hydro_channel_restart_file
                lite_channel_restart_file = restart_parameters.lite_channel_restart_file
                if not (wrf_hydro_channel_restart_file or lite_channel_restart_file):
                    assert restart_parameters.start_datetime, 'No start_datetime provided in config file for cold start (no restart files).'
        except (KeyError, AttributeError, TypeError):
            # Skipping validation if the data does not exist as skip_on_failure is not supported in Pydantic >v2.0
            pass

        return values


    @model_validator(mode='before')
    @classmethod
    def check_flowpath_edge_list(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Verfiy that a flowpath_edge_list file is provided IF using a .json geo-file."""
        try:
            geo_file_path = values['network_topology_parameters'].supernetwork_parameters.geo_file_path
            flowpath_edge_list = values['network_topology_parameters'].supernetwork_parameters.flowpath_edge_list
            if Path(geo_file_path).suffix=='.json':
                assert flowpath_edge_list, "geo_file_path is json, but no flowpath_edge_list is provided."
                assert Path(flowpath_edge_list).suffix=='.json', "geo_file_path is json, but flowpath_edge_list is a different file type."
        except (KeyError, AttributeError, TypeError):
            # Skipping validation if the data does not exist as skip_on_failure is not supported in Pydantic >v2.0
            pass

        return values


    @model_validator(mode='before')
    @classmethod
    def check_lite_restart_directory(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Verify that a lite restart output directory is provided IF lite restart output is enabled."""
        try:
            if values['output_parameters']:
                lite_restart = values['output_parameters'].lite_restart
                if lite_restart is not None:
                    lite_restart_directory = lite_restart.lite_restart_output_directory
                    assert lite_restart_directory, "lite_restart is present in output parameters, but no lite_restart_output_directory is provided."
        except (KeyError, AttributeError, TypeError):
            # Skipping validation if the data does not exist as skip_on_failure is not supported in Pydantic >v2.0
            pass
        
        return values


    @model_validator(mode='before')
    @classmethod
    def check_nts_dt_stream_output_internal_frequency(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Verify that stream output parameters make sense IF writing stream output files is enabled."""
        try:
            compute_params = values.get('compute_parameters')
            output_params = values.get('output_parameters')
            
            if compute_params and output_params:
                # Directly access ForcingParameters
                forcing_params = compute_params.forcing_parameters
                stream_output = output_params.stream_output

                if forcing_params and stream_output:
                    nts = forcing_params.nts
                    dt = forcing_params.dt
                    internal_freq = stream_output.stream_output_internal_frequency

                    # Perform the check
                    if nts and dt and internal_freq:
                        result = nts * dt / (internal_freq * 60)
                        if not result.is_integer():
                            raise ValueError("UPDATE nts. Make sure 'nts' times 'dt' divided by ('stream_output_internal_frequency' times 60) is a whole number in your configuration.")
        except (KeyError, AttributeError, TypeError):
            # Skipping validation if the data does not exist as skip_on_failure is not supported in Pydantic >v2.0
            pass

        return values
