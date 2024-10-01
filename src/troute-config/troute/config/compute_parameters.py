from pydantic import BaseModel, Field, validator
from datetime import datetime

from typing import Optional, List, Union
from typing_extensions import Literal

from .types import FilePath, DirectoryPath
from ._validators import coerce_datetime, coerce_none_to_default


# ---------------------------- Compute Parameters ---------------------------- #

ParallelComputeMethod = Literal[
    "serial",
    "by-network",
    "by-subnetwork-jit",
    "by-subnetwork-jit-clustered",
    "by-subnetwork-diffusive",
    "bmi",
    "serial-hybrid-routing"
]

ComputeKernel = Literal["V02-structured","V02-structured-hybrid-routing", "diffusive"]


class ComputeParameters(BaseModel):
    """
    Parameters specific to the routing simulation.
    """
    parallel_compute_method: ParallelComputeMethod = "by-network"
    """
    parallel computing scheme used during simulation, options below
    - "serial": no parallelization
    - "by-network": parallelization across independent drainage basins
    - "by-subnetwork-jit": parallelization across subnetworks 
    - "by-subnetwork-jit-clustered": parallelization across subnetworks, with clustering to optimize scaling
    """
    compute_kernel: ComputeKernel = "V02-structured"
    """
    routing engine used for simulation
    - "V02-structured" - Muskingum Cunge
    NOTE: There are two other options that were previously being developed for use with the diffusive kernel, 
    but they are now depricated:
    - "diffusive" - Diffusive with adaptive timestepping
    - "diffusice_cnt" - Diffusive with CNT numerical solution
    TODO: Remove these additional options? And this parameter altogether as there is only one option?
    """
    assume_short_ts: bool = False
    """
    If True the short timestep assumption used in WRF hyro is used. if False, the assumption is dropped.
    """
    subnetwork_target_size: int = 10000
    """
    The target number of segments per subnetwork, only needed for "by-subnetwork..." parallel schemes.
    The magnitude of this parameter affects parallel scaling. This is to improve efficiency. Default value has 
    been tested as the fastest for CONUS simultions. For smaller domains this can be reduced.
    """
    cpu_pool: Optional[int] = 1
    """
    Number of CPUs used for parallel computations
    If parallel_compute_method is anything but 'serial', this determines how many cpus to use for parallel processing.
    """
    return_courant: bool = False
    """
    If True, Courant metrics are returnd with simulations. This only works for MC simulations
    """

    restart_parameters: "RestartParameters" = Field(default_factory=dict)
    hybrid_parameters: "HybridParameters" = Field(default_factory=dict)
    forcing_parameters: "ForcingParameters" = Field(default_factory=dict)
    data_assimilation_parameters: "DataAssimilationParameters" = Field(default_factory=dict)


# TODO: determine how to handle context specific required fields
# TODO: consider other ways to handle wrf hydro fields (i.e. subclass)
class RestartParameters(BaseModel):
    """
    Parameters specifying warm-state simulation conditions.
    """
    start_datetime: Optional[datetime] = None
    """
    Time of model initialization (timestep zero). Datetime format should be %Y-%m-%d_%H:%M, e.g., 2023-04-25_00:00
    This start time will control which forcing files and TimeSlice files are required for the simulation. 
    If the start time is erroneously enertered, such that there are no available forcing files, then the simulation will fail. 
    Likewise, if there are no TimeSlice files available, then data assimilation will not occur.
    NOTE: The default is 'None' because the start date can be determined from restart files
    such as 'lite_channel_restart_file' or 'wrf_hydro_channel_restart_file'. But if no restart
    file is provided, this parameter is required.
    """
    lite_channel_restart_file: Optional[FilePath] = None
    """
    Filepath to a 'lite' channel restart file create by a previous t-route simulation. If a file is specified, then it will be 
    given preference over WRF restart files for a simulation restart.
    """
    lite_waterbody_restart_file: Optional[FilePath] = None
    """
    Filepath to a 'lite' waterbody restart file create by a previous t-route simulation. If a file is specified, then it will be 
    given preference over WRF restart files for a simulation restart.
    """

    wrf_hydro_channel_restart_file: Optional[FilePath] = None
    """
    Filepath to WRF Hydro HYDRO_RST file. This file does not need to be timed with start_datetime, which allows initial states
    from one datetime to initialize a simulation with forcings starting at a different datetime. However, if the start_datetime 
    parameter is not specified, then the time attribute in the channel restart file will be used as the starting time of the simulation.
    """
    wrf_hydro_channel_ID_crosswalk_file: Optional[FilePath] = None
    """
    Filepath to channel geometry file.
    NOTE: if `wrf_hydro_channel_restart_file` is given, `wrf_hydro_channel_ID_crosswalk_file` is required
    """
    wrf_hydro_channel_ID_crosswalk_file_field_name: Optional[str] = None
    """
    Field name of segment IDs in restart file.
    """
    wrf_hydro_channel_restart_upstream_flow_field_name: Optional[str] = None
    """
    Field name of upstream flow in restart file.
    """
    wrf_hydro_channel_restart_downstream_flow_field_name: Optional[str] = None
    """
    Field name of downstream flow in restart file.
    """
    wrf_hydro_channel_restart_depth_flow_field_name: Optional[str] = None
    """
    Field name of depth in restart file.
    """

    wrf_hydro_waterbody_restart_file: Optional[FilePath] = None
    """
    Filepath to waterbody restart file. This is often the same as wrf_hydro_channel_restart_file.
    """
    wrf_hydro_waterbody_ID_crosswalk_file: Optional[FilePath] = None
    """
    Filepath to lake parameter file.
    NOTE: required if `wrf_hydro_waterbody_restart_file`
    """
    wrf_hydro_waterbody_ID_crosswalk_file_field_name: Optional[str] = None
    """
    Field name of waterbody ID.
    """
    wrf_hydro_waterbody_crosswalk_filter_file: Optional[FilePath] = None
    """
    Filepath to channel geometry file.
    """
    wrf_hydro_waterbody_crosswalk_filter_file_field_name: Optional[str] = None
    """
    Fieldname of waterbody IDs in channel geometry file.
    """
    

    _coerce_datetime = validator("start_datetime", pre=True, allow_reuse=True)(
        coerce_datetime
    )


# TODO: determine how to handle context specific required fields
class HybridParameters(BaseModel):
    """
    Parameters controlling the use of MC/diffusive hybrid simulations. Only include/populate these parameters if an 
    MC/diffusive hybrid simulations is desired.
    """
    run_hybrid_routing: bool = False
    """
    Boolean parameter whether or not hybrid routing is actived. If it is set to True, the hybrid routing is activated. 
    If false, MC is solely used for channel flow routing.
    NOTE: required for hybrid simulations
    """
    diffusive_domain: Optional[FilePath] = None
    """
    Filepath to diffusive domain dictionary file. This file can be either JSON or yaml and contain a dictionary
    of diffusive network segments, organized by tailwater ID (keys). This is a file such as: 
    https://github.com/NOAA-OWP/t-route/blob/master/test/LowerColorado_TX_v4/domain/coastal_domain_tw.yaml
    This file defines tailwater and head water flowpath IDs for the diffusive domain. See file for more info.
    NOTE: required for hybrid simulations
    """
    use_natl_xsections: bool = False
    """
    Boolean parameter whether or not natural cross section data is used. If it is set to True, diffusive model 
    uses natural cross section data. If False, diffusive model uses synthetic cross section defined by RouteLink.nc
    """
    topobathy_domain: Optional[FilePath] = None
    """
    Filepath to topobathy data for channel cross sections. Currently (June 25, 2024), 3D cross section data
    is contained in a separate file, which this parameter should point to. In the future this data may simply be
    included in the hydrofabric.
    Topobathy data of a channel cross section is defined by comid.
    NOTE: Required for diffusive routing for natural cross sections. 
    """
    run_refactored_network: bool = False
    """
    Boolean parameter whether or not to run the diffusive module on a refactored network. This was necessary on
    the NHD network due to short segments causing computational issues. Not needed for HYFeatures.
    """
    refactored_domain: Optional[FilePath] = None
    """
    A file with refactored flowpaths to eliminate short segments.
    NOTE: Only needed for NHD network. 
    """
    refactored_topobathy_domain: Optional[FilePath] = None
    """
    A file with refactored topobathy data.
    NOTE: Only needed for NHD network.
    """
    coastal_boundary_domain: Optional[FilePath] = None
    """
    File containing crosswalk between diffusive tailwater segment IDs and coastal model output node IDs. 
    This is needed if t-route will use outputs from a coastal model as the downstream boundary condition for
    the diffusive module. See example:
    https://github.com/NOAA-OWP/t-route/blob/master/test/LowerColorado_TX_v4/domain/coastal_domain_crosswalk.yaml
    NOTE: This is related to the ForcingParameters -> coastal_boundary_input_file parameter. 
    """


class QLateralForcingSet(BaseModel):
    """
    Forcing files and number of timesteps associated with each simulation loop. This is optional, only include if 
    explicitly listing the forcing files in each set. If this variable is not present, make sure nts, 
    qlat_file_pattern_filter, and max_loop_size variables are listed.
    NOTE: Using nts, qlat_input_folder, qlat_file_pattern_filter, and max_loop_size is the preferred method.
    """
    nts: "QLateralFiles"
    """
    Number of timesteps in loop iteration 1. This corresponds to the number of files listed in qlat_files.
    This parameter is repeated for as many iterations as are desired.
    """


class QLateralFiles(BaseModel):
    qlat_files: List[FilePath]
    """
    List of forcing file names to be used in a single iteration.
    """


class StreamflowDA(BaseModel):
    """
    Parameters controlling streamflow nudging DA
    """
    streamflow_nudging: bool = False
    """
    Boolean, determines whether or not streamflow nudging is performed.
    NOTE: Mandatory for streamflow DA
    """
    gage_segID_crosswalk_file: Optional[FilePath] = None
    """
    File relating stream gage IDs to segment links in the model domain. This is typically the RouteLink file.
    NOTE: Mandatory for streamflow DA on NHDNetwork. Not necessary on HYFeatures as this information is included
    in the hydrofabric.
    """
    crosswalk_gage_field: Optional[str] = 'gages'
    """
    Column name for gages in gage_segID_crosswalk_file.
    NOTE: Not necessary on HYFeatures.
    """
    crosswalk_segID_field: Optional[str] = 'link'
    """
    Column name for flowpaths/links in gage_segID_crosswalk_file.
    NOTE: Not necessary on HYFeatures.
    """
    lastobs_file: Optional[FilePath] = None
    """
    File containing information on the last streamflow observations that were assimilated from a previous t-route run. 
    This is used for a 'warm' restart. Mostly used for operational NWM settings.
    """
    diffusive_streamflow_nudging: bool = False
    """
    If True, enable streamflow data assimilation in diffusive module. 
    NOTE: Not yet implemented, leave as False. (June 25, 2024)
    """


class ReservoirPersistenceDA(BaseModel):
    """
    Parameters controlling persistence reservoir DA. This if for USGS/USACE reservoirs.
    """
    reservoir_persistence_usgs: bool = False
    """
    If True, USGS reservoirs will perform data assimilation.
    """
    reservoir_persistence_usace: bool = False
    """
    If True, USACE reservoirs will perform data assimilation.
    """
    reservoir_persistence_greatLake: bool = False
    """
    If True, Great Lakes will perform data assimilation.
    """

    crosswalk_usgs_gage_field: str = "usgs_gage_id"
    """
    Column name designation in files for USGS gages.
    """
    crosswalk_usace_gage_field: str = "usace_gage_id"
    """
    Column name designation in files for USACE gages.
    """
    crosswalk_usgs_lakeID_field: str = "usgs_lake_id"
    """
    Column name designation in files for USGS lake IDs.
    """
    crosswalk_usace_lakeID_field: str = "usace_lake_id"
    """
    Column name designation in files for USACE lake IDs.
    """


class ReservoirRfcParameters(BaseModel):
    """
    Parameters controlling RFC reservoirs DA.
    """
    reservoir_rfc_forecasts: Literal[True] = True
    """
    If True, RFC reservoirs will perform data assimilation.
    """
    reservoir_rfc_forecasts_time_series_path: Optional[DirectoryPath] = None
    """
    Directory containing RFC timeseries files.
    NOTE: Required if reservoir_rfc_forecasts is True.
    """
    reservoir_rfc_forecasts_lookback_hours: int = 28
    """
    Hours to look back in time from simulation time for RFC timeseries files.
    """
    reservoir_rfc_forecasts_offset_hours: int = 28
    """
    Offset hours forward in time from simulation time to look for files. 
    This helps find the most recent RFC timeseries files for operational NWM use.
    """
    reservoir_rfc_forecast_persist_days: int = 11
    """
    Days to persist an observation when no new, good observations can be found.
    """


class ReservoirRfcParametersDisabled(BaseModel):
    reservoir_rfc_forecasts: Literal[False] = False


class ReservoirDA(BaseModel):
    """
    Parameters controlling reservoir DA.
    """
    reservoir_persistence_da: Optional[ReservoirPersistenceDA] = None
    reservoir_rfc_da: Optional[
        Union[ReservoirRfcParameters, ReservoirRfcParametersDisabled]
    ] = Field(None, discriminator="reservoir_rfc_forecasts")
    reservoir_parameter_file: Optional[FilePath] = None
    """
    File conaining reservoir parameters (e.g., reservoir_index_AnA.nc).
    NOTE: Needed for NHDNetwork, but not HYFeatures as this information is included in the hydrofabric.
    """


class DataAssimilationParameters(BaseModel, extra='ignore'):
    """
    Parameters controlling data assimilation.
    """
    usgs_timeslices_folder: Optional[DirectoryPath] = None
    """
    Directory path to usgs timeslice files.
    NOTE: required for streamflow nudging and/or USGS reservoir DA
    """
    usace_timeslices_folder: Optional[DirectoryPath] = None
    """
    Directory path to usace timeslice files.
    NOTE: required for USACE reservoir DA
    """
    canada_timeslices_folder: Optional[DirectoryPath] = None
    """
    Directory path to canadian timeslice files. 
    NOTE: required for Lake Erie DA (and streamflow nudging using Canadian gages, though that has not been 
    implemented as of June 25, 2024).
    """
    LakeOntario_outflow: Optional[FilePath] = None
    """
    CSV file containing DA values for Lake Ontario. Needs to be obtained and pre-processed from https://ijc.org/en/loslrb/watershed/flows.
    NOTE: Required for Lake Ontario DA.
    """
    timeslice_lookback_hours: int = 24
    """
    Number of hours to look back in time (from simulation time) for USGS, USACE, and Canadian timeslice data assimilation files.
    """
    interpolation_limit_min: int = 59
    """
    Limit on how many missing values can be replaced by linear interpolation from timeslice files.
    """

    wrf_hydro_lastobs_lead_time_relative_to_simulation_start_time: int = 0
    """
    Lead time of lastobs relative to simulation start time (secs).
    NOTE: Only relevant if using a WRF-Hydro lastobs restart file.
    """
    wrf_lastobs_type: str = "obs-based"
    
    streamflow_da: StreamflowDA = None
    reservoir_da: Optional[ReservoirDA] = None

    qc_threshold: float = Field(1, ge=0, le=1)
    """
    Threshold for determining which observations are deemed acceptable for DA and which are not. If the values is set to 1, 
    then only the very best observations are retained. On the other hand, if the value is set to 0, then all observations will be 
    used for assimilation, even those markesd as very poor quality.
    """

    _coerce_none_to_default = validator(
        "timeslice_lookback_hours", "qc_threshold", pre=True, allow_reuse=True
    )(coerce_none_to_default)


class ForcingParameters(BaseModel):
    """
    Parameters controlling model forcing.
    """
    qts_subdivisions: int = 12
    """
    The number of routing simulation timesteps per qlateral time interval. For example, if dt_qlateral = 3600 secs, 
    and dt = 300 secs, then qts_subdivisions = 3600/300 = 12
    """
    dt: int = 300
    """
    Time step size (seconds). Default is 5 mintues
    """
    qlat_input_folder: Optional[DirectoryPath] = None
    nts: Optional[int] = 288
    """
    Number of timesteps. This value, multiplied by 'dt', gives the total simulation time in seconds.
    """
    max_loop_size: int = 24
    """
    Value is in hours. To handle memory issues, t-route can divvy it's simulation time into chunks, reducing the amount 
    of forcing and data assimilation files it reads into memory at once. This is the size of those time loops.
    """
    qlat_file_index_col: str = "feature_id"
    """
    Name of column containing flowpath/nexus IDs
    """
    qlat_file_value_col: str = "q_lateral"
    """
    Name of column containing q_lateral data.
    """
    qlat_file_gw_bucket_flux_col: str = "qBucket"
    """
    Groundwater bucket flux (to channel) variable name in forcing file.
    NOTE: Only needed if using WRF-Hydro output files (CHRTOUT) as forcing files.
    """
    qlat_file_terrain_runoff_col: str = "qSfcLatRunoff"
    """
    Surface terrain runoff (to channel) variable name in forcing file.
    NOTE: Only needed if using WRF-Hydro output files (CHRTOUT) as forcing files.
    """
    qlat_file_pattern_filter: Optional[str] = "*NEXOUT"
    """
    Globbing file pattern to identify q_lateral forcing files.
    """

    qlat_forcing_sets: Optional[List[QLateralForcingSet]] = None
    binary_nexus_file_folder: Optional[DirectoryPath] = None
    """
    Directory to save converted forcing files. Only needed if running t-route as part of ngen suite AND if t-route is having memory issues.
    NOTE: Exlpanation: Ngen outputs q_lateral files as 1 file per nexus containing all timesteps. t-route requires 1 file per timestep 
    containing all locations. If this parameter is omitted or left blank, t-route will simply read in all of ngen's output q_lateral files 
    into memory and will attempt routing. If the simulation is large (temporally and/or spatially), t-route might crash due to memory issues. 
    By providing a directory to this parameter, t-route will convert ngen's output q_lateral files into parquet files in the format t-route 
    needs. Then, during routing, t-route will only read the required parquet files as determined by 'max_loop_size', thus reducing memory.
    """
    coastal_boundary_input_file: Optional[FilePath] = None
    """
    File containing coastal model output.
    NOTE: Only used if running diffusive routing.
    """


ComputeParameters.update_forward_refs()
QLateralForcingSet.update_forward_refs()
ReservoirRfcParameters.update_forward_refs()
