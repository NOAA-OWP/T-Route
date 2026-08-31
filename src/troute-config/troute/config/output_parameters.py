from pathlib import Path
from pydantic import field_validator, BaseModel, DirectoryPath, FilePath, Field, ValidationInfo

from typing import Optional, List
from typing_extensions import Annotated, Literal

streamOutput_allowedTypes = Literal['.csv', '.nc', '.pkl']


class OutputParameters(BaseModel):
    """
    Parameters controlling model outputs. Output parameters can be left completely blank and no files will be written. 
    However, if 'output_parameters' exists, one of the following specific output file parameters should also be specified. 
    Many of these are meant to mimic WRF-Hydro's outputs.
    """
    chanobs_output: Optional["ChanobsOutput"] = None
    csv_output: Optional["CsvOutput"] = None
    parquet_output: Optional["ParquetOutput"] = None
    chrtout_output: Optional["ChrtoutOutput"] = None
    lite_restart: Optional["LiteRestart"] = None
    hydro_rst_output: Optional["HydroRstOutput"] = None
    wrf_hydro_parity_check: Optional["WrfHydroParityCheck"] = None
    lakeout_output: Optional[DirectoryPath] = None
    test_output: Optional[Path] = None
    stream_output: Optional["StreamOutput"] = None
    lastobs_output: Optional[DirectoryPath] = None


class ChanobsOutput(BaseModel):
    """
    CHANOBS files are outputs from WRF-Hydro containing station observations. This replicates that behavior.
    """
    chanobs_output_directory: Optional[DirectoryPath] = None
    """
    Directory to save CHANOBS output files. If this is None, no CHANOBS will be written.
    """
    chanobs_filepath: Optional[Path] = None
    """
    Filename of CHANOBS output file.
    """


class CsvOutput(BaseModel):
    """
    This is an older alternative to the CSV file writing capabilities of the more recently developed 'stream_output'. 
    This will simply write the full flowveldepth array to a .csv file.
    """
    csv_output_folder: Optional[DirectoryPath] = None
    """
    Directory to save csv output files. If this is None, no csv will be written.
    """
    csv_output_segments: Optional[List[str]] = None
    """
    Subset of segment IDs to include in the output file.
    """


class ParquetOutput(BaseModel):
    # NOTE: required if writing results to parquet
    parquet_output_folder: Optional[DirectoryPath] = None
    parquet_output_segments: Optional[List[str]] = None
    configuration: str = 'None'
    prefix_ids: str = 'wb'


class ChrtoutOutput(BaseModel):
    """
    CHRTOUT files are outputs from WRF-Hydro containing full channel network output. This replicates that behavior.
    """
    wrf_hydro_channel_output_source_folder: Optional[DirectoryPath] = None
    """
    Directory to save CHRTOUT files. No files will be written if this is None.
    """


class LiteRestart(BaseModel):
    """
    Saves final conditions of channel and reservoir dataframes as pickle files to be used in follow up simulation as initial conditions.
    """
    lite_restart_output_directory: Optional[DirectoryPath] = None
    """
    Directory to save lite_restart files. No files will be written if this is None.
    """


class HydroRstOutput(BaseModel):
    """
    Parameters controlling the writing of restart data to HYDRO_RST netcdf files. Mimics WRF-Hydro.
    """
    wrf_hydro_restart_dir: Optional[DirectoryPath] = None
    """
    Directory to save state files.
    """
    wrf_hydro_channel_restart_pattern_filter: str = "HYDRO_RST.*"
    """
    File pattern for state files.
    """
    wrf_hydro_channel_restart_source_directory: Optional[DirectoryPath] = None
    """
    DEPRECATED?
    """
    wrf_hydro_channel_output_source_folder: Optional[DirectoryPath] = None
    """
    DEPRECATED?
    """


class WrfHydroParityCheck(BaseModel):
    """
    Paramters controlling a single-segment parity assessment between t-route and WRF-hydro.
    """
    parity_check_input_folder: Optional[DirectoryPath] = None
    """
    """
    parity_check_file_index_col: str
    """
    """
    parity_check_file_value_col: str
    """
    """
    parity_check_compare_node: str
    """
    """
    parity_check_compare_file_sets: Optional[List["ParityCheckCompareFileSet"]] = None


class ParityCheckCompareFileSet(BaseModel):
    validation_files: List[FilePath]
    """
    """


class StreamOutput(BaseModel):
    """
    t-route's most recent output file type. This will output channel network values (flow, velocity, depth, and nudge values). 
    This has been designed for as much flexibility for user needs as possible, including file type (netcdf, csv, pickle) and how 
    frequently to create output files relative to simulation time and how many output timesteps to include. Only 'stream_output_directory' 
    is required, the other default values will create 1 file per hour of simulation time, containing values at every timestep of 
    simulation. If t-route is run with default dt (300 seconds/5 minutes) for 24 hours, the defaults here would produce 24 output files 
    (1 per hour of simulation), each containing 12 values for each variable (1 value every 5 minutes in the hour of simulation).
    """
    stream_output_directory: Optional[Path] = None
    """
    Directory to save flowveldepth outputs. If this is not None, this form of output will be written.
    """
    mask_output: Optional[FilePath] = None
    """
    Yaml file specifying flowpath/nexus IDs to include in output files.
    """
    stream_output_time: int = 1
    """
    Value is in simulation time hours. This tells t-route how frequently to make output files. '1' would be 1 file per hour 
    of simulation time.
    """
    stream_output_type: streamOutput_allowedTypes = ".nc"
    """
    Output file type.
    """
    stream_output_internal_frequency: Annotated[int, Field(strict=True, ge=5)] = 5
    """
    Value is in minutes. This tells t-route the frequency of t-route's timesteps to include in the output file. For instance, 
    a value of '5' here would output flow, velocity, and depth values every 5 minutes of simulation time. A value of '30' would 
    output values every 30 mintues of simulation time.
    NOTE: This value should not be smaller than dt, and should be a multiple of dt (keep in mind dt is in seconds, while this value 
    is in minutes). So if dt=300(sec), this value cannot be smaller than 5(min) and should be a multiple of 5. 
    """
    
    @field_validator('stream_output_directory')
    @classmethod
    def validate_stream_output_directory(cls, value):
        if value is None:
            return None
            
        # expand ~/output/dir -> /home/user/output/dir
        value = value.expanduser()
        
        if value.exists() and not value.is_dir():
            raise ValueError(f"'stream_output_directory'={value!s} is a file, expected directory.")

        # make directory (and intermediates) if they don't exist
        value.mkdir(parents=True, exist_ok=True)
        return value
    
    @field_validator('stream_output_internal_frequency')
    def validate_stream_output_internal_frequency(cls, value, info: ValidationInfo):
        if value is not None:
            if value % 5 != 0:
                raise ValueError("stream_output_internal_frequency must be a multiple of 5.")
            if info.data.get('stream_output_time') != -1 and value / 60 > info.data['stream_output_time']:
                raise ValueError("stream_output_internal_frequency should be less than or equal to stream_output_time in minutes.")
        return value
 

# Update model references for Pydantic v2
OutputParameters.model_rebuild()
WrfHydroParityCheck.model_rebuild()
