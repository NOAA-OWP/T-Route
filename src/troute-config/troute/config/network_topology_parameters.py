from pydantic import BaseModel, Field, validator

from typing import Optional, List, Union, Dict, Any
from typing_extensions import Literal

from .types import FilePath, DirectoryPath


class NetworkTopologyParameters(BaseModel):
    """
    Parameters controlling how the stream network is synthesized.
    """
    preprocessing_parameters: "PreprocessingParameters" = Field(default_factory=dict)
    supernetwork_parameters: "SupernetworkParameters"
    waterbody_parameters: "WaterbodyParameters" = Field(default_factory=dict)


class PreprocessingParameters(BaseModel):
    """
    Parameters controlling the creation and use of preprocessed network graph data.
    """
    preprocess_only: bool = False
    """
    If True, then network graph objects will be created, saved to disk, and then the execution will stop.
    """
    preprocess_output_folder: Optional[DirectoryPath] = None
    """
    Directory to save preprocessed data to.
    NOTE: required if preprocess_only = True
    """
    preprocess_output_filename: str = "preprocess_output"
    """
    Name to save preprocessed file to (do not include file extension).
    """
    use_preprocessed_data: bool = False
    """
    If True, used preprocessed network data istead of reading from geo_file_path.
    """
    # NOTE: required if use_preprocessed_data = True
    # TODO: determine if str type
    preprocess_source_file: Optional[FilePath] = None
    """
    Filepath of preprocessed data.
    NOTE: required if use_preprocessed_data = True
    """


class SupernetworkParameters(BaseModel):
    """
    Parameters specific to the stream network.
    """
    title_string: Optional[str] = None
    """
    Used for simulation identification. Appears in csv filename, if csv oupt is used.
    Otherwise, this variable is of little use. 
    """
    geo_file_path: FilePath
    """
    Path to the hydrofabric. Currently accepts geopackage (assumes HYFeatures), geojson (assumes HYFeatures), 
    json (assumes HYFeatures), netcdf (assumes NHD).
    """
    network_type: Literal["HYFeaturesNetwork", "NHDNetwork"] = "HYFeaturesNetwork"
    """
    Specify if this is an NHD network or a HYFeatures network.
    """
    flowpath_edge_list: Optional[str] = None
    """
    File containing dictionary of connections between segment IDs and nexus IDs.
    NOTE: Only used if using geojson files for hydrofabric.
    """
    mask_file_path: Optional[FilePath] = None
    """
    File containing channel mask file.
    NOTE: Not implemented for HYFeatures.
    """
    mask_layer_string: str = ""
    mask_driver_string: Optional[str] = None
    mask_key: int = 0

    columns: Optional["Columns"] = None
    """
    Attribute names in channel geometry file.
    Default values depend on newtork type.
    """
    # NOTE: required for CONUS-scale simulations with NWM 2.1 or 3.0 Route_Link.nc data
    synthetic_wb_segments: Optional[List[int]] = Field(
        default_factory=lambda: [
            4800002,
            4800004,
            4800006,
            4800007,
        ]
    )
    """
    Synthetic waterbody segment IDs that are used to construct the Great Lakes
    NOTE: required for CONUS-scale simulations with NWM 2.1 or 3.0 Route_Link.nc data
    """
    synthetic_wb_id_offset: float = 9.99e11
    """
    Arbitrary large number appended to synthetic_wb_segments in their handling process
    """

    terminal_code: int = 0
    """
    Coding in channel geometry dataset for segments draining to ocean. A '0' ID indicates there is nothing downstream.
    """

    driver_string: Union[str, Literal["NetCDF"]] = "NetCDF"
    layer_string: int = 0
    
    @validator("columns", always=True)
    def get_columns(cls, columns: dict, values: Dict[str, Any]) -> dict:
        if columns is None:
            if values['network_type']=="HYFeaturesNetwork":
                default_columns = {
                    'key'       : 'id',
                    'downstream': 'toid',
                    'dx'        : 'Length_m',
                    'n'         : 'n',
                    'ncc'       : 'nCC',
                    's0'        : 'So',
                    'bw'        : 'BtmWdth',
                    'waterbody' : 'rl_NHDWaterbodyComID',
                    'gages'     : 'rl_gages',
                    'tw'        : 'TopWdth',
                    'twcc'      : 'TopWdthCC',
                    'musk'      : 'MusK',
                    'musx'      : 'MusX',
                    'cs'        : 'ChSlp',
                    'alt'       : 'alt',
                    'mainstem'  : 'mainstem',
                    }
            else:
                default_columns = {
                    'key'       : 'link',
                    'downstream': 'to',
                    'dx'        : 'Length',
                    'n'         : 'n',
                    'ncc'       : 'nCC',
                    's0'        : 'So',
                    'bw'        : 'BtmWdth',
                    'waterbody' : 'NHDWaterbodyComID',
                    'gages'     : 'gages',
                    'tw'        : 'TopWdth',
                    'twcc'      : 'TopWdthCC',
                    'alt'       : 'alt',
                    'musk'      : 'MusK',
                    'musx'      : 'MusX',
                    'cs'        : 'ChSlp',
                    }
        else:
            default_columns = columns
        return default_columns


class Columns(BaseModel):
    key: str 
    """
    unique segment identifier
    """
    downstream: str 
    """
    unique identifier of downstream segment
    """
    dx: str 
    """
    segment length
    """
    n: str 
    """
    manning's roughness of main channel
    """
    ncc: str 
    """
    mannings roughness of compound channel
    """
    s0: str 
    """
    channel slope
    """
    bw: str 
    """
    channel bottom width
    """
    waterbody: Optional[str] 
    """
    waterbody identifier
    """
    tw: str 
    """
    channel top width
    """
    twcc: str 
    """
    compound channel top width
    """
    alt: Optional[str] 
    """
    channel bottom altitude
    """
    musk: str 
    """
    muskingum K parameter
    """ 
    musx: str 
    """
    muskingum X parameter
    """
    cs: str 
    """
    channel sideslope
    """
    gages: Optional[str]
    """
    gage ID
    """
    mainstem: Optional[str]
    """
    mainstem ID
    """


class WaterbodyParameters(BaseModel):
    """
    Parameters specifying how (if) waterbodies are handled.
    """
    break_network_at_waterbodies: bool = False
    """
    If True, waterbodies will be treated as reservoirs. If False, the underlying flowpaths will be used for channel routing.
    """
    level_pool: Optional["LevelPool"] = None
    waterbody_null_code: int = -9999
    """
    NULL value to use in flowpath-waterbody crosswalk.
    """


class LevelPool(BaseModel):
    """
    Attributes of the lake geometry file for levelpool simulations.
    """
    level_pool_waterbody_parameter_file_path: Optional[FilePath] = None
    """
    Filepath for NetCDF file containing lake parameters (LAKEPARM). Only used for NHD networks.
    """
    level_pool_waterbody_id: Union[str, Literal["lake_id"]] = "lake_id"
    """
    Column name for waterbody ID.
    """


NetworkTopologyParameters.update_forward_refs()
PreprocessingParameters.update_forward_refs()
SupernetworkParameters.update_forward_refs()
WaterbodyParameters.update_forward_refs()
LevelPool.update_forward_refs()

