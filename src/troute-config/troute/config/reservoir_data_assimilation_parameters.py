from pydantic import BaseModel, DirectoryPath, FilePath

class ReservoirDataAssimilationParameters(BaseModel):
    level_pool: "LevelPool"
    rfc: "Rfc"
    persistence: "Persistence"


class LevelPool(BaseModel):
    level_pool_waterbody_parameter_file_path: FilePath
    reservoir_parameter_file: DirectoryPath


class Rfc(BaseModel):
    reservoir_rfc_timeseries_folder: DirectoryPath  # "./../test/BMI/rfc_timeseries/"
    reservoir_rfc_gage_id: str  # "KNFC1"
    reservoir_rfc_timeseries_offset_hours: int  # 28
    reservoir_rfc_forecast_persist_days: int  # 11


class Persistence(BaseModel):
    reservoir_persistence_usgs: bool  # False
    reservoir_persistence_usace: bool  # False
    gage_lakeID_crosswalk_file: FilePath  # domain/reservoir_index_AnA.nc
    usgs_timeslices_folder: DirectoryPath  # usgs_TimeSlice/
    usace_timeslices_folder: DirectoryPath  # usace_TimeSlice/
    timeslice_lookback_hours: int  # 48
    qc_threshold: int  # 1


ReservoirDataAssimilationParameters.model_rebuild()
