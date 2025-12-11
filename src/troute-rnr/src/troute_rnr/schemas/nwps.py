"""Schema definitions for NWPS data models."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class RFC(BaseModel):
    """
    River Forecast Center information.

    Parameters
    ----------
    abbreviation : str
        The abbreviated name of the RFC.
    name : str
        The full name of the RFC.
    """

    abbreviation: str = Field(description="The abbreviated name of the RFC")
    name: str = Field(description="The full name of the RFC")


class WFO(BaseModel):
    """
    Weather Forecast Office information.

    Parameters
    ----------
    abbreviation : str
        The abbreviated name of the WFO.
    name : str
        The full name of the WFO.
    """

    abbreviation: str = Field(description="The abbreviated name of the WFO")
    name: str = Field(description="The full name of the WFO")


class State(BaseModel):
    """
    State information.

    Parameters
    ----------
    abbreviation : str
        The state's two-letter abbreviation.
    name : str
        The full name of the state.
    """

    abbreviation: str = Field(description="The state's two-letter abbreviation")
    name: str = Field(description="The full name of the state")


class PEDTS(BaseModel):
    """
    Physical Element Data Type and Source information.

    Parameters
    ----------
    observed : str
        The observed PEDTS.
    forecast : str
        The forecast PEDTS.
    """

    observed: str = Field(description="The observed PEDTS")
    forecast: str = Field(description="The forecast PEDTS")


class StatusData(BaseModel):
    """
    Status data for a gauge reading.

    Parameters
    ----------
    primary : float
        The primary measurement value.
    primaryUnit : str
        The unit of the primary measurement.
    secondary : float
        The secondary measurement value.
    secondaryUnit : str
        The unit of the secondary measurement.
    floodCategory : str
        The current flood category.
    validTime : datetime
        The timestamp when this status was recorded.
    """

    primary: float = Field(description="The primary measurement value")
    primaryUnit: str = Field(description="The unit of the primary measurement")
    secondary: float = Field(description="The secondary measurement value")
    secondaryUnit: str = Field(description="The unit of the secondary measurement")
    floodCategory: str = Field(description="The current flood category")
    validTime: datetime = Field(description="The timestamp when this status was recorded")


class Status(BaseModel):
    """
    Overall status including observed and forecast data.

    Parameters
    ----------
    observed : StatusData
        The observed status data.
    forecast : StatusData
        The forecast status data.
    """

    observed: StatusData = Field(description="The observed status data")
    forecast: StatusData = Field(description="The forecast status data")


class FloodCategory(BaseModel):
    """
    Flood category thresholds.

    Parameters
    ----------
    stage : float
        The stage (water level) threshold for this category.
    flow : float
        The flow rate threshold for this category.
    """

    stage: float = Field(description="The stage (water level) threshold for this category")
    flow: float = Field(description="The flow rate threshold for this category")


class FloodCategories(BaseModel):
    """
    Thresholds for different flood categories.

    Parameters
    ----------
    major : FloodCategory
        Thresholds for major flooding.
    moderate : FloodCategory
        Thresholds for moderate flooding.
    minor : FloodCategory
        Thresholds for minor flooding.
    action : FloodCategory
        Thresholds for flood action stage.
    """

    major: FloodCategory = Field(description="Thresholds for major flooding")
    moderate: FloodCategory = Field(description="Thresholds for moderate flooding")
    minor: FloodCategory = Field(description="Thresholds for minor flooding")
    action: FloodCategory = Field(description="Thresholds for flood action stage")


class LRO(BaseModel):
    """
    Long Range Outlook information.

    Parameters
    ----------
    minorCS : str
        Minor flood chance statement.
    moderateCS : str
        Moderate flood chance statement.
    majorCS : str
        Major flood chance statement.
    producedTime : datetime
        Time when the outlook was produced.
    interval : str
        Interval of the outlook.
    """

    minorCS: str = Field(description="Minor flood chance statement")
    moderateCS: str = Field(description="Moderate flood chance statement")
    majorCS: str = Field(description="Major flood chance statement")
    producedTime: datetime = Field(description="Time when the outlook was produced")
    interval: str = Field(description="Interval of the outlook")


class Crest(BaseModel):
    """
    Information about a flood crest.

    Parameters
    ----------
    occurredTime : datetime
        Time when the crest occurred.
    stage : float
        Water stage at crest.
    flow : float
        Flow rate at crest.
    preliminary : str
        Indicator if this is a preliminary crest.
    olddatum : bool
        Indicator if this uses an old datum.
    """

    occurredTime: datetime = Field(description="Time when the crest occurred")
    stage: float = Field(description="Water stage at crest")
    flow: float = Field(description="Flow rate at crest")
    preliminary: str = Field(description="Indicator if this is a preliminary crest")
    olddatum: bool = Field(description="Indicator if this uses an old datum")


class LowWater(BaseModel):
    """
    Information about low water conditions.

    Parameters
    ----------
    occurredTime : datetime
        Time when the low water condition occurred.
    stage : float
        Water stage at low water.
    flow : float
        Flow rate at low water.
    statement : str
        Statement about the low water condition.
    """

    occurredTime: datetime = Field(description="Time when the low water condition occurred")
    stage: float = Field(description="Water stage at low water")
    flow: float = Field(description="Flow rate at low water")
    statement: str = Field(description="Statement about the low water condition")


class Impact(BaseModel):
    """
    Information about flood impacts at a certain stage.

    Parameters
    ----------
    stage : float
        Water stage at which this impact occurs.
    statement : str
        Description of the impact.
    """

    stage: float = Field(description="Water stage at which this impact occurs")
    statement: str = Field(description="Description of the impact")


class Flood(BaseModel):
    """
    Comprehensive flood information.

    Parameters
    ----------
    stageUnits : str
        Units used for stage measurements.
    flowUnits : str
        Units used for flow measurements.
    categories : FloodCategories
        Thresholds for different flood categories.
    lro : LRO | None
        Long Range Outlook information, if available.
    crests : dict[str, list[Crest]]
        Historical and forecasted flood crests.
    lowWaters : dict[str, list[LowWater]]
        Historical and forecasted low water conditions.
    impacts : list[Impact]
        List of flood impacts at different stages.
    """

    stageUnits: str = Field(description="Units used for stage measurements")
    flowUnits: str = Field(description="Units used for flow measurements")
    categories: FloodCategories = Field(description="Thresholds for different flood categories")
    lro: Optional[LRO] = Field(None, description="Long Range Outlook information, if available")
    crests: dict[str, list[Crest]] = Field(description="Historical and forecasted flood crests")
    lowWaters: dict[str, list[LowWater]] = Field(description="Historical and forecasted low water conditions")
    impacts: list[Impact] = Field(description="List of flood impacts at different stages")


class ProbabilityImages(BaseModel):
    """
    Links to probability images.

    Parameters
    ----------
    stage : str
        Link to stage probability image.
    flow : str
        Link to flow probability image.
    volume : str
        Link to volume probability image.
    """

    stage: str = Field(description="Link to stage probability image")
    flow: str = Field(description="Link to flow probability image")
    volume: str = Field(description="Link to volume probability image")


class Probability(BaseModel):
    """
    Probability information for different time ranges.

    Parameters
    ----------
    weekint : ProbabilityImages
        Week interval probability images.
    entperiod : ProbabilityImages
        Entire period probability images.
    shortrange : str
        Link to short range probability image.
    """

    weekint: ProbabilityImages = Field(description="Week interval probability images")
    entperiod: ProbabilityImages = Field(description="Entire period probability images")
    shortrange: str = Field(description="Link to short range probability image")


class Hydrograph(BaseModel):
    """
    Links to hydrograph images.

    Parameters
    ----------
    default : str
        Link to default hydrograph image.
    floodcat : str
        Link to flood category hydrograph image.
    """

    default: str = Field(description="Link to default hydrograph image")
    floodcat: str = Field(description="Link to flood category hydrograph image")


class PhotoGeometry(BaseModel):
    """
    Geometry information for a photo.

    Parameters
    ----------
    type : str
        Type of geometry (e.g., "Point").
    coordinates : list[float]
        Coordinates of the photo location.
    """

    type: str = Field(description='Type of geometry (e.g., "Point")')
    coordinates: list[float] = Field(description="Coordinates of the photo location")


class PhotoProperties(BaseModel):
    """
    Properties of a photo.

    Parameters
    ----------
    image : str
        Link to the image file.
    caption : str
        Caption for the photo.
    """

    image: str = Field(description="Link to the image file")
    caption: str = Field(description="Caption for the photo")


class Photo(BaseModel):
    """
    Information about a photo.

    Parameters
    ----------
    id : str
        Unique identifier for the photo.
    type : str
        Type of the photo data.
    geometry : PhotoGeometry
        Geometry information for the photo.
    properties : PhotoProperties
        Properties of the photo.
    """

    id: str = Field(description="Unique identifier for the photo")
    type: str = Field(description="Type of the photo data")
    geometry: PhotoGeometry = Field(description="Geometry information for the photo")
    properties: PhotoProperties = Field(description="Properties of the photo")


class Images(BaseModel):
    """
    Collection of various images related to the gauge.

    Parameters
    ----------
    probability : Probability
        Probability images for different time ranges.
    hydrograph : Hydrograph
        Hydrograph images.
    photos : list[Photo]
        List of photos related to the gauge location.
    """

    probability: Probability = Field(description="Probability images for different time ranges")
    hydrograph: Hydrograph = Field(description="Hydrograph images")
    photos: list[Photo] = Field(description="List of photos related to the gauge location")


class DataAttribution(BaseModel):
    """
    Attribution information for data sources.

    Parameters
    ----------
    abbrev : str
        Abbreviation of the data source.
    text : str
        Full text of the attribution.
    title : str
        Title of the data source.
    url : str
        URL for more information about the data source.
    """

    abbrev: str = Field(description="Abbreviation of the data source")
    text: str = Field(description="Full text of the attribution")
    title: str = Field(description="Title of the data source")
    url: str = Field(description="URL for more information about the data source")


class ImpactLowWater(BaseModel):
    """
    Information about low water impacts.

    Parameters
    ----------
    value : str
        Value at which the low water impact occurs.
    impact : str
        Description of the low water impact.
    """

    value: str = Field(description="Value at which the low water impact occurs")
    impact: str = Field(description="Description of the low water impact")


class NormalThreshold(BaseModel):
    """
    Normal water level threshold information.

    Parameters
    ----------
    value : float
        The value of the normal threshold.
    units : str
        Units of measurement for the threshold.
    """

    value: float = Field(description="The value of the normal threshold")
    units: str = Field(description="Units of measurement for the threshold")


class Hydronote(BaseModel):
    """
    Hydrologic note information.

    Parameters
    ----------
    statement : str
        The content of the hydrologic note.
    effective : str
        The time when the note becomes effective.
    expiration : str
        The time when the note expires.
    """

    statement: str = Field(description="The content of the hydrologic note")
    effective: str = Field(description="The time when the note becomes effective")
    expiration: str = Field(description="The time when the note expires")


class DatumValue(BaseModel):
    """
    Information about a specific datum value.

    Parameters
    ----------
    label : str
        Label for the datum value.
    abbrev : str
        Abbreviation for the datum value.
    description : str
        Description of the datum value.
    value : float
        The numerical value of the datum.
    """

    label: str = Field(description="Label for the datum value")
    abbrev: str = Field(description="Abbreviation for the datum value")
    description: str = Field(description="Description of the datum value")
    value: float = Field(description="The numerical value of the datum")


class Datums(BaseModel):
    """
    Collection of datum information.

    Parameters
    ----------
    vertical : dict[str, list[DatumValue]]
        Vertical datum information.
    horizontal : dict[str, list[DatumValue]]
        Horizontal datum information.
    notes : dict[str, list[str]]
        Additional notes about the datums.
    """

    vertical: dict[str, list[DatumValue]] = Field(description="Vertical datum information")
    horizontal: dict[str, list[DatumValue]] = Field(description="Horizontal datum information")
    notes: dict[str, list[str]] = Field(description="Additional notes about the datums")


class ZeroDatum(BaseModel):
    """
    Information about the zero datum.

    Parameters
    ----------
    value : float
        The value of the zero datum.
    datum : str
        The type or name of the datum.
    """

    value: float = Field(description="The value of the zero datum")
    datum: str = Field(description="The type or name of the datum")


class Downloads(BaseModel):
    """
    Links to downloadable data.

    Parameters
    ----------
    depthGrids : str
        Link to depth grids data.
    images : str
        Link to image data.
    kmz : str
        Link to KMZ file.
    """

    depthGrids: str = Field(description="Link to depth grids data")
    images: str = Field(description="Link to image data")
    kmz: str = Field(description="Link to KMZ file")


class InundationDataAttribution(BaseModel):
    """
    Attribution information for inundation data.

    Parameters
    ----------
    text : str
        Attribution text.
    title : str
        Title of the data source.
    url : str
        URL for more information.
    image : str
        Link to an image related to the attribution.
    """

    text: str = Field(description="Attribution text")
    title: str = Field(description="Title of the data source")
    url: str = Field(description="URL for more information")
    image: str = Field(description="Link to an image related to the attribution")


class Inundation(BaseModel):
    """
    Information about inundation data and services.

    Parameters
    ----------
    enabled : bool
        Whether inundation data is enabled.
    url : str
        URL for inundation data.
    zeroDatum : ZeroDatum | None
        Zero datum information, if available.
    downloads : Downloads | None
        Links to downloadable data, if available.
    siteSpecificInfo : str
        Site-specific inundation information.
    dataAttribution : list[InundationDataAttribution]
        List of data attributions for inundation data.
    """

    enabled: bool = Field(description="Whether inundation data is enabled")
    url: str = Field(description="URL for inundation data")
    zeroDatum: Optional[ZeroDatum] = Field(None, description="Zero datum information, if available")
    downloads: Optional[Downloads] = Field(None, description="Links to downloadable data, if available")
    siteSpecificInfo: str = Field(description="Site-specific inundation information")
    dataAttribution: list[InundationDataAttribution] = Field(
        description="List of data attributions for inundation data"
    )


class InService(BaseModel):
    """
    Information about the service status of the gauge.

    Parameters
    ----------
    enabled : bool
        Whether the gauge is in service.
    message : str
        Any message related to the service status.
    """

    enabled: bool = Field(description="Whether the gauge is in service")
    message: str = Field(description="Any message related to the service status")


class LowThreshold(BaseModel):
    """
    Information about the low water threshold.

    Parameters
    ----------
    units : str
        Units of measurement for the threshold.
    value : float
        The value of the low threshold.
    """

    units: str = Field(description="Units of measurement for the threshold")
    value: float = Field(description="The value of the low threshold")


class SiteData(BaseModel):
    """
    Comprehensive data about a gauge.

    Parameters
    ----------
    lid : str
        Location ID of the gauge.
    usgsId : str
        USGS ID of the gauge.
    reachId : str
        Reach ID associated with the gauge.
    name : str
        Name of the gauge location.
    description : str
        Description of the gauge location.
    rfc : RFC
        River Forecast Center information.
    wfo : WFO
        Weather Forecast Office information.
    state : State
        State information.
    county : str
        County where the gauge is located.
    timeZone : str
        Time zone of the gauge location.
    latitude : float
        Latitude of the gauge location.
    longitude : float
        Longitude of the gauge location.
    pedts : PEDTS
        Physical Element Data Type and Source information.
    status : Status
        Current status of the gauge.
    flood : Flood
        Flood-related information.
    images : Images
        Collection of related images.
    dataAttribution : list[DataAttribution]
        List of data attributions.
    impactsLowWaters : list[ImpactLowWater]
        List of low water impacts.
    normalThreshold : NormalThreshold | None
        Normal water level threshold, if available.
    hydronotes : list[Hydronote]
        List of hydrologic notes.
    datums : Datums
        Datum information.
    inundation : Inundation
        Inundation data and services information.
    upstreamLid : str
        Location ID of the upstream gauge.
    downstreamLid : str
        Location ID of the downstream gauge.
    inService : InService
        Service status information.
    lowThreshold : LowThreshold | None
        Low water threshold information, if available.
    forecastReliability : str
        Information about the reliability of forecasts.
    TruncateObs : str
        Information about truncation of observations.
    TruncateFcst : str
        Information about truncation of forecasts.
    ObservedFloodCategory : str
        Observed flood category.
    ForecastFloodCategory : str
        Forecast flood category.
    """

    lid: str = Field(description="Location ID of the gauge")
    usgsId: str = Field(description="USGS ID of the gauge")
    reachId: str = Field(description="Reach ID associated with the gauge")
    name: str = Field(description="Name of the gauge location")
    description: str = Field(description="Description of the gauge location")
    rfc: RFC = Field(description="River Forecast Center information")
    wfo: WFO = Field(description="Weather Forecast Office information")
    state: State = Field(description="State information")
    county: str = Field(description="County where the gauge is located")
    timeZone: str = Field(description="Time zone of the gauge location")
    latitude: float = Field(description="Latitude of the gauge location")
    longitude: float = Field(description="Longitude of the gauge location")
    pedts: PEDTS = Field(description="Physical Element Data Type and Source information")
    status: Status = Field(description="Current status of the gauge")
    flood: Flood = Field(description="Flood-related information")
    images: Images = Field(description="Collection of related images")
    dataAttribution: list[DataAttribution] = Field(description="List of data attributions")
    impactsLowWaters: list[ImpactLowWater] = Field(description="List of low water impacts")
    normalThreshold: Optional[NormalThreshold] = Field(
        None, description="Normal water level threshold, if available"
    )
    hydronotes: list[Hydronote] = Field(description="List of hydrologic notes")
    datums: Datums = Field(description="Datum information")
    inundation: Inundation = Field(description="Inundation data and services information")
    upstreamLid: str = Field(description="Location ID of the upstream gauge")
    downstreamLid: str = Field(description="Location ID of the downstream gauge")
    inService: InService = Field(description="Service status information")
    lowThreshold: Optional[LowThreshold] = Field(
        None, description="Low water threshold information, if available"
    )
    forecastReliability: str = Field(description="Information about the reliability of forecasts")
    TruncateObs: str = Field(description="Information about truncation of observations")
    TruncateFcst: str = Field(description="Information about truncation of forecasts")
    ObservedFloodCategory: str = Field(description="Observed flood category")
    ForecastFloodCategory: str = Field(description="Forecast flood category")


class GaugeForecast(BaseModel):
    """
    Forecast data for a gauge.

    Parameters
    ----------
    times : list[datetime]
        List of forecast times.
    primary_name : str
        Name of the primary forecast parameter.
    primary_forecast : list[float]
        List of primary forecast values.
    primary_unit : str
        Unit of measurement for primary forecast.
    latest_observation: list[float]
        The latest observation from NWPS.
    latest_obs_units: str
        The latest observation units.
    secondary_name : str
        Name of the secondary forecast parameter.
    secondary_forecast : list[float]
        List of secondary forecast values.
    secondary_unit : str
        Unit of measurement for secondary forecast.
    """

    times: list[datetime] = Field(description="List of forecast times")
    primary_name: str = Field(description="Name of the primary forecast parameter")
    primary_forecast: list[float] = Field(description="List of primary forecast values")
    primary_unit: str = Field(description="Unit of measurement for primary forecast")
    latest_observation: list[float] = Field(description="The latest observation from NWPS")
    latest_obs_units: str = Field(description="The latest observation units")
    secondary_name: str = Field(description="Name of the secondary forecast parameter")
    secondary_forecast: list[float] = Field(description="List of secondary forecast values")
    secondary_unit: str = Field(description="Unit of measurement for secondary forecast")


class ReachClassification(str, Enum):
    """
    Classification types for reaches.

    Attributes
    ----------
    rfc_point : str
        RFC point classification.
    flowline : str
        Flowline classification.
    """

    rfc_point = Field(description="RFC point classification", default="rfc_point")
    flowline = Field(description="Flowline classification", default="flowline")


class Reach(BaseModel):
    """
    Information about a river reach.

    Attributes
    ----------
    id : int
        Identifier for the reach.
    times : List[datetime]
        List of forecast times.
    primary_name : str
        Name of the primary forecast parameter.
    primary_forecast : List[float]
        List of primary forecast values.
    primary_unit : str
        Unit of measurement for primary forecast.
    latest_obervation: List[float]
        The latest observation from NWPS
    latest_obs_units: str
        The latest observation units
    secondary_name: str
        Name of the secondary forecast parameter
    secondary_forecast : List[float]
        List of secondary forecast values.
    secondary_unit : str
        Unit of measurement for secondary forecast.
    """

    id: int = Field(description="Identifier for the reach")
    times: list[datetime] = Field(description="List of forecast times")
    primary_name: str = Field(description="Name of the primary forecast parameter")
    primary_forecast: list[float] = Field(description="List of primary forecast values")
    primary_unit: str = Field(description="Unit of measurement for primary forecast")
    latest_observation: Optional[float] = Field(description="The latest observation from NWPS", default=None)
    latest_obs_units: Optional[str] = Field(description="The latest observation units", default=None)
    secondary_name: str = Field(description="Name of the secondary forecast parameter")
    secondary_forecast: list[float] = Field(description="List of secondary forecast values")
    secondary_unit: str = Field(description="Unit of measurement for secondary forecast")


class ProcessedData(BaseModel):
    """
    Container for processed data about a location.

    Parameters
    ----------
    lid : str
        Location ID.
    downstream_lid : str
        Downstream location ID.
    reaches : Reach
        The reach we're routing from
    """

    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)
    lid: str = Field(description="Location ID")
    downstream_lid: str = Field(description="Downstream location ID")
    reach: Reach = Field(description="The reach we're routing from")


class ResultItem(BaseModel):
    """
    Represents the result of processing a single RFC entry.

    Parameters
    ----------
    status : str
        The status of the processing operation.
        Possible values: 'success', 'no_forecast', 'api_error', 'error'.
    lid : str
        The location ID (LID) of the processed RFC entry.
    error_type : str | None, optional
        The exception/error that was raised.
    error_message : str | None, optional
        The error message that was raised.
    status_code : str | None, optional
        The status code of the exception if applicable.
    """

    status: str = Field(
        description="The status of the processing operation. Possible values: 'success', 'no_forecast', 'api_error', 'error'"
    )
    lid: str = Field(description="The location ID (LID) of the processed RFC entry")
    error_type: Optional[str] = Field(description="The exception/error that was raised", default=None)
    error_message: Optional[str] = Field(description="The error message that was raised", default=None)
    status_code: Optional[str] = Field(
        description="The status code of the exception if applicable", default=None
    )


class Summary(BaseModel):
    """
    Summarizes the results of processing multiple RFC entries.

    Parameters
    ----------
    total : int
        The total number of RFC entries processed.
    success : int
        The number of entries successfully processed.
    no_forecast : int
        The number of entries that had no forecast available.
    api_error : int
        The number of entries that encountered an API error.
    validation_error : int
        The number of entries that encountered a validation error.
    """

    total: int = Field(description="The total number of RFC entries processed")
    success: int = Field(description="The number of entries successfully processed")
    no_forecast: int = Field(description="The number of entries that had no forecast available")
    api_error: int = Field(description="The number of entries that encountered an API error")
    validation_error: int = Field(description="The number of entries that encountered a validation error")


class PublishMessagesResponse(BaseModel):
    """
    Represents the full response of the publish_messages endpoint.

    Parameters
    ----------
    status : int
        The HTTP status code of the response.
    summary : Summary
        A summary of the processing results.
    results : list[ResultItem]
        Detailed results for each processed RFC entry.
    """

    status: int = Field(description="The HTTP status code of the response")
    summary: Summary = Field(description="A summary of the processing results")
    results: list[ResultItem] = Field(description="Detailed results for each processed RFC entry")


class ConsumerStatus(BaseModel):
    """
    Status of a consumer service.

    Parameters
    ----------
    is_running : bool, default=False
        Whether the consumer is currently running.
    """

    is_running: bool = Field(description="Whether the consumer is currently running", default=False)
