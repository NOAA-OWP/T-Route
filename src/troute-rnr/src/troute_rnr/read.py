"""Module for handling NWM data processing and NWPS integrations."""

from datetime import datetime

import httpx
from pydantic import ValidationError

from troute_rnr.schemas.nwps import ProcessedData, Reach, SiteData
from troute_rnr.schemas.weather import Site
from troute_rnr.settings import Settings
from troute_rnr.utils import get


def convert_to_m3_per_sec(forecast: list[float], unit: str) -> tuple[list[float], str]:
    """Convert forecast units to m3/s.

    Parameters
    ----------
    forecast: List[float]
    - The list of forecasts to convert

    unit: str
    - The units of the forecast

    Returns
    -------
    Tuple[List[float], str]:
    - The forecast, and the units str
    """
    if unit == "kcfs":
        forecast = [flow * 1000 * 0.028316846592 for flow in forecast]
        return forecast, "m3 s-1"
    else:
        raise ValueError(f"Unit conversion not supported for {unit}")


def read_site_data(site: Site, settings: Settings) -> SiteData | None:
    """Retrieves gauge data from the NWPS API for a specific site and validates it meets flood criteria.

    Parameters
    ----------
    site : Site
        The site object containing properties with an 'id' field used to construct the API endpoint
    settings : Settings
        Configuration object containing BASE_URL for the API endpoint and STAGES for flood criteria validation

    Returns
    -------
    SiteData | None
        A validated SiteData object if the site data meets flood criteria requirements,
        None if the site's flood category does not meet the required criteria

    Raises
    ------
    ValidationError
        If the API response cannot be parsed into a valid SiteData object
    httpx.HTTPStatusError
        If the API request fails or returns an error status
    """
    endpoint = f"{settings.BASE_URL}/gauges/{site.properties['id']}"
    try:
        response = get(endpoint)
        forecast = response.json()
        if response.status_code == 404:
            print(f"Gauge not found in NWPS API: {forecast['message']}")
            return None
        try:
            site_data = SiteData(**forecast)
        except ValidationError as e:
            msg = f"ValidationError: Pydantic validation error for the endpoint given: {endpoint}"
            print(msg)
            raise e
        if site_data.ForecastFloodCategory in settings.STAGES:
            return site_data
        else:
            msg = f"{site_data.lid} is not reporting flooding for this forecast. Ignoring routing for RnR"
            print(msg)
            return None
    except httpx.HTTPStatusError as e:
        msg = f"HTTPStatusError: There was no forecast/record at the NWPS API  @ {endpoint}"
        print(msg)
        raise httpx.HTTPStatusError(msg) from e


def read_rfc_flows(forecast: SiteData, settings: Settings) -> ProcessedData | None:
    """
    Pull National Water Model inputs for a given forecast.

    Parameters
    ----------
    forecast : SiteData
        Gauge data containing forecast information.
    settings : Settings
        Configuration settings.

    Returns
    -------
    ProcessedData | None
        Processed data containing reach information or None if data is invalid.
    """
    forecast_endpoint = f"{settings.BASE_URL}/gauges/{forecast.lid}/stageflow/forecast"
    obs_endpoint = f"{settings.BASE_URL}/gauges/{forecast.lid}/stageflow/observation"
    forecast_data = get(forecast_endpoint).json()
    obs_data = get(obs_endpoint).json()
    if forecast_data["data"][0]["secondary"] == -999:
        print(f"Streamflow forecast does not exist. Skipping Forecast for {forecast.lid}")
        return None

    try:
        latest_observation_units = obs_data["secondaryUnits"]
        latest_observation_flow = [obs_data["data"][-1]["secondary"]]

        latest_observation_m3, latest_obs_units = convert_to_m3_per_sec(
            latest_observation_flow, latest_observation_units
        )
    except KeyError:
        # Skiping Obs read. None found
        latest_obs_units = None
        latest_observation_m3 = None
    except ValueError:
        print(f"Streamflow observation returning stage values. Skipping Forecast for {forecast.lid}")
        # Can't route the flows since the units are in ft and not kcfs. Passing
        return None

    times = [datetime.fromisoformat(entry["validTime"].rstrip("Z")) for entry in forecast_data["data"]]
    primary_forecast = [entry["primary"] for entry in forecast_data["data"]]
    secondary_forecast = [entry["secondary"] for entry in forecast_data["data"]]

    if len(secondary_forecast) == 0:
        print(f"No streamflow forecast found. Skipping Forecast for {forecast.lid}")
        return None

    try:
        secondary_m3_forecast, secondary_units = convert_to_m3_per_sec(
            secondary_forecast, forecast_data["secondaryUnits"]
        )
    except ValueError:
        print(f"Streamflow forecast returning stage values. Skipping Forecast for {forecast.lid}")
        # Can't route the flows since the units are in ft and not kcfs. Passing
        return None

    try:
        processed_data = ProcessedData(
            lid=forecast.lid,
            downstream_lid=forecast.downstreamLid,
            reach=Reach(
                id=forecast.reachId,
                times=times,
                primary_name=forecast_data["primaryName"],
                primary_forecast=primary_forecast,
                primary_unit=forecast_data["primaryUnits"],
                latest_observation=latest_observation_m3,
                latest_obs_units=latest_obs_units,
                secondary_name=forecast_data["secondaryName"],
                secondary_forecast=secondary_m3_forecast,
                secondary_unit=secondary_units,
            ),
        )
    except ValidationError as e:
        if forecast_data["wfo"] in settings.unsupported_wfo:
            print(
                f"{forecast.lid} is located in an unsupported RFC Domain {forecast_data['wfo']}. Skipping RnR run"
            )
            return None
        elif forecast.reachId == "":
            print(
                f"No reach ID found for: {forecast.lid}. Skipping RnR run since we do not have the location of the flowline."
            )
            return None
        else:
            print(f"Validation Error for {forecast.lid}. Please check logs for error. Skipping RnR run")
            print(e)
            return None

    return processed_data
