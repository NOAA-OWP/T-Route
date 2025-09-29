"""XML schema models for weather data processing."""

from collections.abc import Mapping
from typing import Optional

from pydantic import Field
from pydantic_xml import BaseXmlModel, element


class Disclaimers(BaseXmlModel, tag="disclaimers"):
    """
    Disclaimers information from XML data.

    Parameters
    ----------
    AHPSXMLversion : str
        The AHPS XML version.
    status : str
        The status of the data.
    quality : str, optional
        The quality rating of the data.
    standing : str, optional
        The standing of the data.
    """

    AHPSXMLversion: str = element(tag="AHPSXMLversion")
    status: str = element(tag="status")
    quality: Optional[str] = element(tag="quality", default=None)
    standing: Optional[str] = element(tag="standing", default=None)


class Datum(BaseXmlModel, tag="datum"):
    """
    Datum information from XML data.

    Parameters
    ----------
    valid : str
        The validity status of the datum.
    primary : str
        The primary value of the datum.
    secondary : str
        The secondary value of the datum.
    """

    valid: str = element(tag="valid")
    primary: str = element(tag="primary")
    secondary: str = element(tag="secondary")


class Observed(BaseXmlModel, tag="observed"):
    """
    Observed weather data from XML.

    Parameters
    ----------
    properties : Mapping[str, str]
        Properties associated with the observed data.
    datum : list[Datum]
        List of datum objects containing measurement data.
    """

    properties: Mapping[str, str]
    datum: list[Datum] = element(tag="datum")


class Site(BaseXmlModel, tag="site"):
    """
    Site information from XML data.

    Parameters
    ----------
    properties : Mapping[str, str]
        Properties associated with the site.
    disclaimers : Disclaimers
        Disclaimers related to the site data.
    observed : Observed, optional
        Observed weather data at the site, if available.
    """

    properties: Mapping[str, str] = Field(description="Properties associated with the site")
    disclaimers: Disclaimers = Field(description="Disclaimers related to the site data")
    observed: Optional[Observed] = Field(description="Disclaimers related to the site data", default=None)
