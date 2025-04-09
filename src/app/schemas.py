from enum import IntEnum

from pydantic import BaseModel


class HttpStatusCode(IntEnum):
    OK = 200
    CREATED = 201
    ACCEPTED = 202
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    INTERNAL_SERVER_ERROR = 500


class TRouteStatus(BaseModel):
    """A schema to define successful t-route output

    Attributes:
    -----------
    status_code: HttpStatusCode
        The HTTP status code output from the code run
    message: str
        The output message from T-Route
    lid : str
        The location ID belonging to the point being routed
    feature_id : str
        The COMID, or hf_id, belonging to the point being routed
    """
    status_code: HttpStatusCode
    message: str
    lid: str
    feature_id: str


class TestStatus(BaseModel):
    """A schema for output from t-route test cases
    
    Attributes:
    -----------
    status_code: HttpStatusCode
        The HTTP status code output from the code run
    message: str
        The output message from T-Route
    """
    status_code: HttpStatusCode
    message: str
