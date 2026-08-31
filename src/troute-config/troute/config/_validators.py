from datetime import datetime
from typing import Tuple, TypeVar, Union


T = TypeVar("T")
"""Some generic type T"""

DATETIME_FORMATS: Tuple[str, ...] = (
    "%Y-%m-%d_%H:%M",
    "%Y-%m-%d_%H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
    "%Y/%m/%d %H:%M:%S",
)
"""Accepted string datetime formats"""


def coerce_datetime(value: Union[str, datetime]) -> datetime:
    """
    If necessary, coerce a datetime string into a datetime object. Raise a ValueError if coercion
    fails.
    """
    if isinstance(value, datetime):
        return value
    elif isinstance(value, str):
        # TODO: hopefully move to ISO8601 in future
        for fmt in DATETIME_FORMATS:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                pass
    raise ValueError(
        f"datetime field must be specified as `datetime.datetime` object or string with format {DATETIME_FORMATS!r}"
    )
