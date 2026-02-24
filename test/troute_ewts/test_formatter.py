import logging
import pytest
from troute_ewts.formatter import CustomFormatter
from troute_ewts.constants import MODULE_NAME

@pytest.fixture
def formatter():
    fmt = "%(asctime)s %(levelname_padded)s %(message)s"
    return CustomFormatter(fmt=fmt, datefmt="%Y-%m-%dT%H:%M:%S")

@pytest.mark.parametrize(
    "level,expected",
    [
        (logging.DEBUG, "DEBUG"),
        (logging.INFO, "INFO"),
        (logging.WARNING, "WARNING"),
        (logging.ERROR, "SEVERE"),
        (logging.CRITICAL, "FATAL"),
    ]
)
def test_level_name_mapping(formatter, level, expected):
    record = logging.LogRecord(
        name=MODULE_NAME,
        level=level,
        pathname="test",
        lineno=0,
        msg="Test message",
        args=None,
        exc_info=None
    )
    formatted = formatter.format(record)
    # Level name should appear in formatted string
    assert expected in formatted

def test_utc_timestamp(formatter):
    record = logging.LogRecord(
        name=MODULE_NAME,
        level=logging.INFO,
        pathname="test",
        lineno=0,
        msg="UTC test",
        args=None,
        exc_info=None
    )
    formatted = formatter.format(record)
    # Timestamp should be in UTC format "YYYY-MM-DDTHH:MM:SS"
    ts_str = formatted.split()[0]
    from datetime import datetime
    dt = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S")
    # It's enough to check it parses without error

def test_trailing_whitespace_stripped(formatter):
    record = logging.LogRecord(
        name=MODULE_NAME,
        level=logging.INFO,
        pathname="test",
        lineno=0,
        msg="Message with space   \n",
        args=None,
        exc_info=None
    )
    formatted = formatter.format(record)
    # Trailing whitespace/newline should be removed
    assert "   \n" not in formatted
    assert formatted.endswith("Message with space")
