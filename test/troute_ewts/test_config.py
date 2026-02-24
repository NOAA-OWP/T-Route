import pytest

import logging
from troute_ewts.config import configure_logging, translate_ngwpc_log_level
from troute_ewts.constants import MODULE_NAME, EV_EWTS_LOGGING

# ------------------------------
def test_configure_logging_default(clean_ewts_env):
    logger = configure_logging()

    assert logger.name == MODULE_NAME
    assert logger.level == logging.INFO
    assert not logger.disabled

# ------------------------------
def test_configure_logging_idempotent(clean_ewts_env):
    logger1 = configure_logging()
    logger2 = configure_logging()

    assert logger1 is logger2
    assert getattr(logger1, "_initialized", False)

# ------------------------------
@pytest.mark.parametrize("inp,expected", [
    ("INFO", "INFO"),
    ("SeVeRe", "ERROR"),
    ("fatal", "CRITICAL"),
    (" debug ", "DEBUG"),
])
def test_translate_ngwpc_log_level(inp, expected):
    assert translate_ngwpc_log_level(inp) == expected

# ------------------------------
@pytest.mark.parametrize("env_value,expected_enabled", [
    (None, True),          # default: enabled
    ("DISABLED", False),
    ("ENABLED", True),
    ("disabled", False),
    ("enabled", True),
    ("anystring", True),
    ("", True),
])
@pytest.mark.parametrize("level_input,expected_level", [
    ("DEBUG", logging.DEBUG),
    ("INFO", logging.INFO),
    ("SEVERE", logging.ERROR),
    ("FATAL", logging.CRITICAL),
])
def test_ewts_logger_matrix(clean_ewts_env, monkeypatch, capsys, env_value, expected_enabled, level_input, expected_level):
    # Set environment variables
    if env_value is None:
        monkeypatch.delenv("NGEN_EWTS_LOGGING", raising=False)
    else:
        monkeypatch.setenv("NGEN_EWTS_LOGGING", env_value)

    monkeypatch.setenv("TROUTE_LOGLEVEL", level_input)

    # Force logger re-initialization
    logger = logging.getLogger(MODULE_NAME)
    logger.handlers.clear()
    logger._initialized = False
    logger.disabled = False  # ensure proper reset

    # Configure logger
    logger = configure_logging()

    # Capture stdout
    captured = capsys.readouterr()

    # Assertions
    assert logger.name == MODULE_NAME
    assert (not logger.disabled) == expected_enabled  # True if enabled
    if expected_enabled:
        assert logger.level == expected_level

    # Assertions for default-enabled print
    if expected_enabled and (env_value is None or env_value not in ("ENABLED", "enabled")):
        assert f"{EV_EWTS_LOGGING} not explicitly set" in captured.out
    else:
        assert f"{EV_EWTS_LOGGING} not explicitly set" not in captured.out

