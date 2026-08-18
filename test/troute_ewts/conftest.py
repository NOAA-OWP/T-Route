import logging
import pytest


@pytest.fixture
def clean_ewts_env(monkeypatch):
    """
    Ensure EWTS-related environment variables are unset and
    logging is reset before each test.
    """
    # EWTS / module env vars
    monkeypatch.delenv("NGEN_LOG_FILE_PATH", raising=False)
    monkeypatch.delenv("TROUTE_LOGLEVEL", raising=False)
    monkeypatch.delenv("TROUTE_LOGFILEPATH", raising=False)
    monkeypatch.delenv("NGEN_EWTS_LOGGING", raising=False)

    # Reset logging state (important!)
    logging.shutdown()
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    yield

    # Cleanup after test (defensive)
    logging.shutdown()
