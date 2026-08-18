import os
import getpass
from datetime import datetime
import pytest
from troute_ewts import paths
from troute_ewts.paths import create_timestamp, get_log_file_path
from troute_ewts.constants import EV_MODULE_LOGFILEPATH, EV_NGEN_LOGFILEPATH

# -------------------------------
# Fixture for a clean log environment
# -------------------------------
@pytest.fixture
def clean_log_env(tmp_path, monkeypatch):
    """Set up a temporary log environment and clean env vars.

    Yields a dict with:
        tmp_dir : Path of temporary base directory
        monkeypatch : the pytest monkeypatch object for further tweaks
    """
    # Clear env vars
    monkeypatch.delenv(EV_MODULE_LOGFILEPATH, raising=False)
    monkeypatch.delenv(EV_NGEN_LOGFILEPATH, raising=False)

    # Patch constants to use tmp_path
    monkeypatch.setattr(paths, "LOG_DIR_NGENCERF", tmp_path)
    monkeypatch.setattr(paths, "LOG_DIR_DEFAULT", "run-logs")

    yield {"tmp_dir": tmp_path, "monkeypatch": monkeypatch}


# -------------------------------
# Tests for create_timestamp()
# -------------------------------
def test_create_timestamp_default():
    ts = create_timestamp()
    assert len(ts) >= 15
    assert "T" in ts

def test_create_timestamp_date_only():
    ts = create_timestamp(date_only=True)
    assert len(ts) == 8

def test_create_timestamp_iso():
    ts = create_timestamp(iso=True)
    assert "T" in ts and "-" in ts and ":" in ts

def test_create_timestamp_append_ms():
    ts = create_timestamp(append_ms=True)
    assert "." in ts


# -------------------------------
# Tests for get_log_file_path()
# -------------------------------
def test_get_log_file_path_uses_module_env(clean_log_env):
    tmp_path = clean_log_env["tmp_dir"]
    monkeypatch = clean_log_env["monkeypatch"]

    logfile = tmp_path / "test_module.log"
    monkeypatch.setenv(EV_MODULE_LOGFILEPATH, str(logfile))

    path, append = get_log_file_path()
    assert path == str(logfile)
    assert append is True


def test_get_log_file_path_uses_ngen_env(clean_log_env):
    monkeypatch = clean_log_env["monkeypatch"]
    tmp_path = clean_log_env["tmp_dir"]

    monkeypatch.delenv(EV_MODULE_LOGFILEPATH, raising=False)
    ngen_file = tmp_path / "ngen.log"
    monkeypatch.setenv(EV_NGEN_LOGFILEPATH, str(ngen_file))

    path, append = get_log_file_path()
    assert path == str(ngen_file)
    assert append is True


def test_get_log_file_path_creates_user_subdir(clean_log_env):
    tmp_path = clean_log_env["tmp_dir"]
    monkeypatch = clean_log_env["monkeypatch"]

    monkeypatch.delenv(EV_MODULE_LOGFILEPATH, raising=False)
    monkeypatch.delenv(EV_NGEN_LOGFILEPATH, raising=False)

    # Use real username
    monkeypatch.setattr(getpass, "getuser", lambda: "alice")

    path, append = get_log_file_path()

    # Subdirectory should be username
    subdir = os.path.basename(os.path.dirname(path))
    assert subdir == "alice"
    assert path.endswith(".log")
    assert os.path.exists(path)


def test_get_log_file_path_fallback_username(clean_log_env):
    tmp_path = clean_log_env["tmp_dir"]
    monkeypatch = clean_log_env["monkeypatch"]

    monkeypatch.delenv(EV_MODULE_LOGFILEPATH, raising=False)
    monkeypatch.delenv(EV_NGEN_LOGFILEPATH, raising=False)

    # Simulate getuser() returning None
    monkeypatch.setattr(getpass, "getuser", lambda: None)

    path, append = get_log_file_path()

    subdir = os.path.basename(os.path.dirname(path))
    # Should fall back to YYYYMMDD
    assert len(subdir) == 8 and subdir.isdigit()
    assert path.endswith(".log")
    assert os.path.exists(path)
