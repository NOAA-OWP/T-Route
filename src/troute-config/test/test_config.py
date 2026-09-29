from pathlib import Path
from test import temporarily_change_dir
from typing import Any, Dict, Tuple

import pytest
from pydantic import ValidationError
from troute.config import Config


def test_config_validation(config_data: Tuple[Path, Dict[str, Any]]) -> None:
    """Validates all config files contained within the `test/` folder

    Parameters
    ----------
    config_data : Tuple[Path, Dict[str, Any]]
        A tuple containing the path to the config file and the parsed config data
        - The first element is a Path object pointing to the config file
        - The second element is a dictionary containing the parsed config yaml file data.

    Raises
    ------
    pytest.fail
        If a ValidationError occurs during Config creation, this function will
        call pytest.fail with a detailed error message showing the config file that fails

    Notes
    -----
    This test function uses the `temporarily_change_dir` context manager to
    change the working directory before attempting to create the Config object
    """
    path, data = config_data
    with temporarily_change_dir(path.parent):
        try:
            Config(**data)
        except ValidationError as e:
            error_details = "\n".join(
                f"{' -> '.join(map(str, err['loc']))}: {err['msg']}"
                for err in e.errors()
            )
            pytest.fail(f"Validation failed for {path}:\n{error_details}")


def test_strict_config_validation(config_data: Tuple[Path, Dict[str, Any]]) -> None:
    """Validates all config files contained within the `test/` folder via strict handling

    Parameters
    ----------
    config_data : Tuple[Path, Dict[str, Any]]
        A tuple containing the path to the config file and the parsed config data
        - The first element is a Path object pointing to the config file
        - The second element is a dictionary containing the parsed config yaml file data.

    Raises
    ------
    pytest.fail
        If a ValidationError occurs during Config creation, this function will
        call pytest.fail with a detailed error message showing the config file that fails

    Notes
    -----
    - This test function uses the `temporarily_change_dir` context manager to
    change the working directory before attempting to create the Config object
    - If this code runs into a "value_error.path.not_exists" error, this is either because:
      1. there is a relative path in the config
      2. the file doesn't exist
      Thus, we will make the relative path absolute and retry the validation. If that fails, we
      know the file does not existt
    """
    path, data = config_data
    parent_path = path.parent
    try:
        with temporarily_change_dir(path.parent):
            Config.with_strict_mode(**data)
    except ValidationError as e:
        for error in e.errors():
            if error["type"] == "value_error.path.not_exists":
                keys = error["loc"]
                invalid_path = error["ctx"]["path"]
                corrected_path = Path(parent_path, invalid_path).__str__()

                # Ensuring the code path exists before changing relative path to absolute
                if Path(corrected_path).exists():
                    current = data
                    for key in keys[:-1]:
                        current = current.setdefault(key, {})
                    current[keys[-1]] = corrected_path
                else:
                    pytest.fail(f"Path does not exist: {corrected_path}")

        try:
            with temporarily_change_dir(path.parent):
                Config.with_strict_mode(**data)
        except ValidationError as e:
            error_details = "\n".join(
                f"{' -> '.join(map(str, err['loc']))}: {err['msg']}"
                for err in e.errors()
            )
            pytest.fail(f"Validation failed for {path}:\n{error_details}")
