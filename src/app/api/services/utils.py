from pathlib import Path
from typing import Dict, List


def update_test_paths_with_prefix(data: Dict[str, str], prefix: Path, paths_to_update: List[List[str]]) -> Dict[str, str]:
    """Update specific paths inside of a config dictionary with the given prefix, if they exist.
    
    Parameters:
    -----------
    data: Dict[str, str]
        The data dictionary read from the yaml config 
    prefix: Path
        The path prefix we want to append
    paths_to_update: List[str]
        The list of paths to update from the config
    
    Returns:
    --------
    Dict[str, str]
        The updated data dictionary
    """

    for keys in paths_to_update:
        current = data
        for key in keys[:-1]:
            if key in current:
                current = current[key]
            else:
                break
        if keys[-1] in current:
            current[keys[-1]] = (prefix / current[keys[-1]]).__str__()

    return data
