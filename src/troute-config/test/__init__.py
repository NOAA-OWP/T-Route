import os

from contextlib import contextmanager
from pathlib import Path


@contextmanager
def temporarily_change_dir(path: Path):
    """Temporarily changes the current working directory

    This context manager changes the current working directory to the specified path,
    yields control back to the caller, and then changes back to the original directory
    when exiting the context

    Parameters
    ----------
    path : Path
        The path to temporarily change the current working directory to

    Yields
    ------
    None
    """
    original_cwd = Path.cwd()
    if original_cwd != path:
        os.chdir(path)
    try:
        yield
    finally:
        if original_cwd != path:
            os.chdir(original_cwd)
