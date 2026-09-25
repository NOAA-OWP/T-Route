"""The LowerColorado demonstration runs, as tests.

These drive the real ``nwm_routing`` CLI over the domain and forcing committed
under ``test/``, so they need no preprocessed data and run anywhere the package
is built. They are the four cases CI used to invoke by hand.

Marked ``endtoend`` so a developer iterating on unit tests can skip them with
``-m "not endtoend"``; they are otherwise part of the default run.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent

# (directory under test/, config, CLI version flag)
_CASES = [
    ("LowerColorado_TX", "test_AnA.yaml", "-V3"),
    ("LowerColorado_TX", "test_AnA_V4_NHD.yaml", "-V4"),
    ("LowerColorado_TX_v4", "test_AnA_V4_HYFeature.yaml", "-V4"),
    ("LowerColorado_TX_v4", "test_AnA_V4_HYFeature_noDA.yaml", "-V4"),
]


@pytest.mark.endtoend
@pytest.mark.parametrize(("subdir", "config", "version"), _CASES, ids=lambda v: str(v))
def test_lower_colorado_runs(subdir: str, config: str, version: str) -> None:
    """The CLI completes on the committed LowerColorado data."""
    workdir = _ROOT / subdir
    if not (workdir / config).is_file():
        pytest.skip(f"{subdir}/{config} is not present")

    proc = subprocess.run(
        [sys.executable, "-m", "nwm_routing", "-f", version, config],
        cwd=workdir,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        # The traceback is the useful part; the run's INFO log is not.
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-25:])
        pytest.fail(f"{subdir}/{config} exited {proc.returncode}\n{tail}")
