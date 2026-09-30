"""Where benchmark and analysis runs put their output.

Every harness used to name its own directory at the repo root or under
``benchmark/``, which left ~170 of them and 40 GB behind, and made the scripts
cwd-sensitive: the same default meant ``./output_ac_noda`` or
``benchmark/output_ac_noda`` depending on where you happened to run from.

They now resolve through :func:`results`, which returns an ABSOLUTE path under
``benchmark/data/results/``. Absolute so a script gives the same answer from any
cwd; under ``benchmark/data/`` because ``benchmark/.gitignore`` already ignores
``data/``, so run output cannot be committed by accident.
"""

from __future__ import annotations

from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BENCH_DIR / "data" / "results"


def results(name: str | Path) -> Path:
    """Absolute path to the run-output directory *name*.

    Only the final component is used, so passing a legacy value like
    ``"benchmark/output_conus_skill"`` resolves to
    ``benchmark/data/results/output_conus_skill`` rather than nesting.
    """
    return RESULTS_DIR / Path(name).name


def ensure(name: str | Path) -> Path:
    """:func:`results`, with the directory created."""
    path = results(name)
    path.mkdir(parents=True, exist_ok=True)
    return path


if __name__ == "__main__":
    # Self-check: cwd-independent, never nests, always under data/results.
    import os

    a = results("output_ac_noda")
    os.chdir("/")
    assert results("output_ac_noda") == a, "results() must not depend on cwd"
    assert results("benchmark/output_conus_skill").name == "output_conus_skill"
    assert RESULTS_DIR in a.parents
    print(f"ok: {a}")
