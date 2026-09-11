"""Tests for the multi-file output merge in ``troute_model``.

A BMI run windows the simulation into chunks and writes one output file per
chunk, then merges them back into a single file named after the first chunk.
That merge is a publish step over the run's only copy of its results, so the
properties worth pinning are transactional rather than numerical: a failure must
leave every input where it was, and a success must not depend on the merged file
and its destination sharing a filesystem.

``_merge_into_first`` needs nothing from the model, so these run without the
optional troute-bmi install being importable as a whole.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

troute_model = pytest.importorskip(
    "troute_nwm_bmi.troute_model",
    reason="troute-bmi is an optional install (pip install -e src/troute-bmi --no-deps)",
)
_merge_into_first = troute_model._merge_into_first


@pytest.fixture
def parts(tmp_path: Path) -> list[Path]:
    """Three chunk outputs, oldest first, as the glob would return them."""
    files = []
    for i in range(3):
        f = tmp_path / f"troute_output_{i}.csv"
        f.write_text(f"chunk{i}\n")
        files.append(f)
    return files


def test_merged_result_lands_on_the_first_part(parts):
    _merge_into_first(parts, lambda dest: dest.write_text("merged\n"))
    assert parts[0].read_text() == "merged\n"
    assert not parts[1].exists() and not parts[2].exists()


def test_no_temporary_file_is_left_behind(parts):
    _merge_into_first(parts, lambda dest: dest.write_text("merged\n"))
    assert sorted(p.name for p in parts[0].parent.iterdir()) == ["troute_output_0.csv"]


def test_temporary_file_is_created_in_the_destination_directory(parts, tmp_path):
    """Otherwise the final rename can cross filesystems and fail with EXDEV.

    The lakeout branch used to create its temporary file in the stream output
    directory, which is a different directory in every shipped config.
    """
    seen: list[Path] = []
    _merge_into_first(parts, lambda dest: (seen.append(dest), dest.write_text("x"))[-1])
    assert seen[0].parent == tmp_path == parts[0].parent


def test_write_failure_leaves_every_input_intact(parts):
    """A failed merge has to be retryable, not destructive.

    Deleting the parts before publishing turned any error here into total loss of
    the run's output, which is the failure this merge path exists to prevent.
    """
    def boom(dest: Path) -> None:
        dest.write_text("half written")
        raise RuntimeError("merge blew up")

    with pytest.raises(RuntimeError, match="merge blew up"):
        _merge_into_first(parts, boom)

    assert [p.read_text() for p in parts] == ["chunk0\n", "chunk1\n", "chunk2\n"]
    assert sorted(p.name for p in parts[0].parent.iterdir()) == [
        "troute_output_0.csv", "troute_output_1.csv", "troute_output_2.csv",
    ]


def test_publish_failure_leaves_every_input_intact(parts, monkeypatch):
    """Same guarantee when the atomic replace itself is what fails."""
    def no_replace(self, target):
        raise OSError(18, "Invalid cross-device link")

    monkeypatch.setattr(Path, "replace", no_replace)
    with pytest.raises(OSError):
        _merge_into_first(parts, lambda dest: dest.write_text("merged\n"))

    assert [p.read_text() for p in parts] == ["chunk0\n", "chunk1\n", "chunk2\n"]


def test_original_permissions_are_preserved(parts):
    """ngen consumes these files, so the merged output must keep the parts' mode."""
    mode = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
    parts[0].chmod(mode)
    _merge_into_first(parts, lambda dest: dest.write_text("merged\n"))
    assert stat.S_IMODE(os.stat(parts[0]).st_mode) == mode


def test_suffix_follows_the_destination_not_the_caller(parts, tmp_path):
    """The lakeout branch borrowed the stream output's suffix, which need not be .nc."""
    seen: list[Path] = []
    _merge_into_first(parts, lambda dest: (seen.append(dest), dest.write_text("x"))[-1])
    assert seen[0].suffix == ".csv"
