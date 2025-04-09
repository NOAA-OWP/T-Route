from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError
from troute.config._utils import use_strict
from troute.config.types import DirectoryPath, FilePath


@pytest.mark.parametrize(
    "file,expected",
    (
        ("some_file", Path("some_file")),
        (Path("some_file"), Path("some_file")),
    ),
)
def test_file_path(file, expected):
    class Foo(BaseModel):
        file: FilePath

    o = Foo(file=file)
    assert o.file == expected


@pytest.mark.parametrize("file", (__file__, Path(__file__)))
def test_file_path_use_strict(file):
    class Foo(BaseModel):
        file: FilePath

    with use_strict():
        o = Foo(file=file)
        assert isinstance(o.file, Path)


@pytest.mark.parametrize("file", ("some_fake_file", Path("some_fake_file")))
def test_file_path_use_strict_raises(file):
    assert not Path(file).exists(), "test expects file not to exist"

    class Foo(BaseModel):
        file: FilePath

    with pytest.raises(ValidationError):
        with use_strict():
            Foo(file=file)


@pytest.mark.parametrize(
    "directory,expected",
    (
        ("some_dir", Path("some_dir")),
        (Path("some_dir"), Path("some_dir")),
    ),
)
def test_directory_path(directory, expected):
    class Foo(BaseModel):
        dir: DirectoryPath

    o = Foo(dir=directory)
    assert o.dir == expected


MOD_DIR = Path(__file__).parent


@pytest.mark.parametrize(
    "directory",
    (
        str(MOD_DIR),
        MOD_DIR,
    ),
)
def test_directory_path_use_strict(directory):
    assert Path(directory).is_dir(), "test expects dir to exist"

    class Foo(BaseModel):
        dir: DirectoryPath

    with use_strict():
        Foo(dir=directory)


@pytest.mark.parametrize(
    "directory",
    (
        "some_fake_dir",
        Path("some_fake_dir"),
    ),
)
def test_directory_path_use_strict_raises(directory):
    assert (
        not Path(directory).exists() and not Path(directory).is_dir()
    ), "test expects dir not to exist"

    class Foo(BaseModel):
        dir: DirectoryPath

    with pytest.raises(ValidationError):
        with use_strict():
            Foo(dir=directory)
