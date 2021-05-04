import os
from pathlib import Path

from solarnet.utils.filesystem import clean_filename, rename_file, rm_rf


def test_rename_file():
    path = Path("test.txt")
    with open(path, 'a'):
        os.utime(path, None)

    assert path.exists()

    new_name = "new.txt"
    new_path = Path(new_name)
    assert not new_path.exists()

    rename_file(path, new_name)
    assert new_path.exists()
    assert not path.exists()

    new_name2 = "new2.txt"
    new_path2 = Path(new_name2)
    assert not new_path2.exists()

    rename_file(new_path, new_name2, keep_old=True)
    assert new_path.exists()
    assert new_path2.exists()

    new_path.unlink()
    new_path2.unlink()
    assert not new_path.exists()
    assert not new_path2.exists()


def test_clean_filename():
    test_name = "this/is\\a path^with(bad)char%:'\"<>|?*"
    test_ok = "thisisapathwithbadchar"

    assert clean_filename(test_name) == test_ok


def test_rm_rf():
    path = Path("rm_rf_test_dir")
    assert not path.exists()

    path.mkdir()
    assert path.exists()

    path_file = path / "test.txt"
    assert not path_file.exists()
    with open(path_file, 'a'):
        os.utime(path_file, None)
    assert path_file.exists()

    rm_rf(path)
    assert not path.exists()
