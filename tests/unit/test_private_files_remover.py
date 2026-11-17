"""The directory remover: links, devices, hostile modes, depth."""
import os

import pytest

from localharness.core import private_files
from localharness.core.private_files import empty_dir, in_background, remove_dir

pytestmark = pytest.mark.skipif(os.name != "posix", reason="descriptor-relative removal")


def _open(path):
    return os.open(path, os.O_RDONLY | os.O_DIRECTORY)


def test_removes_a_tree(tmp_path):
    (tmp_path / "t" / "a" / "b").mkdir(parents=True)
    (tmp_path / "t" / "x.txt").write_text("x")
    (tmp_path / "t" / "a" / "y.txt").write_text("y")
    (tmp_path / "t" / "a" / "b" / "z.txt").write_text("z")
    fd = _open(tmp_path)
    try:
        assert remove_dir("t", fd) == ()
    finally:
        os.close(fd)
    assert not (tmp_path / "t").exists()


def test_a_missing_name_is_nothing_to_remove(tmp_path):
    fd = _open(tmp_path)
    try:
        assert remove_dir("gone", fd) == ()
    finally:
        os.close(fd)


def test_a_link_to_a_directory_is_unlinked_and_its_target_untouched(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep")
    (tmp_path / "t").mkdir()
    (tmp_path / "t" / "link").symlink_to(outside, target_is_directory=True)
    fd = _open(tmp_path)
    try:
        assert remove_dir("t", fd) == ()
    finally:
        os.close(fd)
    assert not (tmp_path / "t").exists()
    assert (outside / "keep.txt").read_text() == "keep"


def test_a_link_handed_as_the_name_is_unlinked_not_followed(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep")
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "t").symlink_to(outside, target_is_directory=True)
    fd = _open(tmp_path / "run")
    try:
        remove_dir("t", fd)
    finally:
        os.close(fd)
    assert not (tmp_path / "run" / "t").is_symlink()
    assert (outside / "keep.txt").exists()


def test_a_fifo_is_unlinked_without_opening_it(tmp_path):
    (tmp_path / "t").mkdir()
    os.mkfifo(tmp_path / "t" / "pipe")
    fd = _open(tmp_path)
    try:
        assert remove_dir("t", fd) == ()
    finally:
        os.close(fd)


def test_another_device_is_reported_and_left(tmp_path, monkeypatch):
    (tmp_path / "t" / "mount").mkdir(parents=True)
    (tmp_path / "t" / "mount" / "f").write_text("f")
    real = os.fstat

    def fake(fd):
        st = real(fd)
        if os.path.samestat(st, os.stat(tmp_path / "t" / "mount")):
            return os.stat_result((*st[:2], st.st_dev + 1, *st[3:]))
        return st

    monkeypatch.setattr(private_files.os, "fstat", fake)
    fd = _open(tmp_path)
    try:
        got = remove_dir("t", fd)
    finally:
        os.close(fd)
    assert got == ("t/mount",)
    assert (tmp_path / "t" / "mount" / "f").exists()


def test_a_tree_past_the_descriptor_cap_goes(tmp_path):
    deep = tmp_path / "t"
    for _ in range(70):
        deep = deep / "d"
    deep.mkdir(parents=True)
    (deep / "f").write_text("f")
    fd = _open(tmp_path)
    try:
        left = remove_dir("t", fd)
    finally:
        os.close(fd)
    assert list(tmp_path.iterdir()) == [] and left == ()


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores modes")
def test_unreadable_and_read_only_trees_are_emptied(tmp_path):
    root = tmp_path / "t"
    (root / "closed").mkdir(parents=True)
    (root / "closed" / "f").write_text("f")
    (root / "ro" / "inner").mkdir(parents=True)
    (root / "ro" / "inner" / "g").write_text("g")
    os.chmod(root / "ro" / "inner", 0o555)
    os.chmod(root / "ro", 0o555)
    os.chmod(root / "closed", 0o000)
    fd = _open(tmp_path)
    try:
        got = remove_dir("t", fd)
    finally:
        os.close(fd)
    assert got == ()
    assert not root.exists()


def test_empty_dir_leaves_the_handed_directory(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "f").write_text("f")
    (tmp_path / "g").write_text("g")
    fd = _open(tmp_path)
    try:
        assert empty_dir(fd) == ()
    finally:
        os.close(fd)
    assert list(tmp_path.iterdir()) == []


def test_the_base_device_is_the_handed_directorys(tmp_path, monkeypatch):
    (tmp_path / "tasks" / "t3" / "sub").mkdir(parents=True)
    (tmp_path / "tasks" / "t3" / "sub" / "doc").write_text("d")
    base_ino = os.stat(tmp_path).st_ino
    real = os.fstat

    def fake(fd):
        st = real(fd)
        if st.st_ino == base_ino:
            return st
        return os.stat_result((*st[:2], st.st_dev + 1, *st[3:]))

    monkeypatch.setattr(private_files.os, "fstat", fake)
    fd = _open(tmp_path / "tasks")
    try:
        assert remove_dir("t3", fd) == ()
    finally:
        os.close(fd)
    assert not (tmp_path / "tasks" / "t3").exists()


def test_a_file_that_will_not_unlink_is_reported(tmp_path, monkeypatch):
    (tmp_path / "t").mkdir()
    (tmp_path / "t" / "f").write_text("f")
    monkeypatch.setattr(private_files.os, "unlink", lambda *a, **k: (_ for _ in ()).throw(PermissionError()))
    fd = _open(tmp_path)
    try:
        got = remove_dir("t", fd)
    finally:
        os.close(fd)
    assert got == ("t/f",)


def test_in_background_hands_back_the_result_or_the_error():
    assert in_background(lambda a: a + 1, 1).result(timeout=2) == 2
    with pytest.raises(ZeroDivisionError):
        in_background(lambda: 1 / 0).result(timeout=2)


def test_a_mode_without_owner_rights_is_restored_before_listing(tmp_path):
    (tmp_path / "t").mkdir()
    (tmp_path / "t" / "f").write_text("f")
    os.chmod(tmp_path / "t", 0o500)
    fd = _open(tmp_path)
    try:
        assert remove_dir("t", fd) == ()
    finally:
        os.close(fd)
    assert not (tmp_path / "t").exists()
