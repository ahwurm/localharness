"""`atomic_write_text` lives in core/private_files.py and takes an optional `dir_fd` (D-28, SC5)."""
from __future__ import annotations

import os
import stat

import pytest

from localharness.config import migrate, overlay, project_records
from localharness.core import private_files


def _dir_fd(path):
    return private_files.open_private_dir(path, create=True)


def test_overlay_re_exports_the_one_writer():
    assert overlay.atomic_write_text is private_files.atomic_write_text
    assert overlay.refuse_unwritable is private_files.refuse_unwritable
    assert migrate.atomic_write_text is private_files.atomic_write_text
    assert project_records.atomic_write_text is private_files.atomic_write_text


def test_overlay_write_still_yields_yaml_at_0600(tmp_path):
    target = tmp_path / "o.yaml"
    overlay.atomic_write_overlay(target, {"a": 1})
    assert target.read_text() == "a: 1\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_dir_fd_write_is_0600_and_leaves_no_temp(tmp_path):
    fd = _dir_fd(tmp_path / "d")
    try:
        private_files.atomic_write_text("f", "hello\n", dir_fd=fd)
        private_files.atomic_write_text("f", "again\n", dir_fd=fd)
    finally:
        os.close(fd)
    d = tmp_path / "d"
    assert sorted(p.name for p in d.iterdir()) == ["f"]
    assert (d / "f").read_text() == "again\n"
    assert stat.S_IMODE((d / "f").stat().st_mode) == 0o600


def test_dir_fd_write_replaces_a_planted_link_without_following_it(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.write_text("keep\n")
    (d / "f").symlink_to(elsewhere)
    fd = _dir_fd(d)
    try:
        private_files.atomic_write_text("f", "new\n", dir_fd=fd)
    finally:
        os.close(fd)
    assert not (d / "f").is_symlink()
    assert (d / "f").read_text() == "new\n"
    assert elsewhere.read_text() == "keep\n"


@pytest.mark.parametrize("name", ["a/b", "", ".", ".."])
def test_dir_fd_write_refuses_a_name_that_is_not_bare(tmp_path, name):
    fd = _dir_fd(tmp_path)
    try:
        with pytest.raises(ValueError):
            private_files.atomic_write_text(name, "x", dir_fd=fd)
    finally:
        os.close(fd)


@pytest.mark.parametrize("kw", [{"refuse_read_only": True}, {"follow_links": True}])
def test_dir_fd_write_refuses_the_path_only_options(tmp_path, kw):
    fd = _dir_fd(tmp_path)
    try:
        with pytest.raises(ValueError):
            private_files.atomic_write_text("f", "x", dir_fd=fd, **kw)
    finally:
        os.close(fd)


def test_a_failed_dir_fd_write_keeps_the_old_body_and_leaves_no_temp(tmp_path, monkeypatch):
    d = tmp_path / "d"
    fd = _dir_fd(d)
    try:
        private_files.atomic_write_text("unit-target", "old\n", dir_fd=fd)

        def boom(*a, **kw):
            raise OSError("simulated")

        monkeypatch.setattr(private_files.os, "fsync", boom)
        with pytest.raises(OSError):
            private_files.atomic_write_text("unit-target", "new\n", dir_fd=fd)
        monkeypatch.undo()
    finally:
        os.close(fd)
    assert (d / "unit-target").read_text() == "old\n"
    assert sorted(p.name for p in d.iterdir()) == ["unit-target"]


def test_open_private_dir_makes_0700_and_refuses_a_link(tmp_path):
    fd = private_files.open_private_dir(tmp_path / "x" / "d")
    os.close(fd)
    assert stat.S_IMODE((tmp_path / "x" / "d").stat().st_mode) == 0o700
    (tmp_path / "ln").symlink_to(tmp_path / "x" / "d")
    with pytest.raises(OSError):
        private_files.open_private_dir(tmp_path / "ln")
    with pytest.raises(OSError):
        private_files.open_private_dir(tmp_path / "missing", create=False)


def test_open_regular_with_dir_fd_refuses_a_link(tmp_path):
    (tmp_path / "real").write_text("x")
    (tmp_path / "ln").symlink_to(tmp_path / "real")
    fd = private_files.open_private_dir(tmp_path, create=False)
    try:
        os.close(private_files.open_regular("real", dir_fd=fd))
        with pytest.raises(OSError):
            private_files.open_regular("ln", dir_fd=fd)
    finally:
        os.close(fd)
