"""LAUNCH-06 / INIT-07 (replay half): a llama.cpp launch reads only the `config.ini` its record carries.

`SpawnedProcessStrategy.activate` rewrites a harness-owned `XDG_CONFIG_HOME` directory,
`<global config dir>/xdg/<port>`, from the record's `config_ini` bytes (or empties it) and points the
child at it, so neither the person's real `~/.config/llama.cpp/config.ini` nor a planted one reaches a
launch (D-30, D-59). Nothing here starts a process: `subprocess.Popen` is the `PopenRecorder`, and
readiness is a stub.
"""
from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from localharness.config.models import ManagedServerConfig
from localharness.provider import lifecycle, server
from tests.unit.launch_support import PopenRecorder, popen  # noqa: F401

BASE = "http://127.0.0.1:8080/v1"


@pytest.fixture(autouse=True)
def _ready(monkeypatch):
    async def wait_ready(base_url, **kw):
        return ["m"]

    monkeypatch.setattr(server, "wait_ready", wait_ready)


def _spec(**over) -> ManagedServerConfig:
    return ManagedServerConfig(**{"runtime": "llamacpp", "binary": "/x/llama-server",
                                  "model": "/x/m.gguf", "port": 8080, **over})


def _ini(root: Path, port: int = 8080) -> Path:
    return root / "xdg" / str(port) / "llama.cpp" / "config.ini"


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.lstat(path).st_mode)


# ------------------------------------------------------------------------------- the names


def test_the_home_is_one_directory_per_port_under_the_global_dir(tmp_path):
    assert server.config_ini_home(tmp_path, 8080) == tmp_path / "xdg" / "8080"
    assert server.config_ini_home(str(tmp_path), 8082) == tmp_path / "xdg" / "8082"
    assert "vllm" not in server.config_ini_home(tmp_path, 8080).parts


# ------------------------------------------------------------------------------- child_environment


def test_the_keyword_adds_xdg_config_home_after_the_scrub():
    assert server.child_environment("llamacpp", {"PATH": "/bin"}, xdg_config_home="/g/xdg/8080") == (
        {"PATH": "/bin", "XDG_CONFIG_HOME": "/g/xdg/8080"}, [])


def test_the_keyword_overrides_a_variable_already_in_the_base():
    env, dropped = server.child_environment(
        "llamacpp", {"XDG_CONFIG_HOME": "/home/u/.config", "LLAMA_ARG_HOST": "0.0.0.0"},
        xdg_config_home="/g/xdg/8080")
    assert env == {"XDG_CONFIG_HOME": "/g/xdg/8080"} and dropped == ["LLAMA_ARG_HOST"]


def test_the_keyword_comes_before_the_key_and_after_the_record_env():
    env, _ = server.child_environment(
        "llamacpp", {}, record_env={"XDG_CONFIG_HOME": "/elsewhere", "A": "1"}, api_key="sk-K",
        xdg_config_home="/g/xdg/8080")
    assert env["XDG_CONFIG_HOME"] == "/g/xdg/8080" and env["A"] == "1"
    assert env[server.KEY_ENV["llamacpp"]] == "sk-K"


def test_without_the_keyword_nothing_changes():
    assert server.child_environment("llamacpp", {"PATH": "/bin"}) == (None, [])
    assert server.child_environment("llamacpp", {"XDG_CONFIG_HOME": "/home/u/.config"}) == (None, [])
    assert server.child_environment("vllm", {"PATH": "/bin"}, xdg_config_home=None) == (None, [])


# ------------------------------------------------------------------------------- activate, primary and peer


async def test_an_activation_writes_the_records_config_ini_and_launches_with_that_home(tmp_path, popen):
    await lifecycle.SpawnedProcessStrategy().activate(_spec(config_ini="tools = all\n"), tmp_path, BASE)
    assert _ini(tmp_path).read_text() == "tools = all\n"
    assert popen[-1]["env"]["XDG_CONFIG_HOME"] == str(tmp_path / "xdg" / "8080")


async def test_argv_and_the_launch_line_are_what_they_were(tmp_path, popen):
    spec = _spec(config_ini="tools = all\n")
    await lifecycle.SpawnedProcessStrategy().activate(spec, tmp_path, BASE)
    assert popen[-1]["cmd"] == server.serve_command(spec, keyed=False, loopback=True)
    assert not any("XDG" in token for token in popen[-1]["cmd"])
    log = server.log_path(tmp_path).read_text()
    assert "=== launch: " in log and "XDG" not in log


async def test_a_record_without_config_ini_empties_the_directory_it_launches_with(tmp_path, popen):
    strategy = lifecycle.SpawnedProcessStrategy()
    await strategy.activate(_spec(config_ini="tools = all\n"), tmp_path, BASE)
    assert _ini(tmp_path).exists()
    await strategy.activate(_spec(), tmp_path, BASE)
    assert not _ini(tmp_path).exists()
    assert (tmp_path / "xdg" / "8080").is_dir()
    assert popen[-1]["env"]["XDG_CONFIG_HOME"] == str(tmp_path / "xdg" / "8080")


async def test_a_peer_gets_its_own_directory(tmp_path, popen):
    await lifecycle.SpawnedProcessStrategy().activate(
        _spec(port=8082, config_ini="ctx-size = 4096\n"), tmp_path, "http://127.0.0.1:8082/v1",
        peer="coder")
    assert _ini(tmp_path, 8082).read_text() == "ctx-size = 4096\n"
    assert not (tmp_path / "xdg" / "8080").exists()
    assert popen[-1]["env"]["XDG_CONFIG_HOME"] == str(tmp_path / "xdg" / "8082")


async def test_a_users_own_variable_never_reaches_the_child(tmp_path, popen, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", "/home/u/.config")
    await lifecycle.SpawnedProcessStrategy().activate(_spec(), tmp_path, BASE)
    assert popen[-1]["env"]["XDG_CONFIG_HOME"] == str(tmp_path / "xdg" / "8080")


async def test_a_directory_that_cannot_be_prepared_is_a_runtime_error_before_any_launch(tmp_path, popen):
    not_a_directory = tmp_path / "config"
    not_a_directory.write_text("a file where the config directory goes")
    with pytest.raises(RuntimeError, match="could not prepare"):
        await lifecycle.SpawnedProcessStrategy().activate(_spec(), not_a_directory, BASE)
    assert popen == []


# ------------------------------------------------------------------------------- the file


def test_the_directories_are_owner_only_and_no_temporary_file_is_left(tmp_path):
    home = server.prepare_config_ini_home(tmp_path, 8080, "tools = all\n")
    assert home == tmp_path / "xdg" / "8080"
    for path in (tmp_path / "xdg", home, home / "llama.cpp"):
        assert _mode(path) == 0o700 and path.is_dir() and not path.is_symlink()
    assert _mode(home / "llama.cpp" / "config.ini") == 0o600
    assert [p.name for p in (home / "llama.cpp").iterdir()] == ["config.ini"]


def test_a_rewrite_replaces_the_bytes(tmp_path):
    server.prepare_config_ini_home(tmp_path, 8080, "a = 1\n")
    server.prepare_config_ini_home(tmp_path, 8080, "b = 2\n")
    assert _ini(tmp_path).read_text() == "b = 2\n"


def test_no_config_ini_leaves_the_directory_with_none(tmp_path):
    home = server.prepare_config_ini_home(tmp_path, 8080, None)
    assert (home / "llama.cpp").is_dir() and not _ini(tmp_path).exists()
    server.prepare_config_ini_home(tmp_path, 8080, "a = 1\n")
    server.prepare_config_ini_home(tmp_path, 8080, None)
    assert not _ini(tmp_path).exists()


# ------------------------------------------------------------------------------- planted components (T-5-37)


def _elsewhere(tmp_path: Path) -> Path:
    """A directory outside the harness's, holding a config.ini a planted link would redirect to."""
    target = tmp_path / "elsewhere"
    target.mkdir()
    (target / "config.ini").write_text("tools = all")
    return target


def _gdir(tmp_path: Path) -> Path:
    g = tmp_path / "g"
    g.mkdir()
    return g


@pytest.mark.parametrize("component", ["xdg", "xdg/8080", "xdg/8080/llama.cpp"])
def test_a_symlinked_component_is_replaced_and_its_target_left_alone(tmp_path, component):
    g = _gdir(tmp_path)
    target = _elsewhere(tmp_path)
    planted = g / component
    planted.parent.mkdir(parents=True, exist_ok=True)
    planted.symlink_to(target, target_is_directory=True)
    server.prepare_config_ini_home(g, 8080, "ctx-size = 8192\n")
    assert (target / "config.ini").read_text() == "tools = all"
    assert [p.name for p in target.iterdir()] == ["config.ini"]
    assert not planted.is_symlink() and planted.is_dir() and _mode(planted) == 0o700
    assert _ini(g).read_text() == "ctx-size = 8192\n"


@pytest.mark.parametrize("component", ["xdg", "xdg/8080", "xdg/8080/llama.cpp"])
def test_a_regular_file_in_the_place_of_a_component_is_replaced(tmp_path, component):
    g = _gdir(tmp_path)
    planted = g / component
    planted.parent.mkdir(parents=True, exist_ok=True)
    planted.write_text("not a directory")
    server.prepare_config_ini_home(g, 8080, "ctx-size = 8192\n")
    assert planted.is_dir() and not planted.is_symlink() and _mode(planted) == 0o700
    assert _ini(g).read_text() == "ctx-size = 8192\n"


def test_a_dangling_symlink_component_is_replaced(tmp_path):
    g = _gdir(tmp_path)
    (g / "xdg").symlink_to(tmp_path / "nowhere", target_is_directory=True)
    server.prepare_config_ini_home(g, 8080, "a = 1\n")
    assert (g / "xdg").is_dir() and not (g / "xdg").is_symlink() and not (tmp_path / "nowhere").exists()


def test_a_directory_that_is_too_open_is_tightened(tmp_path):
    g = _gdir(tmp_path)
    server.prepare_config_ini_home(g, 8080, "a = 1\n")
    for path in (g / "xdg", g / "xdg" / "8080", g / "xdg" / "8080" / "llama.cpp"):
        os.chmod(path, 0o755)
    server.prepare_config_ini_home(g, 8080, "a = 1\n")
    for path in (g / "xdg", g / "xdg" / "8080", g / "xdg" / "8080" / "llama.cpp"):
        assert _mode(path) == 0o700


def test_a_symlinked_config_ini_is_replaced_by_a_rewrite_and_its_target_is_untouched(tmp_path):
    g = _gdir(tmp_path)
    secret = tmp_path / "secret"
    secret.write_text("do not touch")
    server.prepare_config_ini_home(g, 8080, "a = 1\n")
    _ini(g).unlink()
    _ini(g).symlink_to(secret)
    server.prepare_config_ini_home(g, 8080, "ctx-size = 8192\n")
    assert secret.read_text() == "do not touch"
    assert not _ini(g).is_symlink() and _ini(g).read_text() == "ctx-size = 8192\n"


def test_a_symlinked_config_ini_is_removed_when_the_record_has_none(tmp_path):
    g = _gdir(tmp_path)
    secret = tmp_path / "secret"
    secret.write_text("do not touch")
    server.prepare_config_ini_home(g, 8080, None)
    _ini(g).symlink_to(secret)
    server.prepare_config_ini_home(g, 8080, None)
    assert secret.read_text() == "do not touch" and not os.path.lexists(_ini(g))


def test_a_directory_named_config_ini_is_removed(tmp_path):
    g = _gdir(tmp_path)
    server.prepare_config_ini_home(g, 8080, None)
    _ini(g).mkdir()
    (_ini(g) / "inside").write_text("x")
    server.prepare_config_ini_home(g, 8080, None)
    assert not os.path.lexists(_ini(g))
    _ini(g).mkdir()
    server.prepare_config_ini_home(g, 8080, "a = 1\n")
    assert _ini(g).is_file() and _ini(g).read_text() == "a = 1\n"


def test_a_directory_named_config_ini_holding_a_link_is_removed_without_following_it(tmp_path):
    g = _gdir(tmp_path)
    target = _elsewhere(tmp_path)
    server.prepare_config_ini_home(g, 8080, None)
    _ini(g).mkdir()
    (_ini(g) / "link").symlink_to(target, target_is_directory=True)
    server.prepare_config_ini_home(g, 8080, None)
    assert (target / "config.ini").read_text() == "tools = all"


def test_a_failed_write_leaves_no_temporary_file_and_keeps_the_old_bytes(tmp_path, monkeypatch):
    g = _gdir(tmp_path)
    server.prepare_config_ini_home(g, 8080, "old = 1\n")

    def refuse(src, dst):
        raise OSError(28, "no space left")

    monkeypatch.setattr(server.os, "replace", refuse)
    with pytest.raises(OSError):
        server.prepare_config_ini_home(g, 8080, "new = 2\n")
    monkeypatch.undo()
    assert [p.name for p in _ini(g).parent.iterdir()] == ["config.ini"]
    assert _ini(g).read_text() == "old = 1\n"
