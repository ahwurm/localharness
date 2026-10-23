"""`install_route` / `installed_extras`: one spec, six installer rows, nothing spawned."""
from __future__ import annotations

import importlib.metadata
import importlib.util
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from localharness.cli import update_cmd
from localharness.cli.update_cmd import Manual, install_route
from localharness.plugins.plan import installed_extras

EXTRAS = ["browser", "dev", "Exact_Tokenizer", "mobile"]
SPEC = "localharness[browser,dev,mobile]==1.2.3"
PIP_ARGV = [sys.executable, "-I", "-m", "pip", "install", SPEC]
UV_PREREQ = "Install uv first (https://docs.astral.sh/uv/)."


class _Metadata:
    def __init__(self, extras: list[str]) -> None:
        self._extras = extras

    def get_all(self, key: str):
        return list(self._extras) if key == "Provides-Extra" else None


@pytest.fixture(autouse=True)
def nothing_spawns(monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError(f"spawned {a}")
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)


@pytest.fixture
def dist(monkeypatch):
    """localharness 1.2.3 declaring EXTRAS with browser+dev installed, as a pip venv on linux."""
    def distribution(name: str):
        if name == "localharness":
            return SimpleNamespace(metadata=_Metadata(EXTRAS), version="1.2.3")
        raise importlib.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(importlib.metadata, "distribution", distribution)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: distribution(name).version)
    monkeypatch.setattr(update_cmd, "installed_extras", lambda: frozenset({"dev", "browser"}))
    monkeypatch.setattr(update_cmd, "_is_source_install", lambda: False)
    monkeypatch.setattr(update_cmd, "_installer_kind", lambda: ("pip", None))
    monkeypatch.setattr(update_cmd, "_is_windows", lambda: False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())


def _checkout(monkeypatch, tmp_path) -> Path:
    root = tmp_path / "my checkout"
    pkg = root / "src" / "localharness"
    pkg.mkdir(parents=True)
    (root / "pyproject.toml").write_text("")
    monkeypatch.setattr(update_cmd.localharness, "__file__", str(pkg / "__init__.py"))
    monkeypatch.setattr(update_cmd, "_is_source_install", lambda: True)
    return root.resolve()


SYNC = "uv sync --extra browser --extra dev --extra mobile"


# --------------------------------------------------------------------------- the spec

def test_spec_parses_and_names_every_installed_extra_plus_the_new_one(dist):
    route = install_route("mobile")
    req = Requirement(route.spec)
    assert route.spec == SPEC
    assert (req.name, req.extras, str(req.specifier)) == ("localharness", {"browser", "dev", "mobile"}, "==1.2.3")


def test_spec_carries_only_the_one_extra_when_none_is_installed(dist, monkeypatch):
    monkeypatch.setattr(update_cmd, "installed_extras", lambda: frozenset())
    assert install_route("mobile").spec == "localharness[mobile]==1.2.3"


def test_spec_and_command_do_not_depend_on_install_order(dist, monkeypatch):
    routes = []
    for order in (["dev", "browser"], ["browser", "dev"]):
        monkeypatch.setattr(update_cmd, "installed_extras", lambda order=order: frozenset(order))
        routes.append(install_route("mobile"))
    assert routes[0] == routes[1]
    assert routes[0].argv[-1] == SPEC


@pytest.mark.parametrize("extra, spec", [
    ("BROWSER", "localharness[browser,dev]==1.2.3"),
    ("exact-tokenizer", "localharness[browser,dev,exact-tokenizer]==1.2.3"),
    ("Exact_Tokenizer", "localharness[browser,dev,exact-tokenizer]==1.2.3"),
])
def test_members_are_canonicalized(dist, extra, spec):
    assert install_route(extra).spec == spec


@pytest.mark.parametrize("extra", ["browser; rm -rf /", "nope", "", "localharness[browser]"])
def test_unknown_or_hostile_extra_has_no_route(dist, extra):
    assert install_route(extra) is None


def test_no_route_without_an_installed_distribution(dist, monkeypatch):
    monkeypatch.setattr(importlib.metadata, "distribution",
                        lambda name: (_ for _ in ()).throw(importlib.metadata.PackageNotFoundError(name)))
    assert install_route("mobile") is None


# --------------------------------------------------------------------------- the six rows

def test_source_checkout_syncs_the_union_with_dev_surviving(dist, monkeypatch, tmp_path):
    root = _checkout(monkeypatch, tmp_path)
    route = install_route("mobile")
    assert route.argv is None
    assert route.manual == Manual(None, False, f"cd '{root}' && {SYNC}")


def test_source_checkout_on_windows_quotes_with_list2cmdline(dist, monkeypatch, tmp_path):
    root = _checkout(monkeypatch, tmp_path)
    monkeypatch.setattr(update_cmd, "_is_windows", lambda: True)
    assert install_route("mobile").manual == Manual(None, False, f'cd "{root}" && {SYNC}')


def test_uv_tool_install_runs_uv_here(dist, monkeypatch):
    monkeypatch.setattr(update_cmd, "_installer_kind", lambda: ("uv-tool", "/usr/bin/uv"))
    route = install_route("mobile")
    assert (route.argv, route.manual) == (["/usr/bin/uv", "tool", "install", SPEC], None)


@pytest.mark.parametrize("windows, command", [
    (False, f"uv tool install {shlex.quote(SPEC)}"),
    (True, f"uv tool install {SPEC}"),
])
def test_uv_tool_install_without_uv_asks_for_uv_first(dist, monkeypatch, windows, command):
    monkeypatch.setattr(update_cmd, "_installer_kind", lambda: ("uv-tool", None))
    monkeypatch.setattr(update_cmd, "_is_windows", lambda: windows)
    route = install_route("mobile")
    assert route.argv is None
    assert route.manual == Manual(UV_PREREQ, windows, command)


def test_uv_tool_install_on_windows_quits_first(dist, monkeypatch):
    monkeypatch.setattr(update_cmd, "_installer_kind", lambda: ("uv-tool", "C:\\uv\\uv.exe"))
    monkeypatch.setattr(update_cmd, "_is_windows", lambda: True)
    route = install_route("mobile")
    assert route.argv is None
    assert route.manual == Manual(None, True, f"uv tool install {SPEC}")


def test_pip_install_runs_isolated_pip_here(dist):
    route = install_route("mobile")
    assert (route.argv, route.manual) == (PIP_ARGV, None)


def test_pip_install_without_pip_asks_for_ensurepip_first(dist, monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    route = install_route("mobile")
    assert route.argv is None
    assert route.manual == Manual("Run python -m ensurepip --upgrade first.", False, shlex.join(PIP_ARGV))


# --------------------------------------------------------------------------- _installer_kind

@pytest.mark.parametrize("prefix, uv, expected", [
    ("uv/tools/localharness", "/usr/bin/uv", ("uv-tool", "/usr/bin/uv")),
    ("uv/tools/localharness", None, ("uv-tool", None)),
    ("some/venv", "/usr/bin/uv", ("pip", None)),
])
def test_installer_kind(monkeypatch, tmp_path, prefix, uv, expected):
    (tmp_path / prefix).mkdir(parents=True)
    monkeypatch.setattr(update_cmd.sys, "prefix", str(tmp_path / prefix))
    monkeypatch.setattr(update_cmd, "which_outside_cwd", lambda _n: uv)
    assert update_cmd._installer_kind() == expected


@pytest.mark.parametrize("kind, expected", [
    (("uv-tool", "/usr/bin/uv"), ["/usr/bin/uv", "tool", "upgrade", "localharness"]),
    (("uv-tool", None), None),
    (("pip", None), [sys.executable, "-m", "pip", "install", "--upgrade", "localharness"]),
])
def test_upgrade_command_is_byte_identical(monkeypatch, kind, expected):
    monkeypatch.setattr(update_cmd, "_installer_kind", lambda: kind)
    assert update_cmd._upgrade_command() == expected


# --------------------------------------------------------------------------- installed_extras

_REQUIRES = ["pydantic>=2", "starlette<1,>=0.40; extra == 'mobile'", "uvicorn>=0.30; extra == 'mobile'",
             "discord.py>=2.3; extra == 'dispatch'"]


def _fake_dists(monkeypatch, *, extras: list[str], installed: dict[str, str]) -> None:
    def distribution(name: str):
        if name == "localharness":
            return SimpleNamespace(requires=_REQUIRES, metadata=_Metadata(extras))
        if name in installed:
            return SimpleNamespace(version=installed[name])
        raise importlib.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(importlib.metadata, "distribution", distribution)


@pytest.mark.parametrize("extras, installed, expected", [
    (["mobile", "Dispatch"], {"starlette": "0.46.0", "uvicorn": "0.34.0"}, {"mobile"}),
    (["mobile", "Dispatch"], {"starlette": "0.46.0", "uvicorn": "0.34.0", "discord.py": "2.4"}, {"mobile", "dispatch"}),
    (["mobile", "Dispatch"], {"starlette": "0.46.0"}, set()),
    ([], {}, set()),
])
def test_installed_extras_are_the_canonical_names_extra_installed_accepts(monkeypatch, extras, installed, expected):
    _fake_dists(monkeypatch, extras=extras, installed=installed)
    assert installed_extras() == frozenset(expected)


def test_installed_extras_is_empty_without_an_installed_localharness(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "distribution",
                        lambda name: (_ for _ in ()).throw(importlib.metadata.PackageNotFoundError(name)))
    assert installed_extras() == frozenset()


def test_installed_extras_on_this_install():
    """Unmocked: this suite runs under the dev extra (`uv sync --extra dev`)."""
    found = installed_extras()
    assert "dev" in found
    assert all(e == canonicalize_name(e) for e in found)
