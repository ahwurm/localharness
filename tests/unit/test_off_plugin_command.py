"""A BUNDLED plugin that resolves off: its command, run by name, prints the enable hint and exits 4
(G3) — never Click's "No such command" exit 2, which a script reading `experiment run`'s `$?` takes
for the reject-holdout verdict. The stub is never listed in `--help`, is built from the manifest
alone (no command module imported), and covers bundled-off only: an unknown name, or an installed
plugin that is not enabled, still gets Click's exit 2; a needs-extra plugin keeps its own guard.
The last tests run the real `localharness` entry point, so the code is the shell's `$?`."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from localharness.cli import plugin_mount
from localharness.cli.app import app
from localharness.plugins.api import CliDescriptor, PluginManifest
from localharness.plugins.plan import LoadPlan, PlanEntry

runner = CliRunner()
CMD_MODULES = ("localharness.cli.mobile_cmd", "localharness.cli.memory_cli", "localharness.cli.generate_image_cmd")


def _hint(cmd: str, plugin: str) -> str:
    return (f"command '{cmd}' is provided by the {plugin} plugin, which is off — "
            f"run `localharness plugins enable {plugin}`")


def _off(home: Path, *plugins: str) -> None:
    with (home / "config.yaml").open("a", encoding="utf-8") as f:
        f.write("".join(f"{p}:\n  enabled: false\n" for p in plugins))


def _run(*args: str):
    return runner.invoke(app, list(args), env={"COLUMNS": "400"})


@pytest.mark.parametrize("args", [("mobile",), ("mobile", "--help"), ("mobile", "serve", "--port", "1")])
def test_an_off_bundled_command_gets_the_hint_and_exit_4(components_home, args) -> None:
    _off(components_home, "mobile")
    ran = _run(*args)
    assert ran.exit_code == 4, ran.output
    assert ran.stderr.strip() == _hint("mobile", "mobile"), ran.stderr
    assert ran.stdout == "", ran.stdout


def test_every_bundled_off_plugin_gets_its_own_hint(components_home) -> None:
    _off(components_home, "mobile", "memory")  # image is off by default
    for args, cmd, plugin in ((["memory", "list"], "memory", "memory"),
                              (["generate-image", "x"], "generate-image", "image")):
        ran = _run(*args)
        assert (ran.exit_code, ran.stderr.strip()) == (4, _hint(cmd, plugin)), ran.output


def test_the_stub_is_not_listed_and_an_unknown_name_is_still_clicks_exit_2(components_home) -> None:
    _off(components_home, "mobile", "memory", "autoresearch")
    out = _run("--help").output
    # first cell of EVERY panel row: a stub has no help text, so a two-cell row parser would miss it
    listed = {line.strip("│ ").split()[0] for line in out.splitlines() if line.startswith("│") and line.strip("│ ")}
    assert "start" in listed, out
    assert not {"mobile", "memory", "generate-image", "autoresearch", "experiment", "propose"} & listed, out
    bogus = _run("bogus")
    assert bogus.exit_code == 2 and "No such command 'bogus'" in bogus.output, bogus.output


def test_a_needs_extra_plugin_keeps_its_own_guard(components_home, monkeypatch) -> None:
    from localharness.plugins import resolve
    monkeypatch.setitem(resolve.resolve.__kwdefaults__, "extra_installed", lambda e: False)
    on, off = plugin_mount._resolve_commands()
    assert isinstance(on["mobile"], plugin_mount.LazyPluginCommand) and "mobile" not in off


def test_an_installed_plugin_that_is_not_enabled_is_not_named(monkeypatch) -> None:
    """`available` is not bundled-off: the harness does not name a plugin it was not given."""
    from localharness.plugins import resolve
    manifest = PluginManifest(name="extplug", version="0.1.0", kind="tools",
                              cli=(CliDescriptor(name="extcmd", help="x", target="ext_mod:app"),))
    plan = LoadPlan((PlanEntry("extplug", False, "available", "pip", "x", manifest=manifest),), (), None)
    monkeypatch.setattr(resolve, "resolve", lambda *a, **k: type("R", (), {"plan": plan})())
    assert plugin_mount._resolve_commands() == ({}, {})
    ran = _run("extcmd")
    assert ran.exit_code == 2 and "No such command 'extcmd'" in ran.output, ran.output


# ------------------------------------------------------------ a fresh process: imports and the real $?


def _env(tmp_path: Path, off: str) -> dict[str, str]:
    home = tmp_path / "home"
    lh = home / ".localharness"
    lh.mkdir(parents=True)
    (lh / "config.yaml").write_text("version: '1'\nprovider:\n  provider_type: vllm\n  base_url: http://localhost:8000/v1\n"
                                       "  default_model: test-model\n" + off, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith("LOCALHARNESS_")}
    env.update(HOME=str(home), LOCALHARNESS_HOME=str(lh), COLUMNS="100")
    return env


def test_the_stub_imports_no_command_module(tmp_path) -> None:
    code = ("import sys\nfrom localharness.cli.app import app\n"
            "for args in (['mobile'], ['memory', 'list'], ['generate-image', 'x']):\n"
            "    try:\n        app(args, standalone_mode=True)\n    except SystemExit as e:\n"
            "        print(e.code)\n"
            f"print(sorted(m for m in {CMD_MODULES!r} if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
                         env=_env(tmp_path, "mobile: {enabled: false}\nmemory: {enabled: false}\n"),
                         cwd=tmp_path)
    assert out.stdout.split("\n")[:4] == ["4", "4", "4", "[]"], (out.stdout, out.stderr)


LOCALHARNESS = Path(sys.executable).parent / "localharness"


@pytest.mark.skipif(not LOCALHARNESS.exists(), reason="no installed `localharness` entry point")
def test_the_shell_sees_exit_4_for_an_off_plugin_and_2_for_an_unknown_name(tmp_path) -> None:
    env = _env(tmp_path, "mobile: {enabled: false}\n")

    def sh(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([str(LOCALHARNESS), *args], capture_output=True, text=True, timeout=60,
                              env=env, cwd=tmp_path)

    off, bogus = sh("mobile"), sh("bogus")
    assert off.returncode == 4 and _hint("mobile", "mobile") in off.stderr, (off.returncode, off.stderr)
    assert bogus.returncode == 2 and "No such command 'bogus'" in bogus.stderr, bogus.stderr


@pytest.mark.skipif(not LOCALHARNESS.exists(), reason="no installed `localharness` entry point")
def test_with_web_on_the_shell_never_meets_the_stub(tmp_path) -> None:
    on = subprocess.run([str(LOCALHARNESS), "mobile", "--help"], capture_output=True, text=True,
                        timeout=60, env=_env(tmp_path, ""), cwd=tmp_path)
    assert on.returncode == 0 and "is provided by" not in on.stderr, (on.returncode, on.stderr)
