"""Plugin CLI commands mount LAZILY (PAPI-06, criterion 1): an ON plugin's commands are in
`localharness --help` from its manifest without importing the module a command runs; that module is
imported only when the command runs; an off plugin's commands are absent; a core command always
wins a name; and dispatching a core command (`localharness start`) never resolves plugins at all.

The root group resolves plugins from the DEFAULT config dir (the root has no --config-dir), so each
test writes `components_home`'s overrides.yaml. `swatch` is a bundled test plugin whose command
module `lh_test_cli_mod` is written to tmp and put on sys.path; it is removed from sys.modules
before every import-state assertion. The last tests drive the REAL installed example plugin.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest
from typer.core import TyperGroup
from typer.testing import CliRunner

from localharness.cli.app import app
from localharness.plugins import builtin, discovery
from localharness.plugins.api import CliDescriptor, Plugin, PluginManifest

runner = CliRunner()
_REAL_DISCOVER = discovery.discover
_MOD = "lh_test_cli_mod"


def _swatch(name: str = "swatchcmd", help: str = "Do the thing.", target: str = f"{_MOD}:app",
            plugin: str = "swatch", on: bool = True) -> type[Plugin]:
    class Swatch(Plugin):
        """draws swatches"""

        manifest = PluginManifest(name=plugin, version="0.1.0", kind="tools", enabled_by_default=on,
                                  cli=(CliDescriptor(name=name, help=help, target=target),))
    return Swatch


@pytest.fixture(autouse=True)
def mounted(tmp_path: Path, monkeypatch, components_home):
    """The command module on sys.path (not imported); discovery = folders only; wide help."""
    mods = tmp_path / "mods"
    mods.mkdir()
    (mods / f"{_MOD}.py").write_text(textwrap.dedent('''\
        import typer

        app = typer.Typer()


        @app.command()
        def run(flag: str = typer.Option("", "--flag", help="A flag for the thing."),
                fail: bool = typer.Option(False, "--fail"), crash: bool = typer.Option(False, "--crash")):
            """Do the thing, for real."""
            if fail:
                raise typer.Exit(3)
            if crash:
                raise RuntimeError("swatch server on fire")
            typer.echo(f"swatchcmd ran with flag={flag}")
        '''), encoding="utf-8")
    monkeypatch.syspath_prepend(str(mods))
    monkeypatch.setattr(discovery, "discover", lambda global_config_dir: [
        f for f in _REAL_DISCOVER(global_config_dir) if f.source == "folder"])
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (_swatch(),))
    monkeypatch.setenv("COLUMNS", "400")
    sys.modules.pop(_MOD, None)
    yield components_home
    sys.modules.pop(_MOD, None)


def _run(*args: str):
    return runner.invoke(app, list(args))


def _help_rows(output: str) -> dict[str, str]:
    """{command: its one line} from the root --help panel."""
    rows = {}
    for line in output.splitlines():
        cells = line.strip("│ ").split(None, 1)
        if len(cells) == 2 and line.startswith("│"):
            rows[cells[0]] = cells[1].strip()
    return rows


# --------------------------------------------------------------------------- listed, not imported


def test_with_no_plugin_on_the_help_is_exactly_the_core_help(monkeypatch) -> None:
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", ())
    with_group = _run("--help")
    monkeypatch.setattr(app.info, "cls", TyperGroup)
    core_only = _run("--help")

    assert with_group.exit_code == core_only.exit_code == 0
    assert with_group.output == core_only.output
    assert "start" in _help_rows(core_only.output)


def test_an_on_plugins_command_is_listed_from_its_manifest_without_importing_it() -> None:
    result = _run("--help")

    assert result.exit_code == 0, result.output
    assert _help_rows(result.output)["swatchcmd"] == "Do the thing."
    assert _MOD not in sys.modules


def test_running_it_imports_its_module_and_hands_it_the_arguments() -> None:
    result = _run("swatchcmd", "--flag", "x")

    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == "swatchcmd ran with flag=x"
    assert _MOD in sys.modules


def test_its_own_help_usage_and_exit_code_are_its_own() -> None:
    helped = _run("swatchcmd", "--help")
    failed = _run("swatchcmd", "--fail")
    usage = _run("swatchcmd", "--nope")

    assert helped.exit_code == 0 and "Usage: localharness swatchcmd [OPTIONS]" in helped.output
    assert "A flag for the thing." in helped.output
    assert failed.exit_code == 3, failed.output
    assert usage.exit_code == 2 and "No such option '--nope'" in usage.output  # Click 8.4's words
    assert usage.output.startswith("Usage: localharness swatchcmd [OPTIONS]")


def test_an_off_plugins_command_is_absent(mounted) -> None:
    (mounted / "overrides.yaml").write_text("swatch: {enabled: false}\n", encoding="utf-8")

    helped, ran = _run("--help"), _run("swatchcmd")

    assert "swatchcmd" not in _help_rows(helped.output)
    assert ran.exit_code == 2 and "No such command 'swatchcmd'" in ran.output
    assert _MOD not in sys.modules


def test_a_plugin_help_line_is_shown_as_written(monkeypatch) -> None:
    """Rich markup in a plugin's text is escaped: `[the]` would vanish and `[/]` would crash --help."""
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (_swatch(help="Do [the] thing [/] now."),))

    result = _run("--help")

    assert result.exit_code == 0, result.output
    assert _help_rows(result.output)["swatchcmd"] == "Do [the] thing [/] now."


# --------------------------------------------------------------------------- contained


def test_a_command_that_cannot_be_imported_exits_1_naming_plugin_and_command(monkeypatch) -> None:
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (_swatch(target="lh_test_no_such_mod:app"),))

    result = _run("swatchcmd")

    assert result.exit_code == 1, result.output
    assert "plugin swatch: command swatchcmd could not be imported: ModuleNotFoundError" in result.stderr


def test_a_command_that_raises_exits_1_naming_plugin_and_command() -> None:
    result = _run("swatchcmd", "--crash")

    assert result.exit_code == 1, result.output
    assert ("plugin swatch: command swatchcmd raised RuntimeError: swatch server on fire"
            in result.stderr)


def test_plugins_that_cannot_be_resolved_leave_the_core_cli_working(monkeypatch) -> None:
    def broken(loader, **kwargs):
        raise RuntimeError("resolver down")

    monkeypatch.setattr("localharness.plugins.resolve.resolve", broken)

    result = _run("--help")

    assert result.exit_code == 0, result.output
    assert "start" in _help_rows(result.output) and "swatchcmd" not in _help_rows(result.output)


# --------------------------------------------------------------------------- names


def test_a_core_command_always_wins_its_name(monkeypatch) -> None:
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (_swatch(name="doctor", help="Not the doctor."),))

    helped, ran = _run("--help"), _run("doctor", "--help")

    assert [line for line in helped.output.splitlines() if "doctor" in line.split()[:2]] and \
        _help_rows(helped.output)["doctor"].startswith("Run prerequisite checks")
    assert "Not the doctor." not in helped.output
    assert "Run prerequisite checks" in ran.output
    assert _MOD not in sys.modules


def test_two_plugins_with_one_command_name_the_first_in_order_wins(monkeypatch, tmp_path) -> None:
    other = tmp_path / "mods" / "lh_test_cli_other.py"
    other.write_text("import typer\napp = typer.Typer()\n\n@app.command()\ndef run():\n"
                     "    typer.echo('the second plugin')\n", encoding="utf-8")
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (
        _swatch(), _swatch(plugin="swatch2", help="The other one.", target="lh_test_cli_other:app")))

    helped, ran = _run("--help"), _run("swatchcmd", "--flag", "y")

    assert _help_rows(helped.output)["swatchcmd"] == "Do the thing."
    assert ran.stdout.strip() == "swatchcmd ran with flag=y"
    sys.modules.pop("lh_test_cli_other", None)


# --------------------------------------------------------------------------- the cost


@pytest.fixture
def resolves(monkeypatch) -> list[object]:
    """Every call of the resolver, recorded, then passed through."""
    import localharness.plugins.resolve as resolve_mod

    calls: list[object] = []
    real = resolve_mod.resolve

    def spy(loader, **kwargs):
        calls.append(loader)
        return real(loader, **kwargs)

    monkeypatch.setattr(resolve_mod, "resolve", spy)
    return calls


@pytest.mark.parametrize("argv", [["start", "--help"], ["doctor", "--help"], ["plugins", "--help"],
                                  ["--version"]])
def test_dispatching_a_core_command_never_resolves_plugins(resolves, argv) -> None:
    result = _run(*argv)

    assert result.exit_code == 0, result.output
    assert resolves == []


def test_the_command_list_resolves_once(resolves) -> None:
    """The positive control for the test above: the spy sees what the mount does."""
    assert _run("--help").exit_code == 0
    assert len(resolves) == 1


# --------------------------------------------------------------------------- the real example plugin


@pytest.fixture
def example(mounted, tmp_path, monkeypatch):
    """The unmocked entry point; its package writes the sentinel when imported."""
    monkeypatch.setattr(discovery, "discover", _REAL_DISCOVER)
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", ())
    sentinel = tmp_path / "example-imported"
    monkeypatch.setenv("LOCALHARNESS_EXAMPLE_PLUGIN_SENTINEL", str(sentinel))

    def purge() -> None:
        for m in [m for m in sys.modules if m.startswith("localharness_plugin_example")]:
            del sys.modules[m]

    purge()
    yield sentinel
    purge()


def test_the_example_plugin_available_contributes_no_command_and_is_not_imported(example) -> None:
    result = _run("--help")

    assert result.exit_code == 0 and "example" not in _help_rows(result.output)
    assert not example.exists()


def test_the_example_plugin_on_mounts_its_command_lazily(example, mounted) -> None:
    (mounted / "overrides.yaml").write_text("example: {enabled: true}\n", encoding="utf-8")

    started = _run("start", "--help")
    assert started.exit_code == 0 and not example.exists(), "start must not import any plugin"

    helped = _run("--help")
    assert _help_rows(helped.output)["example"] == "Show what the example plugin does."
    assert "localharness_plugin_example.cli" not in sys.modules  # listed from the manifest alone

    ran = _run("example")
    assert ran.exit_code == 0, ran.output
    assert ran.stdout.startswith("example plugin 0.1.0:")
    assert "localharness_plugin_example.cli" in sys.modules
