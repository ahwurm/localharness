"""`localharness memory` is the memory plugin's CliDescriptor (MEMP-03, PAPI-06): listed from the
manifest, imported only when run, absent when memory is off in the global config. Through the root
`app` and a fresh interpreter; the byte-level behaviour is test_memory_cli_golden.py's."""
from __future__ import annotations

import os
import subprocess
import sys

from typer.testing import CliRunner

from localharness.cli.app import app
from localharness.memory.plugin import MemoryPlugin
from tests.integration.test_image_plugin_e2e import _machine
from tests.unit.test_plugin_cli_mount import _help_rows

runner = CliRunner()
HELP = "Browse and edit the agent's persistent memory (list / show / edit / rm / archive / restore)."


def _invoke(*args: str):
    result = runner.invoke(app, list(args), env={"COLUMNS": "200"})
    assert result.exception is None or isinstance(result.exception, SystemExit), (
        f"`localharness {' '.join(args)}` raised {result.exception!r}\n{result.output}")
    return result


def test_memory_command_mounts_from_the_manifest(tmp_path, monkeypatch, fake_home):
    _machine(tmp_path, monkeypatch, fake_home)
    (desc,) = MemoryPlugin.manifest.cli
    assert (desc.name, desc.help, desc.target) == ("memory", HELP, "localharness.cli.memory_cli:memory_app")
    # typer freezes TERMINAL_WIDTH at first import (rich_utils.MAX_WIDTH), so the row may wrap:
    # compare the help text with the panel's borders and wrapping removed.
    out = _invoke("--help").output
    assert "memory" in _help_rows(out), out
    assert HELP in " ".join(out.replace("│", " ").split()), out
    bare = _invoke("memory")  # no_args_is_help kept
    assert bare.exit_code == 2 and "list" in bare.output and "restore" in bare.output, bare.output


def test_memory_command_absent_with_memory_off(tmp_path, monkeypatch, fake_home):
    global_dir, _ = _machine(tmp_path, monkeypatch, fake_home)
    with (global_dir / "config.yaml").open("a", encoding="utf-8") as f:
        f.write("memory:\n  enabled: false\n")
    assert "memory" not in _help_rows(_invoke("--help").output)
    gone = _invoke("memory", "list")
    # off bundled plugin: hint + exit 4, not Click's exit 2 (exit 2 is experiment run's reject-holdout verdict)
    assert gone.exit_code == 4 and "is provided by the memory plugin, which is off" in gone.stderr, gone.output


def test_memory_help_usage_names_localharness_memory(tmp_path, monkeypatch, fake_home):
    _machine(tmp_path, monkeypatch, fake_home)
    helped = _invoke("memory", "--help")
    assert helped.exit_code == 0, helped.output
    assert "localharness memory" in helped.output and "--install-completion" not in helped.output


def test_root_help_does_not_import_the_memory_cli(tmp_path):
    """Lazy mount: `localharness --help` lists `memory` without importing cli/memory_cli.py."""
    code = ("import sys\nfrom localharness.cli.app import app\n"
            "try:\n    app(['--help'], standalone_mode=False)\nexcept SystemExit:\n    pass\n"
            "print('LOADED' if 'localharness.cli.memory_cli' in sys.modules else 'LAZY', file=sys.stderr)")
    home = tmp_path / "home"
    home.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith("LOCALHARNESS_")}
    env.update(HOME=str(home), COLUMNS="200")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
                         env=env, cwd=tmp_path)
    assert out.returncode == 0, out.stderr
    assert "memory" in _help_rows(out.stdout), out.stdout  # listed, so the mount was asked
    assert out.stderr.strip().splitlines()[-1] == "LAZY", out.stderr
