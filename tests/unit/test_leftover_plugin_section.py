"""QA-06: after an uninstall, every everyday command refused a plugin's leftover settings with a
message that never said "plugin", and sometimes named the wrong file.

Release QA (flow G) uninstalled the example plugin with `example:` and `agent.example` still in
overrides.yaml. `start`, `doctor`, `components list` and `config show` refused with
`example (line 1): Extra inputs are not permitted`; `plugins list` said nothing; `plugins disable
example` said `Unknown plugin`; `validate` blamed the agent file for `agent.example`, which lives in
overrides.yaml. The refusal stays: a key that no core model and no installed plugin owns is still
refused, and that is the typo guard. The message now names the file, the line and the fix.

The venv's installed example plugin is "uninstalled" here: discovery sees plugin folders only and
nothing is bundled, so `example` belongs to no plugin.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from localharness.cli.app import app
from localharness.config.loader import ConfigLoader, ConfigValidationError
from localharness.plugins import builtin, discovery
from localharness.plugins.api import Plugin, PluginManifest

runner = CliRunner()
_REAL_DISCOVER = discovery.discover
_CONFIG = """\
version: '1'
provider:
  provider_type: vllm
  base_url: http://127.0.0.1:9/v1
  default_model: test-model
"""
# What `plugins enable example --set color=#112233` and `components set agent.example.size 16` leave.
_LEFTOVER = "example:\n  enabled: true\n  color: '#112233'\n"
_AGENT_LEFTOVER = "agent:\n  temperature: 0.5\n  example:\n    size: 16\n"  # agent.example: line 3
_TAIL = "if a plugin you removed used it, reinstall that plugin or delete"
_HINT = f"not a LocalHarness setting, and no installed plugin is named `example` — {_TAIL} this section"
_AGENT_FILE_HINT = f"not an agent setting, and no installed plugin is named `example` — {_TAIL} this section"
_OVERRIDES_AGENT_HINT = (f"not an agent setting, and no installed plugin is named `example` — {_TAIL} "
                         "`example:` under `agent:`")


class Example(Plugin):
    """the example plugin, installed again"""

    manifest = PluginManifest(name="example", version="0.1.0", kind="tools")


@pytest.fixture(autouse=True)
def uninstalled(monkeypatch):
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", ())
    monkeypatch.setattr(discovery, "discover", lambda global_config_dir: [
        f for f in _REAL_DISCOVER(global_config_dir) if f.source == "folder"])
    monkeypatch.setenv("COLUMNS", "400")  # rich wraps at the console width; paths must arrive whole


@pytest.fixture
def g(tmp_path: Path) -> Path:
    g = tmp_path / "g"
    g.mkdir()
    (g / "config.yaml").write_text(_CONFIG, encoding="utf-8")
    return g


def _agent_file(g: Path, extra: str = "") -> Path:
    path = g / "agents" / "helper.yaml"
    path.parent.mkdir(exist_ok=True)
    path.write_text("name: helper\nrole: helps\n" + extra, encoding="utf-8")
    return path


def _squash(text: str) -> str:
    return " ".join(text.split())


# --------------------------------------------------------------------------- harness level


@pytest.mark.parametrize("where", ["global overrides", "global config", "workspace config",
                                   "workspace overrides"])
def test_a_leftover_section_is_refused_naming_its_file_line_and_fix(g, tmp_path, where) -> None:
    ws = tmp_path / "proj" / ".localharness"
    ws.mkdir(parents=True)
    target = {"global overrides": g / "overrides.yaml", "global config": g / "config.yaml",
              "workspace config": ws / "config.yaml", "workspace overrides": ws / "overrides.yaml"}[where]
    before = target.read_text(encoding="utf-8") if target.exists() else ""
    target.write_text(before + _LEFTOVER, encoding="utf-8")
    line = len(before.splitlines()) + 1

    with pytest.raises(ConfigValidationError) as exc:
        ConfigLoader(config_dir=g, local_config_dir=ws).load_harness()

    assert exc.value.path == str(target)
    assert f"example (line {line}): {_HINT}" in str(exc.value), str(exc.value)


@pytest.mark.parametrize("section", ["org:\n  bogus: 1\n", "agent:\n  temperature: 0.5\n"])
def test_an_unknown_key_in_a_section_core_owns_keeps_pydantics_message(g, section) -> None:
    """Boundary pin: only a top-level key that neither core nor a plugin owns is a leftover. A bad
    key inside `org:`, and `agent:` in config.yaml (a core key, read from overrides.yaml only), keep
    pydantic's own text."""
    (g / "config.yaml").write_text(_CONFIG + section, encoding="utf-8")

    with pytest.raises(ConfigValidationError) as exc:
        ConfigLoader(config_dir=g).load_harness()

    assert "Extra inputs are not permitted" in str(exc.value)
    assert "no installed plugin" not in str(exc.value)


def test_an_installed_plugins_section_is_not_refused(g, monkeypatch) -> None:
    """Control: the same section with the plugin installed loads."""
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (Example,))
    (g / "overrides.yaml").write_text(_LEFTOVER + _AGENT_LEFTOVER, encoding="utf-8")
    loader = ConfigLoader(config_dir=g)

    loader.load_harness()
    loader.load_agent_file(_agent_file(g))

    assert loader.agent_plugin_sections("helper") == {"example": {"size": 16}}


# --------------------------------------------------------------------------- agent level


def test_a_leftover_under_agent_in_overrides_is_attributed_to_overrides(g) -> None:
    overrides = g / "overrides.yaml"
    overrides.write_text(_AGENT_LEFTOVER, encoding="utf-8")
    agent = _agent_file(g)

    with pytest.raises(ConfigValidationError) as exc:
        ConfigLoader(config_dir=g).load_agent_file(agent)

    (err,) = exc.value.errors
    assert exc.value.path == str(overrides)
    assert (err.field_path, err.yaml_line, err.source_path) == ("agent.example", 3, None)
    assert f"agent.example (line 3): {_OVERRIDES_AGENT_HINT}" in str(exc.value), str(exc.value)
    assert str(agent) not in str(exc.value), "the agent file does not hold the key"


def test_a_leftover_in_the_agent_file_is_the_agent_files(g) -> None:
    agent = _agent_file(g, "example:\n  size: 4\n")

    with pytest.raises(ConfigValidationError) as exc:
        ConfigLoader(config_dir=g).load_agent_file(agent)

    (err,) = exc.value.errors
    assert exc.value.path == str(agent)
    assert (err.field_path, err.yaml_line, err.source_path) == ("example", 3, None)
    assert f"example (line 3): {_AGENT_FILE_HINT}" in str(exc.value), str(exc.value)


def test_errors_in_both_files_keep_the_agent_file_header_and_name_overrides_on_its_line(g) -> None:
    overrides = g / "overrides.yaml"
    overrides.write_text(_AGENT_LEFTOVER, encoding="utf-8")
    agent = _agent_file(g, "rol: typo\n")

    with pytest.raises(ConfigValidationError) as exc:
        ConfigLoader(config_dir=g).load_agent_file(agent)

    by_field = {e.field_path: e for e in exc.value.errors}
    assert exc.value.path == str(agent)
    assert (by_field["rol"].source_path, by_field["rol"].yaml_line) == (None, 3)
    assert (by_field["agent.example"].source_path, by_field["agent.example"].yaml_line) == (
        str(overrides), 3)


# --------------------------------------------------------------------------- the commands


def test_start_refuses_a_leftover_before_any_model_listing(g, monkeypatch) -> None:
    (g / "overrides.yaml").write_text(_LEFTOVER, encoding="utf-8")

    def _never(*args, **kwargs):
        raise AssertionError("start reached a model listing before refusing the config")

    monkeypatch.setattr("localharness.cli.model_ops.list_live_models", _never)

    result = runner.invoke(app, ["start", "--no-input", "--config-dir", str(g)])

    assert result.exit_code == 1, result.output
    assert result.exception is None or isinstance(result.exception, SystemExit), repr(result.exception)
    out = _squash(result.output)
    assert f"Cannot load config: {g / 'overrides.yaml'}:" in out, out
    assert f"example (line 1): {_HINT}" in out, out


def test_validate_names_overrides_for_an_agent_leftover(g) -> None:
    (g / "overrides.yaml").write_text(_AGENT_LEFTOVER, encoding="utf-8")
    _agent_file(g)

    result = runner.invoke(app, ["validate", "--config-dir", str(g)])

    assert result.exit_code == 1, result.output
    row = _squash(result.stdout[result.stdout.index("helper.yaml"):])
    assert f"in {g / 'overrides.yaml'}" in row, row
    assert f"Line 3: agent.example: {_OVERRIDES_AGENT_HINT}" in row, row
