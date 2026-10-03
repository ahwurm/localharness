"""R16, extended to parse errors: a YAML file that does not parse is reported by file, line and
column, and never by the source snippet and caret the parser prints, which can show the head of a
key written on the broken line.

Each case breaks the `api_key:` line itself, and checks the key's first 8 characters (and the whole
key) on every surface that reports the file, plus that the line number is named. Explicit
--config-dir; nothing reads the real home; start stops before any model server (a listing raises)."""
from __future__ import annotations

import pytest
import yaml
from typer.testing import CliRunner

from localharness.cli.app import app

runner = CliRunner()
KEY = "sk-SENTINEL-parse-52-0123456789abcdefXYZ"
HEAD, TAIL = KEY[:8], KEY[-8:]  # the parser's snippet is centred on the error: either end can show
_TOP = ('version: "1"\nprovider:\n  provider_type: vllm\n  base_url: http://127.0.0.1:9/v1\n'
        "  default_model: test-model\n")
BROKEN = {  # the api_key line is where each file breaks
    "unclosed-quote": f'proposer:\n  base_url: http://p/v1\n  model: p2\n  api_key: "{KEY}\n',
    "stray-colon": f"proposer:\n  base_url: http://p/v1\n  model: p2\n  api_key: {KEY}: x\n",
}
SURFACES = {
    "doctor": ("doctor",),
    "validate": ("validate",),
    "start": ("start", "--no-input"),
    "components-list": ("components", "list"),
    "components-list-json": ("components", "list", "--json"),
    "model": ("model",),
    "config-show": ("config", "show"),
    "config-migrate": ("config", "migrate"),
}


@pytest.fixture(autouse=True)
def no_model_server(monkeypatch):
    def _never(*args, **kwargs):
        raise AssertionError("start reached a model listing before refusing the config")
    monkeypatch.setattr("localharness.cli.model_ops.list_live_models", _never)
    monkeypatch.setenv("COLUMNS", "400")


def _problem_line(text: str) -> int:
    try:
        yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return exc.problem_mark.line + 1
    raise AssertionError("the fixture parses")


def _context_line(text: str) -> int | None:
    try:
        yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return exc.context_mark.line + 1 if exc.context_mark else None
    raise AssertionError("the fixture parses")


def _clean(*texts: str) -> None:
    for text in texts:
        for piece in (KEY, HEAD, TAIL):
            assert piece not in text, f"{piece!r} reached:\n{text}"


@pytest.mark.parametrize("surface", list(SURFACES))
@pytest.mark.parametrize("case", list(BROKEN))
def test_a_config_that_does_not_parse_names_the_line_never_the_source(tmp_path, case, surface) -> None:
    g = tmp_path / "g"
    g.mkdir()
    text = _TOP + BROKEN[case]
    (g / "config.yaml").write_text(text, encoding="utf-8")
    result = runner.invoke(app, [*SURFACES[surface], "--config-dir", str(g)])

    assert result.exit_code != 0, result.output
    _clean(result.stdout, result.stderr, repr(result.exception))
    if surface in ("doctor", "validate"):
        line = _problem_line(text)
        flat = " ".join(result.output.split())
        assert f":{line}:" in flat or f"Line {line}:" in flat, flat
        started = _context_line(text)  # an unclosed quote fails at the file's end: say where it began
        if started is not None:
            assert f"at line {started}," in flat, flat


def test_an_overrides_file_that_does_not_parse_names_the_line_never_the_source(tmp_path) -> None:
    g = tmp_path / "g"
    g.mkdir()
    (g / "config.yaml").write_text(_TOP, encoding="utf-8")
    text = BROKEN["unclosed-quote"]
    (g / "overrides.yaml").write_text(text, encoding="utf-8")
    result = runner.invoke(app, ["doctor", "--config-dir", str(g)])

    _clean(result.stdout, result.stderr, repr(result.exception))
    assert f"overrides.yaml:{_problem_line(text)}:" in " ".join(result.output.split()), result.output


def test_an_agent_file_that_does_not_parse_names_the_line_never_the_source(tmp_path) -> None:
    g = tmp_path / "g"
    (g / "agents").mkdir(parents=True)
    (g / "config.yaml").write_text(_TOP, encoding="utf-8")
    text = f'name: helper\nrole: helps\nnotes: "{KEY}\n'
    (g / "agents" / "helper.yaml").write_text(text, encoding="utf-8")
    result = runner.invoke(app, ["agent", "list", "--config-dir", str(g)])

    _clean(result.stdout, result.stderr, repr(result.exception))
    assert f"helper.yaml:{_problem_line(text)}:" in " ".join(result.output.split()), result.output


def test_a_generated_agent_yaml_that_does_not_parse_names_the_line_never_the_source() -> None:
    from localharness.orchestrator.workflow import validate_agent_yaml

    text = f'name: helper\nrole: helps\nnotes: "{KEY}\n'
    why = validate_agent_yaml(text)
    assert why is not None and why.startswith("not valid YAML"), why
    _clean(why)
    assert f"line {_problem_line(text)}" in why, why
