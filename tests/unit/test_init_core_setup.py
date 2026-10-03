"""init's core setup: one interactive gate in front of every question, `--no-input`, and an
existing config kept or changed through overrides.yaml — never rewritten without `--force`.

Owner ruling 2026-10-03: init sets up the core (find or name the model server, write the config)
and asks nothing about plugins. A terminal is simulated by swapping init_cmd's OWN `sys` — CliRunner
reassigns the real sys.stdin while the command runs, so patching its isatty does not survive the
invocation. A question that must not be asked is booby-trapped (its `ask` raises), so "asked
nothing" is proven, not inferred from a default answer.
"""
from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from rich.console import Console
from typer.testing import CliRunner

import localharness.cli.init_cmd as init_cmd
from localharness.cli.app import app
from localharness.config.loader import ConfigLoader
from tests.unit.test_init_cmd import _make_capability_result, _make_detector_result

runner = CliRunner()
OLD_CONFIG = ("version: '1'\nprovider:\n  provider_type: ollama\n  base_url: http://original/v1\n"
              "  default_model: old-model\n")


class _Asked:
    """A prompt that must not be asked: answering it fails the run."""

    @staticmethod
    def ask(*_a, **_k):
        raise AssertionError("asked")


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch):
    """No real HTTP: the served-window, runtime and Ollama hot-model probes are stubbed (the
    test_init_cmd pattern); detection and the capability probe are fakes each test points."""
    monkeypatch.setattr(init_cmd, "_detect_max_model_len", lambda *_: None)
    monkeypatch.setattr(init_cmd, "_identify_endpoint_provider", lambda *_: "unknown")
    monkeypatch.setattr(init_cmd, "_get_ollama_hot_model", lambda *_: None)


def _detect(monkeypatch, *models: str, found: bool = True) -> AsyncMock:
    """Detection finds `models` (default one) at http://localhost:11434; the probe answers native."""
    detect = AsyncMock(return_value=_make_detector_result(found=found, models=list(models) or None))
    monkeypatch.setattr(init_cmd, "detect_provider", detect)
    client = MagicMock()
    client.detect_capabilities = AsyncMock(return_value=_make_capability_result())
    monkeypatch.setattr(init_cmd, "LLMClient", MagicMock(return_value=client))
    return detect


def _terminal(monkeypatch, on: bool = True) -> None:
    fake_sys = MagicMock()
    fake_sys.stdin.isatty.return_value = on
    monkeypatch.setattr(init_cmd, "sys", fake_sys)


def _answers(monkeypatch, *confirms: bool) -> MagicMock:
    """Confirm answers in order (one more question than scripted fails the run); IntPrompt and
    Prompt must not be asked."""
    confirm = MagicMock()
    confirm.ask.side_effect = list(confirms)
    monkeypatch.setattr(init_cmd, "Confirm", confirm)
    monkeypatch.setattr(init_cmd, "IntPrompt", _Asked)
    monkeypatch.setattr(init_cmd, "Prompt", _Asked)
    return confirm


def _silent(monkeypatch) -> None:
    for name in ("Confirm", "IntPrompt", "Prompt"):
        monkeypatch.setattr(init_cmd, name, _Asked)


def _init(g: Path, *flags: str):
    return runner.invoke(app, ["init", "--config-dir", str(g), *flags])


def _flat(result) -> str:
    """Rich wraps at 80 columns — normalize whitespace so a phrase survives the wrap."""
    return " ".join((result.output or "").split())


def _exited(result, code: int) -> None:
    """A clean exit with `code` — not a booby-trapped question or a traceback."""
    assert result.exit_code == code, result.output
    assert result.exception is None or isinstance(result.exception, SystemExit), repr(result.exception)


def _seed(g: Path, text: str = OLD_CONFIG) -> Path:
    cfg = g / "config.yaml"
    cfg.write_text(text, encoding="utf-8")
    return cfg


def _overlay(g: Path) -> dict:
    return yaml.safe_load((g / "overrides.yaml").read_text(encoding="utf-8"))


# ------------------------------------------------------------------ the one gate


def test_is_interactive_needs_a_terminal_and_no_no_input(monkeypatch):
    _terminal(monkeypatch, True)
    assert init_cmd.is_interactive(False) is True
    assert init_cmd.is_interactive(True) is False
    _terminal(monkeypatch, False)
    assert init_cmd.is_interactive(False) is False
    detached = MagicMock()
    detached.stdin = None
    monkeypatch.setattr(init_cmd, "sys", detached)
    assert init_cmd.is_interactive(False) is False


def test_no_input_asks_nothing_even_on_a_terminal(tmp_path, monkeypatch):
    _detect(monkeypatch)
    _terminal(monkeypatch)
    _silent(monkeypatch)

    result = _init(tmp_path, "--no-input")

    _exited(result, 0)
    cfg = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["provider"]["default_model"] == "test-model:7b"
    assert "memory" not in cfg and cfg["org"]["permissions"].get("mode") != "read-only"


@pytest.mark.parametrize("flags, picked", [([], "model-a"), (["--model", "model-b"], "model-b")],
                         ids=["first", "named"])
def test_several_models_without_a_question_take_the_first_or_the_named_one(flags, picked, tmp_path,
                                                                           monkeypatch):
    _detect(monkeypatch, "model-a", "model-b")
    _terminal(monkeypatch)
    _silent(monkeypatch)

    result = _init(tmp_path, "--no-input", *flags)

    _exited(result, 0)
    cfg = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert (cfg["provider"]["default_model"], cfg["org"]["default_model"]) == (picked, picked)
    flat = _flat(result)
    assert ("the first of 2 served" in flat) is (picked == "model-a"), flat
    if picked == "model-a":
        assert "localharness init --endpoint http://localhost:11434 --model <name>" in flat, flat


def test_the_quick_path_is_one_question(tmp_path, monkeypatch):
    _detect(monkeypatch)
    _terminal(monkeypatch)
    confirm = _answers(monkeypatch, True)

    result = _init(tmp_path)

    _exited(result, 0)
    assert confirm.ask.call_count == 1
    assert confirm.ask.call_args_list[0].args[0].strip().startswith("Keep the usual settings")
    cfg = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert "memory" not in cfg and cfg["org"]["permissions"].get("mode") != "read-only"


def test_no_to_the_usual_settings_asks_the_two_posture_questions(tmp_path, monkeypatch):
    _detect(monkeypatch)
    _terminal(monkeypatch)
    confirm = _answers(monkeypatch, False, False, True)  # usual: no; host tools: no; memory: yes

    result = _init(tmp_path)

    _exited(result, 0)
    asked = [c.args[0] for c in confirm.ask.call_args_list]
    assert len(asked) == 3, asked
    assert "write/edit files" in asked[1] and "persistent memory" in asked[2], asked
    cfg = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["org"]["permissions"]["mode"] == "read-only" and "memory" not in cfg


@pytest.mark.parametrize("starting", [True, False], ids=["starting", "init"])
def test_a_starting_setup_leaves_out_the_run_start_lines(starting, tmp_path, monkeypatch):
    _detect(monkeypatch)
    _silent(monkeypatch)
    buf = io.StringIO()
    monkeypatch.setattr(init_cmd, "console", Console(file=buf, width=400))
    monkeypatch.setattr(init_cmd, "err_console", Console(file=buf, width=400))

    out = init_cmd.core_setup(str(tmp_path), endpoint=None, model=None, force=False,
                              interactive=False, starting=starting)

    assert out == init_cmd.SetupResult(tmp_path / "config.yaml", True)
    printed = buf.getvalue()
    assert "LocalHarness configured at" in printed
    assert ("Run 'localharness start' to begin." in printed) is (not starting), printed
    assert ("★" in printed) is (not starting), printed


# ------------------------------------------------------------------ an existing config


def test_an_existing_config_is_kept_without_a_terminal(tmp_path, monkeypatch):
    detect = _detect(monkeypatch)
    _silent(monkeypatch)
    cfg = _seed(tmp_path)

    result = _init(tmp_path)

    _exited(result, 0)
    flat = _flat(result)
    assert f"✓ Kept {cfg}." in flat, flat
    assert "To start over: localharness init --force" in flat
    assert cfg.read_bytes() == OLD_CONFIG.encode()
    assert not (tmp_path / "overrides.yaml").exists()
    detect.assert_not_called()


def test_a_terminal_asks_to_keep_the_config_and_yes_keeps_it(tmp_path, monkeypatch):
    detect = _detect(monkeypatch)
    _terminal(monkeypatch)
    confirm = _answers(monkeypatch, True)
    cfg = _seed(tmp_path)

    result = _init(tmp_path)

    _exited(result, 0)
    assert [c.args[0] for c in confirm.ask.call_args_list] == [
        "Config exists: old-model at http://original/v1. Keep it?"]
    assert confirm.ask.call_args.kwargs == {"default": True}
    assert f"✓ Kept {cfg}." in _flat(result)
    assert cfg.read_bytes() == OLD_CONFIG.encode()
    detect.assert_not_called()


def test_no_to_keep_changes_the_server_through_overrides_and_leaves_config_yaml_alone(tmp_path,
                                                                                    monkeypatch):
    _detect(monkeypatch, "new-model")
    _terminal(monkeypatch)
    confirm = _answers(monkeypatch, False)  # Keep it? no — and nothing else is asked on a change
    cfg = _seed(tmp_path)

    result = _init(tmp_path)

    _exited(result, 0)
    assert cfg.read_bytes() == OLD_CONFIG.encode()
    assert _overlay(tmp_path) == {
        "provider": {"provider_type": "ollama", "base_url": "http://localhost:11434",
                     "default_model": "new-model", "available_models": ["new-model"],
                     "supports_function_calling": True},
        "org": {"default_model": "new-model"},
    }
    assert f"✓ Updated {tmp_path / 'overrides.yaml'} — your config.yaml is unchanged." in _flat(result)
    assert confirm.ask.call_count == 1
    # the change is what the next start reads: config.yaml folded with overrides.yaml
    provider = ConfigLoader(config_dir=tmp_path).load_harness().provider
    assert (provider.default_model, provider.base_url) == ("new-model", "http://localhost:11434")


def test_a_change_that_does_not_validate_writes_nothing(tmp_path, monkeypatch):
    _detect(monkeypatch, "new-model")
    _terminal(monkeypatch)
    _answers(monkeypatch, False)
    text = OLD_CONFIG + "proposer:\n  base_url: http://proposer/v1\n  model: new-model\n"
    cfg = _seed(tmp_path, text)

    result = _init(tmp_path)

    _exited(result, 1)
    flat = _flat(result)
    assert "do not validate" in flat and "Nothing was written." in flat, flat
    assert not (tmp_path / "overrides.yaml").exists()
    assert cfg.read_text(encoding="utf-8") == text


@pytest.mark.parametrize("text, reason", [
    ("version: '1'\nprovider: [\n", "while parsing"),
    ("version: '1'\nold: true\n", "provider: Field required"),
], ids=["unparseable", "invalid"])
def test_a_config_that_cannot_be_read_is_named_and_nothing_is_written(text, reason, tmp_path,
                                                                     monkeypatch):
    detect = _detect(monkeypatch)
    _silent(monkeypatch)
    cfg = _seed(tmp_path, text)
    before = sorted(p.name for p in tmp_path.iterdir())

    result = _init(tmp_path)

    _exited(result, 1)
    flat = _flat(result)
    assert f"{cfg} exists but cannot be read" in flat and reason in flat, flat
    assert "localharness init --force" in flat
    assert sorted(p.name for p in tmp_path.iterdir()) == before
    assert cfg.read_text(encoding="utf-8") == text
    detect.assert_not_called()


@pytest.mark.parametrize("flags, url, model", [
    (["--endpoint", "http://localhost:9999/v1", "--model", "custom-model"],
     "http://localhost:9999/v1", "custom-model"),
    (["--model", "new-model"], "http://localhost:11434", "new-model"),
], ids=["endpoint", "model"])
@pytest.mark.parametrize("tty", [True, False], ids=["terminal", "no-terminal"])
def test_an_explicit_endpoint_or_model_on_an_existing_config_is_a_change_not_a_question(
        tty, flags, url, model, tmp_path, monkeypatch):
    _detect(monkeypatch, "test-model:7b", "new-model")
    _terminal(monkeypatch, tty)
    _silent(monkeypatch)
    cfg = _seed(tmp_path)

    result = _init(tmp_path, *flags)

    _exited(result, 0)
    assert cfg.read_bytes() == OLD_CONFIG.encode()
    overlay = _overlay(tmp_path)
    assert (overlay["provider"]["base_url"], overlay["provider"]["default_model"]) == (url, model)
    assert overlay["org"] == {"default_model": model}
    assert "your config.yaml is unchanged" in _flat(result)


def test_a_change_with_no_server_found_keeps_the_config_and_exits_1(tmp_path, monkeypatch):
    _detect(monkeypatch, found=False)
    _terminal(monkeypatch)
    confirm = _answers(monkeypatch, False)  # Keep it? no

    def _no_guided_setup(*_a, **_k):
        raise AssertionError("guided setup on a change")

    monkeypatch.setattr(init_cmd, "_guided_setup", _no_guided_setup)
    cfg = _seed(tmp_path)

    result = _init(tmp_path)

    _exited(result, 1)
    flat = _flat(result)
    assert "No local LLM detected" in flat and "Checked:" in flat, flat
    assert f"Nothing changed: {cfg} is kept." in flat, flat
    assert cfg.read_bytes() == OLD_CONFIG.encode()
    assert not (tmp_path / "overrides.yaml").exists()
    assert confirm.ask.call_count == 1


def test_force_still_regenerates_config_yaml(tmp_path, monkeypatch):
    _detect(monkeypatch)
    _silent(monkeypatch)
    cfg = _seed(tmp_path)

    result = _init(tmp_path, "--force")

    _exited(result, 0)
    text = cfg.read_text(encoding="utf-8")
    assert "old-model" not in text and "http://original/v1" not in text and "test-model:7b" in text
    assert not (tmp_path / "overrides.yaml").exists()


# ------------------------------------------------------------------ no server: skip for now


def _no_server_on_a_terminal(monkeypatch, *answers: str) -> tuple[MagicMock, MagicMock]:
    """Detection finds nothing, a terminal is attached, the guided vLLM offer is declined, and
    Prompt answers `answers` in order. Returns (Prompt, LLMClient) — the probe must not run."""
    _detect(monkeypatch, found=False)
    client_cls = MagicMock()
    monkeypatch.setattr(init_cmd, "LLMClient", client_cls)
    _terminal(monkeypatch)
    _answers(monkeypatch, False)  # "Set up vLLM and a model now?" — no
    prompt = MagicMock()
    prompt.ask.side_effect = list(answers)
    monkeypatch.setattr(init_cmd, "Prompt", prompt)
    return prompt, client_cls


def test_skip_for_now_saves_the_address_and_the_model_unchecked(tmp_path, monkeypatch):
    prompt, client_cls = _no_server_on_a_terminal(monkeypatch, "http://localhost:8081/v1/", "my-model")

    result = _init(tmp_path)

    _exited(result, 0)
    cfg = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert (cfg["provider"]["base_url"], cfg["provider"]["provider_type"],
            cfg["provider"]["default_model"]) == ("http://localhost:8081/v1", "unknown", "my-model")
    assert cfg["org"]["default_model"] == "my-model"
    flat = _flat(result)
    assert "Skip for now: give the address and model you will use" in flat, flat
    assert "not checked yet" in flat and "then run `localharness start`" in flat, flat
    assert [c.args[0] for c in prompt.ask.call_args_list] == [
        "Model server address", "Model name (Enter to skip)"]
    assert prompt.ask.call_args_list[0].kwargs == {"default": "http://localhost:8081/v1"}
    client_cls.assert_not_called()  # saved without checking: no capability probe
    # the saved config is one the next start can read
    assert ConfigLoader(config_dir=tmp_path).load_harness().provider.default_model == "my-model"


def test_skip_for_now_reports_no_server_ready(tmp_path, monkeypatch):
    _no_server_on_a_terminal(monkeypatch, "http://localhost:8081/v1", "my-model")

    out = init_cmd.core_setup(str(tmp_path), endpoint=None, model=None, force=False, interactive=True)

    assert out == init_cmd.SetupResult(tmp_path / "config.yaml", False)


def test_skip_for_now_without_a_model_saves_nothing_and_says_what_to_run(tmp_path, monkeypatch):
    _no_server_on_a_terminal(monkeypatch, "http://localhost:8081/v1", "")

    result = _init(tmp_path)

    _exited(result, 0)
    assert not (tmp_path / "config.yaml").exists()
    assert ("Nothing saved. When a model server answers at http://localhost:8081/v1, run "
            "`localharness init` again.") in _flat(result)


def test_skip_for_now_with_model_flag_asks_only_the_address(tmp_path, monkeypatch):
    prompt, _ = _no_server_on_a_terminal(monkeypatch, "http://10.0.0.5:8000/v1")

    result = _init(tmp_path, "--model", "given-model")

    _exited(result, 0)
    assert prompt.ask.call_count == 1
    cfg = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert (cfg["provider"]["base_url"], cfg["provider"]["default_model"]) == (
        "http://10.0.0.5:8000/v1", "given-model")


def test_skip_for_now_refuses_an_address_without_http(tmp_path, monkeypatch):
    prompt, _ = _no_server_on_a_terminal(monkeypatch, "localhost:8081", "my-model")

    result = _init(tmp_path)

    _exited(result, 0)
    assert not (tmp_path / "config.yaml").exists()
    assert "Nothing saved: the address must start with http:// or https://." in _flat(result)
    assert prompt.ask.call_count == 1  # refused before the model name is asked


def test_no_skip_without_a_terminal_no_server_still_exits_1(tmp_path, monkeypatch):
    _detect(monkeypatch, found=False)
    _silent(monkeypatch)

    result = _init(tmp_path)

    _exited(result, 1)
    assert "No local LLM detected" in _flat(result)
    assert not (tmp_path / "config.yaml").exists()
