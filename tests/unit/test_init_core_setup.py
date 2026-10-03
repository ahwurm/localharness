"""init's core setup: one interactive gate in front of every question, `--no-input`, and an
existing config kept or changed through overrides.yaml — never rewritten without `--force`.

Owner ruling 2026-10-03: init sets up the core (find or name the model server, write the config)
and asks nothing about plugins. A terminal is simulated by swapping init_cmd's OWN `sys` — CliRunner
reassigns the real sys.stdin while the command runs, so patching its isatty does not survive the
invocation. A question that must not be asked is booby-trapped (its `ask` raises), so "asked
nothing" is proven, not inferred from a default answer.
"""
from __future__ import annotations

import asyncio
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from rich.console import Console
from typer.testing import CliRunner

import localharness.cli.init_cmd as init_cmd
from localharness.cli.app import app
from localharness.cli.model_ops import persist_active_endpoint, persist_default_model
from localharness.config.loader import ConfigLoader
from localharness.config.overlay import atomic_write_overlay, load_overlay
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
    """The served window above ContextConfig's 2,000,000-token ceiling: the change would write it."""
    _detect(monkeypatch, "new-model")
    _terminal(monkeypatch)
    _answers(monkeypatch, False)
    monkeypatch.setattr(init_cmd, "_served_window", lambda result: 4_000_000)
    text = OLD_CONFIG
    cfg = _seed(tmp_path, text)

    result = _init(tmp_path)

    _exited(result, 1)
    flat = _flat(result)
    assert "do not validate" in flat and "Nothing was written." in flat, flat
    assert "org.context.max_context_tokens" in flat, flat
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


# ------------------------------------------------------------------ the plugin list init ends with

HEADER = "Plugins — the command beside each one turns it on or sets it up:"
FOOTER = "In a session, /plugins enable <name> does this too and turns it on right away."
PLUGINS = ["image", "web", "memory", "dispatch", "autoresearch"]  # BUILTIN_PLUGINS order


def _plugin_block(result) -> list[str] | None:
    """The rows of the plugin list — which must be the LAST block of the output, after one blank
    line, and end with the in-session line — or None when init printed no list."""
    lines = (result.output or "").rstrip().splitlines()
    if HEADER not in lines:
        return None
    at = lines.index(HEADER)
    assert lines[at - 1] == "", "one blank line comes before the list"
    assert lines[-1] == FOOTER, "the list ends with the in-session line"
    return lines[at + 1:-1]


def test_init_ends_with_every_bundled_plugin_and_its_command(tmp_path, monkeypatch):
    from localharness.plugins import resolve as resolve_mod

    monkeypatch.setitem(resolve_mod.resolve.__kwdefaults__, "extra_installed", lambda extra: False)
    _detect(monkeypatch)
    _terminal(monkeypatch)
    _silent(monkeypatch)  # init asks nothing about plugins
    zed = tmp_path / "plugins" / "zed"  # a plugin found in the plugins/ folder: never listed here
    zed.mkdir(parents=True)
    (zed / "__init__.py").write_text("", encoding="utf-8")

    result = _init(tmp_path, "--no-input")

    _exited(result, 0)
    rows = _plugin_block(result)
    assert rows is not None, result.output
    assert [r.split()[0] for r in rows] == PLUGINS, rows
    image, web, memory, dispatch, autoresearch = rows
    assert image == "  image         off — turn on: localharness plugins enable image"
    assert "on — set up: localharness plugins enable memory" in memory
    assert "on — set up: localharness plugins enable autoresearch" in autoresearch
    assert ("(install `localharness[dispatch]` to use it) — set up: localharness plugins enable "
            "dispatch") in dispatch
    assert "(install `localharness[web]` to use it) — set up: localharness plugins enable web" in web
    out = result.output
    assert out.index("LocalHarness configured at") < out.index("★") < out.index(HEADER)
    assert init_cmd.PLUGINS_HEADER == HEADER


def test_a_kept_config_still_lists_the_plugins(tmp_path, monkeypatch):
    _detect(monkeypatch)
    _silent(monkeypatch)
    _seed(tmp_path)

    result = _init(tmp_path)

    _exited(result, 0)
    rows = _plugin_block(result)
    assert rows is not None and [r.split()[0] for r in rows] == PLUGINS, result.output
    assert result.output.index("Kept") < result.output.index(HEADER)


@pytest.mark.parametrize("outcome", ["changed", "skipped-with-a-model", "nothing-saved"])
def test_the_plugin_list_ends_every_outcome_that_leaves_a_config(outcome, tmp_path, monkeypatch):
    if outcome == "changed":
        _detect(monkeypatch, "new-model")
        _terminal(monkeypatch)
        _answers(monkeypatch, False)
        _seed(tmp_path)
    else:
        _no_server_on_a_terminal(
            monkeypatch, "http://localhost:8081/v1", "my-model" if outcome == "skipped-with-a-model" else "")

    result = _init(tmp_path)

    _exited(result, 0)
    rows = _plugin_block(result)
    if outcome == "nothing-saved":
        assert rows is None, result.output
    else:
        assert rows is not None and [r.split()[0] for r in rows] == PLUGINS, result.output


def test_a_plugin_list_failure_never_fails_init(tmp_path, monkeypatch):
    from localharness.plugins import resolve as resolve_mod

    def _broken(*_a, **_k):
        raise RuntimeError("a plugin problem")

    monkeypatch.setattr(resolve_mod, "resolve", _broken)
    _detect(monkeypatch)
    _silent(monkeypatch)

    result = _init(tmp_path)

    _exited(result, 0)
    assert (tmp_path / "config.yaml").exists()
    assert result.output.rstrip().splitlines()[-1] == "Plugins: run `localharness plugins list` to see them."
    assert HEADER not in result.output and FOOTER not in result.output


def test_the_capability_probe_closes_its_client_in_its_own_loop(monkeypatch):
    """Deferred item 3, the root of 52-04's gc.collect(): init's probe closes its LLMClient inside
    the probe's own event loop. Left open, the SDK finalizes it on whatever loop is running when the
    GC reaches it — a first start's session loop, where closing a transport of the probe's closed
    loop raised "Event loop is closed" on a real terminal."""
    from localharness.provider import client as client_mod

    seen: list[tuple[str, object]] = []

    async def detect(self):
        seen.append(("detect", asyncio.get_running_loop()))
        return _make_capability_result()

    async def aclose(self):
        seen.append(("aclose", asyncio.get_running_loop()))

    monkeypatch.setattr(client_mod.LLMClient, "detect_capabilities", detect)
    monkeypatch.setattr(client_mod.LLMClient, "aclose", aclose)
    assert init_cmd.LLMClient is client_mod.LLMClient  # the probe builds the real client

    cap = init_cmd._probe_capabilities(_make_detector_result(), "test-model:7b")

    assert cap.tool_call_mode == "native"
    assert [what for what, _ in seen] == ["detect", "aclose"]
    assert seen[0][1] is seen[1][1]  # closed in the very loop that probed, before asyncio.run returned


# ------------------------------------------------------------------ --force: a true core reset (R15)
#
# `init --force` regenerates config.yaml AND clears from the machine's overrides.yaml exactly the
# model and server choices the core persists there — a re-run's change, `/model`, a peer-endpoint
# switch. Every plugin section and every other key is the user's and stays. Without --force nothing
# leaves overrides.yaml.

PEER = SimpleNamespace(name="peer", base_url="http://peer:8000/v1", provider_type="vllm", api_key="none")
USER_KEYS = {"provider": {"timeout_seconds": 900.0}, "image": {"comfyui_url": "http://127.0.0.1:8188"}}


def _leaf_paths(d: dict, prefix: str = "") -> set[str]:
    out: set[str] = set()
    for key, value in d.items():
        path = f"{prefix}{key}"
        out |= _leaf_paths(value, path + ".") if isinstance(value, dict) and value else {path}
    return out


def _choices_saved_by_every_writer(g: Path, monkeypatch) -> Path:
    """A managed-server config whose choices were changed by all three writers that persist them to
    overrides.yaml — init's change (through the CLI), `/model` and a peer-endpoint switch (the real
    model_ops functions) — plus the user's own keys: a plugin setting and a provider tuning."""
    _seed(g, OLD_CONFIG + "server:\n  model: old-model\n  binary: /opt/vllm/bin/vllm\n")
    _detect(monkeypatch, "new-model")
    _silent(monkeypatch)
    _exited(_init(g, "--model", "new-model"), 0)
    harness = ConfigLoader(config_dir=g).load_harness()
    asyncio.run(persist_default_model(harness, "slash-model", config_dir=g))
    asyncio.run(persist_active_endpoint(harness, PEER, "peer-model", config_dir=g))
    overlay_path = g / "overrides.yaml"
    overlay = load_overlay(overlay_path)
    assert {"provider.base_url", "org.default_model", "server.model", "active_endpoint.model"} <= (
        _leaf_paths(overlay)), "premise: every writer left its choice"
    overlay["provider"]["timeout_seconds"] = 900.0
    overlay["image"] = dict(USER_KEYS["image"])
    atomic_write_overlay(overlay_path, overlay)
    return overlay_path


@pytest.mark.parametrize("flags, tty", [(["--force"], False), (["--force", "--no-input"], True)],
                         ids=["force", "force-no-input"])
def test_force_clears_the_saved_model_and_server_choices(flags, tty, tmp_path, monkeypatch):
    overlay_path = _choices_saved_by_every_writer(tmp_path, monkeypatch)
    _detect(monkeypatch, "fresh-model")
    _terminal(monkeypatch, tty)
    _silent(monkeypatch)

    result = _init(tmp_path, *flags)

    _exited(result, 0)
    text = (tmp_path / "config.yaml").read_text(encoding="utf-8")
    assert "fresh-model" in text and "old-model" not in text
    assert load_overlay(overlay_path) == USER_KEYS  # the choices gone; plugin key and tuning kept
    flat = _flat(result)
    assert f"Cleared the saved model and server choices in {overlay_path}" in flat, flat
    assert "still sets" not in flat
    harness = ConfigLoader(config_dir=tmp_path).load_harness()  # what the next start reads
    assert (harness.provider.default_model, harness.provider.base_url, harness.org.default_model) == (
        "fresh-model", "http://localhost:11434", "fresh-model")
    assert harness.server is None and harness.active_endpoint is None
    assert harness.provider.timeout_seconds == 900.0


def test_force_leaves_an_overrides_yaml_without_saved_choices_alone(tmp_path, monkeypatch):
    _detect(monkeypatch)
    _silent(monkeypatch)
    _seed(tmp_path)
    overlay_path = tmp_path / "overrides.yaml"
    overlay_path.write_text("# mine\nimage:\n  enabled: true\n", encoding="utf-8")

    result = _init(tmp_path, "--force")

    _exited(result, 0)
    assert overlay_path.read_text(encoding="utf-8") == "# mine\nimage:\n  enabled: true\n"
    assert "Cleared" not in result.output


def test_a_rerun_without_force_removes_nothing(tmp_path, monkeypatch):
    overlay_path = _choices_saved_by_every_writer(tmp_path, monkeypatch)
    before = overlay_path.read_bytes()
    saved = _leaf_paths(load_overlay(overlay_path))

    _exited(_init(tmp_path), 0)  # no terminal: kept
    assert overlay_path.read_bytes() == before

    _detect(monkeypatch, "fresh-model")
    result = _init(tmp_path, "--model", "fresh-model")  # a change: updates, never removes
    _exited(result, 0)
    after = load_overlay(overlay_path)
    assert saved <= _leaf_paths(after), saved - _leaf_paths(after)
    assert after["image"] == USER_KEYS["image"] and after["provider"]["timeout_seconds"] == 900.0
    assert "Cleared" not in result.output


def test_a_fresh_write_without_force_names_the_saved_choices_that_still_win(tmp_path, monkeypatch):
    """config.yaml is missing (deleted by hand, or a first start) while overrides.yaml survived:
    nothing leaves overrides.yaml without --force, so its saved choices still win — say which."""
    _detect(monkeypatch)
    _silent(monkeypatch)
    overlay_path = tmp_path / "overrides.yaml"
    saved = ("provider:\n  base_url: http://localhost:8000/v1\n  default_model: model-x\n"
             "org:\n  default_model: model-x\nimage:\n  enabled: true\n")
    overlay_path.write_text(saved, encoding="utf-8")

    result = _init(tmp_path)

    _exited(result, 0)
    assert (tmp_path / "config.yaml").exists()
    assert overlay_path.read_text(encoding="utf-8") == saved
    assert (f"{overlay_path} still sets provider.base_url, provider.default_model, org.default_model — "
            "those win over config.yaml. To clear them: localharness init --force") in _flat(result)
    assert "Cleared" not in result.output


@pytest.mark.parametrize("overlay_text, said", [
    ("provider:\n  default_model: other-model\nproposer:\n  base_url: http://proposer/v1\n",
     "do not validate"),
    ("provider: [\n", "cannot be read"),
], ids=["reset-does-not-validate", "unreadable"])
def test_force_writes_nothing_when_the_reset_cannot_be_written(overlay_text, said, tmp_path,
                                                               monkeypatch):
    """--force resets both files or neither: a reset whose merged config would not validate (here a
    proposer with no model, which --force keeps: it clears only the saved model and server choices),
    or an overrides.yaml that cannot be read, exits 1 with config.yaml and overrides.yaml untouched."""
    _detect(monkeypatch)
    _silent(monkeypatch)
    cfg = _seed(tmp_path)
    overlay_path = tmp_path / "overrides.yaml"
    overlay_path.write_text(overlay_text, encoding="utf-8")

    result = _init(tmp_path, "--force")

    _exited(result, 1)
    flat = _flat(result)
    assert said in flat and "Nothing was written." in flat, flat
    assert cfg.read_bytes() == OLD_CONFIG.encode()
    assert overlay_path.read_text(encoding="utf-8") == overlay_text


def test_force_then_skip_for_now_also_clears_the_saved_choices(tmp_path, monkeypatch):
    """The other fresh write: --force on a terminal, no server, guided setup declined, a model named
    — the config saved unchecked is the one the next start reads, not overrides.yaml's old choice."""
    _seed(tmp_path)
    overlay_path = tmp_path / "overrides.yaml"
    overlay_path.write_text("provider:\n  base_url: http://localhost:8000/v1\n  default_model: model-x\n"
                            "image:\n  enabled: true\n", encoding="utf-8")
    _no_server_on_a_terminal(monkeypatch, "http://localhost:8081/v1", "my-model")

    result = _init(tmp_path, "--force")

    _exited(result, 0)
    assert load_overlay(overlay_path) == {"image": {"enabled": True}}
    assert "Cleared the saved model and server choices in" in _flat(result)
    provider = ConfigLoader(config_dir=tmp_path).load_harness().provider
    assert (provider.base_url, provider.default_model) == ("http://localhost:8081/v1", "my-model")
