"""init writes a record instead of launching a server (Phase 5, D-12, D-13, D-39, D-55, D-59, D-68).

The guided vLLM setup keeps its install and download questions but starts nothing and waits for
nothing: it hands back an `InitRecords`, which core_setup shows, writes as a loadable config.yaml
without probing a server that is not up, and reports as ready when the first `start` can launch it.
Every call that would launch, wait or probe is booby-trapped, so "launched nothing" is proven.
"""
from __future__ import annotations

import shlex
import subprocess
from dataclasses import replace
from unittest.mock import MagicMock

import pytest
import yaml

import localharness.cli.init_cmd as init_cmd
from localharness.config.loader import ConfigLoader
from localharness.config.models import (
    ContextConfig,
    EndpointRef,
    HarnessConfig,
    ManagedServerConfig,
    TerminalConfig,
    TerminalKeysConfig,
)
from localharness.config.overlay import deep_merge
from localharness.provider import server as managed_server
from localharness.provider.refarch import DGX_SPARK
from tests.unit.test_init_core_setup import (
    _Asked,
    _detect,
    _exited,
    _flat,
    _hermetic,  # noqa: F401  (autouse fixture; real sockets are refused by conftest)
    _init,
    _terminal,
)

VLLM = "/opt/vllm/bin/vllm"
PRESET_LINE = "serving {model} (vLLM) ← {model} · binary: " + VLLM
FLAGS = "flags: " + shlex.join(DGX_SPARK.serve_extra_args)


def _boom(name: str):
    def _raise(*_a, **_k):
        raise AssertionError(f"{name} called: init must launch, wait for and probe nothing")
    return _raise


@pytest.fixture
def guided(tmp_path, monkeypatch):
    """A terminal, nothing detected, the guided setup answered (yes, hardware 1, a local path as
    the model so nothing downloads), and every launch / wait / probe / spawn booby-trapped."""
    _detect(monkeypatch, found=False)
    _terminal(monkeypatch)
    confirm = MagicMock()
    confirm.ask.return_value = True
    monkeypatch.setattr(init_cmd, "Confirm", confirm)
    int_prompt = MagicMock()
    int_prompt.ask.return_value = 1
    monkeypatch.setattr(init_cmd, "IntPrompt", int_prompt)
    model = tmp_path / "checkpoint"
    model.mkdir()
    prompt = MagicMock()
    prompt.ask.return_value = str(model)
    monkeypatch.setattr(init_cmd, "Prompt", prompt)
    monkeypatch.setattr(managed_server, "find_vllm", lambda _d: VLLM)
    for name in ("start_server", "wait_ready"):
        monkeypatch.setattr(managed_server, name, _boom(name))
    monkeypatch.setattr(init_cmd, "_probe_capabilities", _boom("_probe_capabilities"))
    monkeypatch.setattr(init_cmd, "_served_window", _boom("_served_window"))
    monkeypatch.setattr(subprocess, "Popen", _boom("subprocess.Popen"))
    return str(model)


def _written(tmp_path) -> dict:
    return yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))


def _no_none(value) -> bool:
    if isinstance(value, dict):
        return all(_no_none(v) for v in value.values())
    if isinstance(value, list):
        return all(_no_none(v) for v in value)
    return value is not None


# ------------------------------------------------------------------ the guided record, end to end


def test_the_guided_setup_writes_its_record_and_launches_nothing(tmp_path, guided):
    result = _init(tmp_path)

    _exited(result, 0)
    cfg = _written(tmp_path)
    assert cfg["server"]["runtime"] == "vllm"
    assert cfg["server"]["binary"] == VLLM
    assert cfg["server"]["model"] == guided
    assert _no_none(cfg["server"]), cfg["server"]
    assert "effort" not in cfg["server"]
    provider = cfg["provider"]
    assert provider["provider_type"] == "vllm"
    assert provider["base_url"] == "http://127.0.0.1:8081/v1"
    assert provider["default_model"] == guided and provider["available_models"] == [guided]
    assert provider.get("supports_function_calling") is None
    # no context budget of its own: the model default stands and `start`'s auto-fit decides
    assert cfg["org"]["context"]["max_context_tokens"] == ContextConfig().max_context_tokens
    assert cfg["extra_endpoints"] == []
    out = _flat(result)
    for absent in ("No local LLM detected", "Checked:", "Log:", "Launching"):
        assert absent not in out, out
    # the written config is one the next start loads
    loaded = ConfigLoader(config_dir=tmp_path).load_harness()
    assert loaded.provider.default_model == guided and loaded.server.binary == VLLM


def test_the_guided_record_reports_a_server_ready_for_the_first_start(tmp_path, guided):
    out = init_cmd.core_setup(str(tmp_path), endpoint=None, model=None, force=False, interactive=True)

    assert out == init_cmd.SetupResult(tmp_path / "config.yaml", True)


def test_the_docker_route_writes_a_docker_record(tmp_path, guided, monkeypatch):
    monkeypatch.setattr(managed_server, "find_vllm", lambda _d: None)
    monkeypatch.setattr(init_cmd, "which_outside_cwd", lambda _name: "/usr/bin/docker")

    result = _init(tmp_path)

    _exited(result, 0)
    server = _written(tmp_path)["server"]
    assert server["launch"] == "docker" and server["docker_image"] == DGX_SPARK.docker_image
    assert "binary" not in server
    assert f"· docker_image: {DGX_SPARK.docker_image}" in result.output
    out = init_cmd.core_setup(str(tmp_path / "again"), endpoint=None, model=None, force=False,
                              interactive=True)
    assert out.server_ready is True


def test_the_preset_line_and_the_flags_line_are_shown_before_the_write(tmp_path, guided):
    result = _init(tmp_path)

    lines = result.output.splitlines()
    preset = PRESET_LINE.format(model=guided)
    assert preset in lines, lines
    at = lines.index(preset)
    assert lines[at + 1] == FLAGS, lines[at:at + 3]
    assert FLAGS == ("flags: --max-model-len 65536 --gpu-memory-utilization 0.90 "
                     "--enable-auto-tool-choice --tool-call-parser hermes")
    # shown, then adopted: both lines come before the receipt that says the file is on disk
    assert at + 1 < next(i for i, ln in enumerate(lines) if ln.startswith("✓ LocalHarness configured"))
    assert _written(tmp_path)["server"]["extra_args"] == list(DGX_SPARK.serve_extra_args)


def test_no_flags_line_when_the_profile_has_no_extra_args(tmp_path, guided, monkeypatch):
    monkeypatch.setattr(init_cmd, "REF_ARCHS", (replace(DGX_SPARK, serve_extra_args=()),))

    result = _init(tmp_path)

    _exited(result, 0)
    assert PRESET_LINE.format(model=guided) in result.output.splitlines()
    assert "flags:" not in result.output


def test_a_hostile_model_name_is_shown_escaped(tmp_path, guided, monkeypatch):
    hostile = tmp_path / "ev\x1b[31mil\nname"
    hostile.mkdir()
    monkeypatch.setattr(init_cmd.Prompt, "ask", staticmethod(lambda *a, **k: str(hostile)))

    recs = init_cmd._guided_setup(tmp_path)

    shown = "\n".join(recs.lines)
    assert "\x1b" not in shown and "ev\\x1b[31mil\\x0aname" in shown
    assert len(recs.lines) == 2  # a newline in the name did not fake a third line


# ------------------------------------------------------------------ the not-found outcome


def test_declining_the_guided_setup_still_says_nothing_was_found(tmp_path, monkeypatch):
    _detect(monkeypatch, found=False)
    _terminal(monkeypatch)
    confirm = MagicMock()
    confirm.ask.return_value = False
    monkeypatch.setattr(init_cmd, "Confirm", confirm)
    prompt = MagicMock()
    prompt.ask.side_effect = ["http://localhost:8081/v1", "my-model"]
    monkeypatch.setattr(init_cmd, "Prompt", prompt)
    monkeypatch.setattr(init_cmd, "IntPrompt", _Asked)

    result = _init(tmp_path)

    _exited(result, 0)
    flat = _flat(result)
    assert "No local LLM detected" in flat and "Checked:" in flat
    assert flat.index("No local LLM detected") < flat.index("Skip for now")


def test_no_input_with_nothing_detected_says_it_and_exits_1(tmp_path, monkeypatch):
    _detect(monkeypatch, found=False)
    for name in ("Confirm", "IntPrompt", "Prompt"):
        monkeypatch.setattr(init_cmd, name, _Asked)

    result = _init(tmp_path, "--no-input")

    _exited(result, 1)
    flat = _flat(result)
    assert "No local LLM detected" in flat and "Checked:" in flat
    assert not (tmp_path / "config.yaml").exists()


def test_say_not_detected_lists_every_probed_port(capsys):
    init_cmd._say_not_detected()

    out = capsys.readouterr().out
    assert "No local LLM detected" in out
    for port in init_cmd.DEFAULT_PORTS:
        assert f"http://localhost:{port}" in out


# ------------------------------------------------------------------ the writer


def _harness(server=None, peers=()) -> HarnessConfig:
    return HarnessConfig(
        version="1",
        provider={"provider_type": "vllm", "base_url": "http://127.0.0.1:8081/v1",
                  "default_model": "m", "available_models": ["m"]},
        org={"default_model": "m"},
        server=server,
        extra_endpoints=list(peers),
    )


def test_record_dicts_are_exclude_none_dumps_with_revealed_keys():
    server = ManagedServerConfig(runtime="vllm", binary=VLLM, model="m")
    peer = EndpointRef(name="side", base_url="http://127.0.0.1:8082/v1", api_key="s3cret",
                       provider_type="llamacpp")

    server_dict, peer_dicts = init_cmd._record_dicts(_harness(server, [peer]))

    assert server_dict == server.model_dump(mode="python", exclude_none=True)
    assert _no_none(server_dict) and "effort" not in server_dict
    assert len(peer_dicts) == 1 and _no_none(peer_dicts[0])
    assert peer_dicts[0]["api_key"] == "s3cret"  # as typed, not the mask
    assert peer_dicts[0]["name"] == "side"


def test_record_dicts_with_no_server_and_no_peers():
    assert init_cmd._record_dicts(_harness()) == (None, [])


def test_write_harness_writes_exactly_the_record_dicts(tmp_path):
    server = ManagedServerConfig(runtime="vllm", binary=VLLM, model="m")
    peer = EndpointRef(name="side", base_url="http://127.0.0.1:8082/v1", api_key="s3cret")
    harness = _harness(server, [peer])

    init_cmd._write_harness(tmp_path, harness, True)

    cfg = _written(tmp_path)
    server_dict, peer_dicts = init_cmd._record_dicts(harness)
    assert cfg["server"] == server_dict
    assert cfg["extra_endpoints"] == peer_dicts


def test_a_harness_with_no_server_still_writes_server_null_and_empty_peers(tmp_path):
    init_cmd._write_harness(tmp_path, _harness(), True)

    cfg = _written(tmp_path)
    assert cfg["server"] is None
    assert cfg["extra_endpoints"] == []


def test_init_never_writes_a_defaulted_chord_list(tmp_path):
    """A dumped chord list would read as the person's own and make the documented one-key change
    refuse itself (spec 21 §5), so a default harness writes a terminal section with no `keys`."""
    init_cmd._write_harness(tmp_path, _harness(), True)

    terminal = _written(tmp_path)["terminal"]
    assert "keys" not in terminal
    assert terminal["inputbox_enabled"] is True


def test_init_keeps_a_chord_list_the_person_set(tmp_path):
    """A list the person set survives a re-init; the trimmed list beside it is not written."""
    harness = _harness().model_copy(update={"terminal": TerminalConfig(
        keys=TerminalKeysConfig(newline=["alt-enter"]))})
    assert harness.terminal.keys.nudge == []  # trimmed, never set

    init_cmd._write_harness(tmp_path, harness, True)

    assert _written(tmp_path)["terminal"]["keys"] == {"newline": ["alt-enter"]}


def test_a_fresh_init_accepts_the_one_command_line_break_change(tmp_path):
    """After a default write, `components set terminal.keys.newline alt-enter` validates: the
    overlay is merged into what init wrote, exactly as the setter's validation does."""
    init_cmd._write_harness(tmp_path, _harness(), True)
    written = _written(tmp_path)

    merged = deep_merge(written, {"terminal": {"keys": {"newline": "alt-enter"}}})
    reloaded = HarnessConfig.model_validate(merged)

    assert reloaded.terminal.keys.newline == ["alt-enter"]
    assert reloaded.terminal.keys.nudge == []


def test_init_force_accepts_an_overrides_file_that_moved_a_chord(tmp_path):
    """`init --force` validates the fresh harness's dump merged with the saved overrides. The dump
    carries both defaulted lists; an overrides file holding the documented one-command change must
    not turn them into a collision (spec 21 §5: a list at its default value yields)."""
    overrides = tmp_path / "overrides.yaml"
    overrides.write_text(yaml.safe_dump({
        "provider": {"default_model": "saved"},
        "terminal": {"keys": {"newline": "alt-enter"}},
    }), encoding="utf-8")

    reset = init_cmd._saved_choices_reset(tmp_path, _harness())

    assert reset is not None
    _path, overlay, cleared = reset
    assert cleared == ["provider.default_model"]
    assert overlay["terminal"]["keys"] == {"newline": "alt-enter"}


# ------------------------------------------------------------------ the carrier


def _recs(**kw) -> init_cmd.InitRecords:
    return init_cmd.InitRecords(
        server=kw.pop("server", ManagedServerConfig(runtime="vllm", binary=VLLM, model="m", port=9000)),
        served=kw.pop("served", "m"), **kw)


def test_launchable_means_a_binary_or_a_docker_image():
    assert _recs().launchable is True
    assert _recs(server=ManagedServerConfig(runtime="vllm", launch="docker",
                                            docker_image="img", model="m")).launchable is True
    assert _recs(server=ManagedServerConfig(runtime="llamacpp", model="/m.gguf")).launchable is False


def test_the_provider_base_url_follows_the_record_port_unless_one_is_given():
    assert init_cmd._result_from_record(_recs()).base_url == "http://127.0.0.1:9000/v1"
    given = init_cmd._result_from_record(_recs(base_url="http://10.0.0.5:9000/v1"))
    assert given.base_url == "http://10.0.0.5:9000/v1"
    result = init_cmd._result_from_record(_recs())
    assert (result.found, result.provider_type, result.models, result.suggested_model) == (
        True, "vllm", ["m"], "m")


def test_write_records_writes_peers_the_key_and_runs_the_deferred_after_the_file(tmp_path):
    seen: list[bool] = []
    peer = EndpointRef(name="side", base_url="http://127.0.0.1:9001/v1", provider_type="llamacpp")
    recs = _recs(peers=[peer], api_key="k-123",
                 deferred=[lambda: seen.append((tmp_path / "config.yaml").exists())])

    written, peers = init_cmd._write_records(tmp_path, recs)

    assert written == tmp_path / "config.yaml" and seen == [True]
    cfg = _written(tmp_path)
    assert cfg["provider"]["api_key"] == "k-123" and cfg["extra_endpoints"] == []
    overlay = yaml.safe_load((tmp_path / "overrides.yaml").read_text(encoding="utf-8"))
    assert [p["name"] for p in peers] == [p["name"] for p in overlay["extra_endpoints"]] == ["side"]
    assert _no_none(overlay["extra_endpoints"])


def test_a_record_with_a_server_already_up_is_ready_without_a_binary(tmp_path, monkeypatch):
    recs = _recs(server=ManagedServerConfig(runtime="llamacpp", model="/m.gguf"), serving=True)
    monkeypatch.setattr(init_cmd, "_pick_server", lambda *a, **k: recs)
    monkeypatch.setattr(init_cmd, "_probe_capabilities", _boom("_probe_capabilities"))

    out = init_cmd.core_setup(str(tmp_path), endpoint=None, model=None, force=False, interactive=False)

    assert out.server_ready is True


def test_a_declared_only_record_is_not_ready(tmp_path, monkeypatch):
    recs = _recs(server=ManagedServerConfig(runtime="llamacpp", model="/m.gguf"))
    monkeypatch.setattr(init_cmd, "_pick_server", lambda *a, **k: recs)

    out = init_cmd.core_setup(str(tmp_path), endpoint=None, model=None, force=False, interactive=False)

    assert out.server_ready is False
    assert (tmp_path / "config.yaml").exists()


def test_init_cmd_launches_and_waits_for_nothing():
    """The structural half of T-5-18: no start or readiness call anywhere in init_cmd.py."""
    import re
    from pathlib import Path

    src = Path(init_cmd.__file__).read_text(encoding="utf-8")
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    assert re.search(r"start_server\(|wait_ready\(", code) is None
