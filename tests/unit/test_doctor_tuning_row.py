"""doctor's `Tuning:` rows (DOCTOR-01, spec 20 §6.5): what each launchable record's next launch derives.

The world is `test_serve_cmd.serve_world` (a recorded machine, a Swift llama.cpp primary, the box faked at the
autosense seams). The rows come from the same sense + plan `activate` runs; nothing is prepared, started,
decided or written. No process starts and no request leaves the box: doctor's httpx is a mock.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from localharness.agent.context import WindowProbe
from localharness.cli import doctor_cmd, workspace
from localharness.cli.app import app
from localharness.config import trust
from localharness.config.loader import ConfigLoader
from localharness.provider import lifecycle, server, speed_stats
from tests.unit.gguf_support import write_gguf
from tests.unit.launch_support import record_opens
from tests.unit.test_doctor_cmd import (  # noqa: F401  (autouse fixtures, imported for their effect)
    _models_resp,
    _route_the_window_probe,
    _stub_token_counter,
)
from tests.unit.test_serve_cmd import attach_peer, docker_peer, llama_peer, ollama_peer, serve_world
from tests.unit.test_tuning_derive import FLASH_NEXT

runner = CliRunner()
ROWS = re.compile(r"^i\s+(Tuning: .*)$", re.MULTILINE)


@pytest.fixture
def mock_httpx(monkeypatch):
    mock = MagicMock()
    mock.get.return_value = _models_resp({"data": [{"id": "swift"}]})
    monkeypatch.setattr(doctor_cmd, "httpx", mock)
    return mock


@pytest.fixture
def world(tmp_path, monkeypatch, mock_httpx):
    return serve_world(tmp_path, monkeypatch)


def _doctor(world):
    with patch("localharness.agent.context.probe_window", return_value=WindowProbe(None, None)):
        return runner.invoke(app, ["doctor", "--config-dir", str(world.g)])


def _rows(result) -> list[str]:
    return ROWS.findall(result.output)


def _flash(world, name: str = "flash-next", port: int = 8082) -> dict:
    peer = llama_peer(name, port)
    peer["lifecycle"]["model"] = str(write_gguf(world.tmp / f"{name}.gguf", FLASH_NEXT))
    return peer


def _fitted(world, index: int | None = None, **sensing) -> int:
    harness = ConfigLoader(config_dir=world.g, local_config_dir=None).load_harness()
    spec = harness.server if index is None else harness.extra_endpoints[index].lifecycle
    plan = lifecycle.plan_launch(
        spec, lifecycle.sense_launch(spec, world.g, harness.autosense, **sensing), server.launch_decision(spec, False),
        autosense=harness.autosense, api_key_held=False)
    return plan.full_fit


UNREAD = " · weights 18.0 GiB (tensor table not read) · prompt cache 24,576 MiB · RAM reserve 9.6 GiB"


def _files(root: Path) -> dict[str, bytes]:
    return {str(p): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_the_swift_primary_and_a_peer_get_a_row_each_and_an_attach_only_peer_gets_none(world):
    world.edit_config(lambda d: d["extra_endpoints"].extend([_flash(world), attach_peer("attached")]))
    world.rerecord()

    result = _doctor(world)

    assert _rows(result) == [
        f"Tuning: swift — kv q8_0 · mtp (1 layer) · -c {_fitted(world)} (fit){UNREAD}",
        f"Tuning: flash-next — kv q8_0 · -c {_fitted(world, 0)} (fit){UNREAD}"]
    out = result.output
    assert out.index("LLM endpoint reachable") < out.index("Tuning: swift") < out.index("Tuning: flash-next")
    assert out.index("Tuning: flash-next") < out.index("Config directory writable")


def test_the_rows_prepare_nothing_start_nothing_decide_nothing_and_write_nothing(world, monkeypatch):
    def raises(*a, **kw):
        raise AssertionError("a Tuning row must not call this")

    monkeypatch.setattr(server, "prepare_config_ini_home", raises)
    monkeypatch.setattr(speed_stats, "record_retry", raises)
    monkeypatch.setattr(workspace, "decide_machine_trust", raises)
    monkeypatch.setattr(subprocess, "Popen", raises)
    monkeypatch.setattr(doctor_cmd, "_llama_server_linked_backends", lambda binary: None)  # doctor's own ldd probe
    before = _files(world.g)

    result = _doctor(world)

    assert len(_rows(result)) == 1
    assert _files(world.g) == before
    assert not (world.g / "xdg").exists()


def test_a_vllm_binary_record_has_its_parsers_and_no_window(tmp_path, monkeypatch, mock_httpx):
    world = serve_world(tmp_path, monkeypatch, runtime="vllm")

    assert _rows(_doctor(world)) == ["Tuning: swift — reasoning qwen3 · tools qwen3_coder"]


def test_a_record_with_nothing_derived_says_so_and_keeps_its_window(world):
    world.edit_config(lambda d: d.update(autosense={"kv_cache_type": "f16", "draft_mtp": False}))
    world.rerecord()

    assert _rows(_doctor(world)) == [f"Tuning: swift — nothing derived · -c {_fitted(world)} (fit){UNREAD}"]


def test_a_record_without_a_binary_names_the_field_to_set(world):
    world.edit_config(lambda d: (d["server"].pop("binary"), d["extra_endpoints"].append(
        {**llama_peer("flash", 8082), "lifecycle": {**llama_peer("flash", 8082)["lifecycle"], "binary": None}})))
    world.rerecord()

    assert _rows(_doctor(world)) == [
        "Tuning: swift — no llama-server — set server.binary",
        "Tuning: flash — no llama-server — set extra_endpoints[flash].lifecycle.binary"]


def test_a_record_changed_since_it_was_confirmed_is_not_planned(world):
    world.edit_config(lambda d: d["extra_endpoints"].append(_flash(world)))
    world.rerecord()
    world.edit_config(lambda d: d["extra_endpoints"][0]["lifecycle"].update(binary="/x/other-server"))

    rows = _rows(_doctor(world))

    assert rows[1] == "Tuning: flash-next — not planned: its record has changes you have not confirmed"
    assert rows[0].startswith("Tuning: swift — kv q8_0")


def test_a_machine_with_no_record_plans_no_launch_entry(world):
    trust.trust_store_path().unlink()

    assert _rows(_doctor(world)) == [
        "Tuning: swift — not planned: its record has changes you have not confirmed"]


@pytest.mark.parametrize(("change", "window"), [
    ({"context": 32768}, "-c 32768 (the record)"),
    ({"extra_args": ["-c", "4096"]}, "-c from extra_args"),
])
def test_the_window_part_says_where_the_window_comes_from(world, change, window):
    world.edit_config(lambda d: d["server"].update(change))
    world.rerecord()

    assert _rows(_doctor(world)) == [f"Tuning: swift — kv q8_0 · mtp (1 layer) · {window}{UNREAD}"]


def test_an_unknown_free_memory_says_the_window_was_not_computed(world, monkeypatch):
    from localharness.provider import autosense

    monkeypatch.setattr(autosense, "free_memory", lambda ratios: None)

    assert _rows(_doctor(world))[0].startswith("Tuning: swift — kv q8_0 · mtp (1 layer) · -c not computed")


def test_ollama_and_lmstudio_records_get_no_row_and_a_docker_record_does(world):
    lmstudio = {**ollama_peer("lms", 1235), "provider_type": "lmstudio"}
    lmstudio["lifecycle"] = {**lmstudio["lifecycle"], "runtime": "lmstudio"}
    world.edit_config(lambda d: d["extra_endpoints"].extend([ollama_peer(), lmstudio, docker_peer()]))
    world.rerecord()

    rows = _rows(_doctor(world))

    assert [r.split(" — ")[0] for r in rows] == ["Tuning: swift", "Tuning: dock"]


def test_a_key_file_record_is_planned_keyed_and_the_file_is_never_opened(world, tmp_path):
    key = tmp_path / "the.key"
    key.write_text("sk-file\n")
    key.chmod(0o600)
    world.edit_config(lambda d: d["server"].update(api_key_file=str(key), require_api_key=True))
    world.rerecord()

    with record_opens() as opened:
        result = _doctor(world)

    assert len(_rows(result)) == 1
    assert str(key) not in opened and "sk-file" not in result.output


def test_the_rows_print_when_the_endpoint_is_unreachable(world, mock_httpx):
    import httpx

    mock_httpx.get.side_effect = httpx.ConnectError("refused")
    mock_httpx.ConnectError = httpx.ConnectError
    mock_httpx.TimeoutException = httpx.TimeoutException

    result = _doctor(world)

    assert "LLM endpoint unreachable" in result.output and len(_rows(result)) == 1


GIB = 2**30


def _flash_machine(world, monkeypatch, *, available=52.9):
    from localharness.provider import autosense
    from tests.unit.gguf_support import load_fixture, write_recorded, write_sparse

    model = write_recorded(world.tmp, load_fixture("flash-next-iq3-xxs"))[0]
    mmproj = write_sparse(world.tmp / "mmproj.gguf", 907_543_520)
    peer = llama_peer("flash-next", 8082)
    peer["lifecycle"].update(model=str(model), mmproj=str(mmproj), served_name="Flash-Next")
    world.edit_config(lambda d: (d["extra_endpoints"].append(peer), d.update(autosense={"draft_mtp": False})))
    world.rerecord()
    monkeypatch.setattr(autosense, "free_memory", lambda ratios: autosense.FreeMemory(int(30.2 * GIB), int(31.8 * GIB), "nvidia-smi"))
    monkeypatch.setattr(autosense, "_read_meminfo", lambda: {"MemTotal": 64_139_128 * 1024, "MemAvailable": int(available * GIB)})


def test_flash_next_row_carries_the_per_tensor_and_ram_figures(world, monkeypatch):
    _flash_machine(world, monkeypatch)

    assert _rows(_doctor(world))[1] == (
        "Tuning: flash-next — kv q8_0 · -c 262144 (fit) · dense 3.7 GiB · experts 40.0 GiB in RAM · "
        "token tables 27.1 GiB read through mmap · prompt cache 3,849 MiB · RAM reserve 9.2 GiB")


def test_flash_next_without_the_ram_for_its_experts_shows_the_refusal_sentence(world, monkeypatch):
    _flash_machine(world, monkeypatch, available=31.4)

    assert _rows(_doctor(world))[1] == (
        "Tuning: flash-next — Flash-Next cannot run here: its experts need 40.0 GiB of RAM and only 22.2 GiB is "
        "free beside the system's reserve — close something that holds RAM, or pick a smaller quant.")


def test_a_dense_27b_with_a_readable_table_has_no_experts_clause(world, monkeypatch):
    from tests.unit.gguf_support import load_fixture, write_recorded

    model = write_recorded(world.tmp, load_fixture("swift-27b-nvfp4-q8mix"))[0]
    world.edit_config(lambda d: d["server"].update(model=str(model)))
    world.rerecord()

    row = _rows(_doctor(world))[0]

    assert re.search(r"\(fit\) · dense \d+\.\d GiB · token tables \d+\.\d GiB read through mmap · "
                     r"prompt cache 24,576 MiB · RAM reserve 9\.6 GiB$", row)
    assert "experts" not in row


def test_a_model_whose_table_cannot_be_read_shows_its_file_size(world):
    assert _rows(_doctor(world))[0].endswith(f"(fit){UNREAD}")


def test_a_vllm_checkpoint_shows_its_weights_and_the_vllm_budget(tmp_path, monkeypatch, mock_httpx):
    import shutil

    from localharness.provider import autosense
    from tests.unit.gguf_support import write_sparse

    world = serve_world(tmp_path, monkeypatch, runtime="vllm")
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "vllm_checkpoints" / "nvidia-qwen3.8-27b-nvfp4"
    shutil.copy(fixture / "config.json", world.model / "config.json")
    for i in range(2):
        write_sparse(world.model / f"model-0000{i + 1}-of-00002.safetensors", 21_921_697_280 // 2)
    total = int(31.8 * GIB)
    monkeypatch.setattr(autosense, "free_memory", lambda ratios: autosense.FreeMemory(total, total, "nvidia-smi"))
    world.rerecord()

    assert _rows(_doctor(world))[0].endswith("--max-model-len 100864 (fit) · weights 20.4 GiB · vLLM budget 28.6 GiB")



def test_a_verified_launch_of_the_harnesss_own_lends_the_row_the_memory_it_holds(world, monkeypatch):
    from localharness.provider import autosense, procfs

    monkeypatch.setattr(autosense, "free_memory", lambda ratios: autosense.FreeMemory(5 * GIB, 48 * GIB, "nvidia-smi"))
    monkeypatch.setattr(autosense, "held_by", lambda pid: 30 * GIB if pid == 4321 else None)
    monkeypatch.setattr(procfs, "read_launch", lambda cfg, **kw: None)
    assert _rows(_doctor(world)) == [
        "Tuning: swift — swift cannot run here: it needs 18.0 GiB of GPU memory and 5.0 GiB is free — stop what "
        "holds the GPU, or pick a smaller quant."]

    monkeypatch.setattr(procfs, "read_launch",
                        lambda cfg, *, expect=None, **kw: procfs.Launch(4321, None, 8080, "spawned", expect is not None))

    fitted = _fitted(world, memory=autosense.FreeMemory(35 * GIB, 48 * GIB, "nvidia-smi"))
    assert _rows(_doctor(world)) == [f"Tuning: swift — kv q8_0 · mtp (1 layer) · -c {fitted} (fit){UNREAD}"]
