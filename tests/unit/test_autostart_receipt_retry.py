"""The `model` receipt after a swap under the autostart unit names a retry that swap recorded, and only that."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from localharness.cli import start_serving
from localharness.provider import lifecycle, speed_stats
from tests.unit.procfs_support import fake_proc  # noqa: F401 — the fixture
from tests.unit.test_model_cmd_unit_swap import add_docker_peer, edit_config, lines_of, peer_of
from tests.unit.test_model_cmd_unit_swap import cli as _cli

NAME = "qwen-27b-swift-1.5"
PLAIN = f"Serving {NAME} — autostart restarted"
pytest.importorskip("fcntl")
cli = _cli


def retry(world, spec, *, dropped=("draft-mtp",)) -> None:
    speed_stats.record_retry(
        speed_stats.default_speed_stats_path(world.g), lifecycle.record_key(spec), full_fit=262144,
        free_bytes=None, dropped=dropped, binary="vllm", kv_cache_type="q8_0", draft_p_min=0.6)


def swap_recording(cli, *drops):
    """`run_swap` stand-in: each call records the next entry of `drops` (None records nothing), then succeeds."""
    pending = list(drops)

    async def run_swap(harness, incumbent, serving, **kw):
        drop = pending.pop(0)
        if drop is not None:
            retry(cli.world, serving.record.spec, dropped=drop)
        return SimpleNamespace(error=None)

    cli.monkeypatch.setattr(start_serving, "run_swap", run_swap)


def rename_flash(cli, name: str) -> None:
    edit_config(cli, lambda data: peer_of(data, "flash-next").update(name=name))
    cli.record_machine()


def run(cli, name: str = NAME) -> list[str]:
    result = cli.run(name)
    assert result.exit_code == 0, result.output
    return lines_of(result.output)


@pytest.fixture
def swift(cli):
    rename_flash(cli, NAME)
    return cli


def test_a_retry_that_dropped_mtp_says_window_and_speculative_decoding(swift):
    swap_recording(swift, ["draft-mtp"])
    assert f"{PLAIN} (retried with a smaller window, speculative decoding off)" in run(swift)


def test_a_retry_that_only_halved_the_window_says_so(swift):
    swap_recording(swift, [])
    assert f"{PLAIN} (retried with a smaller window)" in run(swift)


def test_a_swap_that_records_no_retry_is_the_plain_receipt(swift):
    swap_recording(swift, None)
    assert PLAIN in run(swift)


def test_a_retry_left_from_before_the_command_is_not_reported(swift):
    retry(swift.world, swift.world.peer("flash-next"))
    swap_recording(swift, None)
    out = run(swift)
    assert PLAIN in out


def test_the_clause_follows_the_swap_that_recorded_it_not_the_next_one(swift):
    swap_recording(swift, ["draft-mtp"], None)
    first = run(swift)
    swift.answering.clear()
    second = run(swift)
    assert f"{PLAIN} (retried with a smaller window, speculative decoding off)" in first
    assert PLAIN in second


def test_a_record_name_with_an_escape_byte_is_shown_escaped(cli):
    rename_flash(cli, "flash\x1b[2Jx")
    swap_recording(cli, None)
    out = run(cli, "flash\x1b[2Jx")
    assert "Serving flash\\x1b[2Jx — autostart restarted" in out
    assert not any("\x1b" in ln for ln in out)


def test_outside_the_unit_the_receipt_carries_no_clause(cli):
    add_docker_peer(cli)
    swap_recording(cli, ["draft-mtp"])
    out = run(cli, "vllm-box")
    assert "Serving vllm-box" in out
    assert not any("retried" in ln for ln in out)


def test_the_clause_builder_needs_a_record_that_changed():
    rec = speed_stats.RetryRecord(262144, None, ("draft-mtp",), "vllm", "q8_0", 0.6, "t1")
    assert start_serving.retry_receipt_clause(None, None) == ""
    assert start_serving.retry_receipt_clause(rec, rec) == ""
    assert start_serving.retry_receipt_clause(rec, rec._replace(at="t2", dropped=())) == " (retried with a smaller window)"
