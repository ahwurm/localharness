"""SEC-11: every file the harness writes into its config folder is owner-only from its first byte.

The config folder holds the provider key, the phone token, grants, the trust store and every
session log. A file created at the umask (0664 on a 002 box) and chmodded afterwards is readable by
every other account on the machine for the moment between the two calls, and forever when the
chmod never comes. So each site is checked with chmod switched OFF (`no_chmod`): a file that is
0600 then was created 0600 — never write-then-chmod.

Helpers first (best effort: a refused chmod or a missing parent never raises), then one test per
creation site, named `test_<file>_is_born_private`. Every refused chmod and every foreign owner is
simulated with monkeypatch; no test changes the mode of a real system directory."""
from __future__ import annotations

import errno
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

posix_only = pytest.mark.skipif(os.name != "posix",
                                reason="POSIX mode bits; Windows access control is an ACL")
pytestmark = posix_only


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


@pytest.fixture(autouse=True)
def umask_002():
    """The umask the research measured on: files 0664, directories 0775 when nothing says otherwise."""
    previous = os.umask(0o002)
    try:
        yield
    finally:
        os.umask(previous)


@pytest.fixture
def no_chmod(monkeypatch):
    """chmod does nothing: whatever mode a file ends with, it was created with."""
    monkeypatch.setattr(os, "chmod", lambda *a, **k: None)
    if hasattr(os, "fchmod"):
        monkeypatch.setattr(os, "fchmod", lambda *a, **k: None)


# --------------------------------------------------------------------------- the helpers


def test_ensure_private_dir_creates_the_leaf_0700_and_parents_at_the_default(tmp_path):
    from localharness.core.private_files import ensure_private_dir

    target = tmp_path / "parent" / "config"
    assert ensure_private_dir(target) is None
    assert _mode(target) == 0o700
    assert _mode(target.parent) == 0o775  # `mkdir -p` semantics for what is above the leaf


def test_ensure_private_dir_tightens_an_open_dir_and_returns_its_old_mode(tmp_path):
    from localharness.core.private_files import ensure_private_dir

    target = tmp_path / "config"
    target.mkdir()
    os.chmod(target, 0o775)
    assert ensure_private_dir(target) == 0o775
    assert _mode(target) == 0o700


def test_ensure_private_dir_leaves_a_private_dir_alone(tmp_path):
    from localharness.core.private_files import ensure_private_dir

    target = tmp_path / "config"
    target.mkdir(mode=0o700)
    assert ensure_private_dir(target) is None
    assert _mode(target) == 0o700


@pytest.mark.parametrize("code", [errno.EROFS, errno.EPERM], ids=["read-only", "another-owner"])
def test_ensure_private_dir_never_raises_when_the_change_is_refused(tmp_path, monkeypatch, code):
    """A read-only mount and a folder another account owns are SIMULATED: os.chmod refuses."""
    from localharness.core import private_files

    target = tmp_path / "config"
    target.mkdir()
    os.chmod(target, 0o775)

    def refuse(*_a, **_k):
        raise OSError(code, os.strerror(code))

    monkeypatch.setattr(private_files.os, "chmod", refuse)
    assert private_files.ensure_private_dir(target) is None
    assert _mode(target) == 0o775


def test_write_private_bytes_creates_0600(tmp_path, no_chmod):
    from localharness.core.private_files import write_private_bytes

    target = tmp_path / "config.yaml"
    write_private_bytes(target, b"version: '1'\n")
    assert _mode(target) == 0o600
    assert target.read_bytes() == b"version: '1'\n"


def test_write_private_bytes_over_an_open_file_leaves_it_0600_with_the_new_bytes(tmp_path):
    from localharness.core.private_files import write_private_bytes

    target = tmp_path / "config.yaml"
    target.write_bytes(b"old and longer than the new text\n")
    os.chmod(target, 0o664)
    write_private_bytes(target, b"new\n")
    assert _mode(target) == 0o600
    assert target.read_bytes() == b"new\n"


def test_touch_private_creates_0600_and_never_changes_an_existing_file(tmp_path, no_chmod):
    from localharness.core.private_files import touch_private

    fresh = tmp_path / "fresh"
    touch_private(fresh)
    assert fresh.read_bytes() == b"" and _mode(fresh) == 0o600

    kept = tmp_path / "kept"
    kept.write_bytes(b"data")
    touch_private(kept)
    assert kept.read_bytes() == b"data" and _mode(kept) == 0o664


def test_touch_private_swallows_a_missing_parent(tmp_path):
    from localharness.core.private_files import touch_private

    touch_private(tmp_path / "no" / "such" / "dir" / "file")  # raises nothing
    assert not (tmp_path / "no").exists()


def test_private_opener_creates_0600(tmp_path, no_chmod):
    from localharness.core.private_files import private_opener

    target = tmp_path / "log.jsonl"
    with open(target, "a", encoding="utf-8", opener=private_opener) as fh:
        fh.write("x\n")
    assert _mode(target) == 0o600


def test_readable_by_others_lists_open_files_below_the_root(tmp_path):
    from localharness.core.private_files import readable_by_others

    root = tmp_path / "config"
    (root / "agents").mkdir(parents=True)
    open_file, closed_file = root / "agents" / "open.yaml", root / "closed.yaml"
    open_file.write_text("a")
    closed_file.write_text("b")
    os.chmod(open_file, 0o644)
    os.chmod(closed_file, 0o600)
    (root / "link").symlink_to(open_file)  # a link is not a file of its own
    assert readable_by_others(root) == [open_file]


# --------------------------------------------------------------------------- the creation sites


async def test_audit_jsonl_is_born_private(tmp_path, no_chmod):
    from localharness.core.bus import EventBus
    from localharness.core.events import ComponentMutated

    path = tmp_path / "audit.jsonl"
    await EventBus(persist_path=path).publish(ComponentMutated(
        path="a.b", before_value=1, after_value=2, layer="user", actor="cli"))
    assert path.exists() and _mode(path) == 0o600


async def test_bus_events_jsonl_and_session_log_are_born_private(tmp_path, no_chmod):
    from localharness.core.bus import EventBus
    from localharness.core.events import ComponentMutated

    path = tmp_path / "agents" / "orchestrator" / "bus-events.jsonl"
    await EventBus(persist_path=path).publish(ComponentMutated(
        path="a.b", before_value=1, after_value=2, layer="user", actor="cli", session_id="s-1"))
    session = path.parent / "sessions" / "s-1.jsonl"
    assert _mode(path) == 0o600
    assert session.exists() and _mode(session) == 0o600


async def test_history_jsonl_is_born_private(tmp_path, no_chmod):
    from localharness.memory.history import HistoryWriter

    path = tmp_path / "agents" / "orchestrator" / "history.jsonl"
    await HistoryWriter(path).append({"v": 1, "type": "user_message", "id": "1", "session_id": "s",
                                      "agent_id": "orchestrator", "ts": 1})
    assert _mode(path) == 0o600


def test_adopted_history_jsonl_is_born_private(tmp_path, no_chmod):
    """The legacy root store's adoption appends its breadcrumb through the builtin open."""
    from localharness.core.agent_dir import _migrate_legacy_root_agent_dir

    (tmp_path / "agents" / "default").mkdir(parents=True)
    _migrate_legacy_root_agent_dir(tmp_path, "orchestrator")
    path = tmp_path / "agents" / "orchestrator" / "history.jsonl"
    assert path.exists() and _mode(path) == 0o600


async def test_repl_history_is_born_private(tmp_path, no_chmod):
    from localharness.channels.terminal import TerminalChannel
    from localharness.core.bus import EventBus

    path = tmp_path / "state" / ".repl_history"
    channel = TerminalChannel(bus=EventBus(), config={}, history_file=str(path))
    await channel.start()
    try:
        assert path.exists() and _mode(path) == 0o600
        channel._history.store_string("a line the user typed")
        assert _mode(path) == 0o600
    finally:
        await channel.stop()


def test_serve_log_and_server_pid_are_born_private(tmp_path, monkeypatch, no_chmod):
    from localharness.provider import server

    monkeypatch.setattr(server.subprocess, "Popen", lambda *a, **k: SimpleNamespace(pid=4242))
    assert server.start_server(tmp_path, ["vllm", "serve", "m"]) == 4242
    assert _mode(server.log_path(tmp_path)) == 0o600
    assert _mode(server.pid_path(tmp_path)) == 0o600


async def test_memory_db_and_its_wal_are_born_private(tmp_path, no_chmod):
    from localharness.memory.sqlite import MemoryStore

    store = MemoryStore("orchestrator", "default", "default", str(tmp_path))
    await store.open()
    try:
        db = tmp_path / "agents" / "orchestrator" / "memory.db"
        assert _mode(db) == 0o600
        wal = db.with_name("memory.db-wal")
        assert wal.exists(), "the premise: opening wrote the schema through the WAL"
        assert _mode(wal) == 0o600  # SQLite gives its side files the database file's mode
    finally:
        await store.close()


def test_memory_md_is_born_private(tmp_path, no_chmod):
    from localharness.memory.markdown import _atomic_write

    path = tmp_path / "MEMORY.md"
    _atomic_write(path, "# notes\n")
    assert _mode(path) == 0o600


def test_compact_md_is_born_private(tmp_path, no_chmod):
    from localharness.agent.context import _write_compact_md

    path = tmp_path / "agents" / "orchestrator" / "compact.md"
    _write_compact_md(path, "summary")
    assert _mode(path) == 0o600


@pytest.mark.parametrize("writer", ["record_tps", "record_tokens_per_chunk"])
def test_speed_ledger_is_born_private(tmp_path, no_chmod, writer):
    from localharness.provider import speed_stats

    path = tmp_path / "speed_stats.json"
    getattr(speed_stats, writer)(path, "vllm", "m", 2.0)
    assert _mode(path) == 0o600


def test_presence_file_is_born_private(tmp_path, no_chmod):
    from localharness.config.session_presence import presence_dir, register

    register(tmp_path, agent="orchestrator", channel="terminal", session_id="s",
             workspace=tmp_path, pid=4242)
    path = presence_dir(tmp_path) / "4242.json"
    assert path.exists() and _mode(path) == 0o600


def test_memory_log_is_born_private(tmp_path, no_chmod):
    from localharness.cli.start_cmd import _route_memory_logs_to_file

    path = _route_memory_logs_to_file(tmp_path)  # conftest restores the shared logger afterwards
    assert path.exists() and _mode(path) == 0o600


async def test_minted_orchestrator_yaml_is_born_private(tmp_path, monkeypatch, no_chmod):
    """A real start with no agent minted the root agent's file."""
    from localharness.cli.start_cmd import _start_async
    from tests.unit.test_start_cmd import _stub_start_boundaries

    _stub_start_boundaries(tmp_path, monkeypatch)
    await _start_async(None, False, False, str(tmp_path))
    path = tmp_path / "agents" / "orchestrator.yaml"
    assert path.exists() and _mode(path) == 0o600


def test_renamed_root_agent_yaml_is_born_private(tmp_path, no_chmod):
    from localharness.cli.start_cmd import _migrate_legacy_root_agent_yaml

    agents = tmp_path / "agents"
    agents.mkdir()
    (agents / "default.yaml").write_text("name: default\nrole: the root\n", encoding="utf-8")
    _migrate_legacy_root_agent_yaml(agents)
    path = agents / "orchestrator.yaml"
    assert path.exists() and _mode(path) == 0o600


def test_packaged_script_is_owner_only_executable(tmp_path):
    from localharness.cli.start_cmd import _ensure_packaged_tools

    _ensure_packaged_tools(tmp_path)
    assert _mode(tmp_path / "tools" / "design-screenshot.js") == 0o700


def test_agent_create_yaml_is_born_private(tmp_path, no_chmod):
    from typer.testing import CliRunner

    from localharness.cli.agent_cmd import agent_app

    result = CliRunner().invoke(agent_app, ["create", "scout", "--global", "--config-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert _mode(tmp_path / "agents" / "scout.yaml") == 0o600


def test_workflow_agent_yaml_is_born_private(tmp_path, no_chmod):
    from localharness.cli.agent_cmd import _build_agent_yaml
    from localharness.orchestrator.workflow import AgentCreationWorkflow

    flow = AgentCreationWorkflow(config_dir=tmp_path)
    flow.set_generated_yaml(yaml.safe_dump(_build_agent_yaml("scout", "Finds things", None)))
    path = flow.deploy_config()
    assert path == tmp_path / "agents" / "scout.yaml" and _mode(path) == 0o600


def test_write_agent_yaml_is_born_private(tmp_path, no_chmod):
    from localharness.cli.agent_cmd import _build_agent_yaml
    from localharness.config.loader import ConfigLoader
    from localharness.config.models import AgentConfig

    path = ConfigLoader(config_dir=tmp_path).write_agent(
        AgentConfig(**_build_agent_yaml("scout", "Finds things", None)))
    assert _mode(path) == 0o600


def _harness():
    from localharness.config.models import HarnessConfig, OrgConfig, ProviderConfig

    return HarnessConfig(version="1", org=OrgConfig(default_model="m"), provider=ProviderConfig(
        provider_type="vllm", base_url="http://127.0.0.1:9/v1", api_key="sk-born-private",
        default_model="m"))


def test_config_yaml_is_born_private_in_an_owner_only_dir(tmp_path, no_chmod):
    from localharness.cli.init_cmd import _write_harness
    from localharness.core.private_files import ensure_private_dir

    config_dir = tmp_path / "config"
    ensure_private_dir(config_dir)
    path = _write_harness(config_dir, _harness(), True)
    assert _mode(config_dir) == 0o700
    assert _mode(path) == 0o600 and "sk-born-private" in path.read_text()


def test_plugins_readme_is_born_private(tmp_path, no_chmod):
    from localharness.cli.init_cmd import _write_harness

    _write_harness(tmp_path, _harness(), True)
    assert _mode(tmp_path / "plugins" / "README.md") == 0o600


def test_init_makes_the_config_dir_owner_only(tmp_path, monkeypatch):
    """The real `init` command end to end, into a folder that does not exist yet."""
    from unittest.mock import AsyncMock, MagicMock

    from typer.testing import CliRunner

    from localharness.cli import init_cmd
    from localharness.cli.app import app
    from localharness.provider.detector import DetectorResult
    from tests.unit.test_init_cmd import _make_capability_result

    monkeypatch.setattr(init_cmd, "detect_provider", AsyncMock(return_value=DetectorResult(
        found=True, provider_type="llamacpp", base_url="http://localhost:8080/v1", models=["qwen"],
        suggested_model="qwen", probe_duration_ms=1.0)))
    client = MagicMock()
    client.detect_capabilities = AsyncMock(return_value=_make_capability_result())
    monkeypatch.setattr(init_cmd, "LLMClient", MagicMock(return_value=client))
    config_dir = tmp_path / "new" / "config"

    result = CliRunner().invoke(app, ["init", "--config-dir", str(config_dir), "--no-input"])

    assert result.exit_code == 0, result.output
    assert _mode(config_dir) == 0o700
    assert _mode(config_dir / "config.yaml") == 0o600


def test_init_force_saves_the_old_config_first(tmp_path, monkeypatch):
    from rich.console import Console

    from localharness.cli import init_cmd

    printed = Console(record=True, width=1000)
    monkeypatch.setattr(init_cmd, "console", printed)
    (tmp_path / "config.yaml").write_bytes(b"old: config\n")
    os.chmod(tmp_path / "config.yaml", 0o664)

    init_cmd._write_harness(tmp_path, _harness(), True, force=True)

    (backup,) = tmp_path.glob("config.yaml.before-init-*")
    assert backup.read_bytes() == b"old: config\n"
    assert _mode(backup) == 0o600
    assert f"Saved your previous config to {backup}" in printed.export_text()
    assert "sk-born-private" in (tmp_path / "config.yaml").read_text()


def test_init_without_force_writes_no_backup(tmp_path):
    from localharness.cli.init_cmd import _write_harness

    (tmp_path / "config.yaml").write_bytes(b"old: config\n")
    _write_harness(tmp_path, _harness(), True)
    assert list(tmp_path.glob("config.yaml.before-init-*")) == []


def test_migrate_backup_is_born_private(tmp_path, no_chmod):
    from localharness.config import migrate

    config = tmp_path / "config.yaml"
    data = {"version": "1", "provider": {"provider_type": "vllm", "base_url": "http://127.0.0.1:9/v1",
                                         "api_key": "sk-born-private", "default_model": "m"},
            "org": {"permissions": {"deny_patterns": ["write(*/.env)"]}}}
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    original = config.read_bytes()
    work = migrate.plan(yaml.safe_load(original))
    assert work is not None, "the premise: an unstamped config has work to do"

    (backup,) = migrate.apply(config, original, work)

    assert backup.read_bytes() == original and _mode(backup) == 0o600


def test_migrate_rewrites_the_config_owner_only(tmp_path):
    from localharness.config import migrate

    config = tmp_path / "config.yaml"
    data = {"version": "1", "provider": {"provider_type": "vllm", "base_url": "http://127.0.0.1:9/v1",
                                         "api_key": "sk-born-private", "default_model": "m"},
            "org": {"permissions": {"deny_patterns": ["write(*/.env)"]}}}
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    os.chmod(config, 0o664)
    original = config.read_bytes()

    migrate.apply(config, original, migrate.plan(yaml.safe_load(original)))

    assert _mode(config) == 0o600


async def test_archive_db_and_its_wal_are_born_private(tmp_path, no_chmod):
    from localharness.autoresearch.archive import ArchiveStore

    db = tmp_path / "autoresearch" / "archive.db"
    store = ArchiveStore(db)
    await store.open()
    try:
        assert _mode(db) == 0o600
        wal = db.with_name("archive.db-wal")
        assert wal.exists(), "the premise: the migrations went through the WAL"
        assert _mode(wal) == 0o600
    finally:
        await store.close()


def test_run_journal_is_born_private(tmp_path, no_chmod):
    from localharness.autoresearch.loop import RunJournal

    journal = RunJournal("run-1", tmp_path)
    journal.write({"event": "start"})
    assert _mode(journal.path) == 0o600


def test_budget_file_is_born_private(tmp_path, no_chmod):
    from localharness.autoresearch.budget import WindowMeter

    path = tmp_path / "autoresearch" / "window.json"
    WindowMeter(window_budget_tokens=1000, state_path=path, clock=lambda: 100.0)
    assert path.exists() and _mode(path) == 0o600


# --------------------------------------------------------------------------- the tighten at start (R10, R16)


def _tighten_records(config_dir: Path) -> list[dict]:
    import json

    audit = config_dir / "audit.jsonl"
    if not audit.exists():
        return []
    rows = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines() if line]
    return [r for r in rows if r.get("path") == "config_dir.mode"]


def _said_anything_about_it(printed: list[str], captured) -> bool:
    text = "\n".join(printed) + captured.out + captured.err
    return any(word in text for word in ("0700", "0o7", "owner-only", "config_dir.mode", "tighten"))


def _open_config_dir(tmp_path: Path) -> Path:
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    os.chmod(config_dir, 0o775)  # what an install from before this release has
    return config_dir


async def test_the_tighten_is_silent_and_once(tmp_path, monkeypatch, capsys):
    from localharness.cli.start_cmd import _start_async
    from tests.unit.test_start_cmd import _capture_start_console, _stub_start_boundaries

    config_dir = _open_config_dir(tmp_path)
    _stub_start_boundaries(config_dir, monkeypatch)
    printed = _capture_start_console(monkeypatch)

    await _start_async(None, False, False, str(config_dir))

    assert _mode(config_dir) == 0o700
    (record,) = _tighten_records(config_dir)
    assert record["event_type"] == "ComponentMutated"
    assert (record["before_value"], record["after_value"]) == ("0o775", "0o700")
    assert (record["layer"], record["actor"]) == ("user", "cli")
    assert not _said_anything_about_it(printed, capsys.readouterr())

    await _start_async(None, False, False, str(config_dir))

    assert len(_tighten_records(config_dir)) == 1, "a second start found nothing to tighten"


async def test_a_config_dir_named_by_localharness_dir_is_tightened_too(tmp_path, monkeypatch):
    from localharness.cli.start_cmd import _start_async
    from tests.unit.test_start_cmd import _stub_start_boundaries

    config_dir = _open_config_dir(tmp_path)
    _stub_start_boundaries(config_dir, monkeypatch)
    monkeypatch.setenv("LOCALHARNESS_DIR", str(config_dir))

    await _start_async(None, False, False, None)

    assert _mode(config_dir) == 0o700
    assert len(_tighten_records(config_dir)) == 1


@pytest.mark.parametrize("code", [errno.EROFS, errno.EPERM], ids=["read-only", "another-owner"])
async def test_a_directory_that_cannot_be_tightened_never_blocks_start(tmp_path, monkeypatch, capsys,
                                                                       code):
    """SIMULATED: os.chmod refuses for the config folder only (a read-only mount, a folder another
    account owns); every other chmod runs as usual."""
    from localharness.cli.start_cmd import _start_async
    from tests.unit.test_start_cmd import _capture_start_console, _stub_start_boundaries

    config_dir = _open_config_dir(tmp_path)
    _stub_start_boundaries(config_dir, monkeypatch)
    printed = _capture_start_console(monkeypatch)
    real_chmod = os.chmod

    def chmod(path, mode, *a, **k):
        if Path(path) == config_dir:
            raise OSError(code, os.strerror(code))
        return real_chmod(path, mode, *a, **k)

    monkeypatch.setattr(os, "chmod", chmod)

    await _start_async(None, False, False, str(config_dir))  # completes

    assert _mode(config_dir) == 0o775
    assert _tighten_records(config_dir) == []
    assert not _said_anything_about_it(printed, capsys.readouterr())
    assert any("startup)" in line for line in printed), "the session came up"


# --------------------------------------------------------------------------- doctor's rows


def _doctor_layout(tmp_path, monkeypatch, fake_home):
    from tests.unit.test_doctor_layer_report import _layout

    return _layout(tmp_path, monkeypatch, fake_home, workspace=False).global_dir


def _rows(output: str, needle: str) -> list[str]:
    return [line for line in output.splitlines() if needle in line]


def test_doctor_names_an_open_folder_it_owns(tmp_path, monkeypatch, fake_home):
    from tests.unit.test_doctor_layer_report import _run_doctor

    config_dir = _doctor_layout(tmp_path, monkeypatch, fake_home)
    os.chmod(config_dir, 0o755)

    (row,) = _rows(_run_doctor(), "readable by other accounts")
    assert (f"{config_dir} is readable by other accounts on this machine — `localharness start` "
            f"makes it owner-only (0700), or run `chmod 700 {config_dir}`") in row
    assert row.lstrip().startswith("⚠")


def test_doctor_names_an_open_folder_another_account_owns(tmp_path, monkeypatch, fake_home):
    from tests.unit.test_doctor_layer_report import _run_doctor

    config_dir = _doctor_layout(tmp_path, monkeypatch, fake_home)
    os.chmod(config_dir, 0o755)
    someone_else = os.stat(config_dir).st_uid + 1
    monkeypatch.setattr(os, "geteuid", lambda: someone_else)  # SIMULATED: never a real chown

    (row,) = _rows(_run_doctor(), "readable by other accounts")
    assert (f"{config_dir} is readable by other accounts and is owned by another account, so it "
            "cannot be made owner-only from here — ask its owner, or move your config to a folder "
            "you own") in row


def test_doctor_names_an_open_folder_on_a_read_only_filesystem(tmp_path, monkeypatch, fake_home):
    from tests.unit.test_doctor_layer_report import _run_doctor

    config_dir = _doctor_layout(tmp_path, monkeypatch, fake_home)
    os.chmod(config_dir, 0o755)
    real_statvfs = os.statvfs

    def statvfs(path):  # SIMULATED: the config folder's mount is read-only
        if Path(path) == config_dir:
            return SimpleNamespace(f_flag=real_statvfs(path).f_flag | os.ST_RDONLY)
        return real_statvfs(path)

    monkeypatch.setattr(os, "statvfs", statvfs)

    (row,) = _rows(_run_doctor(), "readable by other accounts")
    assert (f"{config_dir} is readable by other accounts and is on a read-only filesystem, so start "
            f"could not make it owner-only — make it writable and run `chmod 700 {config_dir}`, or "
            "move your config to a folder you own") in row


def test_doctor_counts_open_files_inside_a_private_folder(tmp_path, monkeypatch, fake_home):
    from tests.unit.test_doctor_layer_report import _run_doctor

    config_dir = _doctor_layout(tmp_path, monkeypatch, fake_home)
    os.chmod(config_dir, 0o700)
    os.chmod(config_dir / "config.yaml", 0o644)

    (row,) = _rows(_run_doctor(), "readable by other accounts")
    assert (f"1 file(s) in {config_dir} are readable by other accounts if the folder is ever opened "
            f"up (e.g. {config_dir / 'config.yaml'}) — `chmod -R go-rwx {config_dir}` makes each "
            "private") in row
    assert row.lstrip().startswith("i")


def test_doctor_says_nothing_about_an_all_private_folder(tmp_path, monkeypatch, fake_home):
    from tests.unit.test_doctor_layer_report import _run_doctor

    config_dir = _doctor_layout(tmp_path, monkeypatch, fake_home)
    os.chmod(config_dir, 0o700)
    os.chmod(config_dir / "config.yaml", 0o600)

    assert _rows(_run_doctor(), "readable by other accounts") == []
