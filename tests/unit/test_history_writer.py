"""Tests for HistoryWriter and memory error hierarchy."""
import json
import pytest
from pathlib import Path
from localharness.memory.errors import (
    MemoryError,
    MemoryWriteError,
    MemoryReadError,
    MemoryCorruptionError,
    DiskFullError,
)
from localharness.memory.history import HistoryWriter

VALID_RECORD = {
    "v": 1,
    "type": "user_message",
    "id": "msg_01",
    "session_id": "sess_01",
    "agent_id": "agent_01",
    "ts": 1748042400,
    "content": "hello",
}


def make_record(**overrides) -> dict:
    r = dict(VALID_RECORD)
    r.update(overrides)
    return r


# --- Error hierarchy tests ---

def test_memory_error_hierarchy():
    assert issubclass(MemoryWriteError, MemoryError)
    assert issubclass(MemoryReadError, MemoryError)
    assert issubclass(MemoryCorruptionError, MemoryError)
    assert issubclass(DiskFullError, MemoryWriteError)


def test_memory_write_error_attributes():
    exc = MemoryWriteError("/tmp/foo.jsonl", ValueError("boom"))
    assert exc.path == "/tmp/foo.jsonl"
    assert isinstance(exc.underlying, ValueError)
    assert "/tmp/foo.jsonl" in str(exc)


def test_memory_corruption_error_attributes():
    exc = MemoryCorruptionError("/tmp/foo.jsonl", "Line 3: bad json")
    assert exc.path == "/tmp/foo.jsonl"
    assert exc.detail == "Line 3: bad json"
    assert "/tmp/foo.jsonl" in str(exc)


# --- HistoryWriter tests ---

@pytest.mark.anyio
async def test_append_creates_file(tmp_path):
    path = tmp_path / "history.jsonl"
    writer = HistoryWriter(path)
    assert not path.exists()
    await writer.append(VALID_RECORD)
    assert path.exists()


@pytest.mark.anyio
async def test_append_and_read_all(tmp_path):
    path = tmp_path / "history.jsonl"
    writer = HistoryWriter(path)
    r1 = make_record(id="msg_01", type="user_message")
    r2 = make_record(id="msg_02", type="assistant_message")
    r3 = make_record(id="msg_03", type="tool_result")
    await writer.append(r1)
    await writer.append(r2)
    await writer.append(r3)
    records = await writer.read_all()
    assert len(records) == 3
    assert records[0]["id"] == "msg_01"
    assert records[1]["id"] == "msg_02"
    assert records[2]["id"] == "msg_03"


@pytest.mark.anyio
async def test_append_required_fields(tmp_path):
    path = tmp_path / "history.jsonl"
    writer = HistoryWriter(path)
    for field in ("v", "type", "id", "session_id", "agent_id", "ts"):
        bad = dict(VALID_RECORD)
        del bad[field]
        with pytest.raises(ValueError, match=field):
            await writer.append(bad)


@pytest.mark.anyio
async def test_append_unknown_type(tmp_path):
    path = tmp_path / "history.jsonl"
    writer = HistoryWriter(path)
    bad = make_record(type="unknown_type")
    with pytest.raises(ValueError, match="unknown_type"):
        await writer.append(bad)


@pytest.mark.anyio
async def test_read_all_empty_file(tmp_path):
    path = tmp_path / "nonexistent.jsonl"
    writer = HistoryWriter(path)
    records = await writer.read_all()
    assert records == []


@pytest.mark.anyio
async def test_read_all_partial_line(tmp_path):
    """Partial last line (crash write) is skipped. Mid-file corruption raises."""
    path = tmp_path / "history.jsonl"
    # Write 2 valid records then a partial line at the end
    r1 = make_record(id="msg_01")
    r2 = make_record(id="msg_02")
    line1 = json.dumps(r1) + "\n"
    line2 = json.dumps(r2) + "\n"
    partial = '{"v": 1, "type": "user_message'  # truncated
    path.write_text(line1 + line2 + partial, encoding="utf-8")

    writer = HistoryWriter(path)
    records = await writer.read_all()
    assert len(records) == 2
    assert records[0]["id"] == "msg_01"
    assert records[1]["id"] == "msg_02"


@pytest.mark.anyio
async def test_read_all_mid_file_corruption(tmp_path):
    """Corruption in middle of file raises MemoryCorruptionError."""
    path = tmp_path / "history.jsonl"
    r1 = make_record(id="msg_01")
    r3 = make_record(id="msg_03")
    bad_line = "not valid json at all\n"
    path.write_text(
        json.dumps(r1) + "\n" + bad_line + json.dumps(r3) + "\n",
        encoding="utf-8",
    )
    writer = HistoryWriter(path)
    with pytest.raises(MemoryCorruptionError):
        await writer.read_all()


@pytest.mark.anyio
async def test_read_last_n(tmp_path):
    path = tmp_path / "history.jsonl"
    writer = HistoryWriter(path)
    records = [make_record(id=f"msg_{i:02d}") for i in range(5)]
    for r in records:
        await writer.append(r)
    last2 = await writer.read_last_n(2)
    assert len(last2) == 2
    assert last2[0]["id"] == "msg_03"
    assert last2[1]["id"] == "msg_04"


@pytest.mark.anyio
async def test_read_last_n_more_than_exists(tmp_path):
    path = tmp_path / "history.jsonl"
    writer = HistoryWriter(path)
    for i in range(3):
        await writer.append(make_record(id=f"msg_{i:02d}"))
    result = await writer.read_last_n(10)
    assert len(result) == 3


@pytest.mark.anyio
async def test_jsonl_line_format(tmp_path):
    path = tmp_path / "history.jsonl"
    writer = HistoryWriter(path)
    await writer.append(VALID_RECORD)
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["id"] == VALID_RECORD["id"]
    assert text.endswith("\n")
