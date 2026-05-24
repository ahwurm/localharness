"""Tests for MarkdownMemory MEMORY.md manager."""
import pytest
from pathlib import Path
from localharness.memory.markdown import MarkdownMemory, VALID_WRITABLE_SECTIONS


# --- Basic existence/read tests ---

def test_exists_false_when_no_file(tmp_path):
    mm = MarkdownMemory(tmp_path / "MEMORY.md")
    assert mm.exists() is False


def test_exists_true_when_file_exists(tmp_path):
    mm = MarkdownMemory(tmp_path / "MEMORY.md")
    mm.regenerate("agent_01", "Test Agent", "A helpful agent.", "", None)
    assert mm.exists() is True


def test_read_empty_when_no_file(tmp_path):
    mm = MarkdownMemory(tmp_path / "MEMORY.md")
    assert mm.read() == ""


def test_read_returns_content(tmp_path):
    mm = MarkdownMemory(tmp_path / "MEMORY.md")
    mm.regenerate("agent_01", "Test Agent", "A helpful agent.", "fact1: value1", None)
    content = mm.read()
    assert len(content) > 0
    assert "Test Agent" in content


# --- regenerate structure tests ---

def test_regenerate_creates_file(tmp_path):
    path = tmp_path / "MEMORY.md"
    mm = MarkdownMemory(path)
    assert not path.exists()
    mm.regenerate("agent_01", "Test Agent", "A helpful agent.", "", None)
    assert path.exists()


def test_regenerate_contains_all_sections(tmp_path):
    mm = MarkdownMemory(tmp_path / "MEMORY.md")
    mm.regenerate("agent_01", "Test Agent", "A helpful agent.", "", None)
    content = mm.read()
    assert "# Memory:" in content
    assert "## Identity" in content
    assert "## Persistent Facts" in content
    assert "## Working Notes" in content
    assert "## Learned Behaviors" in content
    assert "## Session History" in content


def test_regenerate_preserves_working_notes(tmp_path):
    mm = MarkdownMemory(tmp_path / "MEMORY.md")
    mm.regenerate("agent_01", "Test Agent", "role", "", None)
    mm.update_section("working_notes", "My important notes here.")
    # Now regenerate again
    mm.regenerate("agent_01", "Test Agent", "role", "new facts", None)
    assert "My important notes here." in mm.read()


def test_regenerate_preserves_learned_behaviors(tmp_path):
    mm = MarkdownMemory(tmp_path / "MEMORY.md")
    mm.regenerate("agent_01", "Test Agent", "role", "", None)
    mm.update_section("learned_behaviors", "Always respond concisely.")
    mm.regenerate("agent_01", "Test Agent", "role", "", None)
    assert "Always respond concisely." in mm.read()


def test_regenerate_updates_facts(tmp_path):
    mm = MarkdownMemory(tmp_path / "MEMORY.md")
    mm.regenerate("agent_01", "Test Agent", "role", "old fact: old value", None)
    mm.regenerate("agent_01", "Test Agent", "role", "new fact: new value", None)
    content = mm.read()
    assert "new fact: new value" in content


def test_regenerate_appends_session_entry(tmp_path):
    mm = MarkdownMemory(tmp_path / "MEMORY.md")
    mm.regenerate("agent_01", "Test Agent", "role", "", "2026-05-24: Did useful work")
    content = mm.read()
    assert "2026-05-24: Did useful work" in content


def test_regenerate_prepends_session_entry(tmp_path):
    """New session entries appear before old ones in session history."""
    mm = MarkdownMemory(tmp_path / "MEMORY.md")
    mm.regenerate("agent_01", "Test Agent", "role", "", "2026-05-23: First session")
    mm.regenerate("agent_01", "Test Agent", "role", "", "2026-05-24: Second session")
    content = mm.read()
    idx_second = content.index("2026-05-24: Second session")
    idx_first = content.index("2026-05-23: First session")
    assert idx_second < idx_first  # newer entry comes first


# --- get_section tests ---

def test_get_section_returns_content(tmp_path):
    mm = MarkdownMemory(tmp_path / "MEMORY.md")
    mm.regenerate("agent_01", "Test Agent", "role", "", None)
    mm.update_section("working_notes", "Some working notes.")
    result = mm.get_section("working_notes")
    assert "Some working notes." in result


def test_get_section_empty_for_missing(tmp_path):
    mm = MarkdownMemory(tmp_path / "MEMORY.md")
    mm.regenerate("agent_01", "Test Agent", "role", "", None)
    assert mm.get_section("nonexistent") == ""


def test_get_section_empty_when_no_file(tmp_path):
    mm = MarkdownMemory(tmp_path / "MEMORY.md")
    assert mm.get_section("working_notes") == ""


# --- update_section tests ---

def test_update_section_replaces_content(tmp_path):
    mm = MarkdownMemory(tmp_path / "MEMORY.md")
    mm.regenerate("agent_01", "Test Agent", "role", "", None)
    mm.update_section("working_notes", "new text here")
    assert "new text here" in mm.get_section("working_notes")


def test_update_section_rejects_invalid(tmp_path):
    mm = MarkdownMemory(tmp_path / "MEMORY.md")
    mm.regenerate("agent_01", "Test Agent", "role", "", None)
    with pytest.raises(ValueError, match="identity"):
        mm.update_section("identity", "some text")


def test_update_section_rejects_persistent_facts(tmp_path):
    mm = MarkdownMemory(tmp_path / "MEMORY.md")
    mm.regenerate("agent_01", "Test Agent", "role", "", None)
    with pytest.raises(ValueError):
        mm.update_section("persistent_facts", "some text")


def test_update_section_raises_for_missing_file(tmp_path):
    mm = MarkdownMemory(tmp_path / "MEMORY.md")
    with pytest.raises(FileNotFoundError):
        mm.update_section("working_notes", "text")


# --- VALID_WRITABLE_SECTIONS ---

def test_valid_writable_sections():
    assert "working_notes" in VALID_WRITABLE_SECTIONS
    assert "learned_behaviors" in VALID_WRITABLE_SECTIONS
    assert "identity" not in VALID_WRITABLE_SECTIONS
    assert "persistent_facts" not in VALID_WRITABLE_SECTIONS


# --- Atomic write test ---

def test_atomic_write(tmp_path):
    """Verify os.replace is used: tmp file should not remain after regenerate."""
    mm = MarkdownMemory(tmp_path / "MEMORY.md")
    mm.regenerate("agent_01", "Test Agent", "role", "", None)
    # No .tmp file should remain after successful regenerate
    tmp_file = tmp_path / "MEMORY.md.tmp"
    assert not tmp_file.exists()
    # The actual file should be valid
    content = mm.read()
    assert "# Memory:" in content
