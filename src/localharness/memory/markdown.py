"""MEMORY.md file manager for LocalHarness agents."""
import os
from datetime import datetime, timezone
from pathlib import Path

VALID_WRITABLE_SECTIONS = frozenset({"working_notes", "learned_behaviors"})

_SECTION_HEADINGS = {
    "identity": "Identity",
    "persistent_facts": "Persistent Facts",
    "working_notes": "Working Notes",
    "learned_behaviors": "Learned Behaviors",
    "session_history": "Session History",
}


class MarkdownMemory:
    """
    MEMORY.md file manager.

    Parses the markdown file into named sections for selective update.
    Section boundaries are defined by ## headings.
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    def exists(self) -> bool:
        return self._path.exists()

    def read(self) -> str:
        """Read full file content. Returns empty string if file does not exist."""
        if not self._path.exists():
            return ""
        return self._path.read_text(encoding="utf-8")

    def get_section(self, section_slug: str) -> str:
        """
        Extract content of a named section by slug.
        Returns empty string if section not found or file doesn't exist.
        """
        heading = _SECTION_HEADINGS.get(section_slug)
        if heading is None:
            return ""
        content = self.read()
        if not content:
            return ""
        return _extract_section(content, heading)

    def update_section(self, section_slug: str, content: str) -> None:
        """
        Replace content of a named section in place.

        Raises:
            ValueError: If section_slug not in VALID_WRITABLE_SECTIONS.
            FileNotFoundError: If the markdown file does not exist.
        """
        if section_slug not in VALID_WRITABLE_SECTIONS:
            raise ValueError(
                f"Section {section_slug!r} is not writable. "
                f"Valid: {sorted(VALID_WRITABLE_SECTIONS)}"
            )
        if not self._path.exists():
            raise FileNotFoundError(f"MEMORY.md not found: {self._path}")
        heading = _SECTION_HEADINGS[section_slug]
        current = self._path.read_text(encoding="utf-8")
        updated = _replace_section(current, heading, content)
        _atomic_write(self._path, updated)

    def regenerate(
        self,
        agent_id: str,
        agent_name: str,
        role: str,
        facts_text: str,
        session_entry: str | None,
    ) -> None:
        """
        Rewrite the file, preserving working_notes and learned_behaviors verbatim.
        Creates the file if it does not exist.
        """
        # Preserve mutable sections from existing file
        existing_working_notes = ""
        existing_learned_behaviors = ""
        existing_session_lines = ""

        if self._path.exists():
            existing = self._path.read_text(encoding="utf-8")
            existing_working_notes = _extract_section(existing, "Working Notes")
            existing_learned_behaviors = _extract_section(existing, "Learned Behaviors")
            existing_session_lines = _extract_section(existing, "Session History")

        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        working_notes = existing_working_notes or "(No working notes yet.)"
        learned_behaviors = existing_learned_behaviors or "(No learned behaviors yet.)"
        facts_body = facts_text if facts_text.strip() else "(No facts recorded yet.)"

        # Build session history: prepend new entry
        if session_entry:
            if existing_session_lines:
                session_body = session_entry + "\n" + existing_session_lines
            else:
                session_body = session_entry
        else:
            session_body = existing_session_lines or "(No sessions recorded yet.)"

        new_content = (
            f"# Memory: {agent_name}\n\n"
            f"Last updated: {now}\n"
            f"Agent ID: {agent_id}\n\n"
            f"## Identity\n\n"
            f"{role}\n\n"
            f"## Persistent Facts\n\n"
            f"{facts_body}\n\n"
            f"## Working Notes\n\n"
            f"{working_notes}\n\n"
            f"## Learned Behaviors\n\n"
            f"{learned_behaviors}\n\n"
            f"## Session History\n\n"
            f"{session_body}\n"
        )

        self._path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(self._path, new_content)


# --- Internal helpers ---

def _extract_section(content: str, heading: str) -> str:
    """Extract text between ## heading and next ## heading (or EOF). Strip whitespace."""
    marker = f"## {heading}"
    start = content.find(marker)
    if start == -1:
        return ""
    # Find end of heading line
    body_start = content.find("\n", start)
    if body_start == -1:
        return ""
    body_start += 1  # skip newline after heading

    # Find next ## heading
    next_section = content.find("\n## ", body_start)
    if next_section == -1:
        body = content[body_start:]
    else:
        body = content[body_start:next_section]
    return body.strip()


def _replace_section(content: str, heading: str, new_body: str) -> str:
    """Replace the body of a named ## section with new_body."""
    marker = f"## {heading}"
    start = content.find(marker)
    if start == -1:
        # Section not found — append it
        return content.rstrip() + f"\n\n{marker}\n\n{new_body}\n"

    body_start = content.find("\n", start)
    if body_start == -1:
        return content + f"\n\n{new_body}\n"
    body_start += 1

    next_section = content.find("\n## ", body_start)
    if next_section == -1:
        after = ""
        before = content[:body_start]
    else:
        after = content[next_section:]
        before = content[:body_start]

    return before + new_body.strip() + "\n" + after


def _atomic_write(path: Path, content: str) -> None:
    """Write content to path atomically using tmp + os.replace."""
    tmp_path = path.with_suffix(".md.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, path)
