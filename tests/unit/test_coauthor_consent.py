"""config/coauthor.py + cli/coauthor.py — per-project co-author consent (spec 14).

Three properties this file exists to hold down:

1. The record lives in the GLOBAL config dir, keyed by project root (git repo root).
   A workspace that could vouch for its own consent is not a consent boundary.
2. Unknown reads as None (undecided), never False. A session that could not ask must
   not harden into a permanent "no"; only an answered prompt records anything.
3. The key is the resolved realpath, so a symlinked checkout and its real path are
   ONE entry.

The autouse `_isolate_localharness_home` fixture (tests/conftest.py) already points
LOCALHARNESS_HOME at a tmp dir, so the store is hermetic without extra setup.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import yaml


@pytest.fixture(autouse=True)
def _global_dir_is_the_hermetic_home(monkeypatch):
    """LOCALHARNESS_DIR outranks LOCALHARNESS_HOME in resolve_config_dir's chain — clear it so
    a developer who exports it doesn't silently point these tests at their real config dir."""
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)


def _expected_store() -> Path:
    return Path(os.environ["LOCALHARNESS_HOME"]) / "coauthor_consent.yaml"


# --------------------------------------------------------------------------- store


def test_consent_store_lives_in_the_global_config_dir():
    from localharness.config.coauthor import COAUTHOR_CONSENT_FILE, consent_store_path

    assert COAUTHOR_CONSENT_FILE == "coauthor_consent.yaml"
    assert consent_store_path() == _expected_store()


def test_unknown_project_reads_as_undecided(tmp_path):
    """None, not False — 'never asked' is a different answer from 'said no'."""
    from localharness.config.coauthor import consent

    assert consent(str(tmp_path / "proj")) is None


def test_records_and_reads_back_granted(tmp_path):
    from localharness.config.coauthor import consent, record_consent

    root = str(tmp_path / "proj")
    record_consent(root, True)
    assert consent(root) is True


def test_records_and_reads_back_declined(tmp_path):
    from localharness.config.coauthor import consent, record_consent

    root = str(tmp_path / "proj")
    record_consent(root, False)
    assert consent(root) is False


def test_per_project_isolation(tmp_path):
    """Recording True for /a does not affect consent for /b."""
    from localharness.config.coauthor import consent, record_consent

    a = str(tmp_path / "a")
    b = str(tmp_path / "b")
    record_consent(a, True)
    assert consent(a) is True
    assert consent(b) is None  # /b was never asked


def test_multiple_projects_coexist_in_same_file(tmp_path):
    from localharness.config.coauthor import consent, record_consent

    a = str(tmp_path / "a")
    b = str(tmp_path / "b")
    record_consent(a, True)
    record_consent(b, False)
    assert consent(a) is True
    assert consent(b) is False
    data = yaml.safe_load(_expected_store().read_text())
    assert len(data["projects"]) == 2


def test_corrupt_file_reads_as_undecided():
    """A corrupt file means 'undecided', never a crash."""
    from localharness.config.coauthor import consent

    store = _expected_store()
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text("not: valid: yaml: [", encoding="utf-8")
    assert consent("/some/project") is None


def test_malformed_project_entry_reads_as_undecided(tmp_path):
    """A project key with a non-bool value is treated as undecided."""
    from localharness.config.coauthor import consent

    store = _expected_store()
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text(
        yaml.safe_dump({"projects": {str(tmp_path / "proj"): {"co_author": "yes"}}}),
        encoding="utf-8",
    )
    assert consent(str(tmp_path / "proj")) is None


def test_key_normalization_trailing_slash(tmp_path):
    """Trailing slash is stripped: /proj and /proj/ are the same key."""
    from localharness.config.coauthor import consent, record_consent

    root = str(tmp_path / "proj")
    record_consent(root, True)
    assert consent(root + "/") is True


def test_key_normalization_symlink(tmp_path):
    """A symlinked path and its resolved target are the same key."""
    from localharness.config.coauthor import consent, record_consent

    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)

    record_consent(str(real), True)
    assert consent(str(link)) is True


def test_record_consent_writes_0600(tmp_path):
    """The consent file is written with 0600 permissions."""
    from localharness.config.coauthor import record_consent

    record_consent(str(tmp_path / "proj"), True)
    mode = _expected_store().stat().st_mode & 0o777
    assert mode == 0o600


# --------------------------------------------------------------------------- trailer


def test_coauthor_trailer_granted():
    from localharness.config.coauthor import coauthor_trailer

    assert coauthor_trailer(True) == "Co-Authored-By: localharness <localharness.agent@gmail.com>"


def test_coauthor_trailer_declined():
    from localharness.config.coauthor import coauthor_trailer

    assert coauthor_trailer(False) is None


def test_prepare_commit_message_appends_trailer():
    from localharness.config.coauthor import prepare_commit_message

    msg = "Fix the bug"
    result = prepare_commit_message(msg, True)
    assert result == "Fix the bug\n\nCo-Authored-By: localharness <localharness.agent@gmail.com>"


def test_prepare_commit_message_no_trailer_when_declined():
    from localharness.config.coauthor import prepare_commit_message

    msg = "Fix the bug"
    assert prepare_commit_message(msg, False) == msg


def test_prepare_commit_message_idempotent():
    """A message that already has the trailer is returned unchanged."""
    from localharness.config.coauthor import prepare_commit_message

    msg = "Fix the bug\n\nCo-Authored-By: localharness <localharness.agent@gmail.com>"
    assert prepare_commit_message(msg, True) == msg


def test_prepare_commit_message_strips_trailing_newline():
    from localharness.config.coauthor import prepare_commit_message

    msg = "Fix the bug\n"
    result = prepare_commit_message(msg, True)
    assert result == "Fix the bug\n\nCo-Authored-By: localharness <localharness.agent@gmail.com>"


# --------------------------------------------------------------------------- prompt

import asyncio
from types import SimpleNamespace


class _FakeGate:
    """Minimal gate stub: records requests, returns a canned answer."""

    def __init__(self, asker=None, allowed=True):
        self.asker = asker
        self._allowed = allowed
        self.requests: list = []

    def _make_asker(self):
        async def asker(request):
            self.requests.append(request)
            return SimpleNamespace(allowed=self._allowed)
        return asker


def test_prompt_no_asker_fails_closed_and_records_nothing(tmp_path):
    """A run that cannot ask returns False and writes NO record — the project stays undecided."""
    from localharness.cli.coauthor import establish_coauthor_consent
    from localharness.config.coauthor import consent

    root = str(tmp_path / "proj")
    gate = _FakeGate(asker=None)
    result = asyncio.run(establish_coauthor_consent(gate, root))
    assert result is False
    assert consent(root) is None  # nothing recorded


def test_prompt_records_granted_when_asked(tmp_path):
    from localharness.cli.coauthor import establish_coauthor_consent
    from localharness.config.coauthor import consent

    root = str(tmp_path / "proj")
    gate = _FakeGate(asker=_FakeGate(allowed=True)._make_asker(), allowed=True)
    result = asyncio.run(establish_coauthor_consent(gate, root))
    assert result is True
    assert consent(root) is True


def test_prompt_records_declined_when_asked(tmp_path):
    from localharness.cli.coauthor import establish_coauthor_consent
    from localharness.config.coauthor import consent

    root = str(tmp_path / "proj")
    gate = _FakeGate(asker=_FakeGate(allowed=False)._make_asker(), allowed=False)
    result = asyncio.run(establish_coauthor_consent(gate, root))
    assert result is False
    assert consent(root) is False


def test_prompt_does_not_ask_when_already_recorded(tmp_path):
    """A recorded decision beats asking — no second question for this project."""
    from localharness.cli.coauthor import establish_coauthor_consent
    from localharness.config.coauthor import record_consent

    root = str(tmp_path / "proj")
    record_consent(root, True)
    gate = _FakeGate(asker=_FakeGate(allowed=False)._make_asker(), allowed=False)
    result = asyncio.run(establish_coauthor_consent(gate, root))
    assert result is True  # the recorded True wins, the asker is never called
    assert gate.requests == []
