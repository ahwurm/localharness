"""The localharness skills verb: how the model's shell gate classifies it and, from a later phase, what it prints."""
from __future__ import annotations

import pytest

from localharness.agent.gate_types import Verdict
from localharness.agent.shell_classify import classify_shell
from localharness.agent.verdict import evaluate
from tests.unit.agent.test_verdict import SETTINGS, SHELL_META, granting, make_ctx

MODES = ("guarded", "trusted", "auto", "read-only", "unattended")


@pytest.fixture
def ws(tmp_path, monkeypatch):
    home = (tmp_path / "home").resolve()
    (home / ".localharness").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCALHARNESS_DIR", str(home / ".localharness"))
    workspace = (home / "proj").resolve()
    workspace.mkdir()
    return workspace


def test_localharness_skills_is_one_read_only_segment():
    shell = classify_shell("localharness skills", SETTINGS)
    assert shell.signatures == ("localharness skills",)
    assert all(s.read_only for s in shell.segments)


@pytest.mark.parametrize("mode", MODES)
def test_localharness_skills_asks_nothing_in_any_mode(ws, mode):
    result = evaluate("bash_exec", {"command": "localharness skills"}, SHELL_META, make_ctx(ws, mode=mode), SETTINGS)
    assert result.verdict is Verdict.ALLOW
    assert result.request is None
