"""Doctor's workspace layer report (CLI-02) and its migration-state block (dogfood F6).

`doctor` is the command people run when the harness has already surprised them. Phase 39 taught it
to print both LAYER PATHS; this file grades the three things it still could not explain:

1. **Which layer won, per key** (ROADMAP criterion 2). Two catalogue builds — this session's
   layering and the same machine with the workspace switched off — diffed by resolved VALUE.
2. **F5's attribution arriving through DOCTOR.** 43-01 rebuilt `ConfigValidationError` so a bad
   value in a workspace `config.yaml` names the workspace file and its own line. Doctor renders
   `str(exc)` and so gets that for free — which is exactly the kind of claim that is true until it
   is not, so it is asserted here from a real `doctor` run rather than assumed. If these reddens,
   the bug is in `config/loader.py`, not in doctor.
3. **The migration state** (F6). `start` folds new shipped deny-defaults into `config.yaml` on the
   first start after an upgrade and announces it ONCE; that announcement scrolls away in a long or
   failing session, and the only durable trace was a `.bak` file nobody was told to look for.

Every invocation goes through the real Typer app from `proj/src/pkg` — a subdirectory, because
"run from anywhere inside your project" is the behavior the workspace layer exists for and a test
that runs at the project root cannot tell discovery from a literal `./.localharness` read.

The configured `base_url` points at port 9 (discard), refused instantly on loopback: doctor's
endpoint probe fails fast, costs no timeout, and this file never touches the network. Doctor
therefore exits 1 on the endpoint failure in every test here — assertions are on OUTPUT, and the
exit code is only ever compared BETWEEN runs (the stale-revision test), never to 0.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

from localharness.cli.app import app
from localharness.config.paths import discover_workspace_dir

runner = CliRunner()

# Port 9 is discard: refused immediately on loopback, so the probe costs no timeout and the
# machine-safety rule (never reach a live model from the suite) holds by construction.
_UNREACHABLE_BASE_URL = "http://127.0.0.1:9/v1"


# --------------------------------------------------------------------------------- helpers


def _squash(text: str) -> str:
    """Whitespace-stripped comparison — rich wraps at the console width, so a long row can arrive
    in the capture with a newline folded into it (39-04's lesson)."""
    return "".join(text.split())


def _global_config(*, name: str = "GLOBAL-ORG", log_level: str = "info") -> str:
    """A COMPLETE harness config.yaml. `org.name` / `org.log_level` are the probes throughout:
    two plain scalars on two different lines, in two different layers."""
    return yaml.dump(
        {
            "version": "1",
            "provider": {
                "provider_type": "vllm",
                "base_url": _UNREACHABLE_BASE_URL,
                "default_model": "test-model",
                "available_models": ["test-model"],
            },
            "org": {"name": name, "log_level": log_level},
        },
        sort_keys=False,
    )


def _prompt_must_not_fire(monkeypatch) -> None:
    """A rule that forbids a prompt is tested by making the prompt RAISE. Silence proves nothing,
    and an unpatched prompt on a path that should not ask would HANG the runner on stdin."""

    def _boom(*args, **kwargs):
        raise AssertionError("prompted for trust on an in-project workspace, which never asks")

    monkeypatch.setattr("rich.prompt.Confirm.ask", _boom)


def _layout(
    tmp_path: Path,
    monkeypatch,
    *,
    workspace: bool = True,
    global_config: str | None = None,
) -> SimpleNamespace:
    """A fake `$HOME` holding the GLOBAL layer, plus a project with the CWD two levels down.

        <home>/.localharness/      GLOBAL layer: config.yaml
        <home>/proj/.git/          plain empty dir — the project boundary marker
        <home>/proj/.localharness/ WORKSPACE layer: config.yaml, written per test
        <home>/proj/src/pkg/       the CWD every invocation runs from

    All three env moves are load-bearing. Both walks stop at `$HOME`, so a real one would let the
    developer's own `~/.localharness` answer these tests; and the autouse conftest fixture sets
    `LOCALHARNESS_HOME`, which `resolve_workspace_layer` counts as an EXPLICIT selection — leaving
    it set would switch discovery off and every test below would pass for the wrong reason.
    """
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.delenv("LOCALHARNESS_HOME", raising=False)
    home = tmp_path / "home"
    global_dir = home / ".localharness"
    global_dir.mkdir(parents=True)
    (global_dir / "config.yaml").write_text(
        global_config if global_config is not None else _global_config(), encoding="utf-8"
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("COLUMNS", "400")  # keep rich from folding a long key row mid-assertion

    proj = home / "proj"
    (proj / ".git").mkdir(parents=True)
    ws_dir = proj / ".localharness"
    if workspace:
        ws_dir.mkdir(parents=True)

    deep_dir = proj / "src" / "pkg"
    deep_dir.mkdir(parents=True)
    monkeypatch.chdir(deep_dir)
    _prompt_must_not_fire(monkeypatch)

    # Guards: if either flipped, every assertion below would pass for the wrong reason.
    if workspace:
        assert discover_workspace_dir() == ws_dir.resolve()
    else:
        assert discover_workspace_dir() is None

    return SimpleNamespace(
        home=home, global_dir=global_dir, ws_dir=ws_dir, proj=proj, deep_dir=deep_dir
    )


def _write_workspace(layout: SimpleNamespace, data: dict | str) -> Path:
    path = layout.ws_dir / "config.yaml"
    text = data if isinstance(data, str) else yaml.dump(data, sort_keys=False)
    path.write_text(text, encoding="utf-8")
    return path


def _run_doctor() -> str:
    result = runner.invoke(app, ["doctor"])
    # A crash is not a failing check: doctor must survive every layout in this file, and an
    # exception swallowed into stdout would make the substring assertions below meaningless.
    assert result.exception is None or isinstance(result.exception, SystemExit), (
        f"doctor raised {result.exception!r}\n{result.stdout}"
    )
    return result.stdout


def _line_of(path: Path, needle: str) -> int:
    """The 1-based line number of `needle` in `path` — the global file's line number is READ, never
    hardcoded, so re-ordering the fixture config cannot silently make an assertion vacuous."""
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if needle in line:
            return i
    raise AssertionError(f"{needle!r} not found in {path}")


# ------------------------------------------------- criterion 2: which layer won, for which key


def test_doctor_names_the_winning_layer_for_every_changed_key(tmp_path, monkeypatch):
    """ROADMAP criterion 2, from a subdirectory: both layer paths AND the winning layer per key.

    Both the winner and the loser are printed on the row — a user debugging a surprise needs to see
    WHAT the workspace replaced, not only what is in force now. (40-05's e2e made the opposite call
    for a different reason: there, asserting the loser ABSENT was the only way to grade the merge.)
    The DISPLAY shows both; the ASSERTION below pins the winner to the workspace band, which is the
    claim CLI-02 actually makes.
    """
    layout = _layout(tmp_path, monkeypatch)
    _write_workspace(layout, {"org": {"name": "WORKSPACE-ORG", "log_level": "debug"}})

    out = _run_doctor()
    squashed = _squash(out)

    assert "2 key(s) overridden by this workspace:" in out, out
    assert _squash("org.log_level = 'debug'  [workspace-config]  (global: 'info')") in squashed, out
    assert (
        _squash("org.name = 'WORKSPACE-ORG'  [workspace-config]  (global: 'GLOBAL-ORG')")
        in squashed
    ), out
    # The layer paths from 39-05 are still there — criterion 2 is BOTH halves in one run.
    assert _squash(f"Workspace layer: {layout.ws_dir}") in squashed, out
    assert _squash(f"Global layer:    {layout.global_dir}") in squashed, out


def test_a_key_restated_with_the_same_value_is_not_an_override(tmp_path, monkeypatch):
    """"Overridden" is a VALUE difference, not a presence check.

    One body, both assertions, deliberately: a workspace that restates `org.name` with the value the
    global layer already had and genuinely changes `org.log_level` must show exactly one row. An
    implementation that lists nothing would pass the absence half alone, so the presence half is
    asserted from the SAME config — that is what makes the pair discriminating.

    Worth stating because it is counter-intuitive: `org.name`'s winning layer IS `workspace-config`
    here (the workspace file really is the last one to set it). It is absent from this section
    because reporting it would make doctor noise on exactly the configs people copy between
    projects, not because the attribution is wrong.
    """
    layout = _layout(tmp_path, monkeypatch)
    _write_workspace(layout, {"org": {"name": "GLOBAL-ORG", "log_level": "debug"}})

    out = _run_doctor()

    assert "1 key(s) overridden by this workspace:" in out, out
    assert "org.log_level" in out, out
    assert "org.name" not in out, out


def test_a_workspace_that_overrides_nothing_says_so(tmp_path, monkeypatch):
    """One line, not an empty section. A heading with nothing under it reads like a bug in doctor;
    "the global config governs every key" is an answer."""
    layout = _layout(tmp_path, monkeypatch)
    _write_workspace(layout, "# a workspace that only exists to hold agents/\n")

    out = _run_doctor()

    assert "No overrides" in out, out
    assert "key(s) overridden" not in out, out


def test_layr03_no_workspace_prints_no_workspace_vocabulary(tmp_path, monkeypatch):
    """LAYR-03's control: with no `.localharness/` up-tree, NOTHING about layering may appear.

    Every offending token is collected before asserting, rather than checked one `assert` at a
    time — an ordered chain would let the first token shadow the rest, and the mutation that moves
    the section's call outside the `if workspace is not None:` guard reddens on exactly one of
    them (`No overrides`, because the two catalogue builds are then identical). 41-06's lesson,
    fifth shape: make the shadowing impossible instead of arguing about the order.
    """
    _layout(tmp_path, monkeypatch, workspace=False)

    out = _run_doctor()

    leaked = [
        token
        for token in ("workspace-config", "workspace-overrides", "overridden", "No overrides")
        if token in out
    ]
    assert leaked == [], f"workspace vocabulary leaked with no workspace: {leaked}\n{out}"


# ------------------------------------------------- F5, proven through doctor rather than assumed


def test_f5_a_bad_workspace_value_is_reported_against_the_workspace_file(tmp_path, monkeypatch):
    """The post-42 dogfood repro, through DOCTOR.

    Before 43-01 a bad value on line 3 of a workspace `config.yaml` was reported as a line of the
    GLOBAL `config.yaml` — a line where a perfectly valid value sits, so the user opened the wrong
    file and found nothing wrong. Doctor renders `str(exc)`, so 43-01's attribution should arrive
    with no doctor edit; this asserts it does.

    Scoped to the error TEXT, not to the whole capture, because doctor prints the global
    `config.yaml`'s path unconditionally two lines earlier (`Config file:`, step 2, pre-v0.13). The
    claim being graded is that the global file is not named as the CAUSE.
    """
    layout = _layout(tmp_path, monkeypatch)
    ws_config = _write_workspace(layout, "org:\n  name: WS\n  log_level: not-a-level\n")
    global_config = layout.global_dir / "config.yaml"
    global_line = _line_of(global_config, "log_level")
    assert global_line != 3, "fixture broken: the two layers must disagree about the line number"

    out = _run_doctor()
    assert "Config invalid:" in out, out
    error_text = out.split("Config invalid:", 1)[1].split("(skipped:", 1)[0]

    assert _squash(str(ws_config)) in _squash(error_text), error_text
    assert "(line 3)" in error_text, error_text
    assert _squash(str(global_config)) not in _squash(error_text), error_text
    assert f"(line {global_line})" not in error_text, error_text
