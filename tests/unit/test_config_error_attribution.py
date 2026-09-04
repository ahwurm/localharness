"""Config errors name the file that actually owns them (CLI-02 / dogfood F5).

Two failures live in the same code region and this file grades both from the command line, because
what a user READS is the whole point:

1. **Attribution.** `load_harness()` validates the MERGED four-source dict. Before phase 43 it
   hardcoded the GLOBAL `config.yaml` as every error's path and built its line map from only that
   file's text, so a bad value on line 3 of a WORKSPACE `config.yaml` was reported as line 17 of the
   global file — where a perfectly valid value sits. A plausibly-wrong pointer is worse than an
   obviously-bogus one: the user edits a correct line. The post-42 dogfood measured exactly that
   (report-post42.md §5 row F5) and `test_dogfood_f5_repro_*` is that measurement as an assertion.

2. **Malformed overlays.** `load_overlay` did not wrap YAML parse errors, so a malformed
   `overrides.yaml` in EITHER layer escaped `validate_all`'s `except ConfigError` and crashed
   `validate` with a raw `yaml.ParserError` traceback. Phase 40 measured it on both layers and
   deferred it (40's deferred-items.md); the GLOBAL row is the control that proves the crash
   predates workspace layering.

Everything here drives the real Typer app through `CliRunner` against a fake `$HOME` — the
`localharness validate` a user types. Assertions are on OUTPUT, not on exception types alone.
The layout recipe is `tests/integration/test_layer_merge_e2e.py`'s: both discovery walks stop at
`$HOME`, `LOCALHARNESS_DIR`/`LOCALHARNESS_HOME` are cleared (either one counts as an explicit
selection and switches discovery OFF, which would make every assertion below pass for the wrong
reason), `.git` is a plain empty directory so no git binary is needed, and `Confirm.ask` RAISES so
an unexpected trust prompt fails the test instead of hanging it on stdin.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
import yaml
from typer.testing import CliRunner

from localharness.cli.app import app
from localharness.config.paths import discover_workspace_dir

runner = CliRunner()

_UNREACHABLE_BASE_URL = "http://127.0.0.1:9/v1"


_PROVIDER_BLOCK = (
    "provider:\n"
    "  provider_type: vllm\n"
    f"  base_url: {_UNREACHABLE_BASE_URL}\n"
    "  default_model: test-model\n"
)


# --------------------------------------------------------------------------------- layout helpers


def _global_config_with_valid_log_level() -> str:
    """A COMPLETE global config.yaml whose own `org.log_level` is VALID and deliberately far down.

    The dogfood tree was padded with a real provider block, which is why the wrong line the old code
    printed (17) was a plausible one. `_line_of` measures the number instead of hardcoding it, so the
    test cannot silently stop grading if the block is ever reformatted.
    """
    return yaml.dump(
        {
            "version": "1",
            "provider": {
                "provider_type": "vllm",
                "base_url": _UNREACHABLE_BASE_URL,
                "default_model": "test-model",
                "available_models": ["test-model", "other-model"],
                "api_key": "none",
                "timeout_seconds": 120.0,
            },
            "org": {
                "log_level": "info",
            },
        },
        sort_keys=False,
    )


def _line_of(text: str, needle: str) -> int:
    """1-based line number of the first line containing `needle`."""
    for i, line in enumerate(text.splitlines(), start=1):
        if needle in line:
            return i
    raise AssertionError(f"{needle!r} not in:\n{text}")


def _layout(
    tmp_path: Path,
    monkeypatch,
    *,
    global_config: Optional[str] = None,
    global_overlay: Optional[str] = None,
    ws_config: Optional[str] = None,
    ws_overlay: Optional[str] = None,
    workspace: bool = True,
) -> SimpleNamespace:
    """A fake `$HOME` holding the GLOBAL layer, plus a project whose CWD is two levels down.

        <home>/.localharness/      GLOBAL layer: config.yaml (+ overrides.yaml, per test)
        <home>/proj/.git/          plain empty dir — the project boundary marker
        <home>/proj/.localharness/ WORKSPACE layer: config.yaml / overrides.yaml, per test
        <home>/proj/src/pkg/       the CWD every invocation runs from
    """
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.delenv("LOCALHARNESS_HOME", raising=False)
    home = tmp_path / "home"
    global_dir = home / ".localharness"
    global_dir.mkdir(parents=True)
    text = _global_config_with_valid_log_level() if global_config is None else global_config
    (global_dir / "config.yaml").write_text(text, encoding="utf-8")
    if global_overlay is not None:
        (global_dir / "overrides.yaml").write_text(global_overlay, encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("COLUMNS", "400")  # rich wraps at the console width; paths must arrive whole

    proj = home / "proj"
    (proj / ".git").mkdir(parents=True)
    ws_dir = proj / ".localharness"
    if workspace:
        ws_dir.mkdir(parents=True)
        if ws_config is not None:
            (ws_dir / "config.yaml").write_text(ws_config, encoding="utf-8")
        if ws_overlay is not None:
            (ws_dir / "overrides.yaml").write_text(ws_overlay, encoding="utf-8")

    deep_dir = proj / "src" / "pkg"
    deep_dir.mkdir(parents=True)
    monkeypatch.chdir(deep_dir)

    def _boom(*args, **kwargs):
        raise AssertionError("prompted for trust on an in-project workspace, which never asks")

    monkeypatch.setattr("rich.prompt.Confirm.ask", _boom)

    # Guard: if discovery flipped, every assertion below would pass for the wrong reason.
    assert discover_workspace_dir() == (ws_dir.resolve() if workspace else None)

    return SimpleNamespace(
        home=home,
        global_dir=global_dir,
        global_cfg=global_dir / "config.yaml",
        global_ovl=global_dir / "overrides.yaml",
        ws_dir=ws_dir,
        ws_cfg=ws_dir / "config.yaml",
        ws_ovl=ws_dir / "overrides.yaml",
        proj=proj,
        deep_dir=deep_dir,
        global_text=text,
    )


_MALFORMED = "provider: [unterminated\n"


def _assert_clean_report(result) -> None:
    """A malformed file must reach the user as a REPORT, not as a traceback.

    `CliRunner` swallows an uncaught exception into `result.exception` and still returns exit 1, so
    the exit code alone cannot tell a clean failure from a crash — `result.exception` and the
    printed report are what discriminate.
    """
    assert result.exit_code == 1, result.stdout
    assert "invalid" in result.stdout, result.stdout
    assert "ParserError" not in result.stdout, result.stdout
    assert result.exception is None or isinstance(result.exception, SystemExit), (
        f"validate crashed instead of reporting: {result.exception!r}"
    )


# ------------------------------------------------- task 1: a malformed overlay is reported, not raised


def test_malformed_global_overlay_reports_cleanly(tmp_path, monkeypatch):
    """The CONTROL row from phase 40's table: the global overlay's read path predates workspace
    layering and crashed identically, so this proves the fix is not workspace-specific."""
    _layout(tmp_path, monkeypatch, global_overlay=_MALFORMED, workspace=False)

    result = runner.invoke(app, ["validate"])

    _assert_clean_report(result)


def test_malformed_workspace_overlay_reports_cleanly(tmp_path, monkeypatch):
    """The row phase 40 added a second instance of."""
    _layout(tmp_path, monkeypatch, ws_overlay=_MALFORMED)

    result = runner.invoke(app, ["validate"])

    _assert_clean_report(result)


def test_malformed_overlay_error_names_that_overlay_with_a_line(tmp_path, monkeypatch):
    """The error carries the OWNING overlay's own path plus a line/column from the YAML mark —
    the same shape `_load_yaml_file` has always produced for config.yaml."""
    from localharness.config.loader import ConfigParseError
    from localharness.config.overlay import load_overlay

    lay = _layout(tmp_path, monkeypatch, ws_overlay=_MALFORMED)

    with pytest.raises(ConfigParseError) as exc:
        load_overlay(lay.ws_ovl)

    assert exc.value.path == str(lay.ws_ovl)
    assert exc.value.line >= 1
    assert str(lay.ws_ovl) in str(exc.value)


def test_wellformed_missing_and_empty_overlays_are_unchanged(tmp_path):
    """The three non-error paths keep their exact pre-43 behavior."""
    from localharness.config.overlay import load_overlay

    good = tmp_path / "overrides.yaml"
    good.write_text(yaml.dump({"provider": {"default_model": "m"}}), encoding="utf-8")
    assert load_overlay(good) == {"provider": {"default_model": "m"}}

    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    assert load_overlay(empty) == {}

    assert load_overlay(tmp_path / "nope.yaml") == {}


# ---------------------------------------------- task 2: a validation error names its OWNING source


_BAD_WS_CONFIG = 'version: "1"\norg:\n  log_level: not-a-level\n'  # log_level is on LINE 3
_BAD_LOG_LEVEL_LINE = 3


def _loader_for(lay, *, workspace: bool = True):
    """The ConfigLoader shape the CLI builds — global dir plus the discovered workspace dir."""
    from localharness.config.loader import ConfigLoader

    return ConfigLoader(
        config_dir=lay.global_dir, local_config_dir=lay.ws_dir if workspace else None
    )


def test_dogfood_f5_repro_workspace_config_error_names_the_workspace_file(tmp_path, monkeypatch):
    """The post-42 dogfood measurement (report-post42.md §5, F5) as an assertion.

    Measured before this plan: a bad value on line 3 of the WORKSPACE config.yaml was reported as
    line 13 of the GLOBAL file — where a valid `log_level` sits. All four assertions are needed:
    "the right line is printed" also passes when BOTH lines are printed, and a plausibly-wrong
    pointer is worse than an obviously-bogus one because the user edits a correct line.
    """
    lay = _layout(tmp_path, monkeypatch, ws_config=_BAD_WS_CONFIG)
    global_line = _line_of(lay.global_text, "log_level")
    assert global_line != _BAD_LOG_LEVEL_LINE  # premise: the two lines must differ to grade anything

    result = runner.invoke(app, ["validate"])
    out = result.stdout

    assert result.exit_code == 1, out
    assert str(lay.ws_cfg) in out, out  # the workspace file's path IS named
    assert f"Line {_BAD_LOG_LEVEL_LINE}:" in out, out  # its OWN line
    assert str(lay.global_cfg) not in out, out  # the global file is not blamed
    assert f"Line {global_line}:" not in out, out  # nor is the line where its valid value sits


def test_dogfood_f5_repro_the_report_header_is_the_workspace_file(tmp_path, monkeypatch):
    """`doctor` prints the whole `str(exc)`, header included (doctor_cmd.py:164) — that header is
    where the dogfood saw the wrong file named, so it is asserted directly rather than inferred
    from the `validate` rendering."""
    from localharness.config.loader import ConfigValidationError

    lay = _layout(tmp_path, monkeypatch, ws_config=_BAD_WS_CONFIG)

    with pytest.raises(ConfigValidationError) as exc:
        _loader_for(lay).load_harness()

    text = str(exc.value)
    assert text.startswith(f"{lay.ws_cfg}:"), text
    assert str(lay.global_cfg) not in text, text
    assert f"(line {_BAD_LOG_LEVEL_LINE})" in text, text
    # Single owner ⇒ the file heads the report and no error line repeats it (43-05 renders this).
    assert exc.value.errors[0].source_path is None
    assert exc.value.errors[0].field_path == "org.log_level"


def test_global_only_error_text_is_unchanged(tmp_path, monkeypatch):
    """LAYR-03: with no workspace, the FULL rendered error is byte-identical to pre-43.

    `expected` is built through the PRE-43 four-argument `ConfigFieldError` signature, so this is a
    comparison against the old renderer rather than against a string typed out here. It is also the
    guard that reddens if `source_path` is ever set unconditionally.
    """
    from localharness.config.loader import ConfigFieldError, ConfigValidationError

    lay = _layout(
        tmp_path,
        monkeypatch,
        global_config='version: "1"\norg:\n  log_level: not-a-level\n' + _PROVIDER_BLOCK,
        workspace=False,
    )

    with pytest.raises(ConfigValidationError) as exc:
        _loader_for(lay, workspace=False).load_harness()

    fe = exc.value.errors[0]
    assert fe.source_path is None
    expected = ConfigFieldError(fe.field_path, fe.value, fe.message, fe.yaml_line)  # pre-43 form
    assert str(fe) == str(expected)
    assert str(exc.value) == f"{lay.global_cfg}:\n  {expected}"
    assert "overrides.yaml" not in str(exc.value)
    assert str(exc.value).count(str(lay.global_dir)) == 1  # the header, and nowhere else

    # …and the same through the command, where the detail line must gain no path either.
    result = runner.invoke(app, ["validate"])
    assert result.exit_code == 1
    assert f"Line {_BAD_LOG_LEVEL_LINE}: org.log_level:" in result.stdout, result.stdout
    assert str(lay.global_dir) not in result.stdout, result.stdout


def test_workspace_overlay_error_attributes_to_that_overlay(tmp_path, monkeypatch):
    """The highest-priority layer owns what it sets — `overrides.yaml`, not the config.yaml
    beside it."""
    from localharness.config.loader import ConfigValidationError

    lay = _layout(tmp_path, monkeypatch, ws_overlay="org:\n  log_level: not-a-level\n")

    with pytest.raises(ConfigValidationError) as exc:
        _loader_for(lay).load_harness()

    assert str(exc.value).startswith(f"{lay.ws_ovl}:")
    assert exc.value.errors[0].yaml_line == 2  # that overlay's own line, not the config.yaml's


def test_global_overlay_error_attributes_to_that_overlay(tmp_path, monkeypatch):
    """The middle layer is reachable too — all four sources are walked, not just the two files."""
    from localharness.config.loader import ConfigValidationError

    lay = _layout(
        tmp_path, monkeypatch, global_overlay="org:\n  log_level: not-a-level\n", workspace=False
    )

    with pytest.raises(ConfigValidationError) as exc:
        _loader_for(lay, workspace=False).load_harness()

    assert str(exc.value).startswith(f"{lay.global_ovl}:")
    assert exc.value.errors[0].yaml_line == 2


def test_two_bad_files_each_error_names_its_own_file(tmp_path, monkeypatch):
    """Mixed sources: the report keeps the global header and every line names its own owner."""
    from localharness.config.loader import ConfigValidationError

    lay = _layout(
        tmp_path,
        monkeypatch,
        global_config=_PROVIDER_BLOCK + "  timeout_seconds: soon\n",
        ws_config=_BAD_WS_CONFIG,
    )

    with pytest.raises(ConfigValidationError) as exc:
        _loader_for(lay).load_harness()

    by_field = {e.field_path: e for e in exc.value.errors}
    assert set(by_field) == {"provider.timeout_seconds", "org.log_level"}, by_field
    assert exc.value.path == str(lay.global_cfg)  # two owners ⇒ the global file heads the report
    assert by_field["provider.timeout_seconds"].source_path is None  # the header's own file
    assert by_field["org.log_level"].source_path == str(lay.ws_cfg)
    assert by_field["provider.timeout_seconds"].yaml_line == 5  # global file's line
    assert by_field["org.log_level"].yaml_line == _BAD_LOG_LEVEL_LINE  # workspace file's line

    out = runner.invoke(app, ["validate"]).stdout
    assert str(lay.ws_cfg) in out, out


def test_a_list_index_error_attributes_to_its_owning_block(tmp_path, monkeypatch):
    """A Pydantic `loc` can end in a LIST INDEX, which is not a YAML key anywhere.

    `provider.available_models.0` is in no source as a key chain; dropping the last segment finds
    `provider.available_models` in the workspace file. Without that walk-up the error would fall
    all the way back to the global config.yaml and blame the wrong file again.
    """
    from localharness.config.loader import ConfigValidationError

    lay = _layout(
        tmp_path, monkeypatch, ws_config='version: "1"\nprovider:\n  available_models:\n    - 123\n'
    )

    with pytest.raises(ConfigValidationError) as exc:
        _loader_for(lay).load_harness()

    assert exc.value.errors[0].field_path == "provider.available_models.0"
    assert str(exc.value).startswith(f"{lay.ws_cfg}:")


def test_a_field_no_source_sets_falls_back_to_the_global_config(tmp_path, monkeypatch):
    """A cross-field validator's `loc` is EMPTY — no file "sets" it, because the error is about a
    relationship between two values rather than about one line.

    Those fall back to the global config.yaml exactly as they did before this plan. Stated as a
    limit rather than hidden: the block that triggers it here lives in the WORKSPACE file, and the
    report still names the global one.
    """
    from localharness.config.loader import ConfigValidationError

    lay = _layout(
        tmp_path,
        monkeypatch,
        ws_config='version: "1"\nproposer:\n  base_url: http://127.0.0.1:9/v1\n'
        "  model: test-model\n",  # identical to the global provider.default_model
    )

    with pytest.raises(ConfigValidationError) as exc:
        _loader_for(lay).load_harness()

    assert exc.value.path == str(lay.global_cfg)
    assert exc.value.errors[0].source_path is None
    assert "must differ from provider.default_model" in str(exc.value)
