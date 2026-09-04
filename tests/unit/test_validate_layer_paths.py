"""`config validate` checks every file AT ITS OWN PATH, and says which file it checked.

Before this, `validate_all` iterated the files of both layers but re-resolved each one by STEM
through `_find_file` — so with a workspace `agents/foo.yaml` shadowing a global `agents/foo.yaml`
the workspace file was validated twice and the global file never was. Both directions were wrong:
a broken global file was reported valid, and a valid global file was reported with the workspace
file's error. The rows also printed a bare basename, so the two `foo.yaml`s were indistinguishable
in the output a user is meant to act on.
"""
from __future__ import annotations

import yaml

from localharness.config.loader import ConfigLoader


def _seed_agent(base, name: str, **fields):
    d = base / "agents"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.yaml").write_text(yaml.safe_dump({"name": name, **fields}), encoding="utf-8")
    return d / f"{name}.yaml"


def _layers(tmp_path):
    global_dir = tmp_path / "global"
    ws = tmp_path / "proj" / ".localharness"
    (global_dir / "agents").mkdir(parents=True)
    (ws / "agents").mkdir(parents=True)
    (global_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "provider": {
                    "provider_type": "vllm",
                    "base_url": "http://localhost:8000/v1",
                    "default_model": "m",
                },
                "org": {"name": "o"},
            }
        ),
        encoding="utf-8",
    )
    return global_dir, ws


def test_shadowed_bad_global_file_is_caught(tmp_path):
    """A BROKEN global agent yaml shadowed by a VALID workspace one is still reported broken."""
    global_dir, ws = _layers(tmp_path)
    bad = _seed_agent(global_dir, "bar", role="r", temperature=99.0)  # out of range
    good = _seed_agent(ws, "bar", role="the workspace copy, valid")

    verdicts = dict(ConfigLoader(config_dir=global_dir, local_config_dir=ws).validate_all())

    assert verdicts[str(bad)] is not None, "the broken GLOBAL file was reported valid"
    assert verdicts[str(good)] is None


def test_bad_workspace_verdict_is_not_filed_under_the_good_global_file(tmp_path):
    """The reverse direction: a workspace file's error must not be attributed to the global one."""
    global_dir, ws = _layers(tmp_path)
    good = _seed_agent(global_dir, "foo", role="the global copy, valid")
    bad = _seed_agent(ws, "foo", role="r", temperature=99.0)

    verdicts = dict(ConfigLoader(config_dir=global_dir, local_config_dir=ws).validate_all())

    assert verdicts[str(good)] is None, "the valid GLOBAL file inherited the workspace's error"
    assert verdicts[str(bad)] is not None


def test_every_file_of_both_layers_appears_exactly_once(tmp_path):
    """No file is skipped and none is validated twice under a stem that resolves elsewhere."""
    global_dir, ws = _layers(tmp_path)
    paths = [
        _seed_agent(global_dir, "shared", role="global"),
        _seed_agent(ws, "shared", role="workspace"),
        _seed_agent(global_dir, "global-only", role="global"),
        _seed_agent(ws, "ws-only", role="workspace"),
    ]

    reported = [p for p, _ in ConfigLoader(config_dir=global_dir, local_config_dir=ws).validate_all()]

    for path in paths:
        assert reported.count(str(path)) == 1, f"{path} reported {reported.count(str(path))} times"


def test_validate_rows_print_the_full_path(tmp_path, monkeypatch):
    """The CLI row must distinguish the two same-stem files it just checked.

    No --config-dir and no env override: an explicit config dir is a full replacement and skips
    discovery (LAYR-02), so the two-layer row display can only be exercised through the default
    `~/.localharness` chain with HOME pointed at the fixture.
    """
    from typer.testing import CliRunner

    from localharness.cli.app import app

    home = tmp_path / "home"
    global_dir = home / ".localharness"
    (global_dir / "agents").mkdir(parents=True)
    (global_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "provider": {
                    "provider_type": "vllm",
                    "base_url": "http://localhost:8000/v1",
                    "default_model": "m",
                },
                "org": {"name": "o"},
            }
        ),
        encoding="utf-8",
    )
    proj = home / "proj"
    ws = proj / ".localharness"
    (ws / "agents").mkdir(parents=True)
    (proj / ".git").mkdir()  # in-project workspace = loads silently (LAYR-05)
    _seed_agent(global_dir, "twin", role="global")
    _seed_agent(ws, "twin", role="workspace")
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.delenv("LOCALHARNESS_HOME", raising=False)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("COLUMNS", "300")  # else rich crops the row and the path is uncopyable
    monkeypatch.chdir(proj)

    result = CliRunner().invoke(app, ["validate"])

    assert str(global_dir / "agents" / "twin.yaml") in result.output
    assert str(ws / "agents" / "twin.yaml") in result.output


def _two_layer_home(tmp_path, monkeypatch):
    """A configured machine + an in-project workspace, reached through the default HOME chain.

    An explicit --config-dir is a full replacement and skips discovery (LAYR-02), so the
    two-layer behaviour can only be driven this way.
    """
    home = tmp_path / "home"
    global_dir = home / ".localharness"
    (global_dir / "agents").mkdir(parents=True)
    (global_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "provider": {
                    "provider_type": "vllm",
                    "base_url": "http://localhost:8000/v1",
                    "default_model": "m",
                },
                "org": {"name": "o"},
            }
        ),
        encoding="utf-8",
    )
    proj = home / "proj"
    ws = proj / ".localharness"
    ws.mkdir(parents=True)
    (proj / ".git").mkdir()  # in-project workspace = loads silently (LAYR-05)
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.delenv("LOCALHARNESS_HOME", raising=False)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("COLUMNS", "300")  # else rich crops the row and the path is uncopyable
    monkeypatch.chdir(proj)
    return global_dir, ws


def test_workspace_parse_error_names_the_workspace_file(tmp_path, monkeypatch):
    """A syntax error in the WORKSPACE config.yaml must not be reported against the global one.

    The harness row is keyed under the global config.yaml by `validate_all`, and the row only
    deferred to `error.path` for ConfigValidationError — a ConfigParseError carries `.path` too,
    so the one error class that can ONLY come from a specific file was the one class whose file
    was never named. The user was told to fix a config.yaml that is fine.
    """
    from typer.testing import CliRunner

    from localharness.cli.app import app

    global_dir, ws = _two_layer_home(tmp_path, monkeypatch)
    (ws / "config.yaml").write_text("org:\n  name: [unclosed\n", encoding="utf-8")

    result = CliRunner().invoke(app, ["validate"])

    assert result.exit_code == 1, result.output
    assert str(ws / "config.yaml") in result.output, result.output
