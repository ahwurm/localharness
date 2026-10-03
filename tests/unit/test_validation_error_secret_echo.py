"""R16: a validation error never carries a secret value, wherever a command prints one.

The sites besides the loader that turn a refused config into text, each printing pydantic's
str(ValidationError) of a merged config before this: its `input_value` is the whole config, or the
rejected value itself, so a stored or typed `proposer.api_key` could be printed. Each now prints
every error's location and message only, scrubbed of the secret values the config held.

Every case stores a long key and checks the key, its last 8 characters and pydantic's
`input_value` marker (the carrier of this class) in what is printed. Explicit --config-dir; nothing
reads the real home."""
from __future__ import annotations

import pytest
import yaml
from typer.testing import CliRunner

from localharness.cli.app import app

runner = CliRunner()
KEY = "sk-SENTINEL-sites-52-0123456789abcdefXYZ"
_PROVIDER = {"provider_type": "vllm", "base_url": "http://127.0.0.1:9/v1",
             "default_model": "test-model", "available_models": ["test-model"]}
NO_MODEL = {"proposer": {"base_url": "http://p/v1", "api_key": KEY}}  # pydantic's input: the section, key and all
MISSPELLED = {"proposer": {"base_url": "http://p/v1", "model": "p2", "api_kye": KEY}}


def _home(tmp_path, sections: dict):
    g = tmp_path / "g"
    g.mkdir(parents=True)
    cfg = {"version": "1", "provider": _PROVIDER, **sections}
    (g / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return g


def _clean(*texts: str) -> None:
    for text in texts:
        for piece in (KEY, KEY[-8:], "input_value"):
            assert piece not in text, f"{piece!r} reached:\n{text}"


# --- the config migration (`config migrate`, and start's security-defaults update) --------------


def test_config_migrate_refusal_never_echoes_the_key(tmp_path) -> None:
    """Rich used to swallow pydantic's `[type=…, input_value=…]` as a markup tag here, hiding the
    key by accident; the refusal is now each error's location and message, and says so."""
    for i, (sections, why) in enumerate((
            (NO_MODEL, "migrated config fails validation: proposer.model: Field required"),
            (MISSPELLED, "migrated config fails validation: proposer.api_kye: Extra inputs are not permitted"))):
        g = _home(tmp_path / f"case{i}", sections)
        before = (g / "config.yaml").read_bytes()
        result = runner.invoke(app, ["config", "migrate", "--config-dir", str(g)])

        assert result.exit_code == 1, result.output
        _clean(result.stdout, result.stderr, repr(result.exception))
        flat = " ".join(result.output.split())
        assert f"Refusing to write — {why}" in flat, flat
        assert (g / "config.yaml").read_bytes() == before  # refused: nothing written


def test_the_migration_error_itself_carries_no_input(tmp_path) -> None:
    from localharness.config import migrate

    g = _home(tmp_path, MISSPELLED)
    original, plan = migrate.load_plan(g / "config.yaml")
    assert plan is not None
    try:
        migrate.apply(g / "config.yaml", original, plan)
    except migrate.MigrationError as exc:
        _clean(str(exc), repr(exc), repr(exc.__cause__))
        assert "proposer.api_kye: Extra inputs are not permitted" in str(exc)
    else:
        raise AssertionError("a config that fails validation was migrated")


# --- /model and `localharness model <name>`: persisting a new default --------------------------


def _model_home(tmp_path):
    g = _home(tmp_path, {"proposer": {"base_url": "http://p/v1", "model": "p-model", "api_key": KEY}})
    # what `components set proposer.api_key …` leaves: the raw key in overrides.yaml, which the
    # check merges over the loaded config (whose own key is a masked SecretStr)
    (g / "overrides.yaml").write_text(yaml.safe_dump({"proposer": {"api_key": KEY}}), encoding="utf-8")
    return g


def test_model_name_refusal_never_echoes_the_key(tmp_path, monkeypatch) -> None:
    from localharness.cli import model_ops

    g = _model_home(tmp_path)
    monkeypatch.setattr(model_ops, "list_live_models", lambda base_url, *a, **k: (["test-model", "p-model"], True))
    before = (g / "overrides.yaml").read_bytes()
    result = runner.invoke(app, ["model", "p-model", "--config-dir", str(g)])

    assert result.exit_code == 2, result.output
    _clean(result.stdout, result.stderr, repr(result.exception))
    assert "proposer.model must differ" in " ".join(result.output.split()), result.output
    assert (g / "overrides.yaml").read_bytes() == before


def test_persisting_a_colliding_default_raises_without_the_key(tmp_path) -> None:
    """The error /model prints after an in-session swap ("persisting the new default failed")."""
    import asyncio

    from localharness.cli import model_ops
    from localharness.config.loader import ConfigLoader

    g = _model_home(tmp_path)
    harness = ConfigLoader(config_dir=g).load_harness()
    with pytest.raises(ValueError) as info:
        asyncio.run(model_ops.persist_default_model(harness, "p-model", config_dir=g))
    _clean(str(info.value), repr(info.value), repr(info.value.__cause__))
    assert "proposer.model must differ" in str(info.value)


def test_persisting_a_bad_active_endpoint_raises_without_the_input(tmp_path) -> None:
    """The cross-endpoint /model switch validates through the same check (a bad endpoint value)."""
    import asyncio
    from types import SimpleNamespace

    from localharness.cli import model_ops
    from localharness.config.loader import ConfigLoader

    g = _model_home(tmp_path)
    harness = ConfigLoader(config_dir=g).load_harness()
    peer = SimpleNamespace(name="peer", base_url=12345, provider_type="vllm", api_key="none")
    with pytest.raises(ValueError) as info:
        asyncio.run(model_ops.persist_active_endpoint(harness, peer, "m", config_dir=g))
    _clean(str(info.value), repr(info.value), repr(info.value.__cause__))
    assert "active_endpoint.base_url" in str(info.value)


# --- autoresearch adoption -------------------------------------------------------------------------


def test_a_refused_adoption_never_echoes_the_key(tmp_path) -> None:
    """`autoresearch adopt` checks the adopted value against the merged config and prints the
    refusal; a key stored raw in overrides.yaml rides in that merged config."""
    from localharness.autoresearch.adoption import AdoptionRefused, _validate_merged
    from localharness.config.loader import ConfigLoader

    g = _model_home(tmp_path)
    cfg = ConfigLoader(config_dir=g).load_harness()
    new_overlay = {"proposer": {"api_key": KEY}, "org": {"context": {"max_context_tokens": 500}}}
    with pytest.raises(AdoptionRefused) as info:
        _validate_merged(cfg, "org.context.max_context_tokens", new_overlay)
    _clean(str(info.value), repr(info.value), repr(info.value.__cause__))
    assert "adopting 'org.context.max_context_tokens' produces an invalid config" in str(info.value)
    assert "org.context.max_context_tokens: Input should be greater than or equal to 1000" in str(info.value)


# --- a plugin's invalid settings (doctor, plugins list, the start warnings) ------------------------


def test_a_plugin_validator_quoting_its_section_never_reaches_doctor_or_plugins_list(tmp_path, monkeypatch) -> None:
    """resolve() reports a plugin whose settings do not validate with its first error's location and
    message; a plugin's own validator message can quote the section, token included."""
    from localharness.config.loader import ConfigLoader
    from localharness.plugins import builtin, discovery
    from localharness.plugins.resolve import resolve
    from tests.unit.test_components_set_secret_echo import Secq

    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (Secq,))
    monkeypatch.setattr(discovery, "discover", lambda global_config_dir: [])
    g = _home(tmp_path, {"secq": {"n": 4, "tok": KEY}})

    reason = resolve(ConfigLoader(config_dir=g)).plan.entry("secq").reason
    _clean(reason)
    assert "invalid settings — secq: Value error, bad section" in reason, reason
    for args in (("doctor",), ("plugins", "list"), ("plugins", "info", "secq")):
        result = runner.invoke(app, [*args, "--config-dir", str(g)])
        _clean(result.stdout, result.stderr, repr(result.exception))


# --- init: a re-run's change, and --force's reset of the saved choices -----------------------------


def _init_errors(monkeypatch):
    import io

    from rich.console import Console

    from localharness.cli import init_cmd
    buf = io.StringIO()
    monkeypatch.setattr(init_cmd, "err_console", Console(file=buf, width=400))
    return buf


def test_init_change_refusal_never_echoes_the_key(tmp_path, monkeypatch) -> None:
    """A re-run's change that does not validate (a served window above the 2,000,000-token
    ceiling): it is checked against config.yaml merged with overrides.yaml, the stored key in it,
    `org:` before `proposer:` (the key last in the tail)."""
    import click
    from types import SimpleNamespace

    from localharness.cli import init_cmd

    g = _home(tmp_path, {"org": {"log_level": "info"},
                         "proposer": {"base_url": "http://p/v1", "model": "p-old", "api_key": KEY}})
    buf = _init_errors(monkeypatch)
    monkeypatch.setattr(init_cmd, "_served_window", lambda result: 4_000_000)
    found = SimpleNamespace(provider_type="vllm", base_url="http://127.0.0.1:9/v1", models=["p-new"])
    with pytest.raises(click.exceptions.Exit):
        init_cmd._write_change(g, found, "p-new", SimpleNamespace(tool_call_mode="native"))
    _clean(buf.getvalue())
    flat = " ".join(buf.getvalue().split())
    assert ("the new settings do not validate: org.context.max_context_tokens: Input should be less "
            "than or equal to 2000000") in flat, flat
    assert not (g / "overrides.yaml").exists()


def test_init_force_reset_refusal_never_echoes_the_key(tmp_path, monkeypatch) -> None:
    """--force drops the saved choices from overrides.yaml and checks what is left before writing."""
    import click

    from localharness.cli import init_cmd
    from localharness.config.models import HarnessConfig

    g = _home(tmp_path, {})
    (g / "overrides.yaml").write_text(yaml.safe_dump({  # a proposer with no model: --force keeps it
        "provider": {"default_model": "test-model"}, **NO_MODEL}), encoding="utf-8")
    buf = _init_errors(monkeypatch)
    harness = HarnessConfig.model_validate({"version": "1", "provider": _PROVIDER})
    with pytest.raises(click.exceptions.Exit):
        init_cmd._saved_choices_reset(g, harness)
    _clean(buf.getvalue())
    assert "the new settings do not validate: proposer.model: Field required" in " ".join(buf.getvalue().split())
