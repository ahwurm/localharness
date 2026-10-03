"""G2 (50-04): `proposer.api_key` is a SecretStr. Same YAML key, same accepted values; the proposer
client still receives the raw string; every display surface — `components list` (table and JSON),
`components get`, the `components set` echo, `plugins info autoresearch` and `doctor` — shows the
mask, never the key. The value `components set` writes stays raw in the overlay. `_enforce_seal`
(the holdout seal) is byte-identical to the phase base.

The autoresearch plugin is REPLACED into the bundled set (the post-cut truth; never appended)."""
from __future__ import annotations

import ast
import hashlib
import inspect
import subprocess
from pathlib import Path

import pytest
import yaml
from pydantic import SecretStr
from typer.testing import CliRunner

from localharness.autoresearch.plugin import AutoresearchPlugin
from localharness.cli.app import app
from localharness.config.models import HarnessConfig, ProposerConfig
from localharness.plugins import builtin, discovery

pytestmark = pytest.mark.plugin("autoresearch")
runner = CliRunner()
KEY, OTHER, MASK = "sk-SENTINEL", "sk-OTHER", "**********"
_PROVIDER = {"provider_type": "vllm", "base_url": "http://127.0.0.1:9/v1", "default_model": "test-model"}
_PROPOSER = {"base_url": "http://127.0.0.1:9/v1", "model": "p-model", "api_key": KEY}
# sha256 of `_enforce_seal`'s source at the phase base d551b7e (ast.get_source_segment).
SEAL_SHA = "1e45175ab34246a766aa508c42ee28ef32ef81a64a389ec321a98453f3cecf6a"
PROPOSER_PY = Path(__file__).resolve().parents[2] / "src" / "localharness" / "autoresearch" / "proposer.py"


# --------------------------------------------------------------------------- the model


def test_the_key_is_a_secret_with_the_same_values():
    assert ProposerConfig(base_url="u", model="m", api_key="abc").api_key.get_secret_value() == "abc"
    default = ProposerConfig(base_url="u", model="m").api_key
    assert isinstance(default, SecretStr) and default.get_secret_value() == "none"
    cfg = HarnessConfig.model_validate(yaml.safe_load(
        "version: '1'\nprovider: {provider_type: vllm, base_url: 'http://x/v1', default_model: t}\n"
        "proposer: {base_url: 'http://y/v1', model: p, api_key: none}\n"))
    assert cfg.proposer.api_key.get_secret_value() == "none"
    assert KEY not in repr(ProposerConfig(**_PROPOSER)) and KEY not in str(ProposerConfig(**_PROPOSER))


def test_the_main_model_again_is_accepted_and_its_key_stays_secret():
    cfg = HarnessConfig.model_validate({"version": "1", "provider": _PROVIDER,
                                        "proposer": {**_PROPOSER, "model": "test-model"}})
    assert cfg.proposer.model == cfg.provider.default_model
    assert cfg.proposer.api_key.get_secret_value() == KEY and KEY not in repr(cfg)


async def test_the_proposer_client_receives_the_raw_key(proposer_corpus, proposer_results, monkeypatch):
    import localharness.autoresearch.proposer as prop_mod
    from tests.unit.test_proposer import _good_payload
    seen: list[str] = []

    class _Spy:
        def __init__(self, llm_cfg):
            seen.append(llm_cfg.api_key)

        async def detect_capabilities(self):
            return type("C", (), {"tool_call_mode": "xml"})()

        async def stream_complete(self, messages, tools=None, on_token=None):
            return type("M", (), {"content": _good_payload()})(), None

    monkeypatch.setattr(prop_mod, "LLMClient", _Spy)
    cfg = HarnessConfig.model_validate({"version": "1", "provider": _PROVIDER, "proposer": _PROPOSER})
    await prop_mod.propose("agent.role", [proposer_results["train_run_id"]], cfg=cfg,
                           corpus_path=proposer_corpus, results_path=proposer_results["results"])
    assert seen == [KEY, KEY]  # the capability probe and the proposer client, both raw str
    assert all(type(k) is str for k in seen)


# --------------------------------------------------------------------------- display surfaces


@pytest.fixture
def home(components_home, monkeypatch) -> Path:
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (
        *(p for p in builtin.BUILTIN_PLUGINS if p.manifest.name != "autoresearch"), AutoresearchPlugin))
    monkeypatch.setattr(discovery, "discover", lambda global_config_dir: [])
    monkeypatch.setenv("COLUMNS", "400")
    cfg = components_home / "config.yaml"
    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    data["provider"]["base_url"] = _PROVIDER["base_url"]  # discard port: doctor's probe fails fast
    cfg.write_text(yaml.safe_dump({**data, "proposer": _PROPOSER}), encoding="utf-8")
    return components_home


@pytest.mark.parametrize("argv", [
    ["components", "list"], ["components", "list", "--json"], ["components", "get", "proposer.api_key"],
    ["components", "get", "proposer.api_key", "--json"], ["plugins", "info", "autoresearch"],
    ["plugins", "info", "autoresearch", "--json"]])
def test_no_display_surface_shows_the_key(home, argv):
    r = runner.invoke(app, argv)
    assert r.exit_code == 0, r.output
    assert KEY not in r.output
    if argv[1] == "get":
        assert MASK in r.output


def test_doctor_never_shows_the_key(home):
    out = runner.invoke(app, ["doctor", "--config-dir", str(home)]).output
    assert "proposer: p-model at http://127.0.0.1:9/v1" in out, out
    assert KEY not in out


def test_components_set_masks_the_echo_and_writes_the_raw_value(home):
    r = runner.invoke(app, ["components", "set", "proposer.api_key", OTHER])
    assert r.exit_code == 0, r.output
    assert OTHER not in r.output and KEY not in r.output
    overrides = (home / "overrides.yaml").read_text(encoding="utf-8")
    assert OTHER in overrides
    got = runner.invoke(app, ["components", "get", "proposer.api_key", "--json"]).output
    assert OTHER not in got and MASK in got


# --------------------------------------------------------------------------- the seal


def _seal_segment(src: str) -> str:
    fn = next(n for n in ast.parse(src).body if getattr(n, "name", "") == "_enforce_seal")
    return ast.get_source_segment(src, fn)


def test_the_seal_is_byte_identical_to_the_phase_base():
    import localharness.autoresearch.proposer as prop_mod
    now = _seal_segment(PROPOSER_PY.read_text(encoding="utf-8"))
    assert inspect.getsource(prop_mod._enforce_seal).rstrip("\n") == now
    assert hashlib.sha256(now.encode()).hexdigest() == SEAL_SHA
    base = subprocess.run(["git", "show", "d551b7e:src/localharness/autoresearch/proposer.py"],
                          capture_output=True, text=True, cwd=PROPOSER_PY.parent)
    if base.returncode == 0:  # a checkout with history: compare the text itself, not only the hash
        assert _seal_segment(base.stdout) == now
