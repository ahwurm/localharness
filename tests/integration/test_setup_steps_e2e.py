"""memory's, mobile's (web) and autoresearch's setup steps, each through the real
`localharness plugins enable` command: the questions on a (fake) terminal, the one validated write,
the setup action, the check doctor runs, the coding-agent prompt when it does not pass, and the next
step. Nothing reaches a real endpoint: the embedding download is provider.server.download_model
patched, the proposer answers through autoresearch's _TRANSPORT (an httpx.MockTransport), and the
only address that could be dialled is the discard port 127.0.0.1:9.

Plain `def` tests throughout: `plugins enable` calls asyncio.run itself (asyncio_mode is "auto").
Every run passes an explicit --config-dir. Names start with the plugin, so `-k <plugin>` selects one
plugin's tests."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from rich.console import Console
from typer.testing import CliRunner

from localharness.cli import plugins_cmd
from localharness.cli.app import app
from localharness.plugins import setup
from localharness.plugins.api import Check
from localharness.plugins.setup import AGENT_PROMPT_LEAD

runner = CliRunner()
_CONFIG = {  # port 9 (discard): nothing ever reaches a model
    "version": "1",
    "provider": {"provider_type": "vllm", "base_url": "http://127.0.0.1:9/v1",
                 "default_model": "test-model", "available_models": ["test-model"]},
}
MODEL = "Qwen/Qwen3-Embedding-0.6B"
DOWNLOAD_Q = "Download the embedding model now (about 1.2 GB)?"


@pytest.fixture(autouse=True)
def wide(monkeypatch):
    monkeypatch.setenv("COLUMNS", "400")
    monkeypatch.setattr(plugins_cmd, "console", Console(width=400))
    monkeypatch.setattr(setup, "gpu_name", lambda: "NVIDIA GB10")  # no test runs nvidia-smi


@pytest.fixture
def g(tmp_path: Path) -> Path:
    g = tmp_path / "g"
    g.mkdir()
    (g / "config.yaml").write_text(yaml.safe_dump(_CONFIG), encoding="utf-8")
    return g


@pytest.fixture
def terminal(monkeypatch):
    """A terminal: the person types `answers` and answers yes/no with `confirms`, in order; every
    question is recorded."""
    asked: dict[str, list] = {"prompt": [], "confirm": []}
    answers: list[str] = []
    confirms: list[bool] = []

    def prompt(text, default=None, **kw):
        asked["prompt"].append((text, default))
        return answers.pop(0)

    def confirm(text, default=None, **kw):
        asked["confirm"].append(text)
        return confirms.pop(0)

    monkeypatch.setattr(plugins_cmd, "_stdin_is_a_terminal", lambda: True)
    monkeypatch.setattr(plugins_cmd.typer, "prompt", prompt)
    monkeypatch.setattr(plugins_cmd.typer, "confirm", confirm)
    return asked, answers, confirms


@pytest.fixture
def no_terminal(monkeypatch):
    def never(*a, **kw):
        raise AssertionError("asked off a terminal")
    monkeypatch.setattr(plugins_cmd, "_stdin_is_a_terminal", lambda: False)
    monkeypatch.setattr(plugins_cmd.typer, "prompt", never)
    monkeypatch.setattr(plugins_cmd.typer, "confirm", never)


def _enable(g: Path, *args: str):
    return runner.invoke(app, ["plugins", "enable", *args, "--config-dir", str(g)])


def _overrides(g: Path):
    return yaml.safe_load((g / "overrides.yaml").read_text(encoding="utf-8"))


# --- memory: one action, no questions -------------------------------------------------------------

@pytest.fixture
def embed(monkeypatch):
    """The embedding model is missing from the cache until the (fake) download puts it there; the
    sentence_transformers package is installed unless a test says otherwise."""
    from localharness.memory import plugin as memory_plugin
    from localharness.provider import server
    state: dict = {"downloads": [], "cached": False, "package": True}

    def check(model):
        if not state["package"]:
            return Check(name="memory-embedding", status="fail",
                         detail="the sentence_transformers package is not installed — memory search "
                                "and consolidation cannot embed", hint="uv sync --extra embeddings")
        if state["cached"]:
            return Check(name="memory-embedding", status="pass",
                         detail=f"embedding model {model} is in the local cache")
        return Check(name="memory-embedding", status="fail",
                     detail=f"embedding model {model} is not in the local Hugging Face cache")

    def download(repo_id):
        state["downloads"].append(repo_id)
        state["cached"] = True
        return f"/cache/{repo_id}"

    monkeypatch.setattr(memory_plugin, "_embedding_check", check)
    monkeypatch.setattr(memory_plugin, "_embedding_package_installed", lambda: state["package"])
    monkeypatch.setattr(server, "download_model", download)
    return state


@pytest.mark.plugin("memory")
def test_memory_yes_downloads_the_model_then_checks_it(g, terminal, embed) -> None:
    asked, _, confirms = terminal
    confirms.append(True)
    result = _enable(g, "memory")
    out = result.output

    assert result.exit_code == 0, out
    assert asked == {"prompt": [], "confirm": [DOWNLOAD_Q]}
    assert embed["downloads"] == [MODEL]
    assert _overrides(g) == {"memory": {"enabled": True}}
    for text in (f"✓ memory-embedding: downloaded {MODEL}", "Checking it now:",
                 f"✓ memory-embedding: embedding model {MODEL} is in the local cache",
                 "In a session, /memory shows what it keeps."):
        assert text in out, text
    assert out.index("downloaded") < out.index("Checking it now:")
    assert AGENT_PROMPT_LEAD not in out  # the check passes now


@pytest.mark.plugin("memory")
def test_memory_no_downloads_nothing_and_prints_the_prompt(g, terminal, embed) -> None:
    asked, _, confirms = terminal
    confirms.append(False)
    result = _enable(g, "memory")
    out = result.output

    assert result.exit_code == 0, out
    assert asked["confirm"] == [DOWNLOAD_Q] and embed["downloads"] == []
    assert f"✗ memory-embedding: embedding model {MODEL} is not in the local Hugging Face cache" in out
    assert AGENT_PROMPT_LEAD in out and "embeddings" in out[out.index(AGENT_PROMPT_LEAD):]
    assert "In a session, /memory shows what it keeps." in out


@pytest.mark.plugin("memory")
def test_memory_with_the_model_cached_asks_nothing(g, terminal, embed) -> None:
    embed["cached"] = True
    asked, _, _ = terminal
    result = _enable(g, "memory")
    out = result.output

    assert result.exit_code == 0, out
    assert asked == {"prompt": [], "confirm": []}
    assert embed["downloads"] == []
    assert f"✓ memory-embedding: embedding model {MODEL} is in the local cache" in out
    assert AGENT_PROMPT_LEAD not in out
    assert "In a session, /memory shows what it keeps." in out


@pytest.mark.plugin("memory")
def test_memory_without_the_embeddings_package_names_the_install_line(g, terminal, embed) -> None:
    embed["package"] = False
    _, _, confirms = terminal
    confirms.append(True)
    result = _enable(g, "memory")
    out = result.output

    assert result.exit_code == 0, out
    assert embed["downloads"] == []
    assert "memory-embedding: nothing downloaded: the sentence_transformers package is not installed" in out
    assert "uv tool install 'localharness[embeddings]'" in out
    assert AGENT_PROMPT_LEAD in out


@pytest.mark.plugin("memory")
def test_memory_without_a_terminal_downloads_nothing_and_says_what_to_run(g, no_terminal, embed) -> None:
    result = _enable(g, "memory")
    out = result.output

    assert result.exit_code == 0, out
    assert embed["downloads"] == []
    assert _overrides(g) == {"memory": {"enabled": True}}
    assert ("next step — run `localharness plugins enable memory` on a terminal to answer: "
            f"{DOWNLOAD_Q}") in out
    assert "In a session, /memory shows what it keeps." in out


# --- web (mobile): one question, the address the pairing QR sends a phone to ----------------------

PHONE_Q = "Phone address, the URL your phone opens (Enter: `localharness web` guesses it)"
WEB_NEXT = "Run `localharness web`, then scan its pairing QR with your phone."


@pytest.fixture
def web_extra(monkeypatch):
    """The web install extra reads as installed (the resolver's one seam)."""
    from localharness.plugins import resolve
    monkeypatch.setitem(resolve.resolve.__kwdefaults__, "extra_installed", lambda e: True)


def _settings(g: Path, name: str):
    from localharness.config.loader import ConfigLoader
    from localharness.plugins.resolve import resolve
    return resolve(ConfigLoader(config_dir=g)).settings[name].config


@pytest.mark.plugin("web")
def test_web_saves_the_phone_address_then_checks(g, terminal, web_extra) -> None:
    asked, answers, _ = terminal
    answers.append("https://spark.example.ts.net")
    result = _enable(g, "web")
    out = result.output

    assert result.exit_code == 0, out
    assert asked == {"prompt": [(PHONE_Q, "")], "confirm": []}
    assert _settings(g, "web").public_url == "https://spark.example.ts.net"
    for text in ("set web.public_url = 'https://spark.example.ts.net'", "Checking it now:",
                 "web: not enrolled yet", WEB_NEXT):
        assert text in out, text
    assert out.index("Checking it now:") < out.index(WEB_NEXT)
    assert AGENT_PROMPT_LEAD not in out  # answered: "not enrolled yet" is the server's first run


@pytest.mark.plugin("web")
def test_web_refuses_an_address_without_a_scheme_and_writes_nothing(g, terminal, web_extra) -> None:
    _, answers, _ = terminal
    answers.append("spark.local")
    result = _enable(g, "web")

    assert result.exit_code == 2, result.output
    assert "web.public_url must start with http:// or https://" in " ".join(result.output.split())
    assert not (g / "overrides.yaml").exists()


@pytest.mark.plugin("web")
def test_web_enter_saves_nothing_and_prints_the_prompt(g, terminal, web_extra) -> None:
    _, answers, _ = terminal
    answers.append("")
    result = _enable(g, "web")
    out = result.output

    assert result.exit_code == 0, out
    assert _overrides(g) == {"web": {"enabled": True}}
    assert "web: not enrolled yet" in out
    assert AGENT_PROMPT_LEAD in out and "tailscale serve --bg 8765" in out
    assert out.index(AGENT_PROMPT_LEAD) < out.index(WEB_NEXT)


@pytest.mark.plugin("web")
def test_web_without_its_extra_asks_nothing_and_names_the_install_line(g, terminal, monkeypatch) -> None:
    from localharness.plugins import resolve
    monkeypatch.setitem(resolve.resolve.__kwdefaults__, "extra_installed", lambda e: e != "web")
    asked, _, _ = terminal
    result = _enable(g, "web")
    flat = " ".join(result.output.split())

    assert result.exit_code == 0, result.output
    assert asked == {"prompt": [], "confirm": []}
    assert _overrides(g) == {"web": {"enabled": True}}
    assert "web is missing its install extra" in flat and "localharness[web]" in flat
    assert "Checking it now:" not in flat and WEB_NEXT in flat


@pytest.mark.plugin("web")
def test_web_the_saved_address_reaches_the_pairing_qr(g, terminal, web_extra, monkeypatch) -> None:
    """Composed: what the step saved is the address `localharness web` puts in the QR — no guess."""
    pytest.importorskip("starlette")
    pytest.importorskip("uvicorn")
    from localharness.channels.web import auth as web_auth
    from localharness.cli import web_cmd

    _, answers, _ = terminal
    answers.append("https://spark.example.ts.net")
    assert _enable(g, "web").exit_code == 0
    guessed: list[int] = []
    monkeypatch.setattr(web_cmd, "detect_public_url", lambda port, **kw: guessed.append(port))
    monkeypatch.setenv("LOCALHARNESS_DIR", str(g))  # the plugin's command mounts from this machine
    ran = runner.invoke(app, ["web", "--rotate-token", "--config-dir", str(g)])

    assert ran.exit_code == 0, ran.output
    token = web_auth.load_or_create_token(str(g))[0]
    assert f"https://spark.example.ts.net/#t={token}" in ran.output
    assert guessed == []


# --- autoresearch: the proposer's address and model in one write, then one GET <address>/models ---

P_URL = "http://p.test/v1"
AR_NEXT = "Then: `localharness propose --help` shows how to write the first proposal."


@pytest.fixture
def proposer(monkeypatch):
    """The proposer behind autoresearch's _TRANSPORT (every request recorded; `answer[0]` replies),
    and a count of the overrides.yaml writes `plugins enable` makes."""
    import httpx

    from localharness.autoresearch import plugin as autoresearch_plugin
    seen: list = []
    answer = [lambda request: httpx.Response(200, json={"object": "list", "data": [{"id": "p-model"}]})]

    def handler(request):
        seen.append(request)
        return answer[0](request)

    writes: list[dict] = []
    real_write = plugins_cmd.atomic_write_overlay

    def counted(path, overlay):
        writes.append(yaml.safe_load(yaml.safe_dump(overlay)))
        return real_write(path, overlay)

    monkeypatch.setattr(autoresearch_plugin, "_TRANSPORT", httpx.MockTransport(handler))
    monkeypatch.setattr(plugins_cmd, "atomic_write_overlay", counted)
    return seen, answer, writes


@pytest.mark.plugin("autoresearch")
def test_autoresearch_writes_the_proposer_in_one_write_then_checks_it_answers(g, terminal, proposer) -> None:
    asked, answers, _ = terminal
    seen, _, writes = proposer
    answers.extend([P_URL, "p-model"])
    result = _enable(g, "autoresearch")
    out = result.output

    assert result.exit_code == 0, out
    assert [q for q, _ in asked["prompt"]] == ["Proposer address (an OpenAI-compatible base URL)",
                                               "Proposer model id (not your main model)"]
    assert asked["confirm"] == []  # the action has no question: it spends nothing
    expected = {"autoresearch": {"enabled": True}, "proposer": {"base_url": P_URL, "model": "p-model"}}
    assert writes == [expected] and _overrides(g) == expected
    assert [(r.method, str(r.url)) for r in seen] == [("GET", f"{P_URL}/models")]
    for text in (f"✓ autoresearch-proposer: the proposer answers at {P_URL} and serves p-model",
                 "Checking it now:", f"✓ autoresearch: proposer: p-model at {P_URL}", AR_NEXT):
        assert text in out, text
    assert AGENT_PROMPT_LEAD not in out


@pytest.mark.plugin("autoresearch")
def test_autoresearch_refuses_the_main_model_and_never_shows_a_stored_key(g, terminal, proposer) -> None:
    # A stored proposer written by hand: provider first, the key last, so a refusal that echoed
    # the settings would show it.
    (g / "config.yaml").write_text(yaml.safe_dump({**_CONFIG, "proposer": {
        "base_url": "http://old.test/v1", "model": "p-old", "api_key": "SENTINEL-52-05"}},
        sort_keys=False), encoding="utf-8")
    _, answers, _ = terminal
    seen, _, writes = proposer
    answers.extend([P_URL, "test-model"])
    result = _enable(g, "autoresearch")
    flat = " ".join(result.output.split())

    assert result.exit_code == 2, result.output
    assert "proposer.model must differ" in flat
    assert "SENTINEL-52-05" not in result.output
    assert writes == [] and not (g / "overrides.yaml").exists() and seen == []


@pytest.mark.plugin("autoresearch")
def test_autoresearch_a_proposer_that_does_not_answer_is_named_and_the_answers_kept(g, terminal,
                                                                                  proposer) -> None:
    import httpx

    def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)

    _, answers, _ = terminal
    seen, answer, _ = proposer
    answer[0] = refuse
    answers.extend([P_URL, "p-model"])
    result = _enable(g, "autoresearch")
    out = result.output

    assert result.exit_code == 0, out
    assert len(seen) == 1
    assert f"✗ autoresearch-proposer: no answer from {P_URL}/models (ConnectError)" in out
    assert "start the proposer's server, or fix proposer.base_url" in out
    assert _overrides(g)["proposer"] == {"base_url": P_URL, "model": "p-model"}
    assert AR_NEXT in out


@pytest.mark.plugin("autoresearch")
def test_autoresearch_set_values_write_the_same_keys_and_run_no_check(g, no_terminal, proposer) -> None:
    seen, _, writes = proposer
    result = _enable(g, "autoresearch", "--set", f"proposer.base_url={P_URL}", "--set", "proposer.model=p-model")

    assert result.exit_code == 0, result.output
    expected = {"autoresearch": {"enabled": True}, "proposer": {"base_url": P_URL, "model": "p-model"}}
    assert writes == [expected] and _overrides(g) == expected
    assert seen == []
    assert "Checking it now:" not in result.output
