"""Item 9 — /model picker: argument completion for `/model ` from the session's cached
live-model list.

The menu NEVER fetches: the cache is warmed by a best-effort prefetch at REPL start and
refreshed on every /model run. An empty cache or missing supplier means no menu — typing
still works, and /model itself still reports unreachable/malformed servers when run.
Model ids complete case-true (ids are case-sensitive); filtering is case-insensitive.
"""
from __future__ import annotations

from types import SimpleNamespace as NS

from prompt_toolkit.application import create_app_session
from prompt_toolkit.document import Document
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from localharness.channels.terminal import SlashCommandCompleter, _build_persistent_input_app
from localharness.cli.repl import OrchestratorREPL

NAMES = ["qwen3.6-35b-a3b", "qwen2.5-7b-instruct", "llama-3.3-70b"]


def _complete(text: str, names=NAMES):
    c = SlashCommandCompleter(model_names_fn=lambda: list(names))
    return list(c.get_completions(Document(text, len(text)), None))


# ------------------------------------------------------------------ completer argument menu
def test_model_arg_menu_lists_cached_models():
    assert {c.text for c in _complete("/model ")} == set(NAMES)


def test_model_arg_menu_filters_case_insensitive_inserts_case_true():
    comps = _complete("/model QW")
    assert {c.text for c in comps} == {"qwen3.6-35b-a3b", "qwen2.5-7b-instruct"}


def test_model_arg_completion_replaces_only_the_partial_argument():
    comps = _complete("/model qwen3")
    assert comps and all(c.start_position == -len("qwen3") for c in comps)


def test_model_arg_menu_silent_on_empty_cache_and_without_supplier():
    assert _complete("/model ", names=[]) == []
    bare = SlashCommandCompleter()
    assert list(bare.get_completions(Document("/model ", len("/model ")), None)) == []


def test_model_arg_menu_stops_past_the_first_argument():
    assert _complete("/model qwen3.6-35b-a3b extra") == []


def test_other_command_arguments_stay_uncompleted():
    assert _complete("/memory ") == []
    assert _complete("/memory show 12") == []


def test_command_token_completion_unchanged():
    assert {c.text for c in _complete("/mo")} == {"/model"}


# ------------------------------------------------------------------ headless menu drive
async def _drive(feed: str, model_names_fn=None):
    subs: list[str] = []
    holder: dict = {}

    with create_pipe_input() as inp:
        with create_app_session(input=inp, output=DummyOutput()):
            app = _build_persistent_input_app(
                InMemoryHistory(), ">",
                on_submit=subs.append, on_interrupt=lambda: None,
                on_eof=lambda: holder["app"].exit(),
                hint_fn=lambda: [("class:hint", " ")], pct_fn=lambda: None,
                status_fn=lambda: [],
                model_names_fn=model_names_fn,
            )
            holder["app"] = app
            inp.send_text(feed)
            import asyncio
            try:
                await asyncio.wait_for(app.run_async(), timeout=5)
            except asyncio.TimeoutError:  # a hung app is a FAILING drive, never a hung suite
                app.exit()
    return subs


async def test_picker_tab_then_single_enter_submits_the_switch():
    """One-Enter picker: Enter on a HIGHLIGHTED model accepts AND submits in one stroke."""
    subs = await _drive("/model \t\r\x04", model_names_fn=lambda: ["qwen-a", "qwen-b"])
    assert subs == ["/model qwen-a"]


async def test_picker_arrow_navigates_then_single_enter_submits():
    subs = await _drive("/model \t\x1b[B\r\x04", model_names_fn=lambda: ["qwen-a", "qwen-b"])
    assert subs == ["/model qwen-b"]


async def test_command_menu_enter_stays_accept_only():
    """The one-Enter rule is MODEL completions only — command completions keep the
    accept-then-second-Enter contract (Claude Code feel)."""
    subs = await _drive("/mem\t\r\r\x04", model_names_fn=lambda: ["qwen-a"])
    assert subs == ["/memory"]


# ------------------------------------------------------------------ REPL cache plumbing
def _repl(channel, llm):
    return OrchestratorREPL(
        orchestrator=None, agent_loop=NS(_llm=llm), channel=channel, bus=None,
    )


async def test_prefetch_fills_cache_and_channel_hook_serves_it():
    class Chan:
        model_names_fn = None

    chan = Chan()
    llm = NS(config=NS(base_url="http://localhost:1/v1", model="m"))
    r = _repl(chan, llm)

    async def fake_live(base_url):
        return (["a-model", "b-model"], True)

    r._live_models = fake_live
    await r._prefetch_model_cache()
    assert chan.model_names_fn is not None
    assert chan.model_names_fn() == [("a-model", "serving"), ("b-model", "serving")]


async def test_prefetch_swallows_unreachable_server():
    llm = NS(config=NS(base_url="http://localhost:1/v1", model="m"))
    r = _repl(NS(), llm)

    async def boom(base_url):
        raise OSError("connection refused")

    r._live_models = boom
    await r._prefetch_model_cache()  # must not raise
    assert r._model_cache == []


async def test_prefetch_includes_managed_registry_names():
    """Full-swap feature: the picker menu offers the managed server's local checkpoints by
    name alongside whatever is live — live first, registry appended, deduped."""
    llm = NS(config=NS(base_url="http://localhost:1/v1", model="m"))
    managed = NS(local_models=[
        NS(name="live-one", quant=None, tps=None),
        NS(name="qwen3.6-27b", quant="nvfp4 (compressed-tensors)", tps=9.5),
    ])
    r = OrchestratorREPL(
        orchestrator=None, agent_loop=NS(_llm=llm), channel=NS(), bus=None,
        harness_config=NS(server=managed),
    )

    async def fake_live(base_url):
        return (["live-one"], True)

    r._live_models = fake_live
    await r._prefetch_model_cache()
    assert r._model_cache == [
        ("live-one", "serving"),
        ("qwen3.6-27b", "nvfp4 (compressed-tensors) · ~9.5 t/s · swap"),
    ]


def test_completer_renders_meta_from_tuples():
    """Supplier items may be (name, meta) tuples — meta becomes the menu's right column
    (quant + measured t/s). Plain strings keep the generic 'model' meta."""
    c = SlashCommandCompleter(model_names_fn=lambda: [
        ("qwen3.6-35b-a3b", "serving now · nvfp4 (modelopt) · ~30 t/s"),
        "bare-string-model",
    ])
    comps = list(c.get_completions(Document("/model ", len("/model ")), None))
    assert comps[0].text == "qwen3.6-35b-a3b"
    assert comps[0].display_meta_text == "serving now · nvfp4 (modelopt) · ~30 t/s"
    assert comps[1].display_meta_text == "model"


# ------------------------------------------------------------------ peer models in the menu (tree)
def test_menu_includes_peer_models_tagged_with_endpoint():
    """0.10.0 tree: peer-endpoint models join the completion cache after live + registry, each
    tagged `<framework> · <host:port>` so a user tells a vLLM model from an Ollama one. Registry
    (managed) and live entries are unchanged."""
    from localharness.config.models import EndpointRef

    ep = EndpointRef(name="ollama-local", base_url="http://localhost:11434/v1",
                     provider_type="ollama")
    menu = OrchestratorREPL._compose_model_menu(
        ["qwen-live"], None, "qwen-live", {"gpt-oss:20b": ep},
    )
    assert menu[0] == ("qwen-live", "serving now")               # live entry unchanged
    assert menu[-1] == ("gpt-oss:20b", "Ollama · localhost:11434")  # peer tagged with endpoint


def test_menu_without_peers_is_unchanged():
    """No peer_target (the default) leaves the menu exactly live + registry — single-endpoint
    users see no new entries."""
    assert OrchestratorREPL._compose_model_menu(["m"], None, "m") == [("m", "serving now")]


def test_peer_model_completes_as_model_pick_with_endpoint_meta():
    """A peer model in the cache is a normal /model completion: model-pick style (one-Enter
    cross-endpoint switch) with its endpoint tag as the right-column meta."""
    c = SlashCommandCompleter(model_names_fn=lambda: [
        ("qwen-live", "serving now"),
        ("gpt-oss:20b", "Ollama · localhost:11434"),
    ])
    comps = list(c.get_completions(Document("/model gpt", len("/model gpt")), None))
    assert len(comps) == 1
    assert comps[0].text == "gpt-oss:20b"
    assert comps[0].display_meta_text == "Ollama · localhost:11434"
    assert "model-pick" in (comps[0].style or "")
