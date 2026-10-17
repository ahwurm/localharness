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
from localharness.cli.model_ops import Entries, EntryGroup, PlanEntry
from localharness.cli.repl import OrchestratorREPL
from localharness.config.models import HarnessConfig, ManagedServerConfig, OrgConfig, ProviderConfig

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
    # `/mode` joined the table in v0.14 and shares the `/mo` prefix. What this test is about is
    # unchanged: a COMMAND token completes to commands, never to model names.
    assert {c.text for c in _complete("/mo")} == {"/model", "/mode"}


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
                hint_fn=lambda: [("class:hint", " ")], right_fn=lambda: [],
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
    accept-then-second-Enter contract (Claude Code feel). /memory is the memory plugin's row, so
    it is installed here the way a memory-on session's lifecycle installs it."""
    from unittest.mock import AsyncMock

    from localharness.cli.slash_commands import SlashCommand, set_plugin_rows
    from localharness.memory.plugin import MemoryPlugin
    (d,) = MemoryPlugin.manifest.slash
    set_plugin_rows([SlashCommand(d.name, d.help, AsyncMock(), takes_args=True, plugin="memory")])
    try:
        subs = await _drive("/mem\t\r\r\x04", model_names_fn=lambda: ["qwen-a"])
    finally:
        set_plugin_rows(())
    assert subs == ["/memory"]


# ------------------------------------------------------------------ REPL cache plumbing
def _harness(server=None) -> HarnessConfig:
    return HarnessConfig(
        provider=ProviderConfig(provider_type="vllm", base_url="http://localhost:1/v1",
                                default_model="a-model", available_models=["a-model"]),
        org=OrgConfig(default_model="a-model"), server=server)


def _repl(channel, llm, harness=None):
    return OrchestratorREPL(
        orchestrator=None, agent_loop=NS(_llm=llm), channel=channel, bus=None, harness_config=harness,
    )


async def test_prefetch_fills_cache_and_channel_hook_serves_it():
    class Chan:
        model_names_fn = None

    chan = Chan()
    llm = NS(config=NS(base_url="http://localhost:1/v1", model="m"))
    r = _repl(chan, llm, _harness())

    async def fake_live(base_url):
        return (["a-model", "b-model"], True)

    r._live_models = fake_live
    await r._prefetch_model_cache()
    assert chan.model_names_fn is not None
    assert chan.model_names_fn() == [("a-model", "serving"), ("b-model", "serving")]


async def test_prefetch_swallows_unreachable_server():
    llm = NS(config=NS(base_url="http://localhost:1/v1", model="m"))
    r = _repl(NS(), llm, _harness())

    async def boom(base_url):
        raise OSError("connection refused")

    r._live_models = boom
    await r._prefetch_model_cache()  # must not raise
    assert r._model_cache == []


async def test_prefetch_swallows_a_planner_failure(monkeypatch):
    from localharness.cli import model_ops

    llm = NS(config=NS(base_url="http://localhost:1/v1", model="m"))
    r = _repl(NS(), llm, _harness())

    async def fake_live(base_url):
        return (["a-model"], True)

    def broken(*a, **kw):
        raise RuntimeError("planner failed")

    r._live_models = fake_live
    monkeypatch.setattr(model_ops, "plan_entries", broken)
    await r._prefetch_model_cache()  # must not raise
    assert r._model_cache == []


async def test_prefetch_without_a_harness_leaves_the_cache_empty():
    llm = NS(config=NS(base_url="http://localhost:1/v1", model="m"))
    r = _repl(NS(), llm)
    await r._prefetch_model_cache()
    assert r._model_cache == []


async def test_prefetch_offers_a_cold_launch_record_with_its_detail(monkeypatch):
    """The menu holds what the list numbers: a primary that is down is a launch entry with its runtime."""
    from tests.unit.launch_support import stub_liveness

    stub_liveness(monkeypatch, False)
    llm = NS(config=NS(base_url="http://localhost:1/v1", model="m"))
    server = ManagedServerConfig(runtime="llamacpp", binary="/x/llama-server", model="/x/primary.gguf",
                                 port=1, gpu=True, served_name="qwen")
    r = _repl(NS(), llm, _harness(server))

    async def down(base_url):
        return ([], False)

    r._live_models = down
    await r._prefetch_model_cache()
    assert [name for name, _ in r._model_cache] == ["qwen"]
    assert r._model_cache[0][1].startswith("llama.cpp")


def _menu_entries() -> Entries:
    live = [PlanEntry("active-one", "live", "vllm", active=True),
            PlanEntry("other-one", "live", "vllm", rate="34.6 t/s measured"),
            PlanEntry("slow-one", "live", "vllm", rate="9.5 t/s measured")]
    cold = [PlanEntry("flash", "cold", "llamacpp", window=262_144),
            PlanEntry("moe", "cold", "llamacpp", window=32_768, where="experts in RAM")]
    return Entries((EntryGroup("vLLM · localhost:1", tuple(live), True),
                    EntryGroup("this box can launch", tuple(cold), True)))


def test_menu_items_hold_the_numbered_entries_in_listing_order():
    assert OrchestratorREPL._menu_items(_menu_entries()) == [
        ("active-one", "serving now"),
        ("other-one", [("", "serving · "), ("class:tps-green", "34.6 t/s")]),
        ("slow-one", [("", "serving · "), ("class:tps-red", "9.5 t/s")]),
        ("flash", "llama.cpp · 262,144"),
        ("moe", "llama.cpp · experts in RAM · 32,768"),
    ]


def test_menu_items_leave_out_what_the_list_does_not_number():
    unnumbered = Entries((EntryGroup("available once x is stopped", (PlanEntry("flash", "cold", "llamacpp"),), False),))
    assert OrchestratorREPL._menu_items(unnumbered) == []


def test_the_repl_has_one_menu_builder():
    assert hasattr(OrchestratorREPL, "_menu_items") and not hasattr(OrchestratorREPL, "_compose_model_menu")


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


def test_measured_note_prefers_ledger_and_falls_back_to_registry():
    assert OrchestratorREPL._measured_note({"vllm:m": 31.0}, "vllm", "m", NS(tps=9.5)) \
        == "31.0 t/s measured"
    assert OrchestratorREPL._measured_note({}, "vllm", "m", NS(tps=9.5)) == "~9.5 t/s measured"
    assert OrchestratorREPL._measured_note({}, None, "m", None) is None
