"""`/model` prints one row to a model: `name · clause · …`, and a model that cannot run says what it needs and, under it, what to do."""
from __future__ import annotations

import time

import pytest

from localharness.cli import model_ops
from localharness.cli.model_ops import Entries, EntryGroup, PlanEntry
from localharness.provider import autosense
from tests.unit.test_launch_paths_repl import PEER_URL, PRIMARY_URL, _launchable_peer, _session

NAME = "gemma-4-31B-it-DECKARD-HERETIC-UNCENSORED-Thinking-Q4_K_M"
SWIFT = "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS"
BF16 = "gemma-4-31B-DECKARD-bf16"
RAM = autosense.RAM_REFUSAL.format(name=SWIFT, need="40.0", free="16.9")
VRAM = autosense.VRAM_REFUSAL.format(name=BF16, need="57.2", window=" plus a 12,800-token window", free="26.4",
                                     remedy=autosense.VRAM_REMEDY)


def listing(*refused: PlanEntry) -> Entries:
    """A live model, two the box can launch and the refused ones given: the approved screenshot's list."""
    return Entries(
        groups=(EntryGroup("llama.cpp · 127.0.0.1:8080 · the harness",
                           (PlanEntry("qwen-27b-swift-1.5", "live", "llamacpp", active=True, rate="125.4 t/s measured"),), True),
                EntryGroup("this box can launch",
                           (PlanEntry("Qwen3.8-27B-Nvidia-NVFP4", "cold", "llamacpp", 115_456, "gpu", rate="104.2 t/s measured"),
                            PlanEntry(NAME, "cold", "llamacpp", 58_368, "gpu", rate="50.9 t/s measured")), True)),
        not_offered=refused)


def refused(name: str, sentence: str, **kw) -> PlanEntry:
    return PlanEntry(name, "cold", "llamacpp", refusal=sentence, **kw)


def test_the_list_is_one_row_to_a_model_and_a_refused_model_has_its_advice_under_it():
    assert model_ops.entry_lines(listing(refused(SWIFT, RAM), refused(BF16, VRAM))) == [
        "Models:",
        "▸ llama.cpp · 127.0.0.1:8080 · the harness",
        "  1. qwen-27b-swift-1.5 · serving · active · 125.4 t/s measured",
        "▸ this box can launch",
        "  2. Qwen3.8-27B-Nvidia-NVFP4 · llama.cpp · 115,456 window · 104.2 t/s measured",
        f"  3. {NAME} · llama.cpp · 58,368 window · 50.9 t/s measured",
        f"  –  {SWIFT} · not offered · needs 40.0 GiB RAM, 16.9 GiB free",
        "     close something that holds RAM, or pick a smaller quant",
        f"  –  {BF16} · not offered · needs 57.2 GiB GPU memory plus a 12,800-token window, 26.4 GiB free",
        "     stop what holds the GPU, or pick a smaller quant",
    ]


def test_a_served_model_that_is_not_the_active_one_says_serving_alone():
    entries = Entries(groups=(EntryGroup("vLLM · localhost:8081", (
        PlanEntry("a", "live", "vllm", active=True), PlanEntry("b", "live", "vllm", rate="31.0 t/s measured")), True),))
    assert model_ops.entry_lines(entries)[2:] == ["  1. a · serving · active", "  2. b · serving · 31.0 t/s measured"]


def test_the_clauses_of_a_row_that_is_not_set_up_or_keeps_experts_in_ram_or_has_a_strata_offer():
    entries = Entries(groups=(EntryGroup("this box can launch", (
        PlanEntry("lib", "setup", "llamacpp", 8192, None),
        PlanEntry("big", "cold", "llamacpp", 65_536, "experts in RAM", strata="big-strata", rate="12.0 t/s measured")), True),))
    assert model_ops.entry_lines(entries)[2:] == [
        "  1. lib · llama.cpp · 8,192 window · not set up",
        "  2. big · llama.cpp · experts in RAM · 65,536 window · Strata available: /model big-strata · 12.0 t/s measured"]


def test_an_advice_row_is_cut_at_120_with_an_ellipsis_and_a_remedy_that_fits_is_not():
    long_remedy = "a q8_0 KV cache fits it " + "(cache_type_k and cache_type_v: q8_0) " * 5
    sentence = autosense.VRAM_REFUSAL.format(name=BF16, need="57.2", window="", free="26.4", remedy=long_remedy.strip())
    advice = model_ops.entry_lines(listing(refused(BF16, sentence)))[-1]
    assert len(advice) <= 120 and advice.endswith("…") and not advice.endswith(" …")  # 119 characters, trimmed, and the mark
    assert advice.startswith("     a q8_0 KV cache fits it (cache_type_k") and advice == ("     " + long_remedy.strip())[:119].rstrip() + "…"
    assert model_ops.entry_lines(listing(refused(SWIFT, RAM)))[-1] == "     close something that holds RAM, or pick a smaller quant"


@pytest.mark.parametrize("sentence", [
    f"{SWIFT} cannot run here: its record has changes you have not confirmed",
    f"{SWIFT} cannot run here: could not plan it (OSError)",
    f"{SWIFT} cannot run here: it uses custom code — the harness never passes --trust-remote-code.",
])
def test_a_reason_with_no_short_form_is_the_whole_reason_and_prints_no_advice_row(sentence):
    rows = model_ops.entry_lines(listing(refused(SWIFT, sentence)))[-1:]
    assert rows == [f"  –  {SWIFT} · not offered · {model_ops.not_offered_reason(sentence, SWIFT)}"]


def test_a_reason_with_no_short_form_is_cut_at_120_with_an_ellipsis():
    sentence = f"{SWIFT} cannot run here: " + "it will not fit, " * 20
    row = model_ops.entry_lines(listing(refused(SWIFT, sentence)))[-1]
    reason = row.split(" · not offered · ")[1]
    assert len(reason) <= 120 and reason.endswith("…") and not reason.endswith(" …")
    assert reason == ("it will not fit, " * 20)[:119].rstrip() + "…"


def test_the_strata_offer_rides_the_fact_row_not_the_advice_row():
    rows = model_ops.entry_lines(listing(refused(SWIFT, RAM, strata="swift-strata")))[-2:]
    assert rows == [f"  –  {SWIFT} · not offered · needs 40.0 GiB RAM, 16.9 GiB free · Strata available: /model swift-strata",
                    "     close something that holds RAM, or pick a smaller quant"]


def test_a_name_with_control_characters_prints_without_them():
    rows = model_ops.entry_lines(listing(refused("bad\x1b[2Jname", RAM.replace(SWIFT, "bad\x1b[2Jname"))))
    assert all("\x1b" not in row for row in rows)


def test_the_phone_still_gets_the_whole_reason_and_the_sentence_a_person_asks_for_is_whole():
    entries = listing(refused(SWIFT, RAM), refused(BF16, VRAM))
    assert model_ops.not_offered_reason(RAM, SWIFT).endswith("close something that holds RAM, or pick a smaller quant")
    assert entries.pick(SWIFT) == RAM and entries.pick(BF16) == VRAM


@pytest.mark.asyncio
async def test_slash_model_with_a_refused_name_prints_the_full_sentence_and_the_bare_list_the_short_rows(tmp_path):
    _, repl, channel, _ = _session(tmp_path, _launchable_peer())
    entries = listing(refused(SWIFT, RAM), refused(BF16, VRAM))

    async def planned(llm):
        return entries, {model_ops._norm_url(PRIMARY_URL): ("qwen-27b-swift-1.5",), model_ops._norm_url(PEER_URL): None}, set()

    repl._plan_model_entries = planned

    await repl._handle_model_cmd(SWIFT)
    assert channel.messages[-1] == RAM

    await repl._handle_model_cmd("")
    shown = channel.messages[-1].splitlines()
    assert shown[0] == "Models:" and shown[-1] == "/model <name or number> switches · Enter picks from the menu"
    assert f"  –  {SWIFT} · not offered · needs 40.0 GiB RAM, 16.9 GiB free" in shown
    assert not any("Switch with" in row or "[active]" in row or "(serving)" in row for row in shown)


def test_a_name_that_embeds_the_templates_own_words_cannot_stall_the_list():
    """The name is the one part of a refusal sentence the harness did not write; the split reads the rest."""
    name = " cannot run here: its experts need 1 GiB of RAM and only 2" * 800
    sentence = autosense.VRAM_REFUSAL.format(name=name, need="57.2", window="", free="26.4", remedy=autosense.VRAM_REMEDY)
    started = time.perf_counter()
    fact, advice = model_ops.refusal_rows(sentence, name)
    assert time.perf_counter() - started < 0.5
    assert (fact, advice) == ("needs 57.2 GiB GPU memory, 26.4 GiB free", "stop what holds the GPU, or pick a smaller quant")
    # what labels the sentence (a model, or the file a record launches) is not read: it need not be the entry's own name
    assert model_ops.refusal_rows(RAM, "flash-next") == ("needs 40.0 GiB RAM, 16.9 GiB free", "close something that holds RAM, or pick a smaller quant")


@pytest.mark.asyncio
async def test_the_list_hands_the_channel_every_name_it_prints_the_unnumbered_ones_too(tmp_path):
    """A terminal keeps these bright on the dim list: a numbered row's number with its name, the rows of a group a foreign holder
    left unnumbered, and the refused."""
    _, repl, channel, _ = _session(tmp_path, _launchable_peer())
    entries = Entries(
        groups=(EntryGroup("llama.cpp · 127.0.0.1:8080", (PlanEntry("qwen-a", "live", "llamacpp", active=True),), True),
                EntryGroup("available once llm.service is stopped", (PlanEntry("flash-next", "cold", "llamacpp", 8_192, "gpu"),), False)),
        not_offered=(refused(SWIFT, RAM),))

    async def planned(llm):
        return entries, {model_ops._norm_url(PRIMARY_URL): ("qwen-a",), model_ops._norm_url(PEER_URL): None}, set()

    repl._plan_model_entries = planned
    sent = []
    send = channel.send_message

    async def spy(text, metadata=None):
        sent.append(metadata)
        await send(text, metadata=metadata)

    channel.send_message = spy

    await repl._handle_model_cmd("")

    assert sent[-1]["names"] == ["1. qwen-a", "flash-next", SWIFT]
