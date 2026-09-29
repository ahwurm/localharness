"""SAFE-06: a plugin you installed cannot declare its way past the permission gate.

The clamp (plugins/trust.py) rewrites what a NON-bundled plugin's tool declares to the gate, at
registration and before any reader sees it, whenever the declaration would be treated MORE
permissively than saying nothing. Which declarations those are is not argued here; it is MEASURED
over the real verdict (agent/verdict.evaluate): every GateFamily member, all five modes, seven
parameter shapes, a review surface on and off, an empty grant store, under the default settings and
under the two operator knobs that change what a declaration can reach (per-host network asks, a
trusted MCP server).

The guarantee, stated the way the gate can actually give it (SAFE-06: "honoured only when it is
'ask' or stricter than today's default for an unclassified tool — a third-party tool never
self-selects into silent allow"):
1. the clamp set is exactly the families measured to be more permissive than undeclared, no wider;
2. after the clamp, no declaration is more permissive than an undeclared tool, in any probe;
3. in `guarded`, with an empty grant store, no third-party tool is ever allowed.
Why the guarantee is relative, and not "never ALLOW in auto, guarded and trusted": the pin test.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, get_args

from localharness.agent.gate import tool_meta_from_schema
from localharness.agent.gate_types import GateSettings, Mode, ToolMeta, Verdict
from localharness.agent.verdict import MCP_GROUP_PREFIX, GateContext, evaluate
from localharness.config.models import ToolConfig
from localharness.plugins.trust import THIRD_PARTY_CLAMPED_FAMILIES, third_party_overrides
from localharness.tools.base import GATE_FAMILIES, Tool, ToolResult, ToolSchema
from localharness.tools.registry import ToolRegistry

NAME = "thirdparty_tool"  # in none of the gate's name tables: judged by its declaration alone
MODES: tuple[str, ...] = get_args(Mode)
RANK = {Verdict.ALLOW: 0, Verdict.ASK: 1, Verdict.DENY: 2}
TRUSTED_SERVER = "github"
SETTINGS = {
    "default": GateSettings(),
    "knobs on": GateSettings(ask_network_hosts=True, mcp_trusted_servers=frozenset({TRUSTED_SERVER})),
}
MCP_GROUP = f"{MCP_GROUP_PREFIX}{TRUSTED_SERVER}"  # what the gate reads as that server's tool
GROUPS = ("other", MCP_GROUP)
DECLARED = (*sorted(GATE_FAMILIES), None, "read_only")  # every member, undeclared, unrecognised
UNDECLARED = ToolMeta()  # the baseline: a tool that declares nothing to the gate

Probe = tuple[tuple[str, str, dict, bool], GateSettings, GateContext]


class _ThirdParty(Tool):
    """A tool from a plugin you installed, declaring `family` and `group` to the gate."""

    def __init__(self, family: str | None, group: str) -> None:
        super().__init__()
        self._schema = ToolSchema(name=NAME, description="a tool from a plugin you installed",
                                  parameters={"type": "object", "properties": {}}, group=group,
                                  ingest="none", host="safe", result_origin="trusted",
                                  gate_family=family)

    def info(self) -> ToolSchema:
        return self._schema

    async def _execute(self, **kwargs: Any) -> ToolResult:
        return self.ok("ran")


def _probes(ws: Path) -> Iterator[Probe]:
    params = ({}, {"q": "x"}, {"path": str(ws / "notes.txt")}, {"path": "/etc/hosts"},
              {"command": "ls"}, {"command": "rm -rf /tmp/lh-clamp-probe"},
              {"url": "https://example.com/x"})
    for label, settings in SETTINGS.items():
        for mode in MODES:
            for p in params:
                for review in (False, True):
                    ctx = GateContext(boundary=ws, workspace=ws, grants=lambda *_: None,
                                      refusals=None, mode=mode, has_review_surface=review)
                    yield (label, mode, p, review), settings, ctx


def _verdict(meta: ToolMeta, probe: Probe) -> Verdict:
    (_, _, params, _), settings, ctx = probe
    return evaluate(NAME, params, meta, ctx, settings).verdict


async def _effective(family: str | None, group: str) -> ToolMeta:
    """What the gate reads for a non-bundled plugin's tool: the clamp applied at registration, as
    the loader applies it, read back the way agent/loop.py `_tool_facts` reads it."""
    tool = _ThirdParty(family, group)
    overrides, _ = third_party_overrides(tool.info())
    reg = ToolRegistry()
    await reg.register(tool, source_plugin="thirdparty", overrides=overrides)
    return tool_meta_from_schema(reg.lookup_tool(NAME, "root", "", ToolConfig()).info())


def test_the_clamp_set_is_exactly_the_families_measured_more_permissive_than_undeclared(tmp_path):
    ws = tmp_path.resolve()
    looser: dict[str, list] = {}
    for probe in _probes(ws):
        baseline = RANK[_verdict(UNDECLARED, probe)]
        for family in sorted(GATE_FAMILIES):
            if RANK[_verdict(ToolMeta(gate_family=family), probe)] < baseline:
                looser.setdefault(family, []).append(probe[0])

    measured = frozenset(looser)
    assert measured == THIRD_PARTY_CLAMPED_FAMILIES, (
        f"measured {sorted(measured)}, clamped {sorted(THIRD_PARTY_CLAMPED_FAMILIES)}. "
        "The clamp must equal the evidence, no narrower and no wider. Probes that made each "
        "member more permissive than an undeclared tool (settings, mode, params, review): "
        + "; ".join(f"{f}: {len(p)}, e.g. {p[0]}" for f, p in sorted(looser.items())))


async def test_after_the_clamp_no_declaration_is_more_permissive_than_undeclared(tmp_path):
    ws = tmp_path.resolve()
    # Premise: unclamped, both axes this test sweeps DO get through somewhere, so a clamp that
    # missed either one could not pass unnoticed.
    raw = tool_meta_from_schema(_ThirdParty(None, MCP_GROUP).info())
    assert any(_verdict(raw, p) is Verdict.ALLOW and _verdict(UNDECLARED, p) is Verdict.ASK
               for p in _probes(ws))
    assert THIRD_PARTY_CLAMPED_FAMILIES

    looser = []
    for family in DECLARED:
        for group in GROUPS:
            meta = await _effective(family, group)
            looser += [(family, group, probe[0]) for probe in _probes(ws)
                       if RANK[_verdict(meta, probe)] < RANK[_verdict(UNDECLARED, probe)]]
    assert not looser, f"{len(looser)} probes more permissive than undeclared, e.g. {looser[:3]}"


async def test_in_guarded_with_no_grants_no_third_party_tool_is_ever_allowed(tmp_path):
    """Criterion 2's "still asked about": whatever it declares, `read_only` included."""
    ws = tmp_path.resolve()
    allowed = []
    for family in DECLARED:
        for group in GROUPS:
            meta = await _effective(family, group)
            allowed += [(family, group, probe[0]) for probe in _probes(ws)
                        if probe[0][1] == "guarded" and _verdict(meta, probe) is Verdict.ALLOW]
    assert not allowed, f"{len(allowed)} guarded probes allowed, e.g. {allowed[:3]}"


def test_the_undeclared_baseline_is_allowed_in_auto_and_trusted_by_the_mode_rulings(tmp_path):
    """Why SAFE-06 is asserted RELATIVE to an undeclared tool, not as "never ALLOW in auto,
    guarded and trusted" (the first wording in 44-CONTEXT.md and the orchestrator's brief).

    That literal form cannot pass for ANY clamp, because the clamp's own target, an undeclared
    tool (`tool-unfamiliar`), is allowed in two of those three modes by design:
    - `auto`, the default. Owner ruling 2026-09-11 (agent/gate_types.py DEFAULT_MODE): "way too
      intrusive, it stopped me multiple times… the default should be an auto mode that almost
      never triggers unless genuinely risky / dangerous… essentially the thinnest interaction off
      of no interaction". `_decide` drops every ask off the auto blacklist (AUTO_ASK_CLASSES:
      protected-path and shell-destructive), and `tool-unfamiliar` is not on it.
    - `trusted` allows a request whenever every ask in it is grantable, and `tool-unfamiliar` is.
    Making an undeclared tool ask there would change what `auto` and `trusted` MEAN for every
    tool the gate does not know by name: a mode change for the owner to rule on, not a plugin
    rule. SAFE-06's own sentence is relative to exactly this baseline ("'ask' or stricter than
    today's default for an unclassified tool"), and that is what the tests above assert. If this
    pin fails, the modes changed: re-read those tests against the new rulings before touching
    the clamp.
    """
    ws = tmp_path.resolve()
    got = {mode: evaluate(NAME, {}, UNDECLARED,
                          GateContext(boundary=ws, workspace=ws, grants=lambda *_: None, mode=mode),
                          GateSettings()).verdict
           for mode in ("auto", "trusted", "guarded")}
    assert got == {"auto": Verdict.ALLOW, "trusted": Verdict.ALLOW, "guarded": Verdict.ASK}


def test_the_override_rewrites_only_what_it_must_and_the_warning_says_what():
    overrides, warning = third_party_overrides(_ThirdParty("allow", "other").info())
    assert overrides == {"gate_family": None}
    assert f"{NAME!r}" in warning and "'allow'" in warning and "undeclared" in warning

    overrides, warning = third_party_overrides(_ThirdParty("code", MCP_GROUP).info())
    assert overrides == {"group": "other"}  # the honoured family stays; the MCP disguise goes
    assert f"{NAME!r}" in warning and repr(MCP_GROUP) in warning

    for family in ("code", "delegate", None, "read_only"):
        assert third_party_overrides(_ThirdParty(family, "other").info()) == ({}, None)
