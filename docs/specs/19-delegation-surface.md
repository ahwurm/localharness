# Spec 19 — The Delegation Surface: What the `agent` Tool Exposes

**Status:** Draft, revision 2 (pass 1: security REVISE, design SHIP, ux SHIP; pass 2: SHIP from all three; see §7)
**Version target:** next release after 0.17.1
**API version:** `PLUGIN_API_VERSION` unchanged ("1")
**Numbering:** spec 18 is reserved by spec 17 §1 for the task record; this is 19.

---

## 1. Problem

Observed over several sessions driving this repository with a frontier model as the coordinator
(2026-10-06 to 2026-10-08): asked for substantive work, the model repeatedly **authored a new
specialist YAML, or scripted its own runner around the harness**, for work an existing specialist
already covered. Once, while "testing" how the harness should validate pathspecs, it crashed the
system `git` on purpose with an internal-only pathspec magic, which raised an apport dialog on the
owner's desktop.

The owner's ruling (2026-10-08): this is not a documentation problem and must not be solved with a
repository `CLAUDE.md` or `AGENTS.md`, which would collide with a user's own project instructions.
**The harness shapes whichever model drives it through what it exposes**: tool descriptions, the
role prompt, the packet. The `task` tool is the pattern to copy. Its description says when to use
it, what the model does with it, what the runtime owns, and the one ordering rule that makes
evidence bind. It offers no escape hatch.

The `agent` tool's description did the opposite. Reading it top-down, before the first revision of
this spec (`src/localharness/tools/builtin/agent_tool.py:65-85`):

1. "Delegate a task to a subagent. Available agents: explore, web-researcher, cruncher,
   search-verifier, design-critic, …" — **bare names, no purpose**. A model cannot tell
   `cruncher` from `search-verifier` without reading `subagent.py`, so it cannot pick one.
2. "Use this when a specialized agent would handle the task better than you." — no stance on
   what the coordinator itself should and should not do.
3. "You can also BUILD a new specialist: write `~/.localharness/agents/<name>.yaml` (…), then
   delegate to that name **immediately**." — the third sentence, unconditional, with a verb that
   says do it now.
4. The structured brief (`purpose`, `inputs`, `constraints`, `expected_output`, `checks`,
   `stop_condition`) and the HANDOFF block came **last**.

Runtime composition of specialists is a deliberate seam (`agent/subagent.py` docstring: "the
model itself, by WRITING the yaml first, composes new specialists at runtime"; owner ruling
Option 1, pinned by `test_agent_tool_invites_composition_and_names_the_boundary`). The problem
is the shape of the exposure, not the capability.

## 2. Ground truth (what exists)

Line numbers against the tree at `3608c28` plus the uncommitted work on 0.17.1.

| Thing | Where | Fact |
|---|---|---|
| `AgentTool.__init__` | `tools/builtin/agent_tool.py` | `(agent_runner, available_agents: list[str] \| None)`; the list is read **live** at `info()`, because `start_cmd` and the REPL append to it when a yaml is deployed (`tests/unit/test_start_cmd.py::…roster…`, `test_repl_creation_flow.py:230`). |
| Roster source | `cli/start_cmd.py:1496-1497` | `["explore", "web-researcher", "cruncher", "search-verifier"] + [c.name for c in card_registry.all_cards()]`. |
| Agent cards | `orchestrator/cards.py:115-128` | `AgentCard.description = config.role[:200]`; built per config agent by `register_from_config`. Cards exist so a coordinator can hold 50 of them under 8K tokens. **Nothing shows them to the model today.** |
| Built-in roles | `agent/subagent.py:53,95,145,437,445` | `EXPLORE_ROLE`, `WEB_RESEARCHER_ROLE`, `SEARCH_VERIFIER_ROLE`, `CRUNCHER_ROLE`, `CHUNK_SUMMARIZER_ROLE`: full prompts, no one-line purpose anywhere. |
| Nested `agent` tool | `agent/subagent.py:1434` | `AgentTool(agent_runner=child_runner, available_agents=delegatees)` for a non-leaf child (today only `web-researcher → search-verifier`). |
| Bench `agent` tool | `bench/runner.py:598` | `AgentTool(agent_runner=_agent_runner, available_agents=_bench_agents)`. |
| Role prompt | `agent/loop.py:1401-1409` | Says to keep the working record with `task`; says nothing about delegation. Left unchanged by this spec (§6). |
| `task` tool description | `tools/builtin/task_tool.py:66-77` | The reference shape: when, what, what the runtime owns, the binding rule. |

## 3. Design

Two changes, both inside the `agent` tool's own surface. No new config, no new ask, no new
file, no change to the gate, the capability floor, the runner, or the brief.

### 3.1 The roster carries a purpose

`AgentTool` takes an optional `describe_agent: Callable[[str], str | None]`. `info()` renders
each listed name as `name: purpose`, where the purpose is:

1. `describe_agent(name)` when it returns text — `start_cmd` passes a closure over the **live**
   card registry, so a yaml deployed this session is described the moment it is listed;
2. else `BUILTIN_AGENT_PURPOSES[name]`, a five-entry table in `agent_tool.py` for the built-ins;
3. else the bare name.

`purpose_line()` reduces a description to its first sentence, whitespace-collapsed, capped at
110 characters with an ellipsis. A sentence ends at `.`, `!` or `?` followed by whitespace and a
capital (or the end of the text), so "e.g. a reviewer" and "U.S. based" inside the first sentence
do not end it (design-critic, pass 1). For a config specialist that is the first sentence of its
`role` ("You are a security engineer reviewing a change to the localharness project"), which is
what every shipped specialist opens with. A describer that raises degrades to rule 2 or 3 and
logs one debug line per tool, not per request (security-critic, pass 1); the tool schema can
never fail because a registry did.

**Only the global layer lends its text (security-critic, pass 1, MAJOR).** `start_cmd`'s
describer, `describe_agent_for_schema(name, card_registry, workspace)`, returns `None` for any
name whose yaml sits in the workspace layer (`layer_files(workspace, "agents", contained=True)`):
a `.localharness/agents/<name>.yaml` is one a cloned repo can ship, and before this rule a
sentence of it would have landed in the coordinator's system-tier tool schema on every request.
Such a specialist is listed by bare name. The workspace file wins by stem, so a stem present
there masks the global file's text too. This is the line the harness already draws for gate
modes a project layer may not loosen. A yaml the model writes itself lands in the global layer
and is described; its text was already in the model's own context.

**The roster is capped (ux-friction-critic, pass 1).** Purposes are rendered for the first
`ROSTER_PURPOSES_MAX = 20` listed names; the rest are bare. The roster rides on every request,
and a heavy config must not inflate every prompt without bound. `estimated_tokens` on the schema
is `500 + 20 × len(roster)` rather than the old static 500 (design-critic, pass 1).

Cost: ~15–25 tokens per listed specialist on every request. With the four built-ins and the
owner's six config agents, about 200 tokens. The cards were designed for exactly this budget.

### 3.2 The description's order is the guidance

The description is rewritten so that reading it top-down gives, in order:

1. **What it is and who is available.** "Delegate a bounded assignment to a specialist, which
   runs in a fresh context with only its own tools and returns a summary. Specialists: …"
2. **How to pick and how to brief.** Pick the listed specialist whose purpose fits; delegate
   reading, searching and analysis rather than doing it in your own context; for substantial
   work use the six brief fields; the specialist ends with a HANDOFF block.
3. **The coordinator's stance.** "You coordinate: integrate each result, keep your own context
   for decisions. Delegations run one at a time." Then the unchanged
   handle/`grant_handles`/`cruncher` guidance. Revision 1 also said "a specialist cannot
   delegate further"; the design critic noted that is a fact about today's runner, not about
   every `AgentTool` instance, so the depth cap stays enforced by the runner and unadvertised.
4. **Composition, bounded and last.** "Only when no listed specialist's purpose fits the
   assignment, BUILD a new specialist: … with the smallest tool set that does the job, then
   delegate to that name." The word "immediately" is gone. The owner's boundary stays verbatim
   ("compose specialists from the existing tools; never write your own tool for something the
   harness already provides") and gains the one clause this spec exists for: "and never script a
   runner or loop around the harness." The trust notes about MCP servers, looser permissions and
   `~/.localharness/tools/` scripts are unchanged.

Everything the old description said is still said. What changed is the order and two
conditions.

### 3.3 Call sites

| Site | Change |
|---|---|
| `cli/start_cmd.py` | passes `describe_agent=lambda n: describe_agent_for_schema(n, card_registry, workspace)`; the helper is module-level and unit-tested, and reads the registry by key (`AgentCardRegistry.get`), not by scan. |
| `cli/start_cmd.py` (`make_explore_agent_runner(`) | the runner is built with `window=lambda: ctx_mgr.max_context_tokens` in place of the resolved int (spec 22 §9); each child's `ContextManager` takes the value at its spawn. |
| `agent/subagent.py:1434` (nested) | no description change: its delegatees are built-ins, covered by the table; it takes the same `window` callable. |
| `bench/runner.py:598` | no description change: bench agents are built-ins, covered by the table; it passes `lambda: effective_max_context_tokens`. Bench prompts therefore change by the reordered description only, which is the point of the bench. |

## 4. Tests

`tests/unit/test_agent_tool.py`:

- `test_agent_tool_invites_composition_and_names_the_boundary` — unchanged in substance: the
  ruling that composition stays, with the owner's boundary.
- `test_agent_tool_description_order_is_roster_brief_stance_then_composition` — the indices of
  roster < brief fields < "You coordinate" < "Only when no listed specialist" < "BUILD a new
  specialist"; "immediately" absent; "never script a runner or loop around the harness" present.
- `test_agent_tool_roster_carries_a_purpose_per_specialist` — describer's first sentence for a
  config agent, the table for a built-in, bare name otherwise; the roster appears in the
  description.
- `test_agent_tool_roster_is_live_and_describer_failure_is_contained` — the live-list behaviour
  `test_start_cmd` already pins, plus a raising describer degrading to the built-in purpose.
- `test_purpose_line_caps_and_takes_first_sentence`, `test_purpose_line_does_not_end_at_an_abbreviation`.
- `test_roster_caps_purposes_and_lists_the_rest_bare`, `test_estimated_tokens_grows_with_the_roster`,
  `test_describer_failure_is_logged_once_not_per_request`.

`tests/unit/test_start_cmd.py`:

- `test_describe_agent_for_schema_reads_the_global_layer_only` — a workspace stem returns `None`,
  the same name with no workspace layer returns the role text, an unknown name returns `None`.
- `test_agent_tool_roster_shows_workspace_specialists_bare` — end to end through the tool.

Existing tests that construct `AgentTool` without a describer (11 files) are unchanged: the
parameter is optional and the default renders built-ins from the table and anything else bare.

## 5. Measurement

The claim is behavioural: fewer unprompted specialist authorings and no runner scripts for work
an existing specialist covers. The evidence standard is the same as spec 17's: session logs
under `.localharness/agents/<agent>/sessions/`, counting `write` calls to `agents/*.yaml` and
`agent` calls per substantive turn, before and after. No number is claimed here until a bench
or a logged session pair produces one.

## 6. Out of scope, by decision

- **The role prompt** (`loop.py:1401`) says nothing about delegation and still does not. The
  `agent` tool is where delegation lives; putting the same words in two places is how they
  drift. If the critic pass finds the tool description alone insufficient, the role prompt gets
  one sentence, in a revision of this spec.
- **Refusing an `agent` call to a yaml written in the same turn.** Considered and rejected for
  revision 1: it is a runtime rule on the model's behalf, which spec 17 and the task record
  design both avoid ("the runtime records facts and reports them; it does not enforce working
  rules"). Revisit only with evidence that the description change was not enough.
- **A `description:` field on `AgentConfig`.** The first sentence of `role` is what every shipped
  specialist already leads with; a second field is a second thing to keep in sync.
- **A first sentence longer than the card's 200-character `description`** is cut mid-sentence
  before `purpose_line` sees it and ends with an ellipsis (ux-friction-critic, pass 1). No
  shipped specialist is near that; reading the full `role` would mean loading the config in the
  schema path. Accepted for now.
- **Disclosure to a remote model endpoint.** The roster sends a sentence of each global
  specialist's role on every request where only names went before (security-critic, pass 1,
  MINOR). LocalHarness is local-first; a user pointing `provider.base_url` at a remote endpoint is
  already sending every role prompt in full. Noted, not acted on.
- **Pathspec validation for the harness's git-facing tools.** The crash in §1 was the model
  misusing system git, not a harness path; it is noted here as the symptom, not fixed here.

## 7. Critic log

Reviews are delegated from a `localharness start` session in this repository, one `agent` call
per critic with the structured brief, and written verbatim to `reports/spec19-critic-pass-N.md`
(git-ignored). The diff reviewed is `reports/spec19-revN.diff`.

| Rev | Critic | Verdict | Folded in |
|---|---|---|---|
| 1 | security-critic | REVISE | MAJOR: workspace-layer role text reached the tool schema → global layer only (§3.1). MINOR: silent describer degradation → one debug line. MINOR: remote-endpoint disclosure → §6 note. |
| 1 | design-critic | SHIP, 5 MINOR | static `estimated_tokens` → dynamic; naive sentence split → capital-after-stop; "cannot delegate further" → dropped; O(N×M) card scan → `registry.get`; public `roster()`/`purpose_of()` → kept, per the critic. |
| 1 | ux-friction-critic | SHIP, 2 MINOR | unbounded roster → `ROSTER_PURPOSES_MAX`; first sentence > 200 chars → §6 note. Zero new human asks confirmed. |
| 2 | security-critic | SHIP | all three pass-1 findings confirmed resolved; nothing new. |
| 2 | design-critic | SHIP, 2 MINOR | digit after an abbreviation still ended the sentence → `\d` dropped from the lookahead, test added. Per-name `layer_files` listing in the describer → accepted: one small directory listing per listed name per request, and the helper stays a pure function; revisit if the roster or the layer grows. |
| 2 | ux-friction-critic | SHIP | both pass-1 findings confirmed resolved; zero asks beyond the rulings. |
