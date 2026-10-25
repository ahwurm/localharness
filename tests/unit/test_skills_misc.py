"""Skills seams outside the skill modules: the removed microagents key, protected paths, the origins they guard and the children that never hold the tool."""
from __future__ import annotations

import re
from importlib import resources
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from localharness.agent.gate_types import PROTECTED_CONFIG_DIR_ENTRIES, Verdict
from localharness.agent.subagent import CONFIG_CHILD_DEFAULT_TOOLS, _config_child_allowed
from localharness.agent.verdict import AGENTS_SKILLS, GateContext, evaluate
from localharness.config.loader import ConfigLoader, ConfigValidationError
from localharness.config.models import AgentConfig, ContextConfig, ToolConfig
from localharness.config.paths import WORKSPACE_DIR_NAME
from localharness.config import skills
from localharness.cli.app import app
from localharness.config.skills import BUNDLED_NAME, ORIGINS, SkillSet, Skipped, skip_line
from localharness.agent import subagent
from localharness.agent.context import ContentStore
from localharness.agent.task_record import TaskState
from localharness.tools.builtin.skill_tool import SkillTool
from localharness.tools.registry import ToolRegistry
from tests.unit.agent.test_auto_mode import SETTINGS, SHELL, WRITE, _ctx, _shell, _verdict
from tests.unit.test_docs_convergence import check_protected_entries_doc
from tests.unit.test_capabilities import _production_registry
from tests.unit.test_leftover_plugin_section import _CONFIG, _agent_file
from tests.unit.test_skills_discovery import _skill
from tests.unit.test_skills_listing import _start_world


@pytest.fixture
def g(tmp_path: Path) -> Path:
    g = tmp_path / "g"
    g.mkdir()
    (g / "config.yaml").write_text(_CONFIG, encoding="utf-8")
    return g


def test_a_yaml_that_still_sets_context_microagents_fails_naming_the_key(g) -> None:
    agent = _agent_file(g, "context:\n  microagents: [style]\n")

    with pytest.raises(ConfigValidationError) as exc:
        ConfigLoader(config_dir=g).load_agent_file(agent)

    assert "context.microagents (line 4): Extra inputs are not permitted" in str(exc.value), str(exc.value)


def test_the_microagents_field_and_loader_methods_are_gone() -> None:
    assert "microagents" not in ContextConfig.model_fields
    assert not hasattr(ConfigLoader, "microagent_paths")
    assert not hasattr(ConfigLoader, "discover_microagents")


_REFUSED = "Extra inputs are not permitted"


@pytest.mark.parametrize(
    "tail,expected",
    [
        pytest.param("  microagents: []\n", ["context.microagents (line 4): " + _REFUSED], id="empty-list"),
        pytest.param("  microagents:\n", ["context.microagents (line 4): " + _REFUSED], id="null"),
        pytest.param(
            "  microagents: [style]\n  bogus: 1\n",
            ["context.microagents (line 4): " + _REFUSED, "context.bogus (line 5): " + _REFUSED],
            id="beside-another-unknown-key",
        ),
        pytest.param("  microagent: [style]\n", ["context.microagent (line 4): " + _REFUSED], id="singular"),
    ],
)
def test_the_refusal_holds_for_every_form_of_the_key(g, tail, expected) -> None:
    agent = _agent_file(g, "context:\n" + tail)

    with pytest.raises(ConfigValidationError) as exc:
        ConfigLoader(config_dir=g).load_agent_file(agent)

    assert all(text in str(exc.value) for text in expected), str(exc.value)


def test_a_valid_context_block_still_loads(g) -> None:
    agent = _agent_file(g, "context:\n  max_context_tokens: 4096\n")

    assert ConfigLoader(config_dir=g).load_agent_file(agent).context.max_context_tokens == 4096


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    return root


@pytest.fixture
def global_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    folder = tmp_path / "global"
    folder.mkdir()
    monkeypatch.setenv("LOCALHARNESS_DIR", str(folder))
    return folder


def test_skills_is_the_thirteenth_protected_config_entry() -> None:
    assert len(PROTECTED_CONFIG_DIR_ENTRIES) == 13
    assert PROTECTED_CONFIG_DIR_ENTRIES[-1] == "skills"


def test_a_write_under_localharness_skills_is_an_ungrantable_protected_ask(project, global_dir) -> None:
    target = project / ".localharness/skills/release/SKILL.md"

    decision = evaluate("write", {"path": str(target)}, WRITE, _ctx(project), SETTINGS)

    assert decision.verdict is Verdict.ASK
    assert decision.request.klass == "protected-path"
    assert decision.request.grantable is False
    assert decision.request.reason.startswith("writes a protected path ("), decision.request.reason


@pytest.mark.parametrize(
    "tail,verdict",
    [
        pytest.param(".agents/skills/release/SKILL.md", Verdict.ASK, id="below-agents-skills"),
        pytest.param(".agents/skills", Verdict.ASK, id="agents-skills-itself"),
        pytest.param(".agents/skills/.localharness/x", Verdict.ASK, id="nested-localharness-cannot-reopen"),
        pytest.param(".agents", Verdict.ALLOW, id="agents-itself"),
        pytest.param(".agents/skillsx/x", Verdict.ALLOW, id="longer-name"),
        pytest.param(".agents/skill/x", Verdict.ALLOW, id="shorter-name"),
        pytest.param(".agents/notes.md", Verdict.ALLOW, id="sibling-file"),
        pytest.param("sub/.agents/skills/x", Verdict.ALLOW, id="not-at-a-container-root"),
        pytest.param(".agents/./skills/x/SKILL.md", Verdict.ASK, id="dot-part"),
        pytest.param(".agents//skills/x/SKILL.md", Verdict.ASK, id="doubled-slash"),
        pytest.param(".agents/skills/../notes.md", Verdict.ALLOW, id="dotdot-out"),
        pytest.param(".Agents/Skills/x", Verdict.ALLOW, id="case-is-exact"),
    ],
)
def test_agents_skills_is_protected_exactly_at_a_container_root(project, global_dir, tail, verdict) -> None:
    assert _verdict("write", {"path": f"{project}/{tail}"}, _ctx(project), WRITE) is verdict


def test_agents_skills_is_protected_from_the_workspace_and_from_the_boundary(project, global_dir) -> None:
    target = {"path": str(project / ".agents/skills/release/SKILL.md")}

    assert _verdict("write", target, _ctx(project, workspace=project / "sub"), WRITE) is Verdict.ASK
    assert _verdict("write", target, _ctx(project, boundary=None), WRITE) is Verdict.ASK


@pytest.mark.parametrize(
    "command",
    [
        "mv .agents away",
        "rm -rf .agents",
        "rm -rf .agents/skills",
        "mv .agents/skills elsewhere",
        "rm .agents/skills/release/SKILL.md",
        "rm .agents/plugins",
    ],
)
def test_moving_or_deleting_agents_skills_or_its_parent_asks(project, global_dir, command) -> None:
    decision = evaluate("bash_exec", {"command": command}, SHELL, _ctx(project), SETTINGS)

    assert decision.verdict is Verdict.ASK, command
    assert "deletes or moves a protected config entry" in decision.request.reason, decision.request.reason


@pytest.mark.parametrize(
    "command",
    ["rm .agents/notes.md", "mv .agents/notes.md .agents/n.md", "rm -rf skills", "rm -rf docs/skills"],
)
def test_other_children_of_agents_and_plain_skills_folders_do_not_ask(project, global_dir, command) -> None:
    assert _shell(command, _ctx(project)) is Verdict.ALLOW, command


_MODES = ("guarded", "trusted", "auto", "read-only", "unattended")
_AFTER_ASKING = {"guarded": Verdict.ASK, "trusted": Verdict.ASK, "auto": Verdict.ASK,
                 "read-only": Verdict.DENY, "unattended": Verdict.ALLOW}
_SKILL = "skills/release/SKILL.md"


@pytest.fixture
def session_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "session"
    folder.mkdir()
    return folder


def _skill_write(where: str, project: Path, global_dir: Path, session_dir: Path, mode: str):
    sub = project / "sub"
    if where == "session-config-dir":
        ctx = GateContext(boundary=project, workspace=project, grants=lambda workspace, klass, key: None,
                          mode=mode, has_review_surface=True, config_dir=session_dir)
        return session_dir / _SKILL, ctx
    target = {
        "global-config-dir": global_dir / _SKILL,
        "localharness-top": project / ".localharness" / _SKILL,
        "localharness-nested": sub / ".localharness" / _SKILL,
        "agents-workspace": project / ".agents" / _SKILL,
        "agents-boundary": project / ".agents" / _SKILL,
    }[where]
    return target, _ctx(project, mode=mode, workspace=sub if where == "agents-boundary" else None)


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize(
    "where",
    ["global-config-dir", "session-config-dir", "localharness-top", "localharness-nested",
     "agents-workspace", "agents-boundary"],
)
def test_a_skills_write_asks_in_every_mode_that_asks(where, mode, project, global_dir, session_dir) -> None:
    target, ctx = _skill_write(where, project, global_dir, session_dir, mode)

    decision = evaluate("write", {"path": str(target)}, WRITE, ctx, SETTINGS)

    assert decision.verdict is _AFTER_ASKING[mode], decision.reason
    if decision.verdict is Verdict.ASK:
        assert decision.request.klass == "protected-path"
        assert decision.request.grantable is False


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("command", ["mv .agents away", "rm -rf .localharness/skills", "rm -rf {g}/skills"])
def test_moving_or_deleting_a_skills_directory_asks_in_every_mode_that_asks(command, mode, project, global_dir) -> None:
    decision = evaluate("bash_exec", {"command": command.format(g=global_dir)}, SHELL, _ctx(project, mode=mode), SETTINGS)

    assert decision.verdict is _AFTER_ASKING[mode], decision.reason
    if decision.verdict is Verdict.ASK:
        assert decision.request.grantable is False


def _gate_gaps(origins) -> set:
    project = {Path(rel).parts for label, rel in origins if label == "project"}
    machine = {rel for label, rel in origins if label == "machine"}
    protected = {(WORKSPACE_DIR_NAME, name) for name in machine} | {AGENTS_SKILLS}
    return (project ^ protected) | {name for name in machine if name not in PROTECTED_CONFIG_DIR_ENTRIES}


def test_the_gate_protects_exactly_the_origins_that_need_it() -> None:
    assert _gate_gaps(ORIGINS) == set()


def test_the_origins_comparison_bites() -> None:
    project_row = ("project", ".agents/skills")
    machine_row = ("machine", "skills")

    assert _gate_gaps((*ORIGINS, ("project", ".agents/skill")))
    assert _gate_gaps(tuple(row for row in ORIGINS if row != project_row))
    assert _gate_gaps(tuple(("machine", "skill") if row == machine_row else row for row in ORIGINS))
    assert _gate_gaps(tuple(reversed(ORIGINS))) == set()


def test_security_md_names_skills_in_the_thirteen_entry_sentence() -> None:
    text = (Path(__file__).resolve().parents[2] / "SECURITY.md").read_text(encoding="utf-8")
    sentence = re.search(r"one thirteen-entry list protects (.*?)\. ", " ".join(text.split()))

    assert check_protected_entries_doc(text) == []
    assert sentence is not None and "`skills/**`" in sentence.group(1)


def _cfg(add):
    return SimpleNamespace(tools=SimpleNamespace(add=add, deny=[]))


def test_config_children_never_hold_skill() -> None:
    assert _config_child_allowed(_cfg(["read", "skill", "task", "agent"])) == ["read"]
    assert _config_child_allowed(_cfg(["read", "mcp:skill", "plugin:example.skill"])) == ["read"]
    assert _config_child_allowed(_cfg(["read", "skills", "skill-creator"])) == ["read", "skills", "skill-creator"]
    assert _config_child_allowed(_cfg([])) == list(CONFIG_CHILD_DEFAULT_TOOLS)


@pytest.fixture
async def root_registry(tmp_path, monkeypatch) -> ToolRegistry:
    monkeypatch.setattr(skills, "PACKAGE_DIR", tmp_path / "pkg")
    _skill(tmp_path / "cfg" / "skills", "alpha")
    skill_set = SkillSet.discover(tmp_path / "cfg", None)
    base = await _production_registry()
    await base.register(SkillTool(skill_set, ContentStore(), TaskState(None, workspace=str(tmp_path)), result_cap=lambda: 50_000), scope="global")
    assert "skill" in [s.name for s in base.global_schemas()]
    return base


def _yaml_child(add) -> AgentConfig:
    return AgentConfig(name="writer", role="r", tools=ToolConfig(add=add))


@pytest.mark.parametrize("add", [["read", "skill"], ["read", "mcp:skill"], ["read", "plugin:example.skill"]])
async def test_no_config_child_registry_holds_the_skill_tool(root_registry, add) -> None:
    child = ToolRegistry.from_allowed(_config_child_allowed(_yaml_child(add)), base_registry=root_registry)

    assert child.has("read")
    assert not child.has("skill")


async def test_the_grant_target_toolset_omits_skill(root_registry) -> None:
    got = subagent._resolve_target_toolset("writer", lambda n: _yaml_child(["read", "skill"]), root_registry)

    assert got == ["read"]


@pytest.mark.parametrize("name", [*subagent._STATIC_TOOLSETS, *subagent._WEB_ROLES])
async def test_no_built_in_role_holds_the_skill_tool(root_registry, name) -> None:
    assert "skill" not in subagent.builtin_toolset(name, root_registry), name


DESCRIPTION = (
    "Creates or improves a LocalHarness skill, a reusable procedure in a SKILL.md. Use when the person says "
    "make a skill, turn this into a skill, new skill, save this as a procedure or improve this skill."
)


def test_the_bundled_creator_is_a_valid_bundled_skill(tmp_path) -> None:
    found = SkillSet.discover(tmp_path / "cfg", None)

    assert [(s.name, s.origin, s.untrusted) for s in found.skills] == [("skill-creator", "bundled", False)]
    assert found.skills[0].description == DESCRIPTION and len(DESCRIPTION) == 199
    assert found.skipped == ()


def test_a_machine_skill_cannot_take_the_bundled_name(tmp_path) -> None:
    _skill(tmp_path / "cfg" / "skills", "skill-creator")

    found = SkillSet.discover(tmp_path / "cfg", None)

    assert [(s.name, s.origin) for s in found.skills] == [("skill-creator", "bundled")]
    assert [s.fix for s in found.skipped] == [BUNDLED_NAME[1]]


def test_the_verb_lists_the_creator_as_valid(fake_home, monkeypatch) -> None:
    home = fake_home()
    (home / "proj").mkdir()
    monkeypatch.chdir(home / "proj")

    result = CliRunner().invoke(app, ["skills"])

    assert result.exit_code == 0
    assert result.output.splitlines()[0].startswith("skill-creator (bundled, ")
    assert "Skipped:" not in result.output


async def test_a_real_start_lists_the_creator_and_registers_the_tool(tmp_path, monkeypatch) -> None:
    from localharness.cli.start_cmd import _start_async

    _, agents = _start_world(tmp_path, monkeypatch, skill=False)
    await _start_async(None, False, False, str(tmp_path))
    (loop,) = agents

    assert f"- skill-creator: {DESCRIPTION}" in loop._skills
    assert loop._tools.schema_of("skill").parameters["properties"]["name"]["enum"] == ["skill-creator"]


BUNDLED = resources.files("localharness").joinpath("assets", "skills", "skill-creator")
CREATOR = BUNDLED.joinpath("SKILL.md").read_text(encoding="utf-8")
FORMAT = BUNDLED.joinpath("references", "format.md").read_text(encoding="utf-8")
RULE_LINES = [skip_line(Skipped(reason, fix, Path())) for reason, fix in skills.RULES]


def quoted_rules(text: str) -> list[str]:
    return [m.group(1) for line in text.splitlines() if (m := re.fullmatch(r"- `(skipped: [^`]+)`", line))]


def test_format_md_quotes_the_skip_lines_of_the_data() -> None:
    assert quoted_rules(FORMAT) == RULE_LINES


def test_the_quoted_rules_comparison_bites() -> None:
    lines = FORMAT.splitlines()
    first = next(i for i, line in enumerate(lines) if line.startswith("- `skipped: "))
    dropped = [*lines[:first], *lines[first + 1:]]
    edited = [line.replace("rename yours to keep both", "rename it") for line in lines]
    swapped = [*lines[:first], lines[first + 1], lines[first], *lines[first + 2:]]

    assert len(RULE_LINES) == 22
    for mutated in (dropped, edited, swapped):
        assert quoted_rules("\n".join(mutated)) != RULE_LINES


def test_the_worked_example_is_a_valid_skill(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(skills, "PACKAGE_DIR", tmp_path / "pkg")
    example = re.search(r"^```[^\n]*\n(---\n.*?\n)```", FORMAT, re.S | re.M).group(1)
    _skill(tmp_path / "cfg" / "skills", "release-notes", example)

    found = SkillSet.discover(tmp_path / "cfg", None)

    assert [s.name for s in found.skills] == ["release-notes"] and found.skipped == ()
    assert len(found.skills[0].description) <= 250
    assert len(example.split("\n---\n", 1)[1].strip().splitlines()) == 15


def test_format_md_is_short_and_names_what_a_skill_holds() -> None:
    needles = ("name", "description", "allowed-tools", "references/", "scripts/", "assets/")
    places = ("~/.localharness/skills/", ".localharness/skills/", ".agents/skills/")

    assert len(FORMAT.splitlines()) <= 80
    assert all(f"`{needle}`" in FORMAT for needle in needles)
    assert all(f"`{place}" in FORMAT for place in places) and "250" in FORMAT
