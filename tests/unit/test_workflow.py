"""Tests for AgentCreationWorkflow state machine."""
import pytest
import yaml
from pathlib import Path
from localharness.config.loader import ConfigLoader
from localharness.orchestrator.workflow import (
    AgentCreationWorkflow,
    WorkflowState,
    is_cancellation,
    validate_agent_yaml,
)


# #57: the confirm step must never show YAML that will explode at deploy. validate_agent_yaml
# is the pre-confirm gate: None == deployable, else a SHORT human error (no pydantic.dev wall).
def test_validate_agent_yaml_accepts_valid():
    assert validate_agent_yaml("name: good\nrole: monitor and summarize the news") is None


def test_validate_agent_yaml_rejects_bad_enum_cleanly():
    """The live receipt: permissions.mode: read_only (legal: auto|manual) surfaced a raw
    Pydantic wall with a pydantic.dev URL. The error must be short and URL-free."""
    err = validate_agent_yaml("name: good\nrole: does things\npermissions:\n  mode: read_only")
    assert err is not None
    assert "mode" in err  # names the offending field
    assert "pydantic.dev" not in err  # no raw ValidationError URL
    assert "https://" not in err


def test_validate_agent_yaml_rejects_non_mapping_and_missing_name():
    assert validate_agent_yaml("just a string") is not None
    assert validate_agent_yaml("role: no name here") is not None


# #59: deterministic, leading-anchored cancellation. A cancel word buried after real
# content NEVER fires (the subscription example); only input that is WHOLLY a cancel
# phrase (optionally after one 'actually'/'no' prefix, with trailing fillers) cancels.
@pytest.mark.parametrize("text", [
    "cancel",
    "Cancel.",
    "never mind",
    "nevermind",
    "forget it",
    "stop",
    "quit",
    "exit",
    "abort",
    "actually, never mind",
    "actually never mind, forget it",  # the live receipt that became the DESCRIPTION
    "no, forget it",
    "cancel that",          # trailing filler after the phrase
    "  STOP  ",
])
def test_is_cancellation_true(text):
    assert is_cancellation(text) is True


@pytest.mark.parametrize("text", [
    "an agent that helps me cancel subscriptions",  # 'cancel' is DEEPER, not leading
    "stop-loss trading monitor",                    # leads with 'stop' but has real content
    "cancel subscriptions for me automatically",    # leading 'cancel' + real content
    "build me a research agent",
    "it should summarize the news",
    "no",           # bare 'no' at confirm means 'change', not cancel
    "",
    "   ",
    "quitting-time reminder bot",                   # 'quit' is a substring, not a word
])
def test_is_cancellation_false(text):
    assert is_cancellation(text) is False


def test_is_cancellation_is_pure_no_side_effects():
    """No model judgment, no state — same input, same answer, repeatable."""
    for _ in range(3):
        assert is_cancellation("forget it") is True
        assert is_cancellation("an agent to cancel subscriptions") is False


def test_workflow_starts_in_discuss():
    wf = AgentCreationWorkflow()
    assert wf.state == WorkflowState.DISCUSS

def test_workflow_discuss_to_configure():
    wf = AgentCreationWorkflow()
    wf.transition("I want an agent that monitors hacker news for AI articles and summarizes them")
    assert wf.state == WorkflowState.CONFIGURE  # description > 10 chars

def test_workflow_short_description_stays_in_discuss():
    wf = AgentCreationWorkflow()
    wf.transition("agent")
    assert wf.state == WorkflowState.DISCUSS  # too short

def test_workflow_configure_to_confirm():
    wf = AgentCreationWorkflow()
    wf.transition("Build a research agent that searches the web")
    assert wf.state == WorkflowState.CONFIGURE
    wf.transition("")  # configure step processes
    assert wf.state == WorkflowState.CONFIRM

def test_workflow_confirm_yes_to_deploy():
    wf = AgentCreationWorkflow()
    wf.transition("Build a web scraper agent for news monitoring")
    wf.transition("")  # -> confirm
    wf.transition("yes")
    assert wf.state == WorkflowState.DEPLOY

def test_workflow_confirm_no_back_to_discuss():
    wf = AgentCreationWorkflow()
    wf.transition("Build a web scraper agent for news monitoring")
    wf.transition("")  # -> confirm
    wf.transition("change")
    assert wf.state == WorkflowState.DISCUSS

def test_workflow_cancel_from_any_state():
    wf = AgentCreationWorkflow()
    wf.transition("cancel")
    assert wf.state == WorkflowState.CANCELLED

def test_workflow_cancel_from_confirm():
    wf = AgentCreationWorkflow()
    wf.transition("Build a research agent that searches the web")
    wf.transition("")  # -> confirm
    wf.transition("cancel")
    assert wf.state == WorkflowState.CANCELLED

def test_workflow_deploy_config_writes_file(tmp_path: Path):
    wf = AgentCreationWorkflow(loader=ConfigLoader(config_dir=tmp_path))
    wf.set_generated_yaml("name: test-agent\nrole: Test agent role")
    path = wf.deploy_config("test-agent")
    assert path.exists()
    assert path.name == "test-agent.yaml"
    assert "test-agent" in path.read_text()

def test_workflow_aftercare_done_completes():
    wf = AgentCreationWorkflow()
    wf.transition("Build a research agent that does web search for topics")
    wf.transition("")  # -> confirm
    wf.transition("yes")  # -> deploy
    wf.transition("")  # -> aftercare
    wf.transition("done")  # -> complete
    assert wf.state == WorkflowState.COMPLETE

def test_workflow_gathered_stores_description():
    wf = AgentCreationWorkflow()
    wf.transition("I need a fitness tracking agent that logs meals")
    assert "description" in wf.gathered
    assert "fitness" in wf.gathered["description"].lower()


def test_change_followup_updates_generation_input(tmp_path: Path):
    """#56: after 'change' at confirm, the follow-up message must UPDATE the stored
    description so regeneration runs on the CORRECTED input — the old code stored the
    description once (`if 'description' not in gathered`) so 'change' regenerated an
    IDENTICAL config. Choice: append-with-newline (original intent + correction both
    reach the generator)."""
    wf = AgentCreationWorkflow(loader=ConfigLoader(config_dir=tmp_path))
    wf.transition("Track stock prices and summarize the daily moves")  # -> CONFIGURE
    wf.transition("")  # -> CONFIRM
    original_desc = wf.gathered["description"]
    wf.transition("change")  # -> DISCUSS
    assert wf.state == WorkflowState.DISCUSS
    wf.transition("also include crypto and alert on 5% swings")  # the correction
    updated = wf.gathered["description"]
    assert updated != original_desc  # generation input actually changed
    assert "Track stock prices" in updated  # original intent preserved
    assert "crypto" in updated  # correction folded in
    assert wf.state == WorkflowState.CONFIGURE  # advances to regenerate


def test_short_then_long_description_recovers(tmp_path: Path):
    """#56 corollary: a first reply <=10 chars set the description but could never be
    replaced (`if 'description' not in gathered`), wedging DISCUSS forever. A subsequent
    longer description must REPLACE the too-short one and advance."""
    wf = AgentCreationWorkflow(loader=ConfigLoader(config_dir=tmp_path))
    wf.transition("agent")  # 5 chars — too short, stays in DISCUSS
    assert wf.state == WorkflowState.DISCUSS
    wf.transition("monitor hacker news for AI stories and summarize them")
    assert wf.state == WorkflowState.CONFIGURE  # wedge dead — it advanced
    # The too-short stub did NOT contaminate the real description.
    assert wf.gathered["description"] == "monitor hacker news for AI stories and summarize them"


def test_deploy_config_rejects_invalid_yaml(tmp_path: Path):
    """deploy_config must raise ValueError for unparseable YAML."""
    wf = AgentCreationWorkflow(loader=ConfigLoader(config_dir=tmp_path))
    wf.set_generated_yaml("this is: [[[not: valid")
    with pytest.raises(ValueError):
        wf.deploy_config("test-agent")
    assert not (tmp_path / "agents" / "test-agent.yaml").exists()


def test_deploy_config_rejects_bad_schema(tmp_path: Path):
    """deploy_config must raise for YAML that fails AgentConfig validation."""
    wf = AgentCreationWorkflow(loader=ConfigLoader(config_dir=tmp_path))
    # temperature > 2.0 exceeds AgentConfig's le=2.0 constraint
    wf.set_generated_yaml("name: x\nrole: test\ntemperature: 999")
    with pytest.raises((ValueError, Exception)):
        wf.deploy_config("test-agent")
    assert not (tmp_path / "agents" / "test-agent.yaml").exists()


def test_deploy_config_overrides_name(tmp_path: Path):
    """deploy_config must override the name field to match the agent_name parameter."""
    wf = AgentCreationWorkflow(loader=ConfigLoader(config_dir=tmp_path))
    wf.set_generated_yaml("name: llm-chose-this\nrole: A helpful agent")
    path = wf.deploy_config("my-agent")
    data = yaml.safe_load(path.read_text())
    assert data["name"] == "my-agent"


def test_deploy_config_no_name_honors_yaml_name(tmp_path: Path):
    """#19: with no agent_name, deploy what the user confirmed — the YAML's own name."""
    wf = AgentCreationWorkflow(loader=ConfigLoader(config_dir=tmp_path))
    wf.set_generated_yaml("name: llm-chose-this\nrole: A helpful agent")
    path = wf.deploy_config()
    assert path.name == "llm-chose-this.yaml"
    assert yaml.safe_load(path.read_text())["name"] == "llm-chose-this"


def test_deploy_config_no_name_anywhere_fails_explicitly(tmp_path: Path):
    """#28: a generated config with no top-level name must FAIL explicitly, never
    silently become a 'new-agent' placeholder (the project rule: fail loudly, never
    placeholder). Replaces the earlier test that codified the placeholder default."""
    wf = AgentCreationWorkflow(loader=ConfigLoader(config_dir=tmp_path))
    wf.set_generated_yaml("role: A helpful agent")
    with pytest.raises(ValueError, match="name"):
        wf.deploy_config()
    agents = tmp_path / "agents"
    assert not agents.exists() or list(agents.glob("*.yaml")) == []


def test_deploy_config_refuses_to_overwrite_existing(tmp_path: Path):
    """#28: deploy wrote a .tmp then os.replace with NO existence check, silently
    clobbering an existing agent config. It must refuse and name the collision,
    leaving the existing file untouched."""
    wf = AgentCreationWorkflow(loader=ConfigLoader(config_dir=tmp_path))
    wf.set_generated_yaml("name: dup-agent\nrole: The original agent")
    path = wf.deploy_config()
    original = path.read_text()

    wf2 = AgentCreationWorkflow(loader=ConfigLoader(config_dir=tmp_path))
    wf2.set_generated_yaml("name: dup-agent\nrole: A DIFFERENT agent that would clobber")
    with pytest.raises(ValueError, match="exists"):
        wf2.deploy_config()
    assert path.read_text() == original  # original untouched, no silent clobber


# --- config dir resolution (#150 phase 38): chokepoint-routed fallback ---


def test_workflow_config_dir_honors_env_chain(monkeypatch, tmp_path: Path):
    """The bare-construction fallback was Path.home()/".localharness", bypassing the chokepoint
    (research §1). It now follows LOCALHARNESS_DIR > LOCALHARNESS_HOME like everything else."""
    home_env = tmp_path / "home"
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.setenv("LOCALHARNESS_HOME", str(home_env))
    assert AgentCreationWorkflow()._loader.global_config_dir == home_env

    dir_env = tmp_path / "dir"
    monkeypatch.setenv("LOCALHARNESS_DIR", str(dir_env))
    assert AgentCreationWorkflow()._loader.global_config_dir == dir_env


def test_workflow_explicit_config_dir_wins(monkeypatch, tmp_path: Path):
    """The real interactive path (repl.py -> router.py) threads a dir; it still wins."""
    monkeypatch.setenv("LOCALHARNESS_DIR", str(tmp_path / "ignored"))
    explicit = tmp_path / "explicit"
    assert AgentCreationWorkflow(loader=ConfigLoader(config_dir=explicit))._loader.global_config_dir == explicit


def test_workflow_config_dir_unchanged_when_nothing_set(monkeypatch):
    """Zero-behavior-change guarantee: no env, no arg -> ~/.localharness, exactly as before."""
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.delenv("LOCALHARNESS_HOME", raising=False)
    assert AgentCreationWorkflow()._loader.global_config_dir == Path("~/.localharness").expanduser()


def test_the_destination_defaults_to_the_bound_layer(tmp_path: Path):
    m, p = tmp_path / "m", tmp_path / "p"
    assert AgentCreationWorkflow(loader=ConfigLoader(config_dir=m, local_config_dir=p))._layer == "project"
    assert AgentCreationWorkflow(loader=ConfigLoader(config_dir=m))._layer == "machine"
    bare = AgentCreationWorkflow()._loader
    assert isinstance(bare, ConfigLoader) and bare.local_dir is None


def test_deploy_config_writes_the_bound_project_layer_through_the_writer(tmp_path: Path, monkeypatch):
    m, p = tmp_path / "m", tmp_path / "p"
    m.mkdir()
    p.mkdir()
    layers = []
    real = ConfigLoader.write_agent

    def spy(self, name, yaml_text, *, layer, overwrite=False):
        layers.append(layer)
        return real(self, name, yaml_text, layer=layer, overwrite=overwrite)

    monkeypatch.setattr(ConfigLoader, "write_agent", spy)
    loader = ConfigLoader(config_dir=m, local_config_dir=p)
    wf = AgentCreationWorkflow(loader=loader)
    wf.set_generated_yaml("name: x\nrole: r")
    path = wf.deploy_config()
    assert path == p / "agents" / "x.yaml"
    assert path.stat().st_mode & 0o777 == 0o600
    assert layers == ["project"]

    wf2 = AgentCreationWorkflow(loader=loader)
    wf2.set_generated_yaml("name: x\nrole: r2")
    with pytest.raises(ValueError, match="exists") as err:
        wf2.deploy_config()
    assert str(err.value).endswith("remove the existing config.")
    assert str(err.value).count("remove the existing config.") == 1


def _at_confirm(tmp_path: Path, *, bound: bool = True, name: str = "x"):
    m, p = tmp_path / "m", tmp_path / "p"
    m.mkdir(exist_ok=True)
    p.mkdir(exist_ok=True)
    wf = AgentCreationWorkflow(loader=ConfigLoader(config_dir=m, local_config_dir=p if bound else None))
    wf.transition("Build a research agent that searches the web for me")
    wf.set_generated_yaml(f"name: {name}\nrole: r")
    wf.transition("configure_done")
    assert wf.state == WorkflowState.CONFIRM
    return wf, m, p


def test_machine_switches_the_layer_and_project_unbound_is_unknown(tmp_path: Path):
    wf, _, _ = _at_confirm(tmp_path)
    assert wf.transition("machine") == WorkflowState.DEPLOY and wf._layer == "machine"
    wf, _, _ = _at_confirm(tmp_path)
    assert wf.transition("project") == WorkflowState.DEPLOY and wf._layer == "project"
    wf, _, _ = _at_confirm(tmp_path, bound=False)
    assert wf.transition("project") == WorkflowState.CONFIRM


@pytest.mark.parametrize("bound", [True, False])
@pytest.mark.parametrize("token,expected", [
    ("yes", WorkflowState.DEPLOY), ("y", WorkflowState.DEPLOY), ("ok", WorkflowState.DEPLOY),
    ("looks good", WorkflowState.DEPLOY), ("lgtm", WorkflowState.DEPLOY), ("deploy", WorkflowState.DEPLOY),
    ("no", WorkflowState.DISCUSS), ("n", WorkflowState.DISCUSS), ("change", WorkflowState.DISCUSS),
    ("update", WorkflowState.DISCUSS), ("edit", WorkflowState.DISCUSS), ("redo", WorkflowState.DISCUSS),
    ("cancel", WorkflowState.CANCELLED),
])
def test_every_offered_token_and_synonym_is_accepted(tmp_path: Path, bound, token, expected):
    wf, _, _ = _at_confirm(tmp_path, bound=bound)
    assert wf.transition(token) == expected


def test_reply_line_is_built_from_the_offered_tuple(tmp_path: Path):
    assert _at_confirm(tmp_path)[0].reply_line() == "Reply: yes / machine / change / cancel"
    assert _at_confirm(tmp_path, bound=False)[0].reply_line() == "Reply: yes / no / change / cancel"


def test_the_confirm_prompts_are_pinned(tmp_path: Path):
    wf, m, p = _at_confirm(tmp_path)
    assert wf.confirm_prompt() == f"Save to {p / 'agents' / 'x.yaml'}? yes / machine / change / cancel"
    (m / "agents").mkdir()
    (m / "agents" / "x.yaml").write_text("name: x\nrole: r\n")
    assert wf.confirm_prompt() == (f"Save to {p / 'agents' / 'x.yaml'} (shadows {m / 'agents' / 'x.yaml'})? "
                                   "yes / machine / change / cancel")
    wf, m, _ = _at_confirm(tmp_path, bound=False)
    assert wf.confirm_prompt() == (f"Does this look good? It will be saved to {m / 'agents' / 'x.yaml'}. "
                                   "(yes/no/change, or 'cancel')")
    assert "~" not in wf.confirm_prompt()


def test_machine_note_names_the_project_file_it_will_not_override(tmp_path: Path):
    wf, m, p = _at_confirm(tmp_path)
    wf.transition("machine")
    assert wf.machine_note() is None
    (p / "agents").mkdir()
    (p / "agents" / "x.yaml").write_text("name: x\nrole: r\n")
    assert wf.machine_note() == (f"note: {p / 'agents' / 'x.yaml'} shadows {m / 'agents' / 'x.yaml'}; "
                                 "it will not load here")
    wf._layer = "project"
    assert wf.machine_note() is None
