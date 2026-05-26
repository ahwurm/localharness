"""Tests for AgentCreationWorkflow state machine."""
import pytest
import yaml
from pathlib import Path
from localharness.orchestrator.workflow import AgentCreationWorkflow, WorkflowState


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
    wf = AgentCreationWorkflow(config_dir=tmp_path)
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


def test_deploy_config_rejects_invalid_yaml(tmp_path: Path):
    """deploy_config must raise ValueError for unparseable YAML."""
    wf = AgentCreationWorkflow(config_dir=tmp_path)
    wf.set_generated_yaml("this is: [[[not: valid")
    with pytest.raises(ValueError):
        wf.deploy_config("test-agent")
    assert not (tmp_path / "agents" / "test-agent.yaml").exists()


def test_deploy_config_rejects_bad_schema(tmp_path: Path):
    """deploy_config must raise for YAML that fails AgentConfig validation."""
    wf = AgentCreationWorkflow(config_dir=tmp_path)
    wf.set_generated_yaml("name: INVALID_UPPER\nrole: test")
    with pytest.raises((ValueError, Exception)):
        wf.deploy_config("test-agent")
    assert not (tmp_path / "agents" / "test-agent.yaml").exists()


def test_deploy_config_overrides_name(tmp_path: Path):
    """deploy_config must override the name field to match the agent_name parameter."""
    wf = AgentCreationWorkflow(config_dir=tmp_path)
    wf.set_generated_yaml("name: llm-chose-this\nrole: A helpful agent")
    path = wf.deploy_config("my-agent")
    data = yaml.safe_load(path.read_text())
    assert data["name"] == "my-agent"
