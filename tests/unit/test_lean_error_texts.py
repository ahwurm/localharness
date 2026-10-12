"""The three error texts state the fact and the one sanctioned next step, pinned verbatim.

* ``bash_exec`` timeout: the retry sentence appears (on ``error``, which is what the model
  reads for a failed call, and on ``output``) only when a longer timeout is possible.
* ``agent`` not-wired refusal: two lines, the closest roster names then the searched directories.
* ingest-via-exec refusal: one line that still blocks.
"""
from __future__ import annotations

import pytest

from localharness.tools.builtin.bash_tool import (
    BASH_TIMEOUT_MAX_S,
    BashExecTool,
    _timeout_result,
)

_RETRY = "retry with timeout_s up to 300 if the command needs longer."


# --- bash_exec timeout ----------------------------------------------------

def test_timeout_below_the_maximum_carries_the_retry_sentence_on_both_fields():
    result = _timeout_result(10, "sleep 99")
    assert result.output == f"Command timed out after 10s: sleep 99\n{_RETRY}"
    assert result.error == f"Timeout after 10s\n{_RETRY}"
    assert result.success is False
    assert result.error_type == "timeout_error"
    assert result.output.count(_RETRY) == 1
    assert result.error.count(_RETRY) == 1
    assert result.error.count("\n") == 1


def test_timeout_at_the_maximum_carries_the_fact_alone():
    result = _timeout_result(300.0, "x")
    assert result.error == "Timeout after 300.0s"
    assert result.output == "Command timed out after 300.0s: x"
    assert result.error_type == "timeout_error"


def test_timeout_s_500_clamps_to_the_maximum_and_gets_the_fact_alone():
    assert _timeout_result(min(500, BASH_TIMEOUT_MAX_S), "x") == _timeout_result(300.0, "x")


def test_the_schema_maximum_is_the_one_constant():
    props = BashExecTool().info().parameters["properties"]
    assert props["timeout_s"]["maximum"] == BASH_TIMEOUT_MAX_S == 300.0


async def test_a_real_timeout_tells_the_model_it_may_retry_longer():
    result = await BashExecTool().run(command="sleep 5", timeout_s=1)
    assert result.success is False
    assert result.error == f"Timeout after 1s\n{_RETRY}"
    assert result.output.endswith(_RETRY)


# --- agent not-wired refusal ----------------------------------------------

def test_not_wired_near_miss_leads_with_the_closest_name():
    from localharness.agent.subagent import _not_wired_text

    assert _not_wired_text("explorer", "explorer", []) == (
        "explore: the closest available to 'explorer', which is not wired.\nsearched: none"
    )


def test_not_wired_without_a_near_miss_lists_the_roster():
    from localharness.agent.subagent import _not_wired_text

    assert _not_wired_text("zzzz", "zzzz", []) == (
        "cruncher, explore, search-verifier, web-researcher: the closest available to 'zzzz', "
        "which is not wired.\nsearched: none"
    )


def test_not_wired_names_the_searched_directories_and_their_agents(tmp_path):
    from localharness.agent.subagent import _not_wired_text

    ws, gl = tmp_path / "ws" / "agents", tmp_path / "global" / "agents"
    ws.mkdir(parents=True)
    gl.mkdir(parents=True)
    (gl / "data-analyst.yaml").write_text("name: data-analyst\n")
    (gl / "orchestrator.yaml").write_text("name: orchestrator\n")
    text = _not_wired_text(
        "data-analyts", "data-analyts",
        [str(ws / "data-analyts.yaml"), str(gl / "data-analyts.yaml")],
    )
    line1, line2 = text.split("\n")
    assert line1 == "data-analyst: the closest available to 'data-analyts', which is not wired."
    assert line2 == f"searched: {str(ws)!r}, {str(gl)!r}"
    assert "orchestrator" not in line1


def test_not_wired_never_offers_the_requested_name_as_its_own_match(tmp_path):
    from localharness.agent.subagent import _not_wired_text

    d = tmp_path / "agents"
    d.mkdir()
    (d / "broken.yaml").write_text("not: [valid\n")
    line1 = _not_wired_text("broken", "broken", [str(d / "broken.yaml")]).split("\n")[0]
    assert not line1.startswith("broken")
    assert "broken," not in line1


def _runner(load_agent=None):
    import localharness.agent.subagent as subagent

    return subagent.make_explore_agent_runner(
        llm=object(), bus=object(), base_registry=object(),
        permission_evaluator=object(), get_parent_session_id=lambda: "sid",
        **({} if load_agent is None else {"load_agent": load_agent}),
    )


async def test_runner_refusal_is_the_two_lines_with_the_loaders_directories(tmp_path):
    from localharness.config.loader import ConfigNotFoundError

    ws, gl = tmp_path / "ws" / "agents", tmp_path / "global" / "agents"
    ws.mkdir(parents=True)
    gl.mkdir(parents=True)

    def _load(name):
        raise ConfigNotFoundError(name, [str(ws / f"{name}.yaml"), str(gl / f"{name}.yaml")])

    with pytest.raises(ValueError, match="not wired") as ei:
        await _runner(_load)("explorer", "x")
    assert str(ei.value) == (
        "explore: the closest available to 'explorer', which is not wired.\n"
        f"searched: {str(ws)!r}, {str(gl)!r}"
    )


async def test_runner_refusal_without_a_search_says_none():
    def _load(name):
        raise FileNotFoundError(name)  # no searched_paths

    for runner, who in ((_runner(_load), "orchestrator"), (_runner(), "explorer"),
                        (_runner(_load), "explorer")):
        with pytest.raises(ValueError, match="not wired") as ei:
            await runner(who, "x")
        assert str(ei.value).endswith("\nsearched: none")
        assert str(ei.value).count("\n") == 1


# --- ingest-via-exec refusal ----------------------------------------------

def test_ingest_refusal_is_one_line_naming_the_sanctioned_route():
    from localharness.tools.capabilities import IngestViaExecError, assert_no_ingest_via_exec

    schema = BashExecTool().info()
    with pytest.raises(IngestViaExecError) as ei:
        assert_no_ingest_via_exec(
            schema, {"command": "curl http://x"}, agent_id="root", has_ingest=False
        )
    assert str(ei.value) == (
        "BLOCKED: agent 'root' holds no web-ingest tool and may not reach the web through "
        "`bash_exec` (matched http client: 'curl'); delegate the read to web-researcher, and if "
        "that fails, do not work around it; report the failure."
    )


def test_not_wired_suggestion_never_names_an_absent_agent(tmp_path):
    from localharness.agent.subagent import _not_wired_text

    d = tmp_path / "agents"
    d.mkdir()
    (d / "real-one.yaml").write_text("name: real-one\n")
    line1 = _not_wired_text("real-oen", "real-oen", [str(d / "real-oen.yaml")]).split("\n")[0]
    named = line1.split(":")[0].split(", ")
    present = {"cruncher", "explore", "search-verifier", "web-researcher", "real-one"}
    assert named and set(named) <= present
