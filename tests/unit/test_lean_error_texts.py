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
        available_agents=["data-analyst", "orchestrator"],
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
    line1 = _not_wired_text(
        "real-oen", "real-oen", [str(d / "real-oen.yaml")], available_agents=["real-one"]
    ).split("\n")[0]
    named = line1.split(":")[0].split(", ")
    present = {"cruncher", "explore", "search-verifier", "web-researcher", "real-one"}
    assert named and set(named) <= present


def test_not_wired_roster_is_the_registry_not_a_glob_of_the_directories(tmp_path):
    """A yaml the loader will not load (unparseable, a symlink out of the project, a name the
    dispatcher renames) sits in the searched directory but is not in the live roster."""
    from localharness.agent.subagent import _not_wired_text

    d = tmp_path / "agents"
    d.mkdir()
    for stem in ("broken", "linked", "my_agent", "real-one"):
        (d / f"{stem}.yaml").write_text("name: x\n")
    text = _not_wired_text("real-oen", "real-oen", [str(d / "real-oen.yaml")],
                           available_agents=["real-one"])
    line1 = text.split("\n")[0]
    assert line1.startswith("real-one: the closest available to 'real-oen'")
    for ghost in ("broken", "linked", "my_agent"):
        assert ghost not in line1
    # No roster handed in: the built-ins alone, whatever files sit in the directory.
    bare = _not_wired_text("zzzz", "zzzz", [str(d / "zzzz.yaml")]).split("\n")[0]
    assert bare.startswith("cruncher, explore, search-verifier, web-researcher: ")


def test_not_wired_fallback_roster_is_bounded():
    from localharness.agent.subagent import _not_wired_text

    many = [f"agent-{chr(97 + i)}{chr(97 + i)}{chr(97 + i)}" for i in range(20)]
    line1 = _not_wired_text("zzzz", "zzzz", [], available_agents=many).split("\n")[0]
    named = line1.split(": the closest")[0].split(", ")
    assert len(named) == 8
    assert named == sorted({"explore", "web-researcher", "search-verifier", "cruncher", *many})[:8]


async def test_runner_roster_follows_the_live_available_agents_list():
    from localharness.config.loader import ConfigNotFoundError

    def _load(name):
        raise ConfigNotFoundError(name, [])

    live = ["real-one"]
    runner = _runner_with(_load, live)
    with pytest.raises(ValueError, match="not wired") as ei:
        await runner("real-oen", "x")
    assert str(ei.value).startswith("real-one: the closest available to 'real-oen'")
    live.append("real-owl")  # a deploy later in the session reaches the next refusal
    with pytest.raises(ValueError, match="not wired") as ei:
        await runner("real-owx", "x")
    assert "real-owl" in str(ei.value).split("\n")[0]


async def test_runner_surfaces_a_load_error_that_is_not_a_not_found():
    from localharness.config.loader import ConfigParseError, ConfigValidationError

    def _parse(name):
        raise ConfigParseError("/w/agents/broken.yaml", 3, 1, "mapping values are not allowed here")

    with pytest.raises(ValueError) as ei:
        await _runner_with(_parse, ["broken"])("broken", "x")
    text = str(ei.value)
    assert "not wired" not in text and "searched" not in text
    assert text == (
        "agent 'broken' could not be loaded: "
        "/w/agents/broken.yaml:3:1: mapping values are not allowed here"
    )

    def _invalid(name):
        raise ConfigValidationError("/w/agents/bad.yaml", [])

    with pytest.raises(ValueError) as ei:
        await _runner_with(_invalid, [])("bad", "x")
    assert str(ei.value) == "agent 'bad' could not be loaded: /w/agents/bad.yaml:"

    def _boom(name):
        raise RuntimeError("first line\nsecond line")

    with pytest.raises(ValueError) as ei:
        await _runner_with(_boom, [])("bad", "x")
    assert str(ei.value) == "agent 'bad' could not be loaded: first line"


async def test_runner_load_failure_text_does_not_echo_the_rejected_value():
    """The agent file is untrusted: the model sees the file, the field and a short reason, never
    the value the file held (here a secret-looking one, and one the reason text itself repeats)."""
    from localharness.config.loader import (
        ConfigFieldError, ConfigReferenceError, ConfigValidationError)

    secret = "sk-live-SECRET-0123456789abcdef"

    def _invalid(name):
        raise ConfigValidationError("/w/agents/bad.yaml", [
            ConfigFieldError("model.api_key", secret, f"Value error, bad key {secret}"),
            ConfigFieldError("model.name", 7, "Input should be a string"),
        ])

    with pytest.raises(ValueError) as ei:
        await _runner_with(_invalid, [])("bad", "x")
    text = str(ei.value)
    assert secret not in text and "SECRET" not in text and "got:" not in text
    assert text == ("agent 'bad' could not be loaded: "
                    "/w/agents/bad.yaml: invalid model: value rejected by validation (and 1 more)")

    def _dangling(name):
        raise ConfigReferenceError("/w/agents/bad.yaml", "division", secret, "no such division")

    with pytest.raises(ValueError) as ei:
        await _runner_with(_dangling, [])("bad", "x")
    text = str(ei.value)
    assert secret not in text
    assert text == ("agent 'bad' could not be loaded: "
                    "/w/agents/bad.yaml: field 'division' references something that does not exist")


_MARKER = "sk-live-TOPSECRET99"


def _real_refusal_text(tmp_path, agent: dict, sub: str = "") -> str:
    """The text the model gets for an agent file, written to disk and loaded by the real loader."""
    import yaml

    from localharness.agent.subagent import _load_failure_text
    from localharness.config.loader import ConfigError, ConfigLoader

    root = tmp_path / sub if sub else tmp_path
    (root / "agents").mkdir(parents=True)
    (root / "agents" / "bad.yaml").write_text(yaml.safe_dump(agent), encoding="utf-8")
    with pytest.raises(ConfigError) as ei:
        ConfigLoader(config_dir=root).load_agent("bad")
    text = _load_failure_text("bad", ei.value)
    assert text is not None and text.startswith("agent 'bad' could not be loaded: ")
    assert str(root / "agents" / "bad.yaml") in text  # the file is named
    return text


def _assert_no_file_text(text: str) -> None:
    for leaked in (_MARKER, "TOPSECRET", "sk-live", "got:", "Value error", "\\", "\n"):
        assert leaked not in text, (leaked, text)


def test_list_value_with_a_secret_looking_string_is_not_echoed(tmp_path):
    text = _real_refusal_text(tmp_path, {
        "name": "bad", "role": "Work",
        "permissions": {"deny_patterns": [f"Bad PATTERN {_MARKER}"]}})
    _assert_no_file_text(text)
    assert "Bad PATTERN" not in text
    assert "invalid permissions.deny_patterns: value rejected by validation" in text


def test_name_with_newline_quote_backslash_and_secret_is_not_echoed(tmp_path):
    text = _real_refusal_text(tmp_path, {
        "name": f"Bad\n'\"\\{_MARKER}", "role": "Work"})
    _assert_no_file_text(text)
    assert "invalid name: value rejected by validation" in text


def test_timezone_with_quote_and_secret_is_not_echoed(tmp_path):
    text = _real_refusal_text(tmp_path, {
        "name": "bad", "role": "Work",
        "schedule": {"cron": "* * * * *", "timezone": f"No'Such\\{_MARKER}"}})
    _assert_no_file_text(text)
    assert "invalid schedule.timezone: value rejected by validation" in text


def test_unknown_key_named_like_a_secret_is_not_echoed(tmp_path):
    text = _real_refusal_text(tmp_path, {"name": "bad", "role": "Work", _MARKER: "x"})
    _assert_no_file_text(text)
    assert "invalid setting: not a recognized setting" in text

    text = _real_refusal_text(tmp_path, {
        "name": "bad", "role": "Work", "permissions": {_MARKER: True}}, sub="nested")
    _assert_no_file_text(text)
    assert "invalid permissions: not a recognized setting" in text


def test_nested_mapping_value_is_not_echoed(tmp_path):
    text = _real_refusal_text(tmp_path, {
        "name": "bad", "role": "Work",
        "permissions": {"mode": {"k": _MARKER, "nested": {"deep": _MARKER}}}})
    _assert_no_file_text(text)
    assert "invalid permissions.mode:" in text

    text = _real_refusal_text(tmp_path, {
        "name": "bad", "role": "Work", "max_tokens": {"x": {"y": _MARKER}}}, sub="second")
    _assert_no_file_text(text)
    assert "invalid max_tokens: " in text


def test_a_field_the_schema_does_not_declare_is_never_named():
    from localharness.agent.subagent import _declared_field_path

    assert _declared_field_path("permissions.deny_patterns.0") == "permissions.deny_patterns.0"
    assert _declared_field_path("permissions.deny_patterns.x") == "permissions.deny_patterns"
    assert _declared_field_path(f"permissions.{_MARKER}.mode") == "permissions"
    assert _declared_field_path(_MARKER) == ""
    assert _declared_field_path("") == ""


def test_several_errors_still_report_the_count_and_name_only_the_first_declared_field():
    from localharness.agent.subagent import _load_failure_text
    from localharness.config.loader import ConfigFieldError, ConfigValidationError

    exc = ConfigValidationError("/w/agents/bad.yaml", [
        ConfigFieldError(_MARKER, _MARKER, f"Extra {_MARKER}", kind="extra_forbidden"),
        ConfigFieldError("name", _MARKER, f"Value error, {_MARKER!r}", kind="value_error"),
        ConfigFieldError("role", None, "x", kind="missing"),
    ])
    assert _load_failure_text("bad", exc) == (
        "agent 'bad' could not be loaded: "
        "/w/agents/bad.yaml: invalid setting: not a recognized setting (and 2 more)")


def test_a_reference_error_names_only_the_declared_field():
    from localharness.agent.subagent import _load_failure_text
    from localharness.config.loader import ConfigReferenceError

    text = _load_failure_text("bad", ConfigReferenceError("/w/agents/bad.yaml", _MARKER, _MARKER, "gone"))
    assert _MARKER not in text
    assert text == ("agent 'bad' could not be loaded: "
                    "/w/agents/bad.yaml: field 'setting' references something that does not exist")


def _runner_with(load_agent, available_agents):
    import localharness.agent.subagent as subagent

    return subagent.make_explore_agent_runner(
        llm=object(), bus=object(), base_registry=object(),
        permission_evaluator=object(), get_parent_session_id=lambda: "sid",
        load_agent=load_agent, available_agents=available_agents,
    )
