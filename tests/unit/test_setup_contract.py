"""The contract a plugin's setup step is built on: three optional manifest fields (`next_steps`,
`agent_prompt`, `setup_action`), one optional method (`Plugin.setup_action(ctx)`), and the pure
helpers in plugins/setup.py that shape the "paste this into your coding agent" prompt.

What this pins is the CONTRACT, not a flow — nothing here asks, writes or prints. The new fields
default to empty and leave PLUGIN_API_VERSION at "1"; `agent_prompt` may name only {config_dir},
{machine} and the plugin's own setup keys that are not secret and have a default, and anything else
is refused when the manifest is built, so a token can never reach a printed prompt; the rendered
prompt leads with the fixed line, fills the values, and leaves neither placeholder text nor a double
space where a value is empty.
"""
from __future__ import annotations

import pytest

from localharness.plugins import api
from localharness.plugins.api import Check, Plugin, PluginManifest, SetupField
from localharness.plugins.setup import (
    AGENT_PROMPT_LEAD,
    has_setup_action,
    machine_sentence,
    render_agent_prompt,
)

_URL = SetupField(key="url", prompt="Address", default="http://127.0.0.1:1")


def _manifest(**overrides) -> PluginManifest:
    return PluginManifest(**{"name": "x", "version": "1", "kind": "tools", **overrides})


# --- the manifest fields and the closed placeholder set ---------------------------------------


def test_new_fields_default_empty_and_the_api_version_stays_1():
    m = PluginManifest(name="x", version="1", kind="tools")
    assert m.next_steps == m.agent_prompt == m.setup_action == ""
    assert api.PLUGIN_API_VERSION == "1"


def test_agent_prompt_may_name_config_dir_machine_and_a_defaulted_key():
    text = "Run it at {url} under {config_dir}. {machine}"
    assert _manifest(setup=(_URL,), agent_prompt=text).agent_prompt == text


def test_a_dotted_setup_key_is_a_placeholder():
    """A dotted key (autoresearch's `proposer.base_url`) is a name, not plain text: it builds when it
    is a setup key, is refused when it is not, and renders filled."""
    field = SetupField(key="proposer.base_url", prompt="P", default="http://p/v1")
    name = "{proposer.base_url}"
    assert _manifest(setup=(field,), agent_prompt=name).agent_prompt == name
    with pytest.raises(ValueError, match="proposer.base_url"):
        _manifest(agent_prompt=name)
    assert render_agent_prompt(name, {"proposer.base_url": "http://p/v1"}).endswith("\n\n  http://p/v1")


def test_agent_prompt_refuses_a_secret_key():
    tok = SetupField(key="tok", prompt="Token", secret=True, default="x")
    with pytest.raises(ValueError, match="secret"):
        _manifest(setup=(tok,), agent_prompt="Use {tok}.")


def test_agent_prompt_refuses_an_unknown_name():
    with pytest.raises(ValueError, match="nope"):
        _manifest(setup=(_URL,), agent_prompt="{url} and {nope}")


def test_agent_prompt_refuses_a_key_with_no_default():
    with pytest.raises(ValueError, match="no default"):
        _manifest(setup=(SetupField(key="url", prompt="Address"),), agent_prompt="{url}")


# --- rendering ---------------------------------------------------------------------------------


def test_render_leads_with_the_fixed_line_and_fills_values():
    values = {"url": "http://a", "config_dir": "/c"}
    got = render_agent_prompt("Run at {url} in {config_dir}.", values)
    assert got == AGENT_PROMPT_LEAD + "\n\n  Run at http://a in /c."
    assert render_agent_prompt("one\n\ntwo", {}) == AGENT_PROMPT_LEAD + "\n\n  one\n\n  two"


def test_an_unknown_machine_renders_as_nothing_without_a_double_space():
    for values in ({"machine": ""}, {}):
        got = render_agent_prompt("pictures). {machine} On a GB10", values)
        assert got.endswith("  pictures). On a GB10") and "{machine}" not in got, got


def test_an_empty_template_renders_nothing():
    assert render_agent_prompt("", {"config_dir": "/c", "machine": "This machine reports X."}) == ""


def test_braces_that_are_not_names_stay():
    text = "Keep {Upper}, { } and {} as typed."
    assert render_agent_prompt(text, {"Upper": "no"}) == AGENT_PROMPT_LEAD + "\n\n  " + text
    assert _manifest(agent_prompt=text).agent_prompt == text  # plain text, never a refused name


def test_machine_sentence():
    assert machine_sentence("NVIDIA GB10") == "This machine reports NVIDIA GB10."
    assert machine_sentence(None) == "" and machine_sentence("") == ""


# --- the method ----------------------------------------------------------------------------------


def test_has_setup_action():
    class Doer(Plugin):
        manifest = _manifest(name="doer")

        def setup_action(self, ctx):
            return [Check(name="doer", status="pass")]

    class AsyncDoer(Plugin):
        manifest = _manifest(name="async-doer")

        async def setup_action(self, ctx):
            return []

    class Idle(Plugin):
        manifest = _manifest(name="idle")

    assert has_setup_action(Doer) is True and has_setup_action(AsyncDoer) is True
    assert has_setup_action(Idle) is False
