"""The plugins-off test switch (Phase 50 Wave 1): `LOCALHARNESS_TEST_PLUGINS_OFF` decides the config
both home fixtures seed and which `plugin(name)`-marked tests skip. Pure: loads no config.

Only the mechanics are proven here. A real OFF-state suite run needs the plugin registered (the
config loader rejects `autoresearch:` until then), so the first one is 50-06.
"""
from __future__ import annotations

import pytest

from tests.conftest import _minimal_config, plugin_off_reason

TODAY = ("version: '1'\n"
         "provider:\n"
         "  provider_type: vllm\n"
         "  base_url: http://localhost:8000/v1\n"
         "  default_model: test-model\n")


class _Node:
    def __init__(self, *marks):
        self._marks = [m.mark for m in marks]

    def iter_markers(self, name):
        return (m for m in self._marks if m.name == name)


def test_switch_unset_seeds_todays_config(monkeypatch):
    monkeypatch.delenv("LOCALHARNESS_TEST_PLUGINS_OFF", raising=False)
    assert _minimal_config() == TODAY


def test_switch_appends_an_off_section(monkeypatch):
    monkeypatch.setenv("LOCALHARNESS_TEST_PLUGINS_OFF", "autoresearch")
    assert _minimal_config() == TODAY + "autoresearch:\n  enabled: false\n"


def test_switch_takes_a_comma_separated_list(monkeypatch):
    monkeypatch.setenv("LOCALHARNESS_TEST_PLUGINS_OFF", "web, memory")
    assert _minimal_config() == TODAY + "web:\n  enabled: false\nmemory:\n  enabled: false\n"


@pytest.fixture
def _switch_on(monkeypatch):
    monkeypatch.setenv("LOCALHARNESS_TEST_PLUGINS_OFF", "autoresearch")


def test_the_autouse_home_seeds_the_minimal_config(_isolate_localharness_home):
    assert (_isolate_localharness_home / "config.yaml").read_text(encoding="utf-8") == _minimal_config()


def test_components_home_seeds_the_switch(_switch_on, components_home):
    """The autoresearch tests use `components_home`, not the autouse home: it must read the switch."""
    text = (components_home / "config.yaml").read_text(encoding="utf-8")
    assert text == TODAY + "autoresearch:\n  enabled: false\n"


def test_a_marked_test_skips_only_when_its_plugin_is_off(monkeypatch):
    node = _Node(pytest.mark.plugin("autoresearch"))
    monkeypatch.delenv("LOCALHARNESS_TEST_PLUGINS_OFF", raising=False)
    assert plugin_off_reason(node) is None
    monkeypatch.setenv("LOCALHARNESS_TEST_PLUGINS_OFF", "mobile")
    assert plugin_off_reason(node) is None
    monkeypatch.setenv("LOCALHARNESS_TEST_PLUGINS_OFF", "web,autoresearch")
    assert plugin_off_reason(node) == "plugin autoresearch is off (LOCALHARNESS_TEST_PLUGINS_OFF)"
    assert plugin_off_reason(_Node()) is None
