"""The image plugin's roadmap criteria 4 and 5, enforced rather than remembered.

4. The wire moved exactly once (v4, on top of v3's `tool_names`); the retired names — the old
   generated-image id field, the old image route and the old environment-variable prefix — appear
   nowhere in src/ or tests/ (the page and the protocol snapshot included); generate_image declares
   ingest none / host safe / result_origin trusted.
5. Image is merged as a plugin: bundled, off by default, never on the core-import burn-down list,
   and no core module names its tool, its command or ComfyUI.

The retired strings are built from parts so this file does not match itself.
"""
from __future__ import annotations

import ast
from pathlib import Path

from localharness.channels.web import protocol
from localharness.core.events import Observation, TurnCompleted, TurnFailed
from localharness.plugins.builtin import bundled_plugins
from localharness.tools.builtin.image_plugin import ImagePlugin
from localharness.tools.capabilities import ingests_untrusted, is_host_dangerous
from tests.unit.test_image_plugin import _ctx
from tests.unit.test_import_direction import BURN_DOWN, SRC, classify

ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN = ("image" + "_id", "/api/" + "images", "LOCALHARNESS_" + "COMFYUI_")
SUFFIXES = {".py", ".html", ".js", ".json", ".md", ".yaml", ".yml", ".css", ".txt"}
CORE_NEEDLES = ("generate_image", "generate-image", "comfyui")


def test_the_forbidden_strings_appear_nowhere():
    files = [p for d in ("src", "tests") for p in (ROOT / d).rglob("*")
             if p.is_file() and p.suffix in SUFFIXES and "__pycache__" not in p.parts]
    assert any(p.name == "index.html" for p in files) and any(
        p.name == "web_protocol_snapshot.json" for p in files), "the walk missed the page or the snapshot"
    hits = [f"{p.relative_to(ROOT)}:{n}: {needle}"
            for p in files
            for n, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1)
            for needle in FORBIDDEN if needle in line]
    assert not hits, "\n".join(hits)


def test_tool_names_is_preserved_on_both_turn_events():
    assert "tool_names" in TurnCompleted.model_fields and "tool_names" in TurnFailed.model_fields


def test_the_protocol_moved_exactly_once_on_top_of_v3():
    assert protocol.PROTOCOL_VERSION == 4
    tree = ast.parse(Path(protocol.__file__).read_text(encoding="utf-8"))
    body = tree.body
    i = next(n for n, node in enumerate(body) if isinstance(node, ast.Assign)
             and any(getattr(t, "id", None) == "PROTOCOL_VERSION" for t in node.targets))
    doc = body[i + 1].value.value  # the string literal right after the constant is its docstring
    assert isinstance(doc, str)
    for bit in ("v3:", "tool_names", "v4:", "artifact"):
        assert bit in doc, f"the PROTOCOL_VERSION docstring does not mention {bit!r}"
    assert "artifact" in Observation.model_fields


async def test_generate_image_declares_none_safe_trusted(tmp_path):
    (tool,) = await ImagePlugin().tools(_ctx(tmp_path, comfyui_url="http://comfy.test"))
    schema = tool.info()
    assert (schema.ingest, schema.host, schema.result_origin) == ("none", "safe", "trusted")
    assert schema.gate_family is None
    assert not is_host_dangerous(schema) and not ingests_untrusted(schema)


def test_image_is_a_bundled_plugin_and_off_by_default():
    assert ImagePlugin in bundled_plugins()
    assert ImagePlugin.manifest.enabled_by_default is False
    assert ImagePlugin.wants_artifacts is True


def test_no_image_entry_on_the_burn_down_list():
    assert not [pair for pair in BURN_DOWN if any("image" in f for f in pair)], BURN_DOWN


def test_no_core_module_names_the_image_plugin():
    core = [p for p in sorted(SRC.rglob("*.py")) if classify(p.relative_to(SRC).as_posix()) == "core"]
    assert len(core) > 100, f"premise: only {len(core)} core files classified"
    hits = [f"{p.relative_to(SRC).as_posix()}: {needle}" for p in core
            for needle in CORE_NEEDLES if needle in p.read_text(encoding="utf-8").lower()]
    assert not hits, hits
