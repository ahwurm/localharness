"""The public docs say what the code does — checked against the code, not promised.

A docs change cannot be "done" because prose was edited. Each check below is a pure function
`check_*(...) -> list[str]` (one string per disagreement) that reads a document AND the code object
it describes, and every source of truth is imported, never typed out: `BUILTIN_PLUGINS`,
`global_only_paths`, `ASK_GLOBAL_ONLY_FIELDS`, `LAUNCH_KEYS`, the `ToolSchema` declaration
defaults, `plugin_summary`. Each check has a live test on the real file and
a "bites" test that feeds it one deliberate break and asserts a non-empty violation list, so no
check is vacuous.

Every live test is green: a doc found false later fails here, and the fix is to the doc (or the
code), never to the check.

These tests never invoke the CLI (Typer caches the terminal width at first import) and read plugins
from `BUILTIN_PLUGINS` only, never discovery (a dev venv may have other plugins installed).
"""
from __future__ import annotations

import ast
import re
from collections.abc import Callable, Iterable

import pytest

from localharness.agent.gate_types import PROTECTED_CONFIG_DIR_ENTRIES
from localharness.cli.model_switch import LAUNCH_NO, LAUNCH_YES, ROW_LOCKED, ROW_WITHHELD
from localharness.cli.plugins_cmd import CONSENT, THIRD_PARTY_CONFIRM
from localharness.config.loader import (
    AGENT_CORE_GLOBAL_ONLY_SECTIONS, AGENT_GLOBAL_ONLY_FIELDS, ASK_GLOBAL_ONLY_FIELDS, HARNESS_GLOBAL_ONLY_FIELDS,
)
from localharness.config.plugin_sections import CORE_HARNESS_KEYS, global_only_paths
from localharness.config.skills import MAX_ENTRIES, ORIGINS
from localharness.config.trust import LAUNCH_KEYS, LAUNCH_MODEL_KEYS, LAUNCH_MODEL_KIND
from localharness.core.content import MAX_REFERENCE_BYTES
from localharness.plugins.api import Plugin, plugin_summary
from localharness.plugins.builtin import BUILTIN_PLUGINS
from localharness.tools.base import ToolSchema
from tests.unit.test_release_docs import _BLOB, _REPO, _github_anchors

SRC = _REPO / "src" / "localharness"
_TREE = _BLOB.replace("/blob/", "/tree/")


# ---------------------------------------------------------------- helpers: read docs as structure

def _read(rel: str) -> str:
    return (_REPO / rel).read_text(encoding="utf-8")


def _fence(line: str) -> bool:
    return line.lstrip().startswith(("```", "~~~"))


def unfenced_lines(text: str) -> list[str]:
    """The text's lines with every fenced-code line (fences included) blanked; positions kept."""
    out, fenced = [], False
    for line in text.splitlines():
        if _fence(line):
            fenced = not fenced
            out.append("")
        else:
            out.append("" if fenced else line)
    return out


def _heading(line: str) -> tuple[int, str] | None:
    m = re.match(r"(#{1,6}) (.+)", line)
    return (len(m.group(1)), m.group(2).strip()) if m else None


def section(text: str, heading: str, *, prefix: bool = False) -> str | None:
    """The body under `heading` (a whole heading line such as "## Plugins", outside code fences)
    up to the next heading of the same or a higher level; None when the heading is absent. With
    `prefix`, the heading line need only start with `heading` (for "## 5." style numbering)."""
    raw, plain = text.splitlines(), unfenced_lines(text)
    level = len(heading) - len(heading.lstrip("#"))
    for i, line in enumerate(plain):
        if line.strip() == heading or (prefix and line.startswith(heading)):
            for j in range(i + 1, len(plain)):
                h = _heading(plain[j])
                if h and h[0] <= level:
                    return "\n".join(raw[i + 1:j])
            return "\n".join(raw[i + 1:])
    return None


def ticks(text: str) -> list[str]:
    """Every single-backtick code span outside fenced code, in order."""
    return [m.group(1) for line in unfenced_lines(text)
            for m in re.finditer(r"(?<!`)`([^`\n]+)`(?!`)", line)]


def tables(text: str) -> list[list[list[str]]]:
    """Each markdown table outside fenced code as rows of stripped cells (separator rows dropped)."""
    out: list[list[list[str]]] = []
    current: list[list[str]] = []
    for line in [*unfenced_lines(text), ""]:
        if line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                current.append(cells)
        elif current:
            out.append(current)
            current = []
    return out


def table_rows(section_text: str) -> list[list[str]]:
    """The rows of the first table in `section_text` (header first), or []."""
    found = tables(section_text)
    return found[0] if found else []


def bullets(text: str) -> list[tuple[str, str]]:
    """(nearest heading, bullet text with its continuation lines) for every `- ` list item."""
    out: list[tuple[str, str]] = []
    heading, current = "", None
    for line in [*unfenced_lines(text), ""]:
        if (h := _heading(line)) is not None:
            heading = h[1]
        if line.startswith("- ") or (h is not None) or not line.strip() or not line.startswith(" "):
            if current is not None:
                out.append(current)
                current = None
        if line.startswith("- "):
            current = (heading, line)
        elif current is not None and line.startswith(" ") and line.strip():
            current = (current[0], current[1] + "\n" + line)
    return out


def paragraphs(text: str) -> list[str]:
    return [p for p in re.split(r"\n\s*\n", "\n".join(unfenced_lines(text))) if p.strip()]


# ---------------------------------------------------------------- check 3: machine-level-only list

MACHINE_ONLY_HEADING = "### Machine-level-only settings"
_DOTPATH = re.compile(r"(?:<name>|[a-z_][a-z0-9_]*)(?:(?:\.[a-z_][a-z0-9_]*)+(?:\.\*)?|\.\*)")
_FILE_SUFFIXES = (".yaml", ".yml", ".md", ".py", ".json", ".env", ".toml", ".txt")


def machine_only_code_set(plugins: Iterable[type[Plugin]] | None = None) -> frozenset[str]:
    """Every setting only the machine-level (global) config may set, computed from code: each
    bundled plugin's GLOBAL_ONLY fields, the `permissions.ask` keys a project may not set, the core
    harness keys in HARNESS_GLOBAL_ONLY_FIELDS, `<name>.enabled` for a plugin you install (an
    AgentConfigModel may not mark GLOBAL_ONLY at all — the loader refuses such a plugin), and
    AGENT_GLOBAL_ONLY_FIELDS, the agent-file keys the loader re-resolves from the global layer
    (`memory.embedding_model`, `permissions.budget.kill_file`), and `<section>.*` for each core agent
    section only the global layer may set (AGENT_CORE_GLOBAL_ONLY_SECTIONS)."""
    plugins = BUILTIN_PLUGINS if plugins is None else tuple(plugins)
    paths = {f"{P.manifest.name}.{p}" for P in plugins for p in global_only_paths(P.ConfigModel)}
    paths |= ({f"permissions.ask.{f}" for f in ASK_GLOBAL_ONLY_FIELDS} | HARNESS_GLOBAL_ONLY_FIELDS
              | AGENT_GLOBAL_ONLY_FIELDS)
    return frozenset(paths | {"<name>.enabled"} | {f"{s}.*" for s in AGENT_CORE_GLOBAL_ONLY_SECTIONS})


def check_machine_only_list(security_text: str, code_set: frozenset[str] | None = None) -> list[str]:
    """SECURITY.md's `### Machine-level-only settings` list equals the code's set, both ways. The
    list is the backticked dot-paths on the section's bullet lines (`- `), a whole core section
    such as `server` (one token, no dot), and `<section>.*`; prose in the section — e.g. the tighten-only rules no
    code set enumerates — is not compared."""
    code = machine_only_code_set() if code_set is None else code_set
    body = section(security_text, MACHINE_ONLY_HEADING)
    if body is None:
        return [f"SECURITY.md has no '{MACHINE_ONLY_HEADING}' section"]
    doc = {t for _h, b in bullets(body) for t in ticks(b)
           if (_DOTPATH.fullmatch(t) or t in CORE_HARNESS_KEYS) and not t.endswith(_FILE_SUFFIXES)}
    return ([f"machine-level only in code, not listed in SECURITY.md: {p}" for p in sorted(code - doc)]
            + [f"listed in SECURITY.md, not machine-level only in code: {p}" for p in sorted(doc - code)])


def test_machine_only_list_live():
    """SECURITY.md names every setting a project cannot set, exactly the code's set."""
    assert check_machine_only_list(_read("SECURITY.md")) == []


def test_machine_only_list_bites():
    code = machine_only_code_set()
    good = f"{MACHINE_ONLY_HEADING}\n\nA project cannot set these.\n\n" + "".join(
        f"- `{p}`\n" for p in sorted(code))
    assert check_machine_only_list(good, code) == []
    assert check_machine_only_list(good, code | {"image.timeout_s"}) == [
        "machine-level only in code, not listed in SECURITY.md: image.timeout_s"]
    dropped = good.replace("- `image.workflow`\n", "")
    assert check_machine_only_list(dropped, code) == [
        "machine-level only in code, not listed in SECURITY.md: image.workflow"]
    assert check_machine_only_list(good.replace("- `server`\n", ""), code) == [
        "machine-level only in code, not listed in SECURITY.md: server"]
    assert check_machine_only_list(good + "- `org`\n", code) == [
        "listed in SECURITY.md, not machine-level only in code: org"]
    for name in ("task_flow.*", "stuck_detector.*"):
        assert check_machine_only_list(good.replace(f"- `{name}`\n", ""), code) == [
            f"machine-level only in code, not listed in SECURITY.md: {name}"]
    assert check_machine_only_list(good + "- `memory.*`\n", code) == [
        "listed in SECURITY.md, not machine-level only in code: memory.*"]
    assert check_machine_only_list("## Something else\n", code)


# ---------------------------------------------------------------- check: the launch keys the record holds

LLAMA_CONFIG_INI = "/etc/llama.cpp/config.ini"


def check_launch_keys_doc(security_text: str, launch_keys: Iterable[str] | None = None,
                          model_keys: Iterable[str] | None = None) -> list[str]:
    """SECURITY.md names every key of `trust.LAUNCH_KEYS` and `trust.LAUNCH_MODEL_KEYS`, the
    `launch_model` kind, and the root-owned llama.cpp `config.ini` the record cannot cover. The
    names are the backticked tokens of the paragraphs that mention the kind, so a key that only
    appears elsewhere in the file does not count."""
    launch = tuple(LAUNCH_KEYS if launch_keys is None else launch_keys)
    model = tuple(LAUNCH_MODEL_KEYS if model_keys is None else model_keys)
    named = {t for para in paragraphs(security_text) if f"`{LAUNCH_MODEL_KIND}`" in para
             for t in ticks(para)}
    out = [f"launch key not named in SECURITY.md: {k}" for k in (*launch, *model) if k not in named]
    if LAUNCH_MODEL_KIND not in named:
        out.append("the launch_model kind is not named")
    if LLAMA_CONFIG_INI not in named:
        out.append(f"{LLAMA_CONFIG_INI} is not named")
    return out


def test_launch_keys_doc_live():
    """SECURITY.md names all thirteen launch keys, the seven model keys, the kind and the gap."""
    assert check_launch_keys_doc(_read("SECURITY.md")) == []


def test_launch_keys_doc_bites():
    keys, model = LAUNCH_KEYS, LAUNCH_MODEL_KEYS
    good = ("The record holds " + ", ".join(f"`{k}`" for k in keys) + ". The kind `launch_model` holds "
            + ", ".join(f"`{k}`" for k in model) + f". `{LLAMA_CONFIG_INI}` is outside it.\n")
    assert check_launch_keys_doc(good) == []
    assert check_launch_keys_doc(good) == check_launch_keys_doc(good), "stable across two runs"
    assert check_launch_keys_doc(good.replace("`config_ini`", "config_ini")) == [
        "launch key not named in SECURITY.md: config_ini"]
    assert check_launch_keys_doc(good.replace("`cache_ram`", "")) == [
        "launch key not named in SECURITY.md: cache_ram"]
    # without the kind no paragraph is the launch paragraph, so nothing else counts as named either
    unkinded = check_launch_keys_doc(good.replace("`launch_model`", "`launch-model`"))
    assert "the launch_model kind is not named" in unkinded
    assert {f"launch key not named in SECURITY.md: {k}" for k in (*keys, *model)} <= set(unkinded)
    assert check_launch_keys_doc(good.replace(f"`{LLAMA_CONFIG_INI}`", "the root-owned file")) == [
        f"{LLAMA_CONFIG_INI} is not named"]
    assert check_launch_keys_doc(good, [*keys, "brand_new_key"]) == [
        "launch key not named in SECURITY.md: brand_new_key"]
    # a key named only outside the paragraphs that carry the kind does not count
    assert check_launch_keys_doc(good.replace("`binary`", "binary") + "\n\nElsewhere: `binary`.\n") == [
        "launch key not named in SECURITY.md: binary"]
    assert check_launch_keys_doc("## Nothing\n") != []


_NUMBER_WORDS = dict(zip(("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                          "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen"), range(16)))


PROTECTED_LIST = re.compile(r"one (\w+)-entry list protects (.*?)\. ")


def check_protected_entries_doc(text: str, entries: Iterable[str] | None = None) -> list[str]:
    """SECURITY.md's `one <count>-entry list protects` sentence carries the count of
    `PROTECTED_CONFIG_DIR_ENTRIES` and names each entry once; the order is free."""
    entries = tuple(PROTECTED_CONFIG_DIR_ENTRIES if entries is None else entries)
    m = PROTECTED_LIST.search(" ".join(text.split()))
    if m is None:
        return ["the protected config-directory list is not found"]
    named = [t.removesuffix("/**") for t in re.findall(r"`([^`]+)`", m.group(2))]
    out = []
    if _NUMBER_WORDS.get(m.group(1)) != len(entries):
        out.append(f"SECURITY.md says {m.group(1)} protected entries, the code has {len(entries)}")
    out += [f"protected entry not named in SECURITY.md: {e}" for e in entries if e not in named]
    out += [f"not a protected entry: {n}" for n in dict.fromkeys(named) if n not in entries]
    out += [f"protected entry named twice: {n}" for n in dict.fromkeys(named) if named.count(n) > 1]
    return out


def test_protected_entries_doc():
    assert check_protected_entries_doc(_read("SECURITY.md")) == []


def test_protected_entries_doc_bites():
    good = _read("SECURITY.md")
    assert "fourteen-entry" in good
    assert check_protected_entries_doc(good.replace("fourteen-entry", "thirteen-entry")) == [
        f"SECURITY.md says thirteen protected entries, the code has {len(PROTECTED_CONFIG_DIR_ENTRIES)}"]
    assert len(check_protected_entries_doc(good.replace("fourteen-entry", "fifteen-entry"))) == 1
    assert check_protected_entries_doc(good.replace("`overrides.yaml.lock`", "the overlay lock")) == [
        "protected entry not named in SECURITY.md: overrides.yaml.lock"]
    assert check_protected_entries_doc(good.replace("`model-library.json`", "`grants.yaml` `model-library.json`")) == [
        "protected entry named twice: grants.yaml"]
    assert check_protected_entries_doc(good.replace("`grants.yaml`,", "`grants.yaml`, `grants.yml`,", 1)) == [
        "not a protected entry: grants.yml"]
    assert check_protected_entries_doc("## Nothing\n") == ["the protected config-directory list is not found"]
    assert check_protected_entries_doc(good, [*PROTECTED_CONFIG_DIR_ENTRIES, "new.yaml"]) == [
        f"SECURITY.md says fourteen protected entries, the code has {len(PROTECTED_CONFIG_DIR_ENTRIES) + 1}",
        "protected entry not named in SECURITY.md: new.yaml"]


def check_key_file_doc(security_text: str, cap: int | None = None) -> list[str]:
    """SECURITY.md says what `start` does with `server.api_key_file`: that it opens it, the byte rule
    with the code's own `KEY_FILE_CAP`, what a refused file does, where the key may go, and that
    `doctor` still never opens it. The cap is read from the code, never typed here (IN-05)."""
    from localharness.provider.server import KEY_FILE_CAP

    cap = KEY_FILE_CAP if cap is None else cap
    text = " ".join(security_text.split())
    needles = {
        "that start opens the key file": "`server.api_key_file` is the one launch key `start` itself opens",
        "the byte rule": "one line of printable ASCII",
        "the size cap": f"at most {cap} bytes",
        "the refused-file outcome": "the session sends no key",
        "the destination rule": "an endpoint that is this machine",
        "doctor never opens it": "doctor never opens a key file",
        "the narrowing exceptions": "except `bind_all` and `require_api_key`, which are narrowed",
    }
    return [f"SECURITY.md does not say {what}" for what, needle in needles.items() if needle not in text]


def test_key_file_doc_live():
    assert check_key_file_doc(_read("SECURITY.md")) == []


def test_key_file_doc_bites():
    text = _read("SECURITY.md")
    assert check_key_file_doc(text.replace("at most 4096 bytes", "at most 1 bytes")) == [
        "SECURITY.md does not say the size cap"]
    assert check_key_file_doc(text, cap=99) == ["SECURITY.md does not say the size cap"]
    assert check_key_file_doc(text.replace("doctor never opens a key file", "doctor opens")) == [
        "SECURITY.md does not say doctor never opens it"]
    assert len(check_key_file_doc("## Nothing\n")) == 7


# ---------------------------------------------------------------- check: the launch boundary and the model route

LOCK_NEEDLES = ("cold launch or managed restart", "whatever the setting", "not an authorization boundary",
                "one tap and no question", "may be another machine")
STALE_DEFAULT = "today stops working"
MOBILE_NEEDLES = ("except `GET /api/model`, bearer only", "a model launch, whose confirm it can answer")


def _source_constant(rel: str, name: str):
    """The module-level literal `name` in the repo file `rel`, read with `ast` so no optional dependency imports."""
    return next(ast.literal_eval(n.value) for n in ast.parse(_read(rel)).body
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))


def _swap(text: str, phrase: str, new: str) -> str:
    return re.sub(r"\s+".join(map(re.escape, phrase.split())), lambda _: new, text)


def check_model_switch_doc(
    security_text: str, ttl: float = _source_constant("src/localharness/channels/mobile/channel.py", "CONFIRM_TOKEN_TTL_S"),
) -> list[str]:
    """SECURITY.md says the machine lock covers launches, who is refused whatever the setting, how long the confirm lives, and what the model route withholds."""
    mobile, paired = (" ".join((section(security_text, f"### {h}") or "").split())
                      for h in ("`localharness mobile`", "A paired phone or Discord account"))
    found = [f"SECURITY.md's mobile section does not say {n}" for n in (*MOBILE_NEEDLES, f"`{ROW_WITHHELD}`")
             if n not in mobile]
    found += [f"SECURITY.md's paired-phone section does not say {n}" for n in (*LOCK_NEEDLES, f"{int(ttl)} seconds")
              if n not in paired]
    return found + ([f"SECURITY.md's paired-phone section still says {STALE_DEFAULT}"] if STALE_DEFAULT in paired else [])


def test_model_switch_doc_live():
    assert check_model_switch_doc(_read("SECURITY.md")) == []


def test_model_switch_doc_bites():
    text = _read("SECURITY.md")
    for needle in LOCK_NEEDLES:
        assert check_model_switch_doc(_swap(text, needle, "x")) == [
            f"SECURITY.md's paired-phone section does not say {needle}"]
    for needle in MOBILE_NEEDLES:
        assert check_model_switch_doc(_swap(text, needle, "x")) == [f"SECURITY.md's mobile section does not say {needle}"]
    assert check_model_switch_doc(_swap(text, f"`{ROW_WITHHELD}`", "x")) == [
        f"SECURITY.md's mobile section does not say `{ROW_WITHHELD}`"]
    assert check_model_switch_doc(text, ttl=60.0) == ["SECURITY.md's paired-phone section does not say 60 seconds"]
    lock = "**What this does NOT cover.** The lock refuses"
    assert lock in text
    stale = text.replace(lock, f"It is `true` by default, so nothing you do from the phone or Discord {STALE_DEFAULT},\n\n{lock}", 1)
    assert check_model_switch_doc(stale) == [f"SECURITY.md's paired-phone section still says {STALE_DEFAULT}"]
    empty = check_model_switch_doc("## Nothing\n")
    assert len(empty) == len(MOBILE_NEEDLES) + 1 + len(LOCK_NEEDLES) + 1
    assert check_model_switch_doc("## Nothing\n") == empty


DENY_CLASS_NEEDLES = (
    "except each agent's `tasks/` and `sessions/` folders", "never asked, in every mode",
    "lands under `agents/<name>/tasks/` or `agents/<name>/sessions/`", "the delete or move of `agents/<name>` itself",
    "Nothing above those folders is refused", "not a barrier", "write forms that name no static target under those folders",
    "a non-recursive `chmod` at any target")


def check_deny_class_doc(security_text: str) -> list[str]:
    """SECURITY.md's approval-gate section states the deny class on an agent's task and session folders, what it leaves alone, and that it is not a barrier."""
    gate = " ".join((section(security_text, "## Human approval gate") or "").split())
    return [f"SECURITY.md's approval gate does not say {n}" for n in DENY_CLASS_NEEDLES if n not in gate]


def test_deny_class_doc_live():
    assert check_deny_class_doc(_read("SECURITY.md")) == []


def test_deny_class_doc_bites():
    text = _read("SECURITY.md")
    for needle in DENY_CLASS_NEEDLES:
        assert check_deny_class_doc(_swap(text, needle, "x")) == [f"SECURITY.md's approval gate does not say {needle}"]
    assert len(check_deny_class_doc("## Nothing\n")) == len(DENY_CLASS_NEEDLES)


MOBILE_DOC_NEEDLES = ("cannot be cancelled", "two requests for the server", "`active`", "`launches`")


def _page_constant(name: str) -> str:
    """The string of `const <name> = "...";` in the mobile reference page."""
    return re.search(rf'^const {name} = "([^"\n]*)";', _read("src/localharness/channels/mobile/ui/index.html"), re.M).group(1)


def check_mobile_model_doc(
    mobile_text: str, wake: str = _page_constant("MODEL_WAKE"),
    asleep: str = _source_constant("src/localharness/channels/mobile/server.py", "MODEL_LIST_NO_SESSION"),
) -> list[str]:
    """docs/mobile.md's What to expect says the launch confirm and quotes the shipped confirm, lock and asleep strings."""
    expect = " ".join((section(mobile_text, "## What to expect") or "").split())
    return [f"docs/mobile.md does not say {n}" for n in (*MOBILE_DOC_NEEDLES, *(f"`{q}`" for q in (
        LAUNCH_YES, LAUNCH_NO, ROW_LOCKED, asleep, wake))) if n not in expect]


def test_mobile_model_doc_live():
    assert check_mobile_model_doc(_read("docs/mobile.md")) == []


def test_mobile_model_doc_bites():
    text = _read("docs/mobile.md")
    wake = _page_constant("MODEL_WAKE")
    assert check_mobile_model_doc(_swap(text, f"`{wake}`", "x")) == [f"docs/mobile.md does not say `{wake}`"]
    assert check_mobile_model_doc(text, asleep="chat sleeping") == ["docs/mobile.md does not say `chat sleeping`"]
    needle = "two requests for the server"
    assert check_mobile_model_doc(_swap(text, needle, "x")) == [f"docs/mobile.md does not say {needle}"]
    empty = check_mobile_model_doc("## Nothing\n")
    assert len(empty) == len(MOBILE_DOC_NEEDLES) + 5
    assert check_mobile_model_doc("## Nothing\n") == empty


# ---------------------------------------------------------------- check: what the Plugins sentences say

CONSENT_PHRASES = ("from your package index", "runs its build steps")
TRUST_PHRASES = ("with your permissions", "in every session")
NEVER_SHOWN_ANCHORS = ("wherever a command displays it:", "It is never printed:")
INSTALL_TOKEN = "`localharness[<extras>]==<version>`"
TRUST_NEEDLES = ("`/plugins enable <name>`", "Enable in Plugins", *TRUST_PHRASES)
INSTALL_NEEDLES = (INSTALL_TOKEN, "the version you are running", "index your pip or uv configuration names",
                   "unpinned", "no hash checking", *CONSENT_PHRASES)
RESIDUAL_NEEDLES = ("prompt-injected agent", "`pip.conf`", "`uv.toml`", "`PATH`", "`?token=`")


def check_plugins_doc(security_text: str, consent: str = CONSENT[1], confirm: str = THIRD_PARTY_CONFIRM) -> list[str]:
    """The `### Plugins` bullets and paragraphs say what the trust question and the install trust and
    where a secret is never shown, and the consent and the question still say what the doc quotes."""
    whole = " ".join(security_text.split())
    body = section(security_text, "### Plugins") or ""
    flat = lambda parts: " ".join(" ".join(parts).split())
    seen = [(flat(b for _h, b in bullets(body) if "it is your trust grant" in b), TRUST_NEEDLES),
            (flat(b for _h, b in bullets(body) if INSTALL_TOKEN in b), INSTALL_NEEDLES),
            (flat(p for p in paragraphs(body) if "What this does NOT cover" in p), RESIDUAL_NEEDLES)]
    return ([f"SECURITY.md's list after {a} does not name {w}" for a in NEVER_SHOWN_ANCHORS
             for w in ("Plugins", "Configure") if w not in whole.partition(a)[2].split(". ")[0]]
            + [f"SECURITY.md does not say {n}" for text, needles in seen for n in needles if n not in text]
            + [f"the consent no longer says {p}" for p in CONSENT_PHRASES if p not in consent]
            + [f"the trust question no longer says {p}" for p in TRUST_PHRASES if p not in confirm])


def test_plugins_doc_live():
    assert check_plugins_doc(_read("SECURITY.md")) == []


def test_plugins_doc_bites():
    text = _read("SECURITY.md")
    assert check_plugins_doc(text.replace("no hash checking", "")) == ["SECURITY.md does not say no hash checking"]
    assert check_plugins_doc(text.replace("`pip.conf`", "")) == ["SECURITY.md does not say `pip.conf`"]
    assert check_plugins_doc(text.replace("Enable in Plugins", "")) == ["SECURITY.md does not say Enable in Plugins"]
    assert check_plugins_doc(text.replace("`?token=`", "")) == ["SECURITY.md does not say `?token=`"]
    moved = text.replace("Plugins and its Configure, `doctor`", "Plugins, `doctor`").replace(
        "show `**********`. The token comes only", "show `**********`. Configure too. The token comes only")
    assert check_plugins_doc(moved) == ["SECURITY.md's list after It is never printed: does not name Configure"]
    assert check_plugins_doc(text, consent=CONSENT[1].replace("from your package index", "from PyPI")) == [
        "the consent no longer says from your package index"]
    assert check_plugins_doc(text, confirm=THIRD_PARTY_CONFIRM.replace("in every session", "")) == [
        "the trust question no longer says in every session"]
    nothing = check_plugins_doc("## Nothing\n")
    assert len(nothing) == (2 * len(NEVER_SHOWN_ANCHORS) + len(TRUST_NEEDLES) + len(INSTALL_NEEDLES)
                            + len(RESIDUAL_NEEDLES))
    unnamed = text.replace("Plugins (the screen bare", "(the screen bare")
    assert check_plugins_doc(unnamed) == ["SECURITY.md's list after wherever a command displays it: does not name Plugins"]


# ---------------------------------------------------------------- check: the skills section

SKILLS_HEADING = "### Skills"
SKILLS_NEEDLES = (
    "`allowed-tools` grants nothing", "installs nothing", "](#project-instructions-files)",
    "one approval per file", "`guarded`", "`trusted`", "`auto`", "`read-only`", "`unattended`",
    f"{MAX_REFERENCE_BYTES // 1024} KiB", f"{MAX_ENTRIES} entries", "named gaps", "shows only the path",
    "`permissions.workspace_root`", "`mv staged .agents`", "`[budget: …]`", "off switch",
    "`localharness --config-dir X skills`", "whole line", "about a second", "minutes", "once per subcommand",
    "except the bundled one is a protected path", "the bundled set is not protected", "`bash_exec`",
)
ELSEWHERE_NEEDLES = (
    "`.agents/skills/` is protected the same way", "`permissions.ask.protected_config_dir_entries`",
    "A project skill is not that door", "the bundled `skill-creator` writes",
    "A script a skill names is the same case",
)
CLONE_NEEDLES = ("can load agent and division files and skills from it",
                 "config, agent files and skills load with no prompt")


def skills_dirs(origins: Iterable[tuple[str, str]] = ORIGINS) -> list[str]:
    """The directory each origin is named by in the doc, in precedence order."""
    return [("and " if at and label == "project" else "")
            + {"project": f"`{rel}/`", "machine": f"`~/.localharness/{rel}/`", "bundled": "the bundled set"}[label]
            for at, (label, rel) in enumerate(origins)]


def in_order(text: str, needles: Iterable[str]) -> list[str]:
    """Each needle that is absent from `text` or does not first appear after the previous one."""
    out, last = [], -1
    for n in needles:
        if (at := text.find(n)) <= last:
            out.append(n)
        else:
            last = at
    return out


def check_skills_doc(security_text: str) -> list[str]:
    """`### Skills` sits between Plugins and Machine-level-only settings as one prose paragraph that
    names the origins in precedence order and says what skills do and grant; the clone lists name skills."""
    body = section(security_text, SKILLS_HEADING)
    if body is None:
        return ["SECURITY.md has no ### Skills section"]
    whole, flat = (" ".join(t.split()) for t in (security_text, body))
    heads = [line.strip() for line in unfenced_lines(security_text) if _heading(line)]
    at = heads.index(SKILLS_HEADING)
    out = []
    if heads[max(at - 1, 0):at + 2] != ["### Plugins", SKILLS_HEADING, MACHINE_ONLY_HEADING]:
        out.append("### Skills is not between ### Plugins and ### Machine-level-only settings")
    if len(paragraphs(body)) != 1 or any(re.match(r"([-*+]|\d+\.) ", ln.lstrip()) for ln in body.splitlines()):
        out.append("the Skills section is not one paragraph of prose")
    listing = flat.split(" A project's skills load ")[0]
    out += [f"the Skills section does not name {d} in precedence order" for d in in_order(listing, skills_dirs())]
    out += [f"the Skills section does not say {n}" for n in SKILLS_NEEDLES if n not in flat]
    out += [f"SECURITY.md does not say {n}" for n in CLONE_NEEDLES if n not in whole]
    listed = PROTECTED_LIST.search(whole)
    after = whole[listed.end():] if listed else ""
    out += [f"SECURITY.md does not say {n} after the fourteen-entry list" for n in ELSEWHERE_NEEDLES[:2] if n not in after]
    out += [f"SECURITY.md does not say {n}" for n in ELSEWHERE_NEEDLES[2:] if n not in whole]
    if "tools/capabilities.py" in ticks(body):
        out.append("the Skills section names the capability-floor module in backticks")
    if "\N{SECTION SIGN}" in body:
        out.append("the Skills section carries a section sign")
    return out


def test_skills_doc_live():
    assert check_skills_doc(_read("SECURITY.md")) == []


def test_skills_doc_bites():
    text = _read("SECURITY.md")
    body = section(text, SKILLS_HEADING)
    assert check_skills_doc("## Nothing\n") == ["SECURITY.md has no ### Skills section"]
    assert check_skills_doc(text.replace(SKILLS_HEADING + "\n", "### Other\n\nText.\n\n" + SKILLS_HEADING + "\n", 1)) == [
        "### Skills is not between ### Plugins and ### Machine-level-only settings"]
    for more in ("\n- x\n", "\n\nMore.\n"):
        assert check_skills_doc(text.replace(body, body.rstrip() + more + "\n")) == [
            "the Skills section is not one paragraph of prose"]
    kib = f"{MAX_REFERENCE_BYTES // 1024} KiB"
    assert check_skills_doc(text.replace(kib, "100 KiB")) == [f"the Skills section does not say {kib}"]
    assert check_skills_doc(text.replace("`[budget: …]`", "the note")) == [
        "the Skills section does not say `[budget: …]`"]
    assert check_skills_doc(text.replace(body, body.rstrip() + " See `tools/capabilities.py`.\n")) == [
        "the Skills section names the capability-floor module in backticks"]
    assert check_skills_doc(text.replace(body, body.rstrip() + " \N{SECTION SIGN}9.\n")) == [
        "the Skills section carries a section sign"]
    assert check_skills_doc(text.replace("agent files and skills load", "agent files load")) == [
        "SECURITY.md does not say config, agent files and skills load with no prompt"]
    assert check_skills_doc(re.sub(r"is\s+protected the same way", "is protected", text)) == [
        f"SECURITY.md does not say {ELSEWHERE_NEEDLES[0]} after the fourteen-entry list"]
    assert check_skills_doc(text.replace("A project skill is not that door", "A skill is no door")) == [
        f"SECURITY.md does not say {ELSEWHERE_NEEDLES[2]}"]
    for gone, pattern, instead in (("except the bundled one is a protected path", r"except\s+the\s+bundled\s+one\s+is", "is"),
                                   ("the bundled set is not protected", r"the\s+bundled\s+set\s+is\s+not\s+protected", "x"),
                                   ("`bash_exec`", r"`bash_exec`", "x")):
        assert check_skills_doc(re.sub(pattern, instead, text)) == [f"the Skills section does not say {gone}"]
    inside = text.replace("`server-floor.json` (the window", "`.agents/skills/`, `server-floor.json` (the window", 1)
    assert check_protected_entries_doc(inside) == ["not a protected entry: .agents/skills/"]
    assert check_protected_entries_doc(text) == []
    assert skills_dirs() == [
        "`.localharness/skills/`", "and `.agents/skills/`", "`~/.localharness/skills/`", "the bundled set"]
    assert check_skills_doc(re.sub(r",\s+then\s+the\s+bundled\s+set\s+inside\s+the\s+package,\s+whose\s+names\s+no\s+other"
                                   r"\s+skill\s+can\s+take", "", text)) == [
        "the Skills section does not name the bundled set in precedence order"]
    assert check_skills_doc(re.sub(r"\s+and\s+`\.agents/skills/`\s+in\s+a\s+project", "", text)) == [
        "the Skills section does not name and `.agents/skills/` in precedence order"]
    assert in_order("`b` then `a`", ["`a`", "`b`"]) == ["`b`"]
    assert in_order("`a` then `b`", ["`a`", "`b`", "`c`"]) == ["`c`"]
    assert in_order("`a` `a`", ["`a`", "`a`"]) == ["`a`"]


# ---------------------------------------------------------------- check 4: three declaration readers

READERS = {"tools/capabilities.py": None, "agent/gate.py": "gate_family",
           "agent/context.py": "result_origin"}
DECLARATIONS = ("ingest", "host", "result_origin", "gate_family")


def declaration_defaults() -> dict[str, str]:
    """The fail-closed default of each tool declaration, from `ToolSchema` (None renders `none`)."""
    return {d: "none" if (v := ToolSchema.model_fields[d].default) is None else str(v)
            for d in DECLARATIONS}


def check_three_readers(security_text: str, defaults: dict[str, str] | None = None) -> list[str]:
    """SECURITY.md's paragraph that names `tools/capabilities.py` (the floor) also names
    `agent/gate.py` with `gate_family` and `agent/context.py` with `result_origin`; every file it
    names exists and each named identifier is read there; and it states each declaration's
    default as `<declaration>: <default>` equal to `ToolSchema`'s."""
    defaults = declaration_defaults() if defaults is None else defaults
    para = next((p for p in paragraphs(security_text) if "tools/capabilities.py" in ticks(p)), None)
    if para is None:
        return ["SECURITY.md has no paragraph naming `tools/capabilities.py`"]
    spans = ticks(para)
    out = [f"`{t}` does not exist under src/localharness" for t in spans
           if t.endswith(".py") and not (SRC / t).is_file()]
    for path, name in READERS.items():
        if path not in spans:
            out.append(f"the declarations paragraph does not name `{path}`")
        if name is not None:
            if name not in spans:
                out.append(f"the declarations paragraph does not name `{name}` (read by {path})")
            if (SRC / path).is_file() and name not in (SRC / path).read_text(encoding="utf-8"):
                out.append(f"{path} does not read `{name}`")
    stated = {m.group(1): m.group(2) for t in spans if (m := re.fullmatch(r"(\w+): (\S+)", t))}
    for d, want in defaults.items():
        if stated.get(d) != want:
            out.append(f"the declarations paragraph states `{d}: {stated.get(d)}`, "
                       f"ToolSchema's default is `{d}: {want}`")
    return out


def test_three_readers_live():
    """SECURITY.md names the floor, the gate and the context store by file, with the real defaults."""
    assert check_three_readers(_read("SECURITY.md")) == []


def _good_readers_paragraph(defaults: dict[str, str]) -> str:
    stated = ", ".join(f"`{d}: {v}`" for d, v in defaults.items())
    return ("The floor (`tools/capabilities.py`), the gate (`agent/gate.py`, `gate_family`) and the "
            f"context store (`agent/context.py`, `result_origin`) read one declaration: {stated}.\n")


def test_three_readers_bites():
    defaults = declaration_defaults()
    good = _good_readers_paragraph(defaults)
    assert check_three_readers(good) == []
    found = check_three_readers(good.replace("agent/gate.py", "agent/gates.py"))
    assert "`agent/gates.py` does not exist under src/localharness" in found, found
    assert check_three_readers(good, {**defaults, "ingest": "none"})
    assert check_three_readers("Nothing about the floor.\n")


# ---------------------------------------------------------------- check 5: the plugins table

TABLE_HEADER = ["Name", "What it does", "Default", "How to switch"]


def _norm(cell: str) -> str:
    return " ".join(cell.replace("`", "").split())


def check_plugins_table(md_text: str, heading: str = "## Plugins",
                        plugins: Iterable[type[Plugin]] | None = None) -> list[str]:
    """The table under `heading` (the one whose header is Name | What it does | Default | How to
    switch, else the first) has one row per bundled plugin in `BUILTIN_PLUGINS` order: What it
    does == `plugin_summary` (what `plugins list` prints); Default starts `on`/`off` and names
    `localharness[<extra>]` exactly when the plugin needs one; How to switch carries the command
    that changes the default state, and for a plugin off by default `--set <key>` for each of its
    setup keys."""
    plugins = BUILTIN_PLUGINS if plugins is None else tuple(plugins)
    body = section(md_text, heading)
    if body is None:
        return [f"no '{heading}' section"]
    found = tables(body)
    rows = next((t for t in found if t[0] == TABLE_HEADER), found[0] if found else [])
    if not rows:
        return [f"no table under '{heading}'"]
    out = [] if rows[0] == TABLE_HEADER else [f"header is {rows[0]}, want {TABLE_HEADER}"]
    want = [P.manifest.name for P in plugins]
    if (names := [_norm(r[0]) for r in rows[1:]]) != want:
        out.append(f"rows are {names}, want {want} (BUILTIN_PLUGINS order)")
    for P, row in zip(plugins, rows[1:]):
        m = P.manifest
        if len(row) != 4:
            out.append(f"{m.name}: {len(row)} cells, want 4")
            continue
        _name, what, default, switch = (_norm(c) for c in row)
        if what != _norm(plugin_summary(P)):
            out.append(f"{m.name}: What it does is {what!r}, plugins list says {_norm(plugin_summary(P))!r}")
        state = "on" if m.enabled_by_default else "off"
        if not default.startswith(state):
            out.append(f"{m.name}: Default is {default!r}, the plugin is {state} by default")
        extra = f"localharness[{m.requires_extra}]"
        if m.requires_extra and extra not in default:
            out.append(f"{m.name}: Default does not name {extra}")
        if not m.requires_extra and "localharness[" in default:
            out.append(f"{m.name}: Default names an extra the plugin does not need")
        verb = "disable" if m.enabled_by_default else "enable"
        if f"localharness plugins {verb} {m.name}" not in switch:
            out.append(f"{m.name}: How to switch lacks `localharness plugins {verb} {m.name}`")
        if not m.enabled_by_default:
            out += [f"{m.name}: How to switch lacks `--set {s.key}`" for s in m.setup
                    if f"--set {s.key}" not in switch]
    return out


def test_plugins_table_readme_live():
    """README's plugins table is the bundled plugin list, as `plugins list` shows it."""
    assert check_plugins_table(_read("README.md")) == []


def test_index_md_lists_the_bundled_plugins():
    """INDEX.md (internal, git-ignored) names every bundled plugin under its generated plugins section."""
    if not (_REPO / "INDEX.md").is_file():
        pytest.skip("INDEX.md is git-ignored; checked locally at close-out")
    body = section(_read("INDEX.md"), "## Bundled plugins")
    assert body is not None, "no '## Bundled plugins' section"
    rows = tables(body)[0][1:]
    assert sorted(_norm(r[0]) for r in rows) == sorted(P.manifest.name for P in BUILTIN_PLUGINS)


def _good_table() -> str:
    rows = []
    for P in BUILTIN_PLUGINS:
        m = P.manifest
        default = ("on" if m.enabled_by_default else "off") + (
            f"; needs `localharness[{m.requires_extra}]`" if m.requires_extra else "")
        verb = "disable" if m.enabled_by_default else "enable"
        sets = "".join(f" --set {s.key}=..." for s in m.setup) if not m.enabled_by_default else ""
        rows.append(f"| `{m.name}` | {plugin_summary(P)} | {default} | "
                    f"`localharness plugins {verb} {m.name}{sets}` |")
    return "## Plugins\n\n| " + " | ".join(TABLE_HEADER) + " |\n|---|---|---|---|\n" + "\n".join(rows) + "\n"


def test_plugins_table_bites():
    good = _good_table()
    assert check_plugins_table(good) == []
    lines = good.splitlines()
    swapped = "\n".join([*lines[:4], lines[5], lines[4], *lines[6:]])
    assert check_plugins_table(swapped)
    image_row = next(line for line in lines if line.startswith("| `image`"))
    assert check_plugins_table(good.replace(image_row, image_row.replace("| off |", "| on |")))
    assert check_plugins_table(good.replace("Name | What it does", "Plugin | What it does"))


# ---------------------------------------------------------------- check 7: CHANGELOG

def changelog_slice(text: str) -> str:
    """`## [Unreleased]` up to the next `## [` heading, plus `## [0.16.0]` if that heading exists."""
    lines = text.splitlines()

    def entry(title: str) -> list[str]:
        start = next((i for i, line in enumerate(lines) if line.startswith(title)), None)
        if start is None:
            return []
        end = next((j for j in range(start + 1, len(lines)) if lines[j].startswith("## [")), len(lines))
        return lines[start:end]
    return "\n".join(entry("## [Unreleased]") + entry("## [0.16.0]"))


IMAGE_FIRST_RELEASE = "- **Image generation, as a plugin"
GUARDRAILS_BULLET = "- **`GUARDRAILS.md` reaches the model with memory on or off.**"


def check_changelog_items(text: str) -> list[str]:
    """The release entry names the four user-visible changes: the `localharness plugins` command
    and its verbs, the settings keys (`memory.enabled` replacing deprecated `org.memory_enabled`,
    `image.comfyui_url`, `dispatch.discord.*`), and the image plugin's first release under Added."""
    entry = changelog_slice(text)
    if not entry:
        return ["CHANGELOG has no ## [Unreleased] entry"]
    items = bullets(entry)
    out = []
    plugins_cmd = " ".join(b for _h, b in items if "localharness plugins" in b)
    out += [f"no bullet names `localharness plugins` with `{v}`" for v in ("list", "info", "enable", "disable")
            if not re.search(rf"\b{v}\b", plugins_cmd)]
    if not any("org.memory_enabled" in b and "memory.enabled" in b.replace("org.memory_enabled", "")
               and ("deprecat" in b.lower() or "deprecat" in h.lower()) for h, b in items):
        out.append("no bullet says `org.memory_enabled` is deprecated in favour of `memory.enabled`")
    out += [f"the entry does not name `{k}`" for k in ("image.comfyui_url", "dispatch.discord.") if k not in entry]
    if not any(b.startswith(IMAGE_FIRST_RELEASE) and h == "Added" for h, b in items):
        out.append("no Added bullet for the image plugin's first release")
    return out


def check_changelog_structure(text: str) -> list[str]:
    """Every line of the entry is a heading, a list item or its indented continuation, a table row,
    fenced code, or the intro paragraph above the first `###` — so a clobbered bullet head cannot
    hide as a stray line; and the `GUARDRAILS.md` bullet keeps its head."""
    entry = changelog_slice(text)
    out = []
    in_intro = True
    for raw, line in zip(entry.splitlines(), unfenced_lines(entry)):
        if line.startswith(("### ", "## [")):
            in_intro = line.startswith("## [")  # each release entry has its own intro
        if in_intro or not line.strip() or raw != line:
            continue
        if not (line.startswith(("#", "- ", "|")) or re.match(r" {2,}\S", line)):
            out.append(f"stray line in the release entry: {line!r}")
    if GUARDRAILS_BULLET not in entry:
        out.append(f"no bullet starting {GUARDRAILS_BULLET!r}")
    return out


def check_changelog(text: str) -> list[str]:
    return check_changelog_items(text) + check_changelog_structure(text)


def test_changelog_items_live():
    """The CHANGELOG names every user-visible change of the plugin release."""
    assert check_changelog_items(_read("CHANGELOG.md")) == []


def test_changelog_structure_live():
    """No CHANGELOG bullet has lost its head (the GUARDRAILS.md item renders as a bullet)."""
    assert check_changelog_structure(_read("CHANGELOG.md")) == []


def test_the_0_21_0_changelog_entry_tells_the_three_plugins_changes_in_user_terms():
    entry = re.search(r"^## \[0\.21\.0\].*?(?=^## \[|\Z)", _read("CHANGELOG.md"), re.M | re.S).group(0)
    flat = " ".join(entry.replace("**", "").split())
    assert len(re.findall(r"^- ", entry, re.M)) == 3
    for needle in ("kill -INT", "as `/quit` does", "trap '' INT", "a third time", "Turn cancelled.",
                   "/plugins on a terminal now opens Plugins", "after one consent", "your conversation is kept",
                   "`localharness plugins` still works"):
        assert needle in flat, needle


def _good_changelog() -> str:
    return (
        "# Changelog\n\n## [Unreleased]\n\nThis release adds plugins.\n\n### Added\n"
        "- **`localharness plugins`.** `list`, `info NAME`, `enable NAME`, `disable NAME`.\n"
        "- **Image generation, as a plugin (off by default).** Settings: `image.comfyui_url`.\n"
        "### Changed\n"
        f"{GUARDRAILS_BULLET} Core reads it\n  on every turn.\n"
        "- Discord reads `dispatch.discord.*`.\n"
        "### Deprecated\n- `org.memory_enabled` — use `memory.enabled`.\n"
        "\n## [0.15.1] — 2026-10-01\n\nstray text in an older entry is not checked\n")


def test_changelog_bites():
    good = _good_changelog()
    assert check_changelog(good) == []
    broken = good.replace(GUARDRAILS_BULLET, " the model with memory on or off.**")
    assert len(check_changelog_structure(broken)) == 2, check_changelog_structure(broken)
    assert check_changelog_items(good.replace("`info NAME`, ", ""))


# ---------------------------------------------------------------- check 8: denylist

BANNED = (  # identifiers and sentences that described wiring the code no longer has
    "memory_loader", "MemoryLoader", "recall_router", "_seed_memory_store", "legacy_handles",
    "LEGACY_CHANNELS", "run_pre_hooks", "run_post_hooks", "PluginConflictError", "PluginVersionError",
    "discover_all", "config_schema", "still part of core", "Nothing calls it",
    "LOCALHARNESS_" + "COMFYUI_",  # built from parts: test_image_hygiene bans the literal in tests/
    "UNTRUSTED_INGEST", "HOST_DANGEROUS", "_MEMORY_TOOLS", "KIND_BY_GROUP",
)
INTERNAL_IDS = (  # internal phase numbers, plan ids and requirement ids
    re.compile(r"\b[Pp]hases?[ -]\(?\d"),
    re.compile(r"\b(?:4[4-9]|5[0-3])-\d{2}\b"),
    re.compile(r"\b(?:PAPI|PLUG|MEMP|ENAB|SAFE|DISP|AUTO|IMGP|WEBP|CORE|DOCS|SEC)-\d\d\b"),
    re.compile(r"\b(?:LIVE|NUDGE)-\d+\b|\bT-2\d-\d+\b"),  # live-gate and threat ids
    re.compile(r"\b2[12]-0\d\b"),  # plan ids of the browser milestone
    re.compile(r"\bSETUP-(?:0[5-9]|1[0-5])\b"),  # the setup-wizard rows; SETUP-01..04 are v1.0's
)


def public_docs() -> list[str]:
    docs = sorted(p.relative_to(_REPO).as_posix() for p in (_REPO / "docs").rglob("*.md"))
    return ["README.md", "SECURITY.md", "CHANGELOG.md", "examples/README.md", "examples/plugin-template/README.md",
            *[d for d in docs if not d.startswith("docs/competitive/")
              and d != "docs/task-intent-and-clarification-prd.md"]]


def check_denylist(rel: str, text: str) -> list[str]:
    """No public doc names removed wiring, an internal phase / plan / requirement id, a Discord
    fallback variable (the fallback was deleted in 0.17.1, so its doc lines went too), or a
    deprecated name outside its deprecation note: `org.memory_enabled` only within 200 characters
    of "deprecated" / "older", `GuardrailTracker` only where the line or its section heading says
    design / not built. The CHANGELOG is history, so it may name all of these, but never an
    internal id."""
    out = []
    lines = text.splitlines()
    heading = ""
    for no, (line, plain) in enumerate(zip(lines, unfenced_lines(text)), 1):
        where = f"{rel}:{no}"
        heading = h[1] if (h := _heading(plain)) else heading
        for rx in INTERNAL_IDS:
            if rx is INTERNAL_IDS[1] and re.search(r"20\d\d-", line):  # a date, not a plan id
                continue
            out += [f"{where}: internal id {m.group(0)!r}" for m in rx.finditer(line)]
        if rel == "CHANGELOG.md":
            continue
        out += [f"{where}: {b!r}" for b in BANNED if b in line]
        if "GuardrailTracker" in line and not re.search(r"design|not built", line + heading, re.I):
            out.append(f"{where}: 'GuardrailTracker' presented as built")
        out += [f"{where}: {env!r} names the deleted Discord env fallback"
                for env in ("LOCALHARNESS_DISCORD_", "DISCORD_BOT_TOKEN") if env in line]
    if rel != "CHANGELOG.md":
        for m in re.finditer(r"org\.memory_enabled", text):
            near = text[max(0, m.start() - 200):m.end() + 200].lower()
            if "deprecat" not in near and "older" not in near:
                out.append(f"{rel}:{text.count(chr(10), 0, m.start()) + 1}: "
                           "'org.memory_enabled' not beside its deprecation")
    return out


@pytest.mark.parametrize("rel", public_docs())
def test_denylist_live(rel):
    """No public doc carries stale wiring names or internal phase/plan/requirement ids."""
    assert check_denylist(rel, _read(rel)) == []


def test_denylist_bites():
    assert check_denylist("README.md", "Dates like 2026-10-02 and 44-17 on a 2026-10-02 line are fine.\n") == []
    for text in ("the Phase-36 model look", "in Phase 4 we", "Phase 47 shipped it", "plan 51-03 did",
                 "memory is still part of core", "MEMP-02 says", "a future phase (36/37)"):
        assert check_denylist("README.md", text + "\n"), text
    assert check_denylist("CHANGELOG.md", "- Removed `run_pre_hooks`.\n") == []
    assert check_denylist("CHANGELOG.md", "- Shipped in Phase 49.\n")
    assert check_denylist("docs/plugins/dispatch.md", "Set `LOCALHARNESS_DISCORD_TOKEN`.\n")
    assert check_denylist("docs/plugins/dispatch.md", "`LOCALHARNESS_DISCORD_TOKEN` (deprecated).\n")
    assert check_denylist("README.md", "- Set\n  `LOCALHARNESS_DISCORD_TOKEN`.\n- It is deprecated.\n")
    assert check_denylist("docs/task-context.md", "## Tracker\n`GuardrailTracker` runs.\n")
    assert check_denylist("docs/task-context.md", "## Tracker (design, not built)\n`GuardrailTracker`.\n") == []
    assert check_denylist("SECURITY.md", "Set `org.memory_enabled` to turn memory off.\n")
    assert check_denylist("README.md", "SETUP-07 says so.\n")
    assert check_denylist("README.md", "SEC-04 says so.\n")
    assert check_denylist("README.md", "as plan 53-02 did\n")
    assert check_denylist("README.md", "**Requirements:** SETUP-01, SETUP-04\n") == []


# ---------------------------------------------------------------- check 9: links

_LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def _heading_anchor_counts(markdown: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in unfenced_lines(markdown):
        for anchor in _github_anchors(line) if _heading(line) else ():
            counts[anchor] = counts.get(anchor, 0) + 1
    return counts


def check_links(rel: str, text: str, read: Callable[[str], str | None] | None = None) -> list[str]:
    """Every relative link and `#anchor` in `rel` resolves (repo blob/tree URLs count as relative):
    the file or folder exists and the heading exists in the target, exactly once (a duplicated
    heading's anchor is ambiguous on GitHub, so it fails loud)."""
    def default_read(target: str) -> str | None:
        path = _REPO / target
        return path.read_text(encoding="utf-8") if path.is_file() and path.suffix == ".md" else None
    read = read or default_read
    out = []
    body = "\n".join(re.sub(r"`[^`\n]*`", "", line) for line in unfenced_lines(text))
    for m in _LINK.finditer(body):
        url = m.group(1)
        for base in (_BLOB, _TREE):
            if url.startswith(base):
                url = url.removeprefix(base)
                target_base = _REPO
                break
        else:
            if re.match(r"[a-z]+:", url):
                continue
            target_base = (_REPO / rel).parent
        path, _, anchor = url.partition("#")
        if path:
            resolved = (target_base / path).resolve()
            if not resolved.exists():
                out.append(f"{rel}: link to missing {url!r}")
                continue
            target = resolved.relative_to(_REPO).as_posix() if resolved.is_relative_to(_REPO) else None
            content = read(target) if target and resolved.is_file() else None
        else:
            target, content = rel, text
        if anchor and content is not None and target and target.endswith(".md"):
            n = _heading_anchor_counts(content).get(anchor, 0)
            if n != 1:
                out.append(f"{rel}: #{anchor} names {'no heading' if n == 0 else f'{n} headings'} in {target}")
    return out


def test_links_live():
    """Every relative link and anchor in the public docs points at a real file and heading."""
    found = [v for rel in public_docs() for v in check_links(rel, _read(rel))]
    assert found == []


def test_links_bites():
    good = "## Here\n\n[a](#here) [b](SECURITY.md#plugins) [c](docs/plugins/) [d](https://example.com)\n"
    assert check_links("README.md", good) == []
    assert check_links("README.md", good.replace("## Here", "## There")) == [
        "README.md: #here names no heading in README.md"]
    assert check_links("README.md", "[x](docs/nope.md)\n") == ["README.md: link to missing 'docs/nope.md'"]
    assert check_links("README.md", "## A\n## A\n[l](#a)\n") == ["README.md: #a names 2 headings in README.md"]
    assert check_links("README.md", f"[s]({_BLOB}docs/plugins/memory.md#no-such-heading)\n")


TASK_FLOW_NEEDLES = (
    "held between a floor and a ceiling fixed in code", "never from org.children_budget",
    "can still hold one turn for the flow cap")


def check_task_flow_doc(security_text: str) -> list[str]:
    """SECURITY.md's machine-level-only list says `task_flow.*` is clamped in code, apart from `org.children_budget`, and what a writable global agent file still costs."""
    body = " ".join((section(security_text, MACHINE_ONLY_HEADING) or "").split())
    return [f"SECURITY.md's machine-level-only list does not say {n}" for n in TASK_FLOW_NEEDLES if n not in body]


def test_task_flow_doc_live():
    assert check_task_flow_doc(_read("SECURITY.md")) == []


def test_task_flow_doc_bites():
    text = _read("SECURITY.md")
    for needle in TASK_FLOW_NEEDLES:
        assert check_task_flow_doc(_swap(text, needle, "x")) == [
            f"SECURITY.md's machine-level-only list does not say {needle}"]
    assert len(check_task_flow_doc("## Nothing\n")) == len(TASK_FLOW_NEEDLES)


FILER_WALK_NEEDLES = (
    "follows no link below `agents/`", "a machine config folder or `agents/` folder that is itself a link keeps working",
    "A task folder is created fresh", "`sessions/jtbd.md`")


def check_filer_walk_doc(security_text: str) -> list[str]:
    """SECURITY.md says the task and session folders are walked without following a link below `agents/`, what still works, and that the session docs are written through that walk."""
    body = " ".join(((section(security_text, "## Trust boundaries") or "") + (section(security_text, "## Files the harness writes") or "")).split())
    return [f"SECURITY.md does not say {n}" for n in FILER_WALK_NEEDLES if n not in body]


def test_filer_walk_doc_live():
    assert check_filer_walk_doc(_read("SECURITY.md")) == []


def test_filer_walk_doc_bites():
    text = _read("SECURITY.md")
    for needle in FILER_WALK_NEEDLES:
        assert check_filer_walk_doc(_swap(text, needle, "x")) == [f"SECURITY.md does not say {needle}"]
    assert len(check_filer_walk_doc("## Nothing\n")) == len(FILER_WALK_NEEDLES)


_PANEL = "src/localharness/orchestrator/task_flow/panel.py"
REPLY_PANEL_NEEDLES = (
    "A clean panel is a quality signal, never a security one",
    "a poisoned file can as easily make a critic answer `NONE` as object",
    f"`{_source_constant(_PANEL, 'FENCE_LABEL')}`", f"`{_source_constant(_PANEL, 'FENCE_CLOSE')}`",
    "the same class as the laundered influence above",
    "the first blocking finding reaches a paired phone's lock screen as the push body",
    "cut to 120 characters and encrypted to the device",
    "one generation with no tools offered, no retry and no compaction",
    "is delivered as text and never run")


def check_reply_panel_doc(security_text: str) -> list[str]:
    """SECURITY.md's prompt-injection section says a clean panel is no security signal, names the fenced findings path with the shipped fence, and the tool-less revision."""
    body = " ".join((section(security_text, "## Threat model: prompt injection") or "").split())
    return [f"SECURITY.md's prompt-injection section does not say {n}" for n in REPLY_PANEL_NEEDLES if n not in body]


def test_reply_panel_doc_live():
    assert check_reply_panel_doc(_read("SECURITY.md")) == []


def test_reply_panel_doc_bites():
    text = _read("SECURITY.md")
    for needle in REPLY_PANEL_NEEDLES:
        assert check_reply_panel_doc(_swap(text, needle, "x")) == [
            f"SECURITY.md's prompt-injection section does not say {needle}"]
    assert len(check_reply_panel_doc("## Nothing\n")) == len(REPLY_PANEL_NEEDLES)


SLEEP_FILE_NEEDLES = ("restamped as an ordinary request", "a wake holds no trusted subtype")


def check_sleep_file_doc(security_text: str) -> list[str]:
    """SECURITY.md says a slept conversation's human-origin messages are restamped as ordinary requests, so a wake holds no trusted subtype."""
    body = " ".join((section(security_text, "## Files the harness writes") or "").split())
    return [f"SECURITY.md's files section does not say {n}" for n in SLEEP_FILE_NEEDLES if n not in body]


def test_sleep_file_doc_live():
    assert check_sleep_file_doc(_read("SECURITY.md")) == []


def test_sleep_file_doc_bites():
    text = _read("SECURITY.md")
    for needle in SLEEP_FILE_NEEDLES:
        assert check_sleep_file_doc(_swap(text, needle, "x")) == [f"SECURITY.md's files section does not say {needle}"]
    assert len(check_sleep_file_doc("## Nothing\n")) == len(SLEEP_FILE_NEEDLES)


TYPED_TURN_NEEDLES = (
    "Four places mark a turn as the person's `typed` words",
    "a line typed at the terminal or sent from the phone",
    "an ACP prompt that is not a command",
    "the objective of `/task new <objective>` and the change of `/task plan <change>`",
    "a missed caller demotes a turn rather than promotes it",
    "Only the person's `/task new` and `/task plan` give the model one more call on their stage")


def check_typed_turn_doc(security_text: str) -> list[str]:
    """SECURITY.md's prompt-injection section names the four places a turn is the person's `typed` words, says a missed caller demotes, and says which verbs give the model one more call."""
    body = " ".join((section(security_text, "## Threat model: prompt injection") or "").split())
    return [f"SECURITY.md's prompt-injection section does not say {n}" for n in TYPED_TURN_NEEDLES if n not in body]


def test_typed_turn_doc_live():
    assert check_typed_turn_doc(_read("SECURITY.md")) == []


def test_typed_turn_doc_bites():
    text = _read("SECURITY.md")
    for needle in TYPED_TURN_NEEDLES:
        assert check_typed_turn_doc(_swap(text, needle, "x")) == [f"SECURITY.md's prompt-injection section does not say {needle}"]
    assert len(check_typed_turn_doc("## Nothing\n")) == len(TYPED_TURN_NEEDLES)


STOP_NEEDLES = (
    "`/task stop` ends the run after the running unit",
    "Ctrl+C, the phone's stop and ACP's cancel end the running unit at once and can leave its files half-edited",
    "nothing reverts what the unit wrote")


def check_stop_doc(security_text: str) -> list[str]:
    """SECURITY.md's prompt-injection section says `/task stop` ends the run after the unit and a hard cancel can leave files half-edited."""
    body = " ".join((section(security_text, "## Threat model: prompt injection") or "").split())
    return [f"SECURITY.md's prompt-injection section does not say {n}" for n in STOP_NEEDLES if n not in body]


def test_stop_doc_live():
    assert check_stop_doc(_read("SECURITY.md")) == []


def test_stop_doc_bites():
    text = _read("SECURITY.md")
    for needle in STOP_NEEDLES:
        assert check_stop_doc(_swap(text, needle, "x")) == [f"SECURITY.md's prompt-injection section does not say {needle}"]
    assert len(check_stop_doc("## Nothing\n")) == len(STOP_NEEDLES)

TASK_DOCS_NEEDLES = (
    "`/task show` prints a task doc only after its digest matches what was filed",
    "A folder this session did not make is shown read-only, headed `not adopted · not made by this session`",
    "The header `/task accept` prints first is display, not a confirmation",
    "`/task clear` removes only the open task's folder and an old `task.json`")


def check_task_docs_doc(security_text: str) -> list[str]:
    """SECURITY.md's file-writes section says how `/task show`, `/task accept` and `/task clear` treat a task doc and a folder."""
    body = " ".join((section(security_text, "## Files the harness writes") or "").split())
    return [f"SECURITY.md's file-writes section does not say {n}" for n in TASK_DOCS_NEEDLES if n not in body]


def test_task_docs_doc_live():
    assert check_task_docs_doc(_read("SECURITY.md")) == []


def test_task_docs_doc_bites():
    text = _read("SECURITY.md")
    for needle in TASK_DOCS_NEEDLES:
        assert check_task_docs_doc(_swap(text, needle, "x")) == [f"SECURITY.md's file-writes section does not say {needle}"]
    assert len(check_task_docs_doc("## Nothing\n")) == len(TASK_DOCS_NEEDLES)


TASK_RUN_NEEDLES = (
    ("## Threat model: prompt injection", "The `task` tool declares its results untrusted, a label"),
    ("## Threat model: prompt injection",
     "`research.md`, `plan.md`, `decisions.md`, the step logs and a research `question` are written by the model from repository files"),
    ("## Threat model: prompt injection",
     "An executor's summarizer reads its own tool output into a harness-origin message, new on a child agent that holds host tools, the same class as the root's exposure"),
    ("## Files the harness writes",
     "A wake begins with no task: the sleep file carries no task record, and an unfinished task folder is named, never resumed"),
    ("## Human approval gate",
     "The session's state folder, a project's `.localharness/` when it has one, is on that list from start"),
    ("## Threat model: prompt injection", "No person's go-ahead stands between a plan and its steps"),
    ("## Threat model: prompt injection", "so an executor's ordinary repository writes and commands run unasked"),
    ("## Threat model: prompt injection", "The root already held the same tools, so no capability is new"))


def check_task_run_doc(security_text: str) -> list[str]:
    """SECURITY.md says, each in its section, what the task pipeline adds and leaves to the person: the label, the paths, the gone go-ahead, a wake with no task, and the gate's list."""
    bodies = {h: " ".join((section(security_text, h) or "").split()) for h, _ in TASK_RUN_NEEDLES}
    return [f"SECURITY.md's {h} section does not say {n}" for h, n in TASK_RUN_NEEDLES if n not in bodies[h]]


def test_task_run_doc_live():
    assert check_task_run_doc(_read("SECURITY.md")) == []


def test_task_run_doc_bites():
    text = _read("SECURITY.md")
    for heading, needle in TASK_RUN_NEEDLES:
        assert check_task_run_doc(_swap(text, needle, "x")) == [f"SECURITY.md's {heading} section does not say {needle}"]
    assert len(check_task_run_doc("## Nothing\n")) == len(TASK_RUN_NEEDLES)


PHONE_TASK_NEEDLES = (
    ("### A paired phone or Discord account", "A paired phone can also type `/task new <objective>`"),
    ("### A paired phone or Discord account", "outside the `remote_unattended` lock"),
    ("### A paired phone or Discord account", "ACP's `/task run` and `/task accept` are the local operator's"),
    ("### `localharness mobile`", "closes both paths, though not `/task new`, `run` or `accept`"))


def check_phone_task_doc(security_text: str) -> list[str]:
    """SECURITY.md says, each in its section, that a paired phone's `/task new`, `run` and `accept` sit outside the `remote_unattended` lock and that ACP's are the local operator's."""
    bodies = {h: " ".join((section(security_text, h) or "").split()) for h, _ in PHONE_TASK_NEEDLES}
    return [f"SECURITY.md's {h} section does not say {n}" for h, n in PHONE_TASK_NEEDLES if n not in bodies[h]]


def test_phone_task_doc_live():
    assert check_phone_task_doc(_read("SECURITY.md")) == []


def test_phone_task_doc_bites():
    text = _read("SECURITY.md")
    for heading, needle in PHONE_TASK_NEEDLES:
        assert check_phone_task_doc(_swap(text, needle, "x")) == [f"SECURITY.md's {heading} section does not say {needle}"]
    assert len(check_phone_task_doc("## Nothing\n")) == len(PHONE_TASK_NEEDLES)


STUCK_TASK_NEEDLES = (
    "The `task` tool's results are declared untrusted, so its manifest counts as progress only through its last line, which the harness computes from the task record",
    "repeated `task` calls are bounded by the flow cap, `/task stop` and the action budget rather than by the identical-call ceiling")


def check_stuck_task_doc(security_text: str) -> list[str]:
    """SECURITY.md's prompt-injection section says a `task` result counts as progress only through the manifest's last line and what bounds repeated calls."""
    body = " ".join((section(security_text, "## Threat model: prompt injection") or "").split())
    return [f"SECURITY.md's prompt-injection section does not say {n}" for n in STUCK_TASK_NEEDLES if n not in body]


def test_stuck_task_doc_live():
    assert check_stuck_task_doc(_read("SECURITY.md")) == []


def test_stuck_task_doc_bites():
    text = _read("SECURITY.md")
    for needle in STUCK_TASK_NEEDLES:
        assert check_stuck_task_doc(_swap(text, needle, "x")) == [f"SECURITY.md's prompt-injection section does not say {needle}"]
    assert len(check_stuck_task_doc("## Nothing\n")) == len(STUCK_TASK_NEEDLES)


_ROOT_PARAGRAPH = (
    "Answer ordinary questions directly, and make a single edit or lookup directly: read first, then change. "
    "When the work is bigger than that, run it through the task: write what you need to know and what you will read "
    "before you read it, plan the steps before you change anything, and let the critics judge the plan and each "
    "result, calling the task again after each result until it says done, blocked or halted. Follow the person's "
    "latest request and corrections and stop where they asked. Say plainly what is done, what the critics still "
    "flag, and what you do not know.")
_SELF_CHECK_LINE = "agent.self_check dropped from config; the critics replace it"
_GONE_SPELLINGS = ("_SELF_CHECK" + "_NUDGE", "SelfCheck" + "Config", "Answer ordinary questions directly. " + "For substantive work")


def _joined(source: str) -> str:
    return re.sub(r'"\s*\n\s*"', "", source)


def check_self_check_gone(src_tree_text: str, loop_text: str, migrate_line: str) -> list[str]:
    """No source spells the deleted self-review nudge, config class or old root paragraph; the root prompt holds the task paragraph; migrate's dropped-setting line is the spec's."""
    src = _joined(src_tree_text)
    out = [f"src/localharness still spells {s}" for s in _GONE_SPELLINGS if s in src]
    out += [] if _ROOT_PARAGRAPH in _joined(loop_text) else ["agent/loop.py does not hold the root paragraph"]
    return out + ([] if migrate_line == _SELF_CHECK_LINE else ["config/migrate.py's SELF_CHECK_DROPPED differs from the spec's line"])


def _self_check_args() -> tuple[str, str, str]:
    return ("\n".join(p.read_text(encoding="utf-8") for p in sorted(SRC.rglob("*.py"))),
            _read("src/localharness/agent/loop.py"), _source_constant("src/localharness/config/migrate.py", "SELF_CHECK_DROPPED"))


def test_self_check_gone_live():
    assert check_self_check_gone(*_self_check_args()) == []


def test_self_check_gone_bites():
    tree, loop, line = _self_check_args()
    for planted in _GONE_SPELLINGS:
        assert check_self_check_gone(tree + f'\nX = "{planted}"\n', loop, line) == [f"src/localharness still spells {planted}"]
    assert check_self_check_gone(tree, _swap(loop, "let the critics judge the plan and each result", "x"), line) == [
        "agent/loop.py does not hold the root paragraph"]
    assert check_self_check_gone(tree, loop, line + " ") == ["config/migrate.py's SELF_CHECK_DROPPED differs from the spec's line"]
