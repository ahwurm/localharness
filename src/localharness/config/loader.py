"""ConfigLoader: YAML parse, validate, inheritance resolve, write."""
from __future__ import annotations

import json
import logging
import math
import os
import re
import reprlib as repr_lib
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, Callable, Collection, Mapping, Optional, get_args

import yaml
from pydantic import BaseModel, SecretStr, ValidationError
from pydantic_yaml import to_yaml_str

from localharness.agent.gate_types import DEFAULT_MODE, MODE_STRICTNESS, GateSettings

from .models import (
    ASK_TO_GATE_FIELD,
    LEGACY_MODE_ALIASES,
    AgentConfig,
    BudgetConfig,
    ChildBudget,
    ChildrenBudgetConfig,
    ContextConfig,
    DivisionConfig,
    HarnessConfig,
    OrgConfig,
    PermissionConfig,
)
from localharness.config.overlay import (
    deep_merge,
    load_overlay,
    restrict_config_file,
    _resolve_user_overlay_path,
)
from localharness.config.paths import resolve_config_dir
from localharness.config.redact import (
    SECRET_MASK, at_secret, scrub, secret_values, yaml_problem, yaml_where,
)
from localharness.config.plugin_sections import (
    CORE_AGENT_KEYS, CORE_HARNESS_KEYS, split_plugin_keys, unowned_hint,
)

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Error hierarchy (spec 06 section 6.2)
# ------------------------------------------------------------------ #

class ConfigError(Exception):
    """Base class for all configuration errors."""


class ConfigParseError(ConfigError):
    """YAML is malformed."""
    def __init__(self, path: str, line: int, column: int, message: str) -> None:
        self.path = path
        self.line = line
        self.column = column
        self.message = message
        super().__init__(f"{path}:{line}:{column}: {message}")


# How much of a rejected value an error message may show. Long enough to recognize what you typed,
# short enough that rendering it is never the expensive part of failing. The bound is on the
# MESSAGE, not on the value: `_short_repr` never builds the full repr in the first place.
VALUE_REPR_LIMIT = 500

_BOUNDED_REPR = repr_lib.Repr()
_BOUNDED_REPR.maxlevel = 4
_BOUNDED_REPR.maxstring = 120
_BOUNDED_REPR.maxother = 120
_BOUNDED_REPR.maxlist = _BOUNDED_REPR.maxtuple = _BOUNDED_REPR.maxset = 8
_BOUNDED_REPR.maxdict = 8


class _MaskingRepr(repr_lib.Repr):
    """_BOUNDED_REPR's limits, with every secret value shown as SECRET_MASK (R16). It masks while
    the bounded repr walks, so nothing is copied and an amplified value stays cheap."""

    def __init__(self, secrets: frozenset[str]) -> None:
        super().__init__()
        self.__dict__.update(vars(_BOUNDED_REPR))
        self.secrets = secrets

    def repr_str(self, x: str, level: int) -> str:
        return repr(SECRET_MASK) if x in self.secrets else super().repr_str(x, level)

    def repr_int(self, x: int, level: int) -> str:
        return repr(SECRET_MASK) if repr(x) in self.secrets else super().repr_int(x, level)

    def repr_float(self, x: float, level: int) -> str:
        return repr(SECRET_MASK) if repr(x) in self.secrets else repr(x)


def _short_repr(value: Any, limit: int = VALUE_REPR_LIMIT,
                secrets: frozenset[str] = frozenset()) -> str:
    """A bounded, honest repr of the value a config field rejected.

    `repr()` on a rejected value is unbounded work on attacker-shaped input: a YAML file using
    alias amplification (`*a` referenced nine times, five levels deep) parses in milliseconds and
    reprs to gigabytes — measured at 6GB and 41 seconds before this, from a file under 1KB, in the
    command a user runs BECAUSE something is already wrong.

    `reprlib` is what keeps the work bounded: it walks a limited number of elements to a limited
    depth and never materializes the whole structure. The size we report alongside is one that is
    cheap to know — a string's length, a container's top-level item count — never a byte count of
    a repr we deliberately did not build.

    `secrets`: values shown as SECRET_MASK wherever they appear, the value itself included (R16).
    """
    if secrets and not isinstance(value, bool) and (
            (isinstance(value, str) and value in secrets)
            or (isinstance(value, (int, float)) and repr(value) in secrets)):
        return repr(SECRET_MASK)
    if value is None or isinstance(value, (bool, int, float)):
        return repr(value)
    if isinstance(value, str):
        if len(value) <= limit:
            return repr(value)
        return f"{value[:limit]!r} … (truncated, {len(value)} chars)"
    text = (_MaskingRepr(secrets) if secrets else _BOUNDED_REPR).repr(value)
    if isinstance(value, (list, tuple, set, frozenset, dict)) and "..." in text:
        return f"{text[:limit]} (truncated, {len(value)} top-level items)"
    return text[:limit]


class ConfigFieldError:
    """One validation failure for one field. Its rejected value is printed only as shown_value()
    gives it (R16): `secrets` are the secret values the validated data held, `secret` says the
    error is on or inside a secret field, `withheld` that no schema says what in it is secret."""
    def __init__(
        self,
        field_path: str,
        value: Any,
        message: str,
        yaml_line: Optional[int] = None,
        source_path: Optional[str] = None,
        *,
        secrets: frozenset[str] = frozenset(),
        secret: bool = False,
        withheld: bool = False,
    ) -> None:
        self.field_path = field_path
        self.value = value
        self.message = message
        self.yaml_line = yaml_line
        # The file that actually SET this field, when it is not the file the surrounding
        # ConfigValidationError is headed with. None whenever the two agree — which keeps every
        # pre-43 message (agent yamls, org.yaml, a global-only harness error) byte-identical.
        self.source_path = source_path
        self.secrets, self.secret, self.withheld = secrets, secret, withheld

    def shown_value(self) -> Optional[str]:
        """The rejected value as it may be printed: the mask on or inside a secret field; None when
        withheld (a key that is not a setting, or a section no installed plugin owns — nothing says
        which part of it is secret); else the bounded repr with every secret value masked."""
        if self.withheld:
            return None
        if self.secret:
            return repr(SECRET_MASK)
        return _short_repr(self.value, secrets=self.secrets)

    def __str__(self) -> str:
        loc = f" (line {self.yaml_line})" if self.yaml_line else ""
        origin = f"{self.source_path}: " if self.source_path else ""
        shown = self.shown_value()
        return f"{origin}{self.field_path}{loc}: {self.message}" + (
            f" (got: {shown})" if shown is not None else "")


class ConfigValidationError(ConfigError):
    """One or more Pydantic validation failures."""
    def __init__(self, path: str, errors: list[ConfigFieldError]) -> None:
        self.path = path
        self.errors = errors
        lines = [f"{path}:"] + [f"  {e}" for e in errors]
        super().__init__("\n".join(lines))


class ConfigNotFoundError(ConfigError):
    """Agent or division config file not found."""
    def __init__(self, name: str, searched_paths: list[str]) -> None:
        self.name = name
        self.searched_paths = searched_paths
        paths_str = ", ".join(searched_paths)
        super().__init__(f"Config for {name!r} not found. Searched: {paths_str}")


class ConfigReferenceError(ConfigError):
    """A config field references something that doesn't exist."""
    def __init__(self, path: str, field: str, ref: str, message: str) -> None:
        self.path = path
        self.field = field
        self.ref = ref
        super().__init__(f"{path}: field '{field}' references missing {ref!r}: {message}")


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _build_line_map(yaml_text: str) -> dict[str, int]:
    """Return mapping from dot-notation field paths to 1-based line numbers."""
    line_map: dict[str, int] = {}
    indent_stack: list[tuple[int, str]] = []  # (indent, key)

    for lineno, raw_line in enumerate(yaml_text.splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        match = re.match(r"^(\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:", raw_line)
        if not match:
            continue
        indent = len(match.group(1))
        key = match.group(2)
        # Pop stack to find parent at strictly lower indent
        while indent_stack and indent_stack[-1][0] >= indent:
            indent_stack.pop()
        if indent_stack:
            path = f"{indent_stack[-1][1]}.{key}"
        else:
            path = key
        line_map[path] = lineno
        indent_stack.append((indent, path))

    return line_map


def _file_line_map(path: Path) -> dict[str, int]:
    """_build_line_map of a file's text; {} when it cannot be read."""
    try:
        return _build_line_map(path.read_text(encoding="utf-8"))
    except OSError:
        return {}


def _dotpath_in(raw: object, dotpath: str) -> bool:
    """True iff `raw` is a nested dict carrying this dot-path as a key chain."""
    cur: Any = raw
    for part in dotpath.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False
        cur = cur[part]
    return True


def _owning_source(
    loc: str,
    sources: list[tuple[Path, dict]],
) -> Optional[tuple[Path, dict]]:
    """The HIGHEST-priority source that actually sets `loc`, or None.

    `sources` arrives highest-priority-FIRST — the reverse of _raw_config_sources()'s ruled
    order — because the winner of the merge is the file whose value Pydantic actually rejected.

    A Pydantic `loc` can end in a list index or a discriminator that is not a YAML key
    (`provider.available_models.0`). Progressively dropping the last segment attributes such an
    error to its owning BLOCK rather than dropping to the global fallback. A model-level validator
    has an EMPTY loc — no file sets it, because the error is about a relationship between values
    rather than about one line — and that falls through to None on purpose.
    """
    parts = loc.split(".")
    while parts:
        probe = ".".join(parts)
        for path, raw in sources:
            if _dotpath_in(raw, probe):
                return path, raw
        parts.pop()
    return None


def _load_yaml_file(path: Path) -> dict:
    """Read file, safe_load, return dict (empty dict if file is empty/None)."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ConfigNotFoundError(str(path), [str(path)]) from e
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        # file, line and column, never the parser's snippet of the line (R16); not chained either
        raise _parse_error(path, e) from None
    return data or {}


def _parse_error(path: Any, exc: BaseException) -> ConfigParseError:
    """A YAML parse error as ConfigParseError: the file, line and column, and the parser's problem
    text, never the snippet and caret it prints, which can show a key on the broken line (R16)."""
    line, column = yaml_where(exc)
    return ConfigParseError(str(path), line, column, yaml_problem(exc))


def _field_error(field_path: str, err: Any, message: str, yaml_line: Optional[int], *,
                 model: Any, secrets: frozenset[str]) -> ConfigFieldError:
    """One pydantic error as a ConfigFieldError that is safe to print (R16): its message scrubbed of
    `secrets` (what the data `model` validated holds at secret fields), its value masked on or
    inside a secret field, and withheld for a key the model does not have."""
    return ConfigFieldError(field_path, err.get("input"), scrub(message, secrets), yaml_line,
                            secrets=secrets, secret=at_secret(model, err["loc"]),
                            withheld=err["type"] == "extra_forbidden")


def _pydantic_error_to_field_errors(
    exc: ValidationError,
    path: str,
    line_map: dict[str, int],
    model: Any = None,
    data: Any = None,
) -> list[ConfigFieldError]:
    secrets = secret_values(model, data)
    errors: list[ConfigFieldError] = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"])
        errors.append(_field_error(loc, err, err["msg"], line_map.get(loc), model=model, secrets=secrets))
    return errors


def _resolve_scalar(
    field: str,
    agent_val: Any,
    division_val: Any,
    org_val: Any,
    default: Any,
) -> Any:
    """Return most specific non-inherit/non-None value, or default."""
    if agent_val not in (None, "inherit"):
        return agent_val
    if division_val not in (None, "inherit"):
        return division_val
    if org_val not in (None, "inherit"):
        return org_val
    return default


def _overlay_default(overlay: dict, field: str, default: Any) -> Any:
    """The `agent:` overlay's value for `field`, or the shipped default when it says nothing.

    `None`/`"inherit"` in the overlay mean "not set" for the same reason they do everywhere else in
    `_resolve_scalar`: they are the harness's two ways of spelling "look further up".
    """
    value = overlay.get(field)
    return default if value in (None, "inherit") else value


def _normalize_mode(value: Any) -> Optional[str]:
    """A raw `permissions.mode` as one of the v0.14 session modes, or None if it is neither.

    The narrow-only union runs on RAW yaml, before `PermissionConfig` validates, so it meets the
    legacy `auto`/`manual` spellings that models.py maps to `guarded` (PRD §3.4). Comparing
    strictness on the mapped value is what stops a workspace `mode: auto` from reading as an
    unknown mode and slipping through unchecked. An unrecognized value returns None and is left
    for Pydantic to reject with a proper field error.
    """
    if not isinstance(value, str):
        return None
    mapped = LEGACY_MODE_ALIASES.get(value, value)
    return mapped if mapped in MODE_STRICTNESS else None


# ------------------------------------------------------------------ #
# PRD §3.3: the `permissions.ask.*` half of the narrow-only union
# ------------------------------------------------------------------ #
#
# The v0.14 spine critic found the hole these two sets close: `_narrow_project_layer_permissions`
# narrowed `mode` and `workspace_root`, while EVERY `permissions.ask.*` key a project layer wrote
# passed straight through the merge into `AskConfig.to_gate_settings()`. A cloned repo whose
# `.localharness/agents/<name>.yaml` said
#   permissions: {ask: {destructive_signatures: [], protected_paths_home: [],
#                       mcp_trusted_servers: ["evil"]}}
# emptied the UNGRANTABLE tier and pre-trusted an MCP server without a single prompt — the exact
# thing PRD §3.3 says a repo cannot do: it can only TIGHTEN.
#
# Every `AskConfig` field belongs to exactly one of the sets below, except `network_hosts`, which
# is handled on its own because it is a bool: a project layer may switch the ask ON, never off.
# A rule set added to `AskConfig` without a classification here is simply left alone by the
# narrowing — add it to one of the two sets when you add the field.

ASK_TIGHTEN_ONLY_FIELDS: frozenset[str] = frozenset({
    "destructive_signatures",
    "protected_paths_home",
    "protected_paths_workspace",
    "protected_paths_system",
    "protected_config_dir_entries",
    "write_shaped_commands",
    "payload_commands",
    "pipe_to_shell_sources",
    "pipe_to_shell_sinks",
    "interpreter_commands",
    "inline_by_nature",
    "source_commands",
    "git_config_dangerous_keys",
})
"""The `permissions.ask` rule sets a project layer may ADD to (PRD §3.3).

Every one of them names calls that ASK or REFUSE, so a longer list is a stricter session and a
shorter one is a hole. The project layer's value is therefore unioned onto the global layer's
(or onto the shipped `GateSettings` default when the global layer is silent) — a repo that knows
its own dangerous command gets to say so, and a repo that deletes `rm -rf` from the ungrantable
class changes nothing.
"""

ASK_GLOBAL_ONLY_FIELDS: frozenset[str] = frozenset({
    "read_only_signatures",
    "dropped_commands",
    "wrapper_commands",
    "subcommand_tools",
    "mcp_trusted_servers",
    "timeout_s",
})
"""The `permissions.ask` keys a project layer may not set at all (PRD §3.3).

There is no tightening direction for these: ADDING to any of them loosens the gate. A signature
added to `read_only_signatures` ALLOWS outright; a command added to `dropped_commands`,
`wrapper_commands` or `subcommand_tools` changes what a segment signs as and so what a grant key
means; a name in `mcp_trusted_servers` is a grant the operator writes themselves; a longer
`timeout_s` moves the deadline a human answers inside. A project-layer value is dropped back to
the global layer's, with a warning naming the key.
"""

def _drops(value: Any, shipped: Any) -> list[str]:
    """The shipped entries `value` leaves out, sorted (a shipped set's own order varies by run)."""
    return sorted(str(d) for d in shipped or () if not isinstance(value, list) or d not in value)


def permission_loosenings(perms: Any) -> list[tuple[str, str]]:
    """The keys of one `permissions` mapping that are LOOSER than the shipped defaults, as sorted
    (dotted key, shown value) pairs — what a machine agent, division or org file must have
    confirmed before it applies (orchestrator ruling R12). Tightenings are never listed:
    `mode` counts only when less strict than DEFAULT_MODE (MODE_STRICTNESS); `workspace_root`
    counts whenever set (an explicit root can widen the leash); `ask.network_hosts` never (on is
    tighter, off is the default); an `ask.<f>` in ASK_TIGHTEN_ONLY_FIELDS counts when it drops an
    entry of the shipped GateSettings default (shown "drops a, b"); an `ask.<f>` in
    ASK_GLOBAL_ONLY_FIELDS and `budget.kill_file` count whenever they differ from the shipped
    default; `deny_patterns` never (the org rung's list is unioned into every agent at resolution,
    `load_agent_file` step 5, so a shorter list in an agent or division file removes nothing — the
    legacy org.yaml, which IS that rung, is `org_deny_loosenings`'s); budgets and
    `defaults_revision` never. Values are shown as written (a list as compact JSON) — none of these
    is a secret."""
    if not isinstance(perms, dict):
        return []

    def shown(value: Any) -> str:
        return value if isinstance(value, str) else json.dumps(value, separators=(",", ":"),
                                                               default=str)

    out: list[tuple[str, str]] = []
    mode = _normalize_mode(perms.get("mode"))
    if mode is not None and MODE_STRICTNESS[mode] < MODE_STRICTNESS[DEFAULT_MODE]:
        out.append(("permissions.mode", shown(perms["mode"])))
    if perms.get("workspace_root") is not None:
        out.append(("permissions.workspace_root", shown(perms["workspace_root"])))
    ask = perms.get("ask")
    for field, value in ask.items() if isinstance(ask, dict) else ():
        if value is None:
            continue
        shipped = _gate_default(field)
        if field in ASK_TIGHTEN_ONLY_FIELDS and (dropped := _drops(value, shipped)):
            out.append((f"permissions.ask.{field}", "drops " + ", ".join(dropped)))
        elif field in ASK_GLOBAL_ONLY_FIELDS and not (
                set(value) == set(shipped or ()) if isinstance(value, list) else value == shipped):
            out.append((f"permissions.ask.{field}", shown(value)))
    budget = perms.get("budget")
    if isinstance(budget, dict) and "kill_file" in budget and (
            budget["kill_file"] != BudgetConfig.model_fields["kill_file"].default):
        out.append(("permissions.budget.kill_file", shown(budget["kill_file"])))
    return sorted(out)


def org_deny_loosenings(perms: Any) -> list[tuple[str, str]]:
    """The legacy org.yaml's `deny_patterns` when it leaves out a shipped pattern. That file is the
    base rung the deny union starts from (`load_org` → step 5), so unlike an agent or division
    file a shorter list there does remove patterns — measured: `deny_patterns: []` dropped all of
    them on a config.yaml carrying no deny list of its own."""
    deny = perms.get("deny_patterns") if isinstance(perms, dict) else None
    dropped = _drops(deny, PermissionConfig().deny_patterns) if deny is not None else []
    return [("permissions.deny_patterns", "drops " + ", ".join(dropped))] if dropped else []


HARNESS_GLOBAL_ONLY_FIELDS: frozenset[str] = frozenset({
    "provider.base_url", "provider.api_key", "extra_endpoints", "active_endpoint", "server",
    "org.audit_log_path", "org.hooks", "org.enforce_capability_floor",
    "org.web_fetch_allow_private", "channels.remote_unattended", "channels.unattended_max_actions",
    "autosense", "effort_budget", "org.default_effort",
    "proposer.base_url", "proposer.api_key"})
"""The core harness keys only the global config (config.yaml or overrides.yaml) may set.

A project's `.localharness/` loads with no prompt when you stand in the project, so it may set only
what the agent does there. These keys say where a request or a credential goes (the provider's and
the proposer's address and key, the peer endpoints and the recorded active endpoint), what the
harness launches (`server`, the whole section: executable, image, arguments, address), a file it
writes anywhere on disk (`org.audit_log_path`), free-form settings a hook plugin may read a URL or a
command from (`org.hooks`), and the protections' own switches (`org.enforce_capability_floor`,
`org.web_fetch_allow_private`, `channels.remote_unattended`) and the action ceiling a remote
channel's unattended session runs under (`channels.unattended_max_actions`: dropped, never merely
lowered, so only the machine moves it). An address alone is enough to leak:
every message and every file the agent reads travels to it, key or no key. A member may be a whole
section or list (`server`, `extra_endpoints`); readers test a dot-path with
`is_harness_global_only`, which also matches the keys inside a member section.

`autosense`, `effort_budget` and `org.default_effort` decide how the harness sizes and bounds what
it serves on this machine's hardware (memory margins, model folders, the reasoning share of every
reply), so a project may not move them. A whole-section workspace value is warned even when it
restates the defaults, because the comparison is against the section's default object.

`org.enforce_capability_floor` is the capability floor's off switch. No direction of it belongs
to a repository: a cloned repo must not be able to switch the floor off for the sessions started
inside it, and switching it back on over the machine owner's off is the owner's call too. So, as
for `ASK_GLOBAL_ONLY_FIELDS`, a workspace value that differs from the global one (or from the
field's default, when the global layer is silent) is dropped with one warning naming the key and
the file; a workspace value equal to it is the operator's own and is no warning.

`proposer.base_url` and `proposer.api_key` are a network destination and the credential sent to it
(`plugins enable autoresearch` checks the proposer with the key): a repository that could set the
address would receive the key, and one that could set the key could have the proposer's traffic
billed to and logged by an account of its own. They are the machine's, as every bundled plugin's
endpoints and credentials are. `proposer.base_url` has no default, so when the global layers set no
address the workspace's `proposer:` section cannot stand without one: it gives way whole to the
global layers' section, or to none. `proposer.model` and the rest of the section stay project
settings.

`provider.base_url` has no default either, but a session cannot run without a provider: a machine
whose global layers set no `provider:` at all, in a project that supplies one, is refused with the
one line NO_MACHINE_PROVIDER rather than run against the project's address.
"""

NO_MACHINE_PROVIDER = (
    "no model server is set on this machine: set provider.base_url in {global_file} (the value in "
    "{ws_file} is ignored — a project may not choose where requests go)")


def is_harness_global_only(path: str) -> bool:
    """Is this harness dot-path machine-level only — a member, or a key inside a member section?"""
    return any(path == f or path.startswith(f + ".") for f in HARNESS_GLOBAL_ONLY_FIELDS)


AGENT_GLOBAL_ONLY_FIELDS: frozenset[str] = frozenset({
    "memory.embedding_model", "permissions.budget.kill_file"})
"""Agent-file keys only the global layer decides. `permissions.budget.kill_file` has its own rule at
the budget resolution (the kill switch is one machine-wide file); a key under a plugin's agent
section (`memory.embedding_model`: the model sentence-transformers loads, and a local folder it
imports Python from) is re-resolved by `_narrow_agent_global_only`. Never marked GLOBAL_ONLY on the
plugin's AgentConfigModel: the resolver refuses such a plugin (plugins/resolve.py)."""

AGENT_CORE_GLOBAL_ONLY_SECTIONS: frozenset[str] = frozenset({"stuck_detector"})
"""Core agent-file SECTIONS only the global layer may set. A workspace agent file (and a workspace
overlay of a built-in) gets such a section restored WHOLE from the global layer: the global
`agents/<name>.yaml` section, else the global overrides' `agent:` section, else the schema default.
Never key by key — a workspace file that merely OMITS a key would fall to the schema default and
lift a stricter global value. Distinct from AGENT_GLOBAL_ONLY_FIELDS, which names dotted keys under
plugin sections (and the kill switch) and is a label read by the docs gate."""

AGENT_CORE_LOWER_ONLY_KEYS: frozenset[str] = frozenset({
    "permissions.budget.max_actions", "permissions.budget.max_duration_minutes"})
"""Core agent-file KEYS a workspace layer may only LOWER. Against the global layer's own value for the
agent (its agents/<name>.yaml, the division, the legacy org.yaml, the overrides' `agent:` section) a
workspace agent file, a workspace division file that shadows the global one, or a workspace overlay
of a built-in keeps a value only when it is at or below that baseline; where the global layer says
nothing the workspace's number is kept as written (and marked, so the seam can cap it at use). The
sibling of AGENT_CORE_GLOBAL_ONLY_SECTIONS, which restores whole sections; a later phase adds its
`context.*` window-ratio paths here."""


def layer_files(layer_dir: Path, subdir: str, *, contained: bool = False) -> list[Path]:
    """The `<layer_dir>/<subdir>/*.yaml` files of ONE config layer, sorted — the one enumeration
    the roster (`ConfigLoader.agent_yaml_paths`, `list_divisions`) and the trust fingerprints
    (config/trust.py) share, so what is fingerprinted is exactly what can load.

    `contained` (a PROJECT layer): a file that leads outside `layer_dir` — a symlink to elsewhere,
    or a file in a `subdir` folder that is itself such a symlink — is left out. A repository can
    ship `.localharness/agents/orchestrator.yaml -> ../../docs/orch.yaml` (git keeps symlinks); a
    project's agent files must be files inside its `.localharness/`."""
    folder = Path(layer_dir) / subdir
    files = sorted(folder.glob("*.yaml")) if folder.exists() else []
    return [f for f in files if not leaves_layer(f, layer_dir)] if contained else files


def in_layer(path: Path, layer_dir: Path) -> bool:
    """Was `path` found in this layer? Decided by WHERE THE FILE SITS — the path as found,
    normalised but with no symlink followed — never by where it resolves: a project's
    `.localharness/agents/orchestrator.yaml` is the project's file even when it links to
    `docs/orch.yaml`. Resolving first let such a file skip every project rule (the untrusted-project
    MCP strip, the machine-only embedding model), while the trust question still listed it."""
    return Path(os.path.abspath(path)).is_relative_to(os.path.abspath(layer_dir))


def leaves_layer(path: Path, layer_dir: Path) -> bool:
    """Does `path` — the file itself, or a folder on the way to it — link to somewhere outside
    `layer_dir`? Both sides resolved, so a link that stays inside the layer is fine; a path that
    cannot be resolved counts as leaving."""
    try:
        return not Path(path).resolve().is_relative_to(Path(layer_dir).resolve())
    except (OSError, RuntimeError):
        return True

_UNSET = object()


def _section_tried(section: str, declared: Any, restored: Any) -> bool:
    """Did a workspace file's `section` ask for anything the restored one does not hold? The one
    predicate both doors note by (IN-01): the declared section merged over the restored one,
    validated, against the restored one validated. A key equal to what is restored (a schema
    default under a silent global included) changes nothing and earns no note; a different value,
    an unknown key, or a section that is not a mapping does."""
    if declared is _UNSET:
        return False
    if not isinstance(declared, dict):
        return True
    model = AgentConfig.model_fields[section].annotation
    base = restored if isinstance(restored, dict) else {}
    try:
        return model.model_validate(deep_merge(base, declared)) != model.model_validate(base)
    except ValidationError:
        return True


def _dig(data: Any, parts: list[str]) -> Any:
    for part in parts:
        if not isinstance(data, dict) or part not in data:
            return _UNSET
        data = data[part]
    return data


def _model_default(model: Any, parts: list[str]) -> Any:
    """The value a HarnessConfig path takes when no layer sets it, as config data (a SecretStr
    default as its text), or _UNSET when the field has none (`proposer.base_url` is required). An
    Optional section (`proposer`) is looked into."""
    field = model.model_fields[parts[0]]
    if len(parts) > 1:
        inner = next((a for a in get_args(field.annotation)
                      if isinstance(a, type) and issubclass(a, BaseModel)), field.annotation)
        return _model_default(inner, parts[1:])
    if field.is_required():
        return _UNSET
    value = field.get_default(call_default_factory=True)
    return value.get_secret_value() if isinstance(value, SecretStr) else value


def _put(data: dict, parts: list[str], value: Any) -> dict:
    """`data` with `value` at `parts`, or without that key when `value` is _UNSET. Never mutates and
    never creates a section: one a later workspace file replaced (`proposer: null`) holds nothing
    to put back, and the global layers always reach `merged` before the workspace's."""
    head, rest = parts[0], parts[1:]
    if not rest:
        return {k: v for k, v in data.items() if k != head} if value is _UNSET else {**data, head: value}
    inner = data.get(head)
    return {**data, head: _put(inner, rest, value)} if isinstance(inner, dict) else data


def _narrow_raw(merged: dict, sources: tuple[dict, ...], ws_files: tuple[str, str],
                paths: Collection[str], *, baseline: Callable[[list[str], Any], Any],
                accept: Callable[[Any, Any], bool],
                note: Callable[[str, Any, Any], Optional[str]],
                keep_accepted: bool = False) -> tuple[dict, list[str]]:
    """Walk `paths` over the raw config sources and undo what a workspace source set that
    `accept` refuses; the one walker behind the machine-level-only drop and the lower-only rule.

    `baseline(parts, machine)` is the value the global layers allow (`machine`: what they set at
    that path, or _UNSET); `accept(ws, allowed)` judges one workspace source's value (_UNSET, the
    key being absent, is the caller's to accept); `note(path, file, ws)` is the warning for a value
    refused, or None for a silent restore. One warning per such path and file. The global value
    goes back; when there is none the key goes, so the default applies; and when there is no
    baseline either, the workspace's whole section gives way to the global layers' (or to none).
    Removing, never writing a default, keeps the result the same in any order of paths. With
    `keep_accepted` (the lower-only rule), a value another workspace source wrote that `accept`
    takes wins over the global one: the highest-priority accepted workspace value goes back, so one
    refused source never discards the other's valid lowering (WR-05). Never mutates."""
    global_view = deep_merge(sources[0], sources[1])
    warnings: list[str] = []
    for path in sorted(paths):
        parts = path.split(".")
        machine = _dig(global_view, parts)
        allowed = baseline(parts, machine)
        refused = [(file, ws) for source, file in zip(sources[2:], ws_files)
                   for ws in (_dig(source, parts),) if not accept(ws, allowed)]
        if not refused:
            continue
        warnings += [text for file, ws in refused if (text := note(path, file, ws)) is not None]
        accepted = [ws for source in sources[2:] for ws in (_dig(source, parts),)
                    if ws is not _UNSET and accept(ws, allowed)] if keep_accepted else []
        if accepted:
            merged = _put(merged, parts, accepted[-1])  # sources are in merge order
        elif machine is not _UNSET:
            merged = _put(merged, parts, machine)
        elif allowed is not _UNSET:
            merged = _put(merged, parts, _UNSET)
        else:
            merged = _put(merged, parts[:-1], _dig(global_view, parts[:-1]))
    return merged, warnings


def _narrow_harness_global_only(merged: dict, sources: tuple[dict, ...],
                                ws_files: tuple[str, str]) -> tuple[dict, list[str]]:
    """Undo every HARNESS_GLOBAL_ONLY_FIELDS value a workspace source (config.yaml, overrides.yaml)
    set differently from the global layers (or from the field's default, when they are silent); one
    warning per such path and file. See `_narrow_raw` for the restore."""
    return _narrow_raw(
        merged, sources, ws_files, HARNESS_GLOBAL_ONLY_FIELDS,
        baseline=lambda parts, machine: _model_default(HarnessConfig, parts) if machine is _UNSET else machine,
        accept=lambda ws, allowed: ws is _UNSET or ws == allowed,
        note=lambda path, file, ws: f"ignoring {path} in {file}: only the global config may set it")


_CHILD_ENTRIES = ("explore", "web_researcher", "search_verifier", "cruncher", "chunk_summarizer",
                  "default")
"""The `org.children_budget` entries, `default` included on purpose, each with `actions` and
`minutes`."""

HARNESS_LOWER_ONLY_FIELDS: frozenset[str] = frozenset(
    {f"org.children_budget.{entry}.{key}" for entry in _CHILD_ENTRIES for key in ("actions", "minutes")}
    | {"org.children_budget.lookup_share", "org.children_budget.review_share"}
    | {f"org.context.{name}" for name in ("tool_output_ratio", "evict_ratio", "result_cap_ratio")})
"""The core harness keys a project may make smaller and never larger: each child agent's own
ceiling, the shares a delegated child takes of its parent's budget, and the three window ratios
(the share of the live context window one tool result may take, the size above which a result is
evicted, and what one result may put into the conversation). A workspace value is kept only when
it is a finite number not above what the global layers set (or, where they are silent, the shipped
default); a larger, non-numeric or out-of-range one (a zero ratio included) is dropped with a
warning naming the key and the file, and a `null` is dropped quietly."""

# The bounds the models give these fields. Stated once; a workspace value above the baseline is
# refused whatever its bound, so the upper bound only matters to a value that carries no baseline.
_ACTIONS_BOUNDS = {"ge": 1, "le": 10_000}
_MINUTES_BOUNDS = {"gt": 0, "le": 1440.0}
_SHARE_BOUNDS = {"gt": 0, "le": 1.0}
_RATIO_BOUNDS = {"gt": 0, "le": 1.0}  # the three window ratios of ContextConfig (tool_output_ratio, ...)


def _finite_int(v: Any, *, ge: Optional[int] = None, le: Optional[int] = None) -> bool:
    """Is `v` a whole number (a bool is not one) inside the bounds given? Judged on a raw YAML
    value, where a string, a list or `true` can sit in a numeric field."""
    return (isinstance(v, int) and not isinstance(v, bool)
            and (ge is None or v >= ge) and (le is None or v <= le))


def _finite_number(v: Any, *, gt: Optional[float] = None, ge: Optional[float] = None,
                   le: Optional[float] = None) -> bool:
    """Is `v` an int or float (a bool is neither), finite, and inside the bounds given? Only a
    float can be infinite or NaN: `math.isfinite` on a huge YAML int (400 digits) would raise
    OverflowError converting it, and an int compares with the bounds exactly (WR-04)."""
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return False
    if isinstance(v, float) and not math.isfinite(v):
        return False
    return (gt is None or v > gt) and (ge is None or v >= ge) and (le is None or v <= le)


_INVALID = object()
"""A global-layer value that is present but does not validate: never read as silence."""


def _coerce_global(model: type[BaseModel], field: str, v: Any, **fill: Any) -> Any:
    """What `model`'s own validation makes of a GLOBAL layer's raw value at `field` (`"50"` and
    `50.0` are 50, `"30"` is 30.0, as the loaded config would hold them), or _INVALID when it does
    not validate. A lower-only rule compares against this, never against the raw YAML: a value
    pydantic accepts is the operator's setting whatever its spelling, and reading a differently
    typed one as "the global layer is silent" would let a project lift it (CR-01). `fill` supplies
    the model's other required fields."""
    try:
        return getattr(model.model_validate({**fill, field: v}), field)
    except (ValidationError, OverflowError, ValueError, TypeError):
        return _INVALID


def _refuse_invalid_global_budget(field: str, value: Any, file: Optional[Path], key_prefix: str) -> None:
    """Raise the ConfigValidationError the operator's own budget value earns, naming the global
    file that holds it (WR-22). Reached when that value is present but does not validate and a
    workspace file is on the cascade, whether it declares the same key or omits it (WR-31): the
    load fails as it fails with no workspace file over it, never falls to the schema default
    (None, no limit)."""
    key = f"{key_prefix}permissions.budget.{field}"
    try:
        BudgetConfig.model_validate({field: value})
    except ValidationError as exc:
        line = _file_line_map(file).get(key) if file is not None else None
        errors = [_field_error(key, err, err["msg"], line, model=BudgetConfig, secrets=frozenset())
                  for err in exc.errors()]
        raise ConfigValidationError(str(file), errors) from None
    except (OverflowError, ValueError, TypeError):
        pass
    line = _file_line_map(file).get(key) if file is not None else None
    raise ConfigValidationError(str(file), [ConfigFieldError(
        key, value, "is not a valid budget value", line)]) from None


def _budget_value_ok(parts: list[str], v: Any) -> bool:
    """Is `v` a usable value at this `org.children_budget` path, on the field's own bounds?"""
    if parts[-1] == "actions":
        return _finite_int(v, **_ACTIONS_BOUNDS)
    return _finite_number(v, **(_MINUTES_BOUNDS if parts[-1] == "minutes" else _SHARE_BOUNDS))


_CHILDREN_BUDGET_DEFAULTS = ChildrenBudgetConfig()


def _children_budget_default(parts: list[str]) -> int | float:
    """The shipped value at an `org.children_budget` path, read off a default instance. Never
    `_model_default`: that descends into `ChildBudget`, whose fields are required, and returns
    _UNSET, which would leave the lower-only rule without a bound."""
    node: Any = _CHILDREN_BUDGET_DEFAULTS
    for part in parts[2:]:
        node = getattr(node, part)
    return node


def _ratio_baseline(parts: list[str], machine: Any) -> float:
    """What a project may not exceed at an `org.context` ratio path: the global layers' value as
    `ContextConfig` validates it, else the field's shipped default. Always a finite float in (0, 1]
    (a global value that is present but unusable falls to the default: the schema refuses it on
    its own at validation)."""
    allowed: Any = _INVALID
    if machine is not _UNSET and machine is not None:
        allowed = _coerce_global(ContextConfig, parts[-1], machine)
    if allowed is _INVALID or not _finite_number(allowed, **_RATIO_BOUNDS):
        allowed = _model_default(HarnessConfig, parts)
    return float(allowed)


def _lower_only_baseline(parts: list[str], machine: Any) -> int | float:
    """What a project may not exceed at this path: the global layers' value as the model validates
    it (a global `"20"` is 20), else the shipped default. Always a finite `int` (actions) or
    `float` (minutes, shares, ratios), so the kind of a value to accept is read off it."""
    if parts[:2] == ["org", "context"]:
        return _ratio_baseline(parts, machine)
    allowed: Any = _INVALID
    if machine is not _UNSET and machine is not None:
        if parts[-1] in ("actions", "minutes"):
            allowed = _coerce_global(ChildBudget, parts[-1], machine, actions=1, minutes=1.0)
        else:
            allowed = _coerce_global(ChildrenBudgetConfig, parts[-1], machine)
    if allowed is _INVALID or not _budget_value_ok(parts, allowed):
        allowed = _children_budget_default(parts)
    return allowed if parts[-1] == "actions" else float(allowed)


def _accepted_lower(ws: Any, allowed: int | float) -> bool:
    """Is this workspace value absent, or a finite number of the baseline's kind not above it?
    The baseline is inside the field's bounds, so only the lower bound is left to check."""
    if ws is _UNSET:
        return True
    ok = _finite_int(ws, ge=1) if isinstance(allowed, int) else _finite_number(ws, gt=0)
    return ok and ws <= allowed


def _guard_children_budget_sections(merged: dict, sources: tuple[dict, ...],
                                    ws_files: tuple[str, str]) -> tuple[dict, list[str]]:
    """Restore `org.children_budget`, or one of its entries, to the global layers' when a workspace
    source set it to something that is not a mapping (`5`, a string, `null`): the merged config
    would not validate. The block is judged first, then each entry. One warning per path and
    file. Never mutates."""
    global_view = deep_merge(sources[0], sources[1])
    warnings: list[str] = []
    for parts in [["org", "children_budget"]] + [["org", "children_budget", e] for e in _CHILD_ENTRIES]:
        path = ".".join(parts)
        for source, file in zip(sources[2:], ws_files):
            value = _dig(source, parts)
            if value is _UNSET or isinstance(value, dict):
                continue
            warnings.append(f"ignoring {path} in {file}: a project may only lower it")
            merged = _put(merged, parts, _dig(global_view, parts))
    return merged, warnings


def _gate_default(ask_field: str) -> Any:
    """The shipped `GateSettings` default behind one `AskConfig` field, or None.

    The baseline a tighten-only union starts from when the global layer declared nothing: `None`
    in `AskConfig` means "use the shipped default in `agent/gate_types.py`", so the union has to
    reach the same value the gate would have used. Read off the dataclass rather than restated
    here, exactly as `AskConfig.to_gate_settings()` does.
    """
    gate_name = ASK_TO_GATE_FIELD.get(ask_field, ask_field)
    for gate_field in dataclass_fields(GateSettings):
        if gate_field.name == gate_name:
            return gate_field.default
    return None


def _org_deny_patterns(raw: object) -> list[str]:
    """The `org.permissions.deny_patterns` list declared by ONE raw config source.

    Reads raw YAML, never a validated model: this feeds the enforcement union, and a deny list
    must not shrink because some unrelated key in the same file failed validation. Anything that
    is not a list of strings contributes nothing.
    """
    org = raw.get("org") if isinstance(raw, dict) else None
    perms = org.get("permissions") if isinstance(org, dict) else None
    patterns = perms.get("deny_patterns") if isinstance(perms, dict) else None
    if not isinstance(patterns, list):
        return []
    return [p for p in patterns if isinstance(p, str)]


def _org_mode(raw: object) -> Optional[str]:
    """The `org.permissions.mode` ONE raw config source declares, normalized, or None."""
    org = raw.get("org") if isinstance(raw, dict) else None
    perms = org.get("permissions") if isinstance(org, dict) else None
    return _normalize_mode(perms.get("mode")) if isinstance(perms, dict) else None


# ------------------------------------------------------------------ #
# ConfigLoader
# ------------------------------------------------------------------ #

class ConfigLoader:
    """
    Loads, validates, and resolves LocalHarness configuration files.

    Usage:
        loader = ConfigLoader()
        config = loader.load_agent("hn-monitor")
        harness_config = loader.load_harness()
    """

    def __init__(
        self,
        *,
        config_dir: Optional[Path] = None,
        local_config_dir: Optional[Path] = None,
        project_trusted: bool = True,
        machine_withheld: Optional[Mapping[str, Collection[tuple[str, str]]]] = None,
    ) -> None:
        # #35: one precedence chain — explicit arg > LOCALHARNESS_DIR (what --config-dir binds)
        # > LOCALHARNESS_HOME (legacy) > ~/.localharness. The overlay + runtime paths resolve
        # against THIS dir, so --config-dir now actually isolates.
        self._config_dir = resolve_config_dir(config_dir)
        # None = no workspace layer applies. Phase 39: the old literal `./.localharness` default
        # made a stray CWD directory outrank an explicit --config-dir (LAYR-02 said "full
        # replacement"; it wasn't). Callers now NAME their workspace layer — cli/workspace.py's
        # resolve_workspace_layer() is the only thing that computes one.
        self._local_dir = Path(local_config_dir) if local_config_dir is not None else None
        # What start's trust decision withheld (cli/workspace.decide_project_trust and
        # decide_machine_trust). The defaults load everything, as every command but `start` does:
        # only `start` decides trust, and nothing else starts a server.
        self._project_trusted = project_trusted
        self._machine_withheld = {k: frozenset(v) for k, v in (machine_withheld or {}).items()}
        # Workspace agent files whose MCP servers start's trust decision withheld (start's one stderr line).
        self.project_mcp_skipped: dict[str, list[str]] = {}
        self._agent_cache: dict[str, AgentConfig] = {}
        self._division_cache: dict[str, DivisionConfig] = {}
        self._harness_cache: Optional[HarnessConfig] = None
        # Workspace values load_harness() dropped at HARNESS_GLOBAL_ONLY_FIELDS paths (start's summary).
        self.harness_warnings: list[str] = []
        # Workspace agent-file values dropped at AGENT_GLOBAL_ONLY_FIELDS paths (start's summary).
        self.agent_warnings: list[str] = []
        # Project agent/division files ignored because they link outside the project (also in
        # agent_warnings; kept apart so `validate` can print exactly these lines).
        self.ignored_links: list[str] = []
        self._org_cache: Optional[OrgConfig] = None
        self._raw_harness_dict: Optional[dict] = None
        self._raw_sources_cache: Optional[tuple[dict, dict, dict, dict]] = None
        self._org_mode_cache: Optional[tuple[Optional[str]]] = None  # (mode,) once org_mode() ran
        # ENAB-01: the names that own a plugin settings section (discovered once per loader), and
        # the `agent.<name>` sections each agent load split off, by agent name.
        self._plugin_names: Optional[frozenset[str]] = None
        self._agent_plugin_sections: dict[str, dict[str, Any]] = {}

    # ---------------------------------------------------------------- #
    # Internal helpers
    # ---------------------------------------------------------------- #

    def _search_bases(self) -> tuple[Path, ...]:
        """Config bases in FIRST-WINS order: the workspace layer (when one applies), then global."""
        if self._local_dir is None:
            return (self._config_dir,)
        return (self._local_dir, self._config_dir)

    def _find_file(self, subdir: str, name: str) -> Optional[Path]:
        """Return path to first existing {name}.yaml in local_dir or config_dir. A project file
        that leads outside the project's `.localharness/` is ignored, with one warning, and the
        machine's file of that name is used instead (`layer_files(contained=True)`)."""
        for base in self._search_bases():
            candidate = base / subdir / f"{name}.yaml"
            if candidate.exists():
                if base == self._local_dir and self._project_file_leaves(candidate):
                    continue
                return candidate
        return None

    def _project_file_leaves(self, path: Path) -> bool:
        """True (and one warning, once) when a project-layer file leads outside the project's
        `.localharness/`: such a file is never loaded, fingerprinted or listed."""
        if not leaves_layer(path, self._local_dir):  # type: ignore[arg-type]
            return False
        folder = path.parent
        shown = folder if leaves_layer(folder, self._local_dir) else path  # type: ignore[arg-type]
        note = (f"ignoring {str(shown)!r}: it is a symlink leading outside "
                f"{str(self._local_dir)!r} — a project's agent and division files must be files "
                "inside its .localharness/")
        if note not in self.agent_warnings:
            self.agent_warnings.append(note)
            self.ignored_links.append(note)
        return True

    def _project_layer_files(self, subdir: str) -> list[Path]:
        """layer_files for the project layer, contained, warning once per file left out."""
        found = layer_files(self._local_dir, subdir)  # type: ignore[arg-type]
        return [f for f in found if not self._project_file_leaves(f)]

    def _validate_dict(self, model_cls: type, data: dict, path: str, yaml_text: str = "") -> Any:
        """Validate data dict through model_cls, raising ConfigValidationError on failure."""
        line_map = _build_line_map(yaml_text) if yaml_text else {}
        try:
            return model_cls.model_validate(data)
        except ValidationError as exc:
            errors = _pydantic_error_to_field_errors(exc, path, line_map, model_cls, data)
            # from None: the chained pydantic error would carry its input to any traceback (R16)
            raise ConfigValidationError(path, errors) from None

    # ---------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------- #

    def _raw_config_sources(self) -> tuple[dict, dict, dict, dict]:
        """The four raw config sources, in the owner-ruled merge order (2026-09-03, Option A):

            1. global config.yaml
            2. global overrides.yaml   (`agent:` section excluded)
            3. workspace config.yaml
            4. workspace overrides.yaml (`agent:` section excluded)

        The SPECIFIC beats the GENERAL: a later source wins any key an earlier one also sets, and
        the global layer still governs every key the workspace is silent about. Every source is
        OPTIONAL here and absent ones are `{}` — only the global config.yaml is required, and
        `load_harness` is what enforces that. A workspace never makes a file mandatory.

        The `agent:` section is stripped from BOTH overlays: it is an agent-scope default layer
        (issue #22) consumed by `load_agent`, not a `HarnessConfig` field (extra="forbid"), so it
        must not reach harness merge/validation from any of the four sources.

        Workspace overlay path is built with plain path arithmetic, and deliberately NOT by
        handing the workspace dir to `_resolve_user_overlay_path`: that helper's contract is the
        GLOBAL env chain (explicit arg > LOCALHARNESS_DIR > LOCALHARNESS_HOME > ~/.localharness).
        Passing it a workspace dir short-circuits correctly today by accident of the explicit-arg
        branch, and would silently break if that chain's precedence ever changed.
        """
        if self._raw_sources_cache is not None:
            return self._raw_sources_cache

        global_cfg_path = self._config_dir / "config.yaml"
        global_cfg = _load_yaml_file(global_cfg_path) if global_cfg_path.exists() else {}
        # What start withheld from these two files (an `org:` permission loosening, a `server:`
        # launch command, config/trust.machine_snapshot) is absent from every read of them.
        global_cfg = self._without_withheld(global_cfg, global_cfg_path)

        global_overlay_path = _resolve_user_overlay_path(self._config_dir)
        global_overlay = load_overlay(global_overlay_path)
        global_overlay = self._without_withheld(
            {k: v for k, v in global_overlay.items() if k != "agent"}, global_overlay_path)
        # What start withheld from `server:` (any launch or launch_model entry) is enforced by the
        # entry's name: see `_apply_launch_withholds` for the two posture keys that narrow and the
        # rest that drop the section.
        global_cfg, global_overlay = self._apply_launch_withholds(global_cfg, global_overlay)

        ws_cfg: dict = {}
        ws_overlay: dict = {}
        if self._local_dir is not None:
            ws_cfg_path = self._local_dir / "config.yaml"
            if ws_cfg_path.exists():
                ws_cfg = _load_yaml_file(ws_cfg_path)
            ws_overlay = load_overlay(self._local_dir / "overrides.yaml")
            ws_overlay = {k: v for k, v in ws_overlay.items() if k != "agent"}

        self._raw_sources_cache = (global_cfg, global_overlay, ws_cfg, ws_overlay)
        return self._raw_sources_cache

    def _apply_launch_withholds(self, global_cfg: dict, global_overlay: dict) -> tuple[dict, dict]:
        """Enforce the withheld `server.*` entries (kind `launch` and `launch_model`) by NAME.

        - `server.bind_all` is put to `false` and `server.require_api_key` to `true` in the withheld
          file's raw dict: the launch is narrowed (loopback, keyed), never prevented, and `server:`
          survives. The put also adds the key where the file's section lacks it. When
          `require_api_key` is narrowed to true and the merged record declares no key source (the
          same declared-sources test as the fingerprint's `keyed|open`), `bind_all` is narrowed to
          `false` in both files in the same pass: keyed with no key must not open every interface.
          Narrowing runs here, before validation, so that record still loads.
        - Any other `server.*` name (binary, docker_image, extra_args, env, api_key_file, runtime,
          launch, local_models, config_ini, and every launch_model key) withholds the whole `server:`
          section of both files: the session attaches to whatever answers and launches nothing this
          run. There is no safe value to substitute for them, and dropping one key would leave a
          section that no longer validates (a binary launch needs its binary).
        - `extra_endpoints.<peer>.lifecycle.<key>` names are the per-peer rule of
          `_without_withheld` and say nothing about `server:`.

        Never mutates its arguments."""
        from localharness.config.overlay import deep_merge
        from localharness.config.trust import LAUNCH_KIND, LAUNCH_MODEL_KIND, _key_state

        files = {"config.yaml": global_cfg, "overrides.yaml": global_overlay}
        drop_server = False
        narrow_require_key = False
        for file in files:
            for kind, name in sorted((self._machine_withheld or {}).get(file, ())):
                if kind not in (LAUNCH_KIND, LAUNCH_MODEL_KIND) or name.startswith("extra_endpoints."):
                    continue
                if name == "server.bind_all":
                    files[file] = _put(files[file], ["server", "bind_all"], False)
                elif name == "server.require_api_key":
                    files[file] = _put(files[file], ["server", "require_api_key"], True)
                    narrow_require_key = True
                else:
                    drop_server = True
        if drop_server:
            return ({k: v for k, v in files["config.yaml"].items() if k != "server"},
                    {k: v for k, v in files["overrides.yaml"].items() if k != "server"})
        if narrow_require_key:
            def section(raw: dict) -> dict:
                value = raw.get("server")
                return value if isinstance(value, dict) else {}

            merged = {**deep_merge(section(files["config.yaml"]), section(files["overrides.yaml"])),
                      "require_api_key": True}
            if _key_state(merged, files, merged.get("runtime", "vllm")) == "open":
                for file in files:  # _put creates no section: a file without `server:` is untouched
                    files[file] = _put(files[file], ["server", "bind_all"], False)
        return files["config.yaml"], files["overrides.yaml"]

    def _layered_org_deny(self) -> list[str]:
        """`org.permissions.deny_patterns` unioned across all four config sources.

        Order-preserving and deduplicating, the same shape as load_agent's org→division→agent
        union: safety ACCUMULATES across layers and a workspace can never subtract a global deny
        (MERG-02). `deep_merge` REPLACES lists wholesale, which is the right default for every
        other key and exactly wrong for this one — a workspace `org:` block that declared its own
        deny list would otherwise DELETE the global org's denials.

        An explicit `deny_patterns: []` in any layer therefore contributes nothing and removes
        nothing. That is the contract, not an oversight.
        """
        seen: set[str] = set()
        out: list[str] = []
        for source in self._raw_config_sources():
            for pattern in _org_deny_patterns(source):
                if pattern not in seen:
                    seen.add(pattern)
                    out.append(pattern)
        return out

    def _layered_raw_org(self) -> dict:
        """The raw `org:` section, deep-merged across all four config sources in the ruled order.

        `raw_harness_dict()` is deliberately the GLOBAL config.yaml alone (it guards an overlay
        write that always targets the global layer), so agent-level org inheritance needs its own
        layered view — otherwise a workspace could set `org.context.max_context_tokens`, the merged
        HarnessConfig would report it, and the agent that actually runs would use the global value.
        """
        out: dict = {}
        for source in self._raw_config_sources():
            section = source.get("org") if isinstance(source, dict) else None
            if isinstance(section, dict):
                out = deep_merge(out, section)
        return out

    def org_mode(self) -> Optional[str]:
        """`org.permissions.mode` as it reaches a session, or None when no layer sets one (QA-18).

        Each layer's overrides.yaml beats its config.yaml, as in the merge. The one exception is
        the one `permissions.mode` gets everywhere (PRD §3.3): the workspace layer may TIGHTEN the
        global value (or the default, when the global layer is silent), never loosen it. A looser
        workspace value is dropped with a warning, once per loader. `load_agent_file` hands this
        to every agent that sets no mode of its own.
        """
        if self._org_mode_cache is None:
            g_cfg, g_over, ws_cfg, ws_over = self._raw_config_sources()
            glob, ws = _org_mode(g_over) or _org_mode(g_cfg), _org_mode(ws_over) or _org_mode(ws_cfg)
            floor = glob or DEFAULT_MODE
            if ws is not None and MODE_STRICTNESS[ws] < MODE_STRICTNESS[floor]:
                log.warning(
                    "ignoring workspace org.permissions.mode %r: a project layer may only tighten "
                    "the session mode, and the global layer asks for %r", ws, floor,
                )
                ws = None
            self._org_mode_cache = (ws or glob,)
        return self._org_mode_cache[0]

    def load_harness(self) -> HarnessConfig:
        if self._harness_cache is not None:
            return self._harness_cache
        cfg_path = self._config_dir / "config.yaml"
        if not cfg_path.exists():
            raise ConfigNotFoundError("config.yaml", [str(cfg_path)])
        text = cfg_path.read_text(encoding="utf-8")
        sources = self._raw_config_sources()
        # The GLOBAL config.yaml only. `components set` validates a new GLOBAL overlay against
        # this before writing, and every overlay writer stays global in v0.13 — so folding
        # workspace data in here would widen a write-time check with data the write can never
        # target. Deliberate scope choice, not an oversight (phase 43 owns the effective view).
        self._raw_harness_dict = sources[0]

        # Ruled order (owner, 2026-09-03 — Option A): global config < global overrides <
        # workspace config < workspace overrides. Folding from {} rather than from the raw global
        # dict guarantees `merged` is never an alias of self._raw_harness_dict, so the deny splice
        # below cannot reach back and mutate the raw dict components_cmd reads.
        merged: dict = {}
        for source in sources:
            merged = deep_merge(merged, source)
        # Returned, not logged, as merge_plugin_layers does: start prints them in its summary line,
        # and a log record would reach the terminal a second time through logging.lastResort.
        ws_files = self.plugin_layer_files()[2:]
        merged, self.harness_warnings = _narrow_harness_global_only(merged, sources, ws_files)
        # A project may only lower a child's ceiling or a share (the raw sources, before the
        # merge has folded a `null` into a value). Sections that are not mappings give way first.
        merged, section_notes = _guard_children_budget_sections(merged, sources, ws_files)
        merged, lower_notes = _narrow_raw(
            merged, sources, ws_files, HARNESS_LOWER_ONLY_FIELDS,
            baseline=_lower_only_baseline, accept=_accepted_lower,
            note=lambda path, file, ws: None if ws is None else
            f"ignoring {path} in {file}: a project may only lower it", keep_accepted=True)
        self.harness_warnings += section_notes + lower_notes
        # An overrides.yaml `extra_endpoints` REPLACES config.yaml's list wholesale (lists do not
        # merge), so the peers config.yaml names vanish without a word. Say so once — only over a
        # non-empty list (the `[]` an `init`-written config.yaml holds shadows nothing) and only when a
        # peer by name actually vanishes: Phase 4's peer-posture flow and `init`'s import copy the whole
        # config.yaml list into overrides.yaml, and then nothing is hidden (D-40, WR-08).
        config_peers = sources[0].get("extra_endpoints")
        over_peers = sources[1].get("extra_endpoints")
        if "extra_endpoints" in sources[1] and isinstance(config_peers, list) and config_peers:
            kept_names = {p.get("name") for p in over_peers if isinstance(p, dict)} \
                if isinstance(over_peers, list) else set()
            hidden = [p for p in config_peers if not (isinstance(p, dict) and p.get("name") in kept_names)]
            if hidden:
                self.harness_warnings.append(
                    f"overrides.yaml extra_endpoints replaces the {len(hidden)} peers in config.yaml")
        # The one refusal: the global layers set no provider and the workspace's gave way whole,
        # so there is nothing to run against. One line naming the file to set it in — never a field
        # report, which would show the project's address back.
        if "provider" not in merged:
            ignored = [f for src, f in zip(sources[2:], ws_files)
                       if _dig(src, ["provider", "base_url"]) is not _UNSET]
            if ignored:
                raise ConfigError(NO_MACHINE_PROVIDER.format(
                    global_file=self._config_dir / "config.yaml", ws_file=ignored[-1]))

        # deny_patterns carve-out (MERG-02): see _layered_org_deny. Spliced with deep_merge rather
        # than assigned in place for the same non-aliasing reason. Guarded on non-empty so a config
        # that declares no deny rules anywhere does not GAIN an explicit empty list here, which
        # would displace PermissionConfig's shipped defaults in this view. That is all the guard
        # does: a layer's own explicit `deny_patterns: []` still arrives through `merged` and still
        # reports as empty on the returned HarnessConfig. Enforcement does not read this list —
        # `load_agent` unions load_org()'s shipped defaults with the same layered union (step 5
        # there), so the evaluator's list stays complete whatever this merged view says.
        union_deny = self._layered_org_deny()
        if union_deny:
            merged = deep_merge(merged, {"org": {"permissions": {"deny_patterns": union_deny}}})

        # NOT `_validate_dict`: that helper serves the genuinely single-file configs (agent,
        # division, org yamls) and must keep reporting exactly one path. The harness config is the
        # only MERGED one, so each of its errors is attributed to the source that actually set the
        # field — before this, every error was hardcoded to `cfg_path` and given a line number read
        # off the GLOBAL file's text, so a bad value on line 3 of a workspace config.yaml was
        # reported as a line of the global file where a valid value sat (dogfood F5, CLI-02).
        #
        # ENAB-01: plugin-owned `<name>:` sections leave first — HarnessConfig is extra="forbid".
        # They are read back per source through plugin_layers(), so each value's layer survives for
        # global-only narrowing and provenance. A key that is neither core nor a known plugin stays
        # in `merged` and is refused below: the typo guard.
        merged, _plugins = split_plugin_keys(merged, self.plugin_names(), CORE_HARNESS_KEYS)
        try:
            result = HarnessConfig.model_validate(merged)
        except ValidationError as exc:
            # Highest priority FIRST: the file whose value survived the merge is the one Pydantic
            # rejected. `sources` is (global_cfg, global_overlay, ws_cfg, ws_overlay) in ruled order.
            ranked: list[tuple[Path, dict]] = []
            if self._local_dir is not None:
                ranked.append((self._local_dir / "overrides.yaml", sources[3]))
                ranked.append((self._local_dir / "config.yaml", sources[2]))
            ranked.append((_resolve_user_overlay_path(self._config_dir), sources[1]))
            ranked.append((cfg_path, sources[0]))

            line_maps: dict[Path, dict[str, int]] = {str(cfg_path): _build_line_map(text)}
            errors: list[ConfigFieldError] = []
            owners: list[Path] = []
            secrets = secret_values(HarnessConfig, merged)
            for err in exc.errors():
                loc = ".".join(str(p) for p in err["loc"])
                owner = _owning_source(loc, ranked)
                # No source sets it (a cross-field validator, a pure-default field): fall back to
                # the global config.yaml, exactly as every harness error did before phase 43.
                owner_path = owner[0] if owner is not None else cfg_path
                key = str(owner_path)
                if key not in line_maps:
                    try:
                        line_maps[key] = _build_line_map(owner_path.read_text(encoding="utf-8"))
                    except OSError:
                        line_maps[key] = {}
                owners.append(owner_path)
                # QA-06: a top-level key no core model and no installed plugin owns is refused (the
                # typo guard), most often a removed plugin's section: say so, and what to do.
                # `agent:` is core's (read from overrides.yaml only) and keeps pydantic's text.
                message = err["msg"]
                if (err["type"] == "extra_forbidden" and len(err["loc"]) == 1
                        and loc not in CORE_HARNESS_KEYS):
                    message = unowned_hint(loc)
                errors.append(_field_error(loc, err, message, line_maps[key].get(loc),
                                           model=HarnessConfig, secrets=secrets))

            # If every error came from ONE file, that file heads the report and no error repeats
            # it — so a workspace-only mistake reads exactly like a global-only one always has,
            # just pointing at the right file. Mixed sources keep the global header and name each
            # owner on its own line.
            header = owners[0] if len(set(owners)) == 1 else cfg_path
            for field_err, owner_path in zip(errors, owners):
                if owner_path != header:
                    field_err.source_path = str(owner_path)
            # from None: the chained pydantic error would carry its input to any traceback (R16)
            raise ConfigValidationError(str(header), errors) from None

        self._harness_cache = result
        return result

    def raw_harness_dict(self) -> dict:
        """Return the parsed project YAML dict (NO overlay applied).

        This is the GLOBAL `config.yaml` ONLY — it never includes workspace data, even when a
        workspace layer applies. Its consumer (`localharness components set`) validates a
        candidate overlay before writing it, and every overlay writer targets the global layer in
        v0.13, so folding workspace data in here would widen a write-time check with data the
        write can never target.

        Used by `localharness components set` to rebuild the merged config for
        validation BEFORE writing the overlay. Side-effect: triggers load_harness
        if not yet called (populates _raw_harness_dict).
        """
        if self._harness_cache is None:
            self.load_harness()
        return dict(self._raw_harness_dict)  # defensive copy

    @property
    def user_overlay_path(self) -> Path:
        """Path to the user overlay file (LOCALHARNESS_HOME-aware).

        Used by `localharness components set` as the atomic_write_overlay target.
        Always resolved at call time so test monkeypatching takes effect.
        """
        return _resolve_user_overlay_path(self._config_dir)

    def invalidate_cache(self) -> None:
        """Drop the cached HarnessConfig (and the discovered plugin names) so the next
        load_harness() re-reads disk.

        Used by `localharness components set` after writing the overlay. The agent cache is not
        dropped here, so neither are the agent plugin sections that belong to it — reload() drops
        both.
        """
        self._harness_cache = None
        self._raw_harness_dict = None
        self._raw_sources_cache = None
        self._org_mode_cache = None
        self._plugin_names = None

    @property
    def global_config_dir(self) -> Path:
        """The GLOBAL layer (or its --config-dir replacement) — where plugin folders are discovered
        and GUARDRAILS.md lives. Never the workspace."""
        return self._config_dir

    def plugin_names(self) -> frozenset[str]:
        """Every plugin that may own a settings section: BUILTIN_PLUGINS' manifests plus discovered
        METADATA — entry points and the GLOBAL plugins/ folder (self._config_dir), never the
        workspace. Nothing is imported."""
        if self._plugin_names is None:
            from localharness.plugins import discovery
            from localharness.plugins.builtin import bundled_plugins
            self._plugin_names = (frozenset(c.manifest.name for c in bundled_plugins())
                                  | frozenset(d.name for d in discovery.discover(self._config_dir)))
        return self._plugin_names

    def plugin_layers(self) -> dict[str, tuple[Any, Any, Any, Any]]:
        """Each plugin's raw `<name>:` section in the four ruled sources — global config, global
        overrides, workspace config, workspace overrides — None where a source is silent. Kept per
        layer so global-only narrowing and provenance know where a value came from. A plugin named
        after a core key is not listed: that key is core's."""
        g_cfg, g_over, ws_cfg, ws_over = self._raw_config_sources()
        return {n: (g_cfg.get(n), g_over.get(n), ws_cfg.get(n), ws_over.get(n))
                for n in sorted(self.plugin_names() - CORE_HARNESS_KEYS)}

    def legacy_org_flags(self) -> tuple[Any, Any, Any, Any]:
        """The raw `org.memory_enabled` of each ruled source (global config, global overrides,
        workspace config, workspace overrides), None where a source does not set it — read so the
        memory plugin's `memory.enabled` can honour the deprecated key at every layer (MEMP-06)."""
        return tuple((s.get("org") or {}).get("memory_enabled") if isinstance(s.get("org"), dict)
                     else None for s in self._raw_config_sources())

    def plugin_layer_files(self) -> tuple[str, str, str, str]:
        """The four files plugin_layers() reads, in the same order ("" for the workspace pair when
        no workspace applies)."""
        ws = self._local_dir
        return (str(self._config_dir / "config.yaml"),
                str(_resolve_user_overlay_path(self._config_dir)),
                str(ws / "config.yaml") if ws is not None else "",
                str(ws / "overrides.yaml") if ws is not None else "")

    def workspace_value(self, dotpath: str) -> Any:
        """What the workspace layer writes at `dotpath` — its overrides.yaml over its config.yaml,
        raw, before any machine-level-only narrowing drops it — or None when it writes nothing
        there or no workspace applies. For naming a project value that is ignored."""
        parts = dotpath.split(".")
        found = [v for v in (_dig(s, parts) for s in self._raw_config_sources()[2:]) if v is not _UNSET]
        return found[-1] if found else None

    def agent_plugin_sections(self, agent_name: str) -> dict[str, Any]:
        """{plugin name: its `agent.<name>` section} for the agent whose `name:` is `agent_name`, as
        split off when this loader loaded it (the agent file over the overrides file's `agent:`
        defaults, as for every agent key). Empty for an agent this loader never loaded or one that
        sets none."""
        return dict(self._agent_plugin_sections.get(agent_name, {}))

    def unowned_sections(self) -> list[tuple[str, Path, Optional[int]]]:
        """(dot-path, file, line) for each key load_harness / load_agent refuse as unowned (QA-06),
        most often a removed plugin's settings — what `plugins list` names: each top-level key of
        the four config sources that is neither core nor a known plugin, and `agent.<key>` for such
        a key in the GLOBAL overrides' `agent:` section (the one agents read). Nothing is validated
        or imported. Keys inside agent yaml files are not listed: `validate` and `start` report
        them against their own file."""
        known = self.plugin_names()
        overlay = _resolve_user_overlay_path(self._config_dir)
        ws = self._local_dir
        files = (self._config_dir / "config.yaml", overlay,
                 ws / "config.yaml" if ws is not None else None,
                 ws / "overrides.yaml" if ws is not None else None)
        found = [(str(key), f) for f, source in zip(files, self._raw_config_sources())
                 if f is not None for key in source if key not in CORE_HARNESS_KEYS | known]
        agent = load_overlay(overlay).get("agent")
        found += [(f"agent.{key}", overlay) for key in (agent if isinstance(agent, dict) else {})
                  if key not in CORE_AGENT_KEYS | known]
        maps = {f: _file_line_map(f) for _, f in found}
        return [(dotpath, f, maps[f].get(dotpath)) for dotpath, f in found]

    def load_org(self) -> OrgConfig:
        if self._org_cache is not None:
            return self._org_cache
        org_path = self._config_dir / "org.yaml"
        if not org_path.exists():
            self._org_cache = OrgConfig()
            return self._org_cache
        text = org_path.read_text(encoding="utf-8")
        data = self._without_withheld(_load_yaml_file(org_path), org_path)
        result = self._validate_dict(OrgConfig, data, str(org_path), text)
        self._org_cache = result
        return result

    def load_division(self, name: str, *, bypass_cache: bool = False) -> DivisionConfig:
        if not bypass_cache and name in self._division_cache:
            return self._division_cache[name]
        path = self._find_file("divisions", name)
        if path is None:
            searched = [str(b / "divisions" / f"{name}.yaml") for b in self._search_bases()]
            raise ConfigNotFoundError(name, searched)
        result = self.load_division_file(path)
        self._division_cache[name] = result
        return result

    def load_division_file(self, path: Path) -> DivisionConfig:
        """Load and validate the division yaml AT THIS PATH — no stem re-resolution.

        `load_division` answers "which file wins for this name"; this one answers "is THIS file
        valid". `validate_all` needs the second question: with a workspace file shadowing a global
        file of the same stem, resolving by stem validated the winner twice and never opened the
        shadowed file at all.
        """
        text = path.read_text(encoding="utf-8")
        data = self._without_withheld(_load_yaml_file(path), path)
        return self._validate_dict(DivisionConfig, data, str(path), text)

    def _raw_org_context(self) -> dict:
        """Explicitly-set org-level `context:` block. The org config lives in config.yaml's
        `org:` section — the single user-facing source of truth — so read it there first, across
        BOTH layers (v0.13: a workspace's org.context is a config.yaml key like any other); a
        standalone org.yaml is only a legacy override."""
        try:
            org_section = self._layered_raw_org()
        except Exception:
            org_section = {}
        ctx = org_section.get("context") if isinstance(org_section, dict) else None
        if isinstance(ctx, dict) and ctx:
            return ctx
        org_path = self._config_dir / "org.yaml"
        if org_path.exists():
            legacy = self._without_withheld(_load_yaml_file(org_path), org_path).get("context")
            return legacy if isinstance(legacy, dict) else {}
        return {}

    def _raw_division_context(self, name: str) -> dict:
        """Explicitly-set `context:` block from a division yaml (empty if absent)."""
        path = self._find_file("divisions", name)
        if path is None:
            return {}
        ctx = self._without_withheld(_load_yaml_file(path), path).get("context")
        return ctx if isinstance(ctx, dict) else {}

    def load_agent(self, name: str, *, bypass_cache: bool = False) -> AgentConfig:
        if not bypass_cache and name in self._agent_cache:
            return self._agent_cache[name]

        # 1. Find and load raw YAML
        path = self._find_file("agents", name)
        if path is None:
            searched = [str(b / "agents" / f"{name}.yaml") for b in self._search_bases()]
            raise ConfigNotFoundError(name, searched)
        result = self.load_agent_file(path)
        self._agent_cache[name] = result
        return result

    def load_agent_file(self, path: Path) -> AgentConfig:
        """Resolve and validate the agent yaml AT THIS PATH — no stem re-resolution, no cache.

        Same split as `load_division_file`, for the same reason: `validate_all` must open the file
        it is reporting on. Inheritance (division, org, overlay, workspace_root) resolves exactly
        as it would for a session, so a file that validates here is a file that loads.
        """
        text = path.read_text(encoding="utf-8")
        raw = self._without_withheld(_load_yaml_file(path), path)
        if self._local_dir is not None and in_layer(path, self._local_dir):
            raw = self._without_project_mcp(raw, path)

        # 2. Load org
        org = self.load_org()

        # 3. Load division (if any), raising ConfigReferenceError if missing
        div_name = raw.get("division")
        if div_name:
            try:
                division = self.load_division(div_name)
            except ConfigNotFoundError as exc:
                raise ConfigReferenceError(
                    str(path),
                    "division",
                    div_name,
                    f"Division '{div_name}' not found",
                ) from exc
        else:
            division = None

        # 4. Build merged dict via scalar resolution
        merged = dict(raw)  # start with agent raw values

        # The user overlay's `agent:` section (issue #22) — the layer `components set agent.*`
        # writes and `components get` reads back. Read HERE, before the scalar resolution, because
        # it is the DEFAULT slot for these three fields: step 5b's deep_merge layers the overlay
        # UNDER `merged`, and `merged` always carries a resolved value for model/temperature/
        # max_tokens, so an overlay scalar could never win — stored, confirmed, and ignored at load
        # (pre-existing <=0.12.5). Precedence shipped: agent yaml > division > org > overlay >
        # schema default, which is 5b's stated contract with the last two rungs now real.
        overlay_path = _resolve_user_overlay_path(self._config_dir)
        overlay_agent_raw = self._overlay_agent()
        overlay_agent = overlay_agent_raw if isinstance(overlay_agent_raw, dict) else {}

        # Resolve scalar fields: model, temperature, max_tokens
        agent_model = raw.get("model", "inherit")
        div_model = division.model if division else "inherit"
        org_model = org.default_model if org.default_model else "inherit"
        merged["model"] = _resolve_scalar(
            "model", agent_model, div_model, org_model,
            _overlay_default(overlay_agent, "model", "inherit"),
        )

        # Effort: agent > division only. The loader never sees the harness, so the endpoint and org
        # rungs live in config/effort.py::resolve_effort; the overlay does not carry this field.
        merged["effort"] = _resolve_scalar("effort", raw.get("effort"), division.effort if division else None, None, "inherit")

        # `model_fields_set`: OrgConfig's temperature/max_tokens carry SCHEMA defaults (0.6, 4096),
        # never None — so reading them straight made the org layer "set" on every install and the
        # rung below it (the overlay, and the shipped default it stands on) unreachable. Only a
        # value the org config actually declared counts as org inheritance. `default_model` needs
        # no such guard: its unset spelling is the empty string, already handled above.
        agent_temp = raw.get("temperature")
        div_temp = division.temperature if division else None
        org_temp = org.default_temperature if "default_temperature" in org.model_fields_set else None
        merged["temperature"] = _resolve_scalar(
            "temperature", agent_temp, div_temp, org_temp,
            _overlay_default(overlay_agent, "temperature", 0.6),
        )

        # max_tokens needs no `model_fields_set` guard: its unset spelling is None at every rung
        # (org `default_max_tokens`, division, agent), so an undeclared value already falls
        # through. None all the way down is the answer "nobody chose a cap", and it stays None all
        # the way to the wire: the request omits max_tokens and the model generates until it is
        # done. A number, 4,096 included, is a choice and is carried through untouched.
        agent_mt = raw.get("max_tokens")
        div_mt = division.max_tokens if division else None
        # `org` is load_org(): the LEGACY standalone org.yaml, which nothing in src/ writes —
        # `init` writes `org:` INSIDE config.yaml. So an org-level cap in the config users
        # actually have never reached this resolution at all, exactly as the deny-pattern union
        # below documents for its own field. Read the layered raw `org:` section first (the same
        # precedence `_raw_org_context` uses for `context:`), legacy org.yaml second.
        # `default_temperature` and `default_model` have the identical hole and are deliberately
        # NOT changed here: switching on a temperature that has been inert in someone's
        # config.yaml is a behaviour change that needs its own decision, not a silent ride along
        # with a fix about the output cap.
        org_mt = self._layered_raw_org().get("default_max_tokens", org.default_max_tokens)
        merged["max_tokens"] = _resolve_scalar(
            "max_tokens", agent_mt, div_mt, org_mt,
            _overlay_default(overlay_agent, "max_tokens", None),
        )

        # Resolve the `context` block agent->division->org (per-field). Previously the
        # agent's raw context (or the schema default) was the ONLY source, so an org-level
        # `context:` (e.g. the window init fits to the served max_model_len) was write-only
        # and never reached a context-less agent. Read EXPLICIT yaml context blocks only —
        # parsed org/division ContextConfig objects always carry schema defaults, which
        # would otherwise shadow a more-specific value with a generic default.
        org_ctx = self._raw_org_context()
        div_ctx = self._raw_division_context(div_name) if div_name else {}
        agent_ctx = raw.get("context") if isinstance(raw.get("context"), dict) else {}
        merged_ctx = {**org_ctx, **div_ctx, **(agent_ctx or {})}
        if merged_ctx:
            merged["context"] = merged_ctx

        # 5. Union deny_patterns: org + division + agent (additive)
        agent_perms = raw.get("permissions") or {}
        agent_deny = agent_perms.get("deny_patterns", []) if isinstance(agent_perms, dict) else []

        div_deny: list[str] = []
        if division:
            div_deny = division.permissions.deny_patterns

        # `org` above is load_org(): the LEGACY standalone org.yaml, or PermissionConfig's shipped
        # defaults when that file is absent — which it is on every real install, since nothing in
        # src/ writes one and `init` writes `org:` INSIDE config.yaml. So the org config users
        # actually have has never reached enforcement. Add it, from BOTH layers, via the same
        # raw-source union load_harness splices (MERG-02): safety accumulates, and a workspace can
        # never subtract a global deny.
        #
        # Deliberately NOT wrapped in try/except: if a config file cannot be read this must fail
        # loudly rather than hand the agent a silently shorter deny list. A deny list that shrinks
        # on a parse error is a fail-open security regression.
        org_deny = [*org.permissions.deny_patterns, *self._layered_org_deny()]

        # The overlay's own `agent.permissions.deny_patterns`. Step 5b layers the overlay UNDER
        # `merged`, and `merged["permissions"]["deny_patterns"]` is always set below — so a list
        # written here through `components set` was stored, read back, and then replaced wholesale
        # at load. Unioned like every other layer: deny accumulates in the restrictive direction
        # only, so an overlay can ADD a rule and can never drop one (MERG-02).
        overlay_perms = overlay_agent.get("permissions")
        overlay_deny = (
            overlay_perms.get("deny_patterns", []) if isinstance(overlay_perms, dict) else []
        )
        if not isinstance(overlay_deny, list):
            overlay_deny = []

        # Build union preserving order, deduplicating
        seen: set[str] = set()
        union_deny: list[str] = []
        for pat in (*org_deny, *overlay_deny, *div_deny, *agent_deny):
            if pat not in seen:
                seen.add(pat)
                union_deny.append(pat)

        # Merge permissions section
        perms_merged: dict = {}
        if isinstance(agent_perms, dict):
            perms_merged = dict(agent_perms)

        # Budget: agent wins if set, else division, else org defaults
        agent_budget_raw = perms_merged.get("budget") or {}
        if isinstance(agent_budget_raw, dict):
            div_budget = division.permissions.budget if division else None
            org_budget = org.permissions.budget

            resolved_budget: dict = {}
            for field in ("max_actions", "max_duration_minutes", "kill_file"):
                a_val = agent_budget_raw.get(field)
                d_val = getattr(div_budget, field) if div_budget else None
                o_val = getattr(org_budget, field)
                resolved_budget[field] = _resolve_scalar(field, a_val, d_val, o_val, None)

            # C2-family invariant (F5): the kill switch is a machine-global CONTROL artifact — one
            # file stops every agent on the box — so a workspace must not be able to point a
            # session at a different one. start_cmd and subagent already pin its DIRECTORY to the
            # global config dir; the VALUE came off this resolved config, so a workspace
            # `agents/<name>.yaml` with an absolute `kill_file` silently detached the session from
            # the operator's KILL. Re-resolved from the GLOBAL layer alone whenever a workspace
            # applies. Inert without one (LAYR-03).
            if self._local_dir is not None:
                global_kill = self._global_layer_kill_file(path.stem, div_name, org_budget)
                if resolved_budget.get("kill_file") != global_kill:
                    log.warning(
                        "ignoring workspace kill_file %r for agent %r: the kill switch resolves "
                        "from the global config layer only (%r)",
                        resolved_budget.get("kill_file"), path.stem, global_kill,
                    )
                resolved_budget["kill_file"] = global_kill
            # Remove None values so Pydantic uses its own defaults
            resolved_budget = {k: v for k, v in resolved_budget.items() if v is not None}
            perms_merged["budget"] = resolved_budget

        # `permissions.ask` cascades like every other permission key — and did not (C1). Step 4
        # resolves the scalars, the union above resolves `deny_patterns`, and `ask` had NOTHING:
        # `perms_merged` starts as the AGENT file's own permissions, so an `org.permissions.ask`
        # in the config.yaml `init` writes reached a merged agent only when a workspace layer
        # happened to apply, as `_narrow_project_layer_ask`'s baseline. Without one — the default
        # install — a globally declared `destructive_signatures` or `network_hosts: true` was
        # stored, read back by `config show`, and never reached `AskConfig.to_gate_settings()`.
        # A tightening the operator asked for and silently did not get is the worst direction for
        # this particular block to fail in.
        #
        # The same agent > division > org > overlay cascade, merged PER KEY (`ask` is a block of
        # independent knobs, not one value), read from the GLOBAL layer alone and applied
        # unconditionally. The agent file AT THIS PATH layers on top, because it may be a
        # workspace file that `_global_layer_ask` deliberately cannot see — and step 5c's
        # narrow-only union then runs on the result exactly as before, dropping only the keys the
        # project layer moved AWAY from this baseline.
        cascaded_ask = self._global_layer_ask(path.stem, div_name)
        agent_ask = agent_perms.get("ask") if isinstance(agent_perms, dict) else None
        if isinstance(agent_ask, dict):
            cascaded_ask = {
                **cascaded_ask,
                **{k: v for k, v in agent_ask.items() if v is not None},
            }
        if cascaded_ask:
            perms_merged["ask"] = cascaded_ask

        perms_merged["deny_patterns"] = union_deny
        merged["permissions"] = perms_merged

        # 5b. Overlay the user layer's `agent:` section as a LOW-priority default (issue #22):
        #     the same layer `localharness components set agent.*` writes and `components get`
        #     reads back. Per-agent yaml (and org/division inheritance) WINS — `merged` is
        #     layered ON TOP of the overlay. Resolved through THIS loader's config_dir like
        #     load_harness (#35 — no longer env-only, so --config-dir isolates the agent overlay).
        #     The three SCALAR fields already resolved against it in step 4 (see the note there);
        #     this covers every other agent-scope key the overlay may carry.
        if overlay_agent:
            merged = deep_merge(overlay_agent, merged)

        # 5c. The project-layer narrowing (PRD §3.3). It used to also auto-fill
        # `permissions.workspace_root` with the folder containing `.localharness/` (v0.13
        # CONF-01, "the confinement leash comes free with the workspace layer"). That default is
        # GONE as of v0.14.1, and the reason is a live end-to-end run: with `auto` as the mode,
        # `bash_exec("cat > /tmp/notes/x")` ran and `write(path="/tmp/notes/x")` was hard-refused
        # — same file, same session, opposite answers. The refusal came from the per-tool leash
        # (`tools/base.Tool._outside_workspace`), which this line had switched on without anyone
        # asking for it, while the shell tool's own string never reached it.
        #
        # The gate owns the boundary now. It DERIVES it from where you stand
        # (`verdict.derive_boundary`), applies it to every tool the same way, and `auto`'s
        # contract is that a write outside the project runs unless it lands somewhere protected.
        # A second, invisible, tool-shaped copy of the same idea could only disagree with it.
        #
        # An EXPLICIT `permissions.workspace_root` is untouched and still a hard confinement:
        # the tool returns permission_denied, no prompt, no mode that overrides it. That is what
        # harness-run evals set, and it is now the only way the leash switches on — a human wrote
        # it down.
        if self._local_dir is not None:
            self._narrow_project_layer_permissions(merged, path.stem, div_name)

        # 5c'. The core sections only the global layer may set (AGENT_CORE_GLOBAL_ONLY_SECTIONS),
        #      restored WHOLE for a workspace file: the global section, else the schema default.
        #      Never key by key — a workspace file that omits `escalation_threshold` would fall to
        #      the schema's looser default over a stricter global one (spec 17 §6). A file that
        #      declared the section differently earns one note; one that left it out, none.
        if self._local_dir is not None and in_layer(path, self._local_dir):
            for section in sorted(AGENT_CORE_GLOBAL_ONLY_SECTIONS):
                restored = self._global_agent_section(path.stem, section)
                if restored is _UNSET:
                    merged.pop(section, None)
                else:
                    merged[section] = restored
                if _section_tried(section, raw.get(section, _UNSET), restored):
                    note = f"ignoring {section} in {str(path)!r}: only the global config may set it"
                    if note not in self.agent_warnings:
                        self.agent_warnings.append(note)

        # 5c''. The budget keys a workspace layer may only LOWER (AGENT_CORE_LOWER_ONLY_KEYS). The
        #      rule reads what the workspace FILES DECLARED (raw), never the merged value: after the
        #      merge a `null` and an omission are indistinguishable, and the schema default must
        #      never be taken for a workspace's number. It runs whenever a workspace file is on this
        #      agent's cascade: the agent file itself, or a workspace division file that shadows
        #      the global division (the division door lifts every member's budget). The baseline is
        #      one per file, `_global_layer_budget` — the loader knows layers, never callers
        #      (spec §13).
        budget_kept = self._lower_only_agent_budget(merged, raw, path, div_name)

        # 5d. QA-18: an agent that sets no `permissions.mode` (in its file, or under `agent:` in
        #     overrides.yaml) runs in the org's, the mode `init`'s posture question writes. Key
        #     presence, never a comparison with the default: an explicit `mode: auto` is the
        #     agent's choice. Resolved here, where all agent inheritance is, so the gate, the trust
        #     question, `/mode` and a gate-less loop read one value. After 5c, which judges only
        #     what a project layer wrote.
        perms = merged.get("permissions")
        if isinstance(perms, dict) and "mode" not in perms and self.org_mode() is not None:
            perms["mode"] = self.org_mode()

        # 6. Validate merged dict
        line_map = _build_line_map(text)
        # ENAB-01, agent level: `agent.<name>` sections leave before AgentConfig (extra="forbid")
        # validates, and reach the plugin through agent_plugin_sections(). Unknown keys still fail.
        merged, agent_sections = split_plugin_keys(merged, self.plugin_names(), CORE_AGENT_KEYS)
        if self._local_dir is not None and in_layer(path, self._local_dir):
            agent_sections = self._narrow_agent_global_only(agent_sections, path)
        # ENAB-01 / MEMP-06: a bundled plugin's agent section fails here exactly as a core agent key
        # does — once memory's settings left AgentConfig, a bad `memory:` would otherwise become a
        # soft plugin failure (`validate` never resolves plugins). Bundled only: a plugin you
        # installed is never imported before it is enabled (ENAB-06). The RAW section is still what
        # agent_plugin_sections() returns; resolve() re-validates and builds the instance.
        from localharness.plugins.builtin import bundled_plugins
        plugin_errors: list[ConfigFieldError] = []
        plugin_settings: dict[str, Any] = {}
        for cls in bundled_plugins():
            name = cls.manifest.name
            if cls.AgentConfigModel is None:
                continue
            try:
                plugin_settings[name] = cls.AgentConfigModel.model_validate(
                    agent_sections.get(name, {}))
            except ValidationError as exc:
                secrets = secret_values(cls.AgentConfigModel, agent_sections.get(name, {}))
                for err in exc.errors():
                    loc = ".".join([name, *map(str, err["loc"])])
                    plugin_errors.append(_field_error(loc, err, err["msg"], line_map.get(loc),
                                                      model=cls.AgentConfigModel, secrets=secrets))
        try:
            result = AgentConfig.model_validate(merged)
        except ValidationError as exc:
            # QA-06: a top-level key only the overrides' `agent:` section sets (an unknown key can
            # come from nowhere else) is that file's, as `agent.<key>` on its own line — never the
            # agent file's, which does not contain it. Every other error stays the agent file's.
            errors: list[ConfigFieldError] = []
            owners: list[Path] = []
            secrets = secret_values(AgentConfig, merged)
            for err in exc.errors():
                loc = ".".join(str(p) for p in err["loc"])
                owner, field, line, message = path, loc, line_map.get(loc), err["msg"]
                if err["type"] == "extra_forbidden" and len(err["loc"]) == 1:
                    under = err["loc"][0] not in raw and err["loc"][0] in overlay_agent
                    message = unowned_hint(loc, agent=True, under_agent=under)
                    if under:
                        owner, field = overlay_path, f"agent.{loc}"
                        line = _file_line_map(overlay_path).get(field)
                owners.append(owner)
                errors.append(_field_error(field, err, message, line, model=AgentConfig, secrets=secrets))
            # load_harness's rule: one owner heads the report; mixed owners keep the agent file and
            # name each foreign owner on its own line.
            header = owners[0] if len(set(owners)) == 1 else path
            for field_err, owner in zip(errors, owners):
                if owner != header:
                    field_err.source_path = str(owner)
            for field_err in plugin_errors:
                if header != path:
                    field_err.source_path = str(path)
            # from None: the chained pydantic error would carry its input to any traceback (R16)
            raise ConfigValidationError(str(header), errors + plugin_errors) from None
        if plugin_errors:
            raise ConfigValidationError(str(path), plugin_errors)

        self._agent_plugin_sections[result.name] = agent_sections
        result._plugin_settings = plugin_settings  # read as attributes: result.memory.recall_scope
        result._budget_kept_from_workspace = frozenset(budget_kept)
        return result

    def _lower_only_agent_budget(
        self, merged: dict, raw: dict, path: Path, div_name: Optional[str]
    ) -> set[str]:
        """Apply AGENT_CORE_LOWER_ONLY_KEYS to `merged` (the agent's cascade, in place) and return
        the budget fields kept from a workspace file while the global layer was silent.

        Against a finite global baseline B the workspace may only lower: a value at or below B is
        kept, a higher or non-finite one becomes B with a note, a `null` or an omission becomes B
        silently. Where the global layer declares nothing a finite workspace value is kept as
        written, no note, and its field is returned (the seam caps a marked value on a session
        with no local operator, in any mode: WR-70); a non-finite one is removed with a note."""
        local = self._local_dir
        if local is None:
            return set()
        ws_agent = in_layer(path, local)
        div_path = self._find_file("divisions", div_name) if div_name else None
        ws_div = div_path is not None and in_layer(div_path, local)
        if not ws_agent and not ws_div:
            return set()  # no workspace file on this cascade: nothing a repo wrote reaches it
        div_raw: Any = None
        if ws_div:
            try:
                div_raw = self._without_withheld(_load_yaml_file(div_path), div_path)
            except ConfigError:
                div_raw = None
        perms = merged.get("permissions")
        budget = perms.get("budget") if isinstance(perms, dict) else None
        if not isinstance(budget, dict):
            return set()

        def _unset(v: Any) -> bool:
            return v is None or v == "inherit"

        kept: set[str] = set()
        for key_path in sorted(AGENT_CORE_LOWER_ONLY_KEYS):
            field = key_path.rsplit(".", 1)[-1]
            parts = ["permissions", "budget", field]
            in_actions = field == "max_actions"

            def finite(v: Any) -> bool:
                return (_finite_int(v, **_ACTIONS_BOUNDS) if in_actions
                        else _finite_number(v, ge=0.1, le=1440.0))

            a_val = _dig(raw, parts)
            a_val = None if a_val is _UNSET else a_val
            if not ws_agent and not _unset(a_val):
                continue  # the global agent file decided; the cascade never reached the division
            # W: the workspace declaration the cascade uses, and the file that wrote it.
            decl, src = _UNSET, None
            if ws_agent and not _unset(a_val):
                decl, src = a_val, path
            elif ws_div:
                d_val = _dig(div_raw, parts)
                if d_val is not _UNSET and not _unset(d_val):
                    decl, src = d_val, div_path

            # The global value as the model validates it (CR-01): a global `"50"` or `50.0` is the
            # operator's 50, never silence. One that is present and does not validate fails the
            # load, as it does with no workspace file over it (WR-22), whether or not the
            # workspace file declares the key (WR-31): a shadowing file that omits it would
            # otherwise load the schema default, None, which is no limit at all.
            raw_global, g_file, g_key = self._global_layer_budget_source(path.stem, div_name, field)
            baseline = (raw_global if raw_global is None
                        else _coerce_global(BudgetConfig, field, raw_global))
            note = (f"ignoring permissions.budget.{field} in {str(src)!r}: "
                    "a project may only lower it")
            if baseline is _INVALID:
                _refuse_invalid_global_budget(field, raw_global, g_file, g_key)
            if finite(baseline):
                if decl is not _UNSET and finite(decl) and decl <= baseline:
                    result = decl
                else:
                    result = baseline
                    if decl is not _UNSET and note not in self.agent_warnings:
                        self.agent_warnings.append(note)
                budget[field] = result
            elif decl is _UNSET:
                continue  # silent global, nothing declared: the cascade's own answer stands
            elif finite(decl):
                budget[field] = decl
                kept.add(field)
            else:
                budget.pop(field, None)
                if note not in self.agent_warnings:
                    self.agent_warnings.append(note)
        return kept

    def _without_project_mcp(self, raw: dict, path: Path) -> dict:
        """An untrusted project's agent-file `tools.mcp_servers` is removed before the merge, so no
        AgentConfig built from that file names a server to start; the file and the server names are
        recorded for start's one line. Never mutates `raw` (the YAML reader's object)."""
        tools = raw.get("tools") if isinstance(raw, dict) else None
        servers = tools.get("mcp_servers") if isinstance(tools, dict) else None
        if self._project_trusted or not servers:
            return raw
        self.project_mcp_skipped[str(path)] = (
            [str(s.get("name") or "?") if isinstance(s, dict) else "?" for s in servers]
            if isinstance(servers, list) else ["?"])
        return {**raw, "tools": {k: v for k, v in tools.items() if k != "mcp_servers"}}

    def _machine_file(self, path: Path) -> Optional[str]:
        """`path` relative to the machine's config dir, posix ("agents/orchestrator.yaml"), or None
        for a file outside it. Lexical first, so a symlinked agent file keeps its own name; then
        with the directories resolved, never the file itself."""
        path = Path(path)
        for root, candidate in ((self._config_dir, path),
                                (self._config_dir.resolve(), path.parent.resolve() / path.name)):
            try:
                return candidate.relative_to(root).as_posix()
            except ValueError:
                continue
        return None

    def _overlay_agent(self) -> Any:
        """The global overrides.yaml's `agent:` section — every agent's default layer — with what
        start withheld from it removed: it is fingerprinted like an agent file (file
        "overrides.yaml"), so a server put there waits for the same one Yes."""
        path = _resolve_user_overlay_path(self._config_dir)
        return self._without_withheld(load_overlay(path).get("agent"), path)

    def _without_withheld(self, raw: Any, path: Path) -> Any:
        """What start withheld from a machine agent, division or org file, or from config.yaml or
        overrides.yaml — an MCP server by name (every server of that name in the file: two may share
        one, and over-withholding is the safe side), `memory.embedding_model`, a dotted
        `permissions.…` or `org.permissions.…` key — is absent from every read of that file this
        run, baselines included (a withheld `server.…` launch key narrows or drops the section, in
        `_apply_launch_withholds`). Never mutates `raw`.

        The per-peer rule (kind `launch` or `launch_model`, the grammar
        `extra_endpoints.<peer>.lifecycle.<key>`): the peer name is recovered by stripping the
        `extra_endpoints.` prefix and then `rpartition(".lifecycle.")` — never `split(".")`, which
        would read `qwen3.6` as `qwen3` and let that peer's lifecycle launch. Like an MCP server, it
        applies to every peer of that name, in both settings files (a name filed under one file
        still names the peer in the other: lists replace wholesale, so the one that loads is the
        winner's, and over-withholding is the safe side). A withheld `bind_all` / `require_api_key`
        narrows that key inside the lifecycle of every peer of the name (`false` / `true`); any
        other key removes the lifecycle (`lifecycle: None`, the peer stays attachable and
        validates). The peer list is rebuilt, never `_put`: `_put` has no way into a list."""
        machine_file = self._machine_file(path) or ""
        withheld = self._machine_withheld.get(machine_file) if self._machine_withheld else None
        peer_withheld = self._peer_withholds(machine_file)
        if not isinstance(raw, dict) or not (withheld or peer_withheld):
            return raw
        withheld = withheld or frozenset()
        servers = {name for kind, name in withheld if kind == "mcp_server"}
        tools = raw.get("tools")
        if servers and isinstance(tools, dict) and isinstance(tools.get("mcp_servers"), list):
            raw = {**raw, "tools": {**tools, "mcp_servers": [
                s for s in tools["mcp_servers"]
                if not (isinstance(s, dict) and str(s.get("name") or "") in servers)]}}
        for kind, name in sorted(withheld):
            if kind in ("embedding_model", "permission", "org_permission"):
                raw = _put(raw, name.split("."), _UNSET)
        peers = raw.get("extra_endpoints")
        if peer_withheld and isinstance(peers, list):
            raw = {**raw, "extra_endpoints": [
                self._without_withheld_lifecycle(p, peer_withheld) for p in peers]}
        return raw

    def _peer_withholds(self, machine_file: str) -> list[tuple[str, str]]:
        """The `(peer, lifecycle key)` pairs withheld for `machine_file`: from config.yaml AND
        overrides.yaml when it is either (see `_without_withheld`), else from its own entries (which
        name no peer). Names that do not carry the `.lifecycle.` marker name nothing and are skipped."""
        from localharness.config.trust import LAUNCH_KIND, LAUNCH_MODEL_KIND

        files = ("config.yaml", "overrides.yaml") if machine_file in (
            "config.yaml", "overrides.yaml") else (machine_file,)
        pairs: list[tuple[str, str]] = []
        for file in files:
            for kind, name in sorted((self._machine_withheld or {}).get(file, ())):
                if kind not in (LAUNCH_KIND, LAUNCH_MODEL_KIND) or not name.startswith("extra_endpoints."):
                    continue
                peer, sep, key = name[len("extra_endpoints."):].rpartition(".lifecycle.")
                if sep:
                    pairs.append((peer, key))
        return pairs

    @staticmethod
    def _without_withheld_lifecycle(peer: Any, peer_withheld: list[tuple[str, str]]) -> Any:
        """`peer` with each withheld lifecycle key applied by name; `peer` itself when it is not a
        named peer with a lifecycle. Never mutates."""
        if not isinstance(peer, dict) or not isinstance(peer.get("lifecycle"), dict):
            return peer
        for name, key in peer_withheld:
            if str(peer.get("name") or "") != name or peer.get("lifecycle") is None:
                continue
            if key in ("bind_all", "require_api_key"):
                peer = {**peer, "lifecycle": {**peer["lifecycle"], key: key == "require_api_key"}}
            else:
                peer = {**peer, "lifecycle": None}
        return peer

    def _narrow_agent_global_only(self, sections: dict, path: Path) -> dict:
        """A plugin-section key in AGENT_GLOBAL_ONLY_FIELDS that a WORKSPACE agent file sets: the
        global layer's value stands (its agents/<same name>.yaml, else the global overrides'
        `agent:` section), else the plugin's own default; a different workspace value is dropped
        with one warning naming the key and the file. Never mutates `sections`."""
        for dotted in sorted(AGENT_GLOBAL_ONLY_FIELDS):
            head, _, key = dotted.partition(".")
            section = sections.get(head)
            if not isinstance(section, dict) or key not in section:
                continue
            machine = self._global_agent_value(path.stem, head, key)
            allowed = machine if machine is not _UNSET else self._plugin_agent_default(head, key)
            if section[key] == allowed:
                continue
            # Returned, not logged, as load_harness's are: start prints them in its summary line,
            # and a log record would reach the terminal again through logging.lastResort — on
            # every reload of this file, mid-session included.
            note = f"ignoring {dotted} in {path}: only the global config may set it"
            if note not in self.agent_warnings:
                self.agent_warnings.append(note)
            kept = {k: v for k, v in section.items() if k != key}
            if machine is not _UNSET:
                kept[key] = machine
            sections = {**sections, head: kept}
        return sections

    def _global_agent_value(self, stem: str, head: str, key: str) -> Any:
        """`<head>.<key>` as the GLOBAL layer alone sets it for agent `stem`: its agents/<stem>.yaml,
        else the global overrides' `agent:` section, else _UNSET. Raw YAML, as
        `_global_layer_kill_file` reads it: a file that does not parse sets nothing."""
        def declared(data: Any) -> Any:
            section = data.get(head) if isinstance(data, dict) else None
            return section[key] if isinstance(section, dict) and key in section else _UNSET

        def read(load: Callable[[], Any]) -> Any:
            try:
                return declared(load())
            except ConfigError:
                return _UNSET

        agent_file = self._config_dir / "agents" / f"{stem}.yaml"
        found = (read(lambda: self._without_withheld(_load_yaml_file(agent_file), agent_file))
                 if agent_file.exists() else _UNSET)
        if found is _UNSET:
            found = read(self._overlay_agent)
        return found

    def _global_agent_section(self, stem: str, head: str, *, overlay: bool = True) -> Any:
        """The raw `head` section as the GLOBAL layer alone sets it for agent `stem`: its
        agents/<stem>.yaml's section merged over (when `overlay`) the global overrides' `agent:`
        section, in step 5b's order, so the restore is exactly what a global-only load of the agent
        holds (WR-03: a key the operator set through `components set agent.stuck_detector.*` is
        not lost because the agent file also has the section). Either alone when only one has it,
        else _UNSET. A section that is not a mapping counts as absent, and so does a file that does
        not parse. Returns a copy: the caller may edit it, the file's dict stays untouched."""
        def declared(data: Any) -> Any:
            section = data.get(head) if isinstance(data, dict) else None
            return dict(section) if isinstance(section, dict) else _UNSET

        def read(load: Callable[[], Any]) -> Any:
            try:
                return declared(load())
            except ConfigError:
                return _UNSET

        agent_file = self._config_dir / "agents" / f"{stem}.yaml"
        found = (read(lambda: self._without_withheld(_load_yaml_file(agent_file), agent_file))
                 if agent_file.exists() else _UNSET)
        if not overlay:
            return found
        under = read(self._overlay_agent)
        if under is _UNSET:
            return found
        return under if found is _UNSET else deep_merge(under, found)

    @staticmethod
    def _plugin_agent_default(head: str, key: str) -> Any:
        """The default of `key` in the bundled plugin `head`'s AgentConfigModel, or _UNSET."""
        from localharness.plugins.builtin import bundled_plugins
        cls = next((c for c in bundled_plugins()
                    if c.manifest.name == head and c.AgentConfigModel is not None), None)
        return _model_default(cls.AgentConfigModel, [key]) if cls is not None else _UNSET

    def _global_layer_kill_file(
        self, stem: str, div_name: Optional[str], org_budget: Any
    ) -> Any:
        """`permissions.budget.kill_file` as the GLOBAL layer alone declares it, or None.

        Same agent > division > org cascade as the merged budget, read from the global dir's files
        only — so the global layer's own kill_file still applies through a workspace file that
        shadows it, and a workspace-only agent simply has none (which is what an agent that
        declares nothing has always had). Raw YAML, not the validated models, because this must
        answer even when some unrelated key in the same file is invalid: a kill switch that goes
        missing on a validation error is the wrong direction to fail.
        """
        def _declared(path: Path) -> Any:
            if not path.exists():
                return None
            try:
                raw = self._without_withheld(_load_yaml_file(path), path)
            except ConfigError:
                return None
            perms = raw.get("permissions") if isinstance(raw, dict) else None
            budget = perms.get("budget") if isinstance(perms, dict) else None
            return budget.get("kill_file") if isinstance(budget, dict) else None

        agent_val = _declared(self._config_dir / "agents" / f"{stem}.yaml")
        div_val = (
            _declared(self._config_dir / "divisions" / f"{div_name}.yaml") if div_name else None
        )
        return _resolve_scalar(
            "kill_file", agent_val, div_val, getattr(org_budget, "kill_file", None), None
        )

    def _global_layer_budget(self, stem: str, div_name: Optional[str], field: str) -> Any:
        """`permissions.budget.<field>` as the GLOBAL layer alone declares it, or None.

        One baseline per file: the global agents/<stem>.yaml, else the global divisions/<div>.yaml,
        else the legacy global org.yaml, else the global overrides' `agent:` section — the first
        value that is neither `null` nor "inherit". Raw YAML, as `_global_layer_kill_file` reads
        it, so it answers when an unrelated key in the same file is invalid; a file that does not
        parse is silent. No caller role enters (spec §13).
        """
        return self._global_layer_budget_source(stem, div_name, field)[0]

    def _global_layer_budget_source(
        self, stem: str, div_name: Optional[str], field: str
    ) -> tuple[Any, Optional[Path], str]:
        """`_global_layer_budget`'s value, the file that declared it, and the key prefix the value
        sits under in that file (`"agent."` in overrides.yaml, else `""`); `(None, None, "")` when
        the global layer is silent. The file is what an error about that value names (WR-22)."""
        def _declared(source: Any) -> Any:
            perms = source.get("permissions") if isinstance(source, dict) else None
            budget = perms.get("budget") if isinstance(perms, dict) else None
            return budget.get(field) if isinstance(budget, dict) else None

        def _declared_file(path: Path) -> Any:
            if not path.exists():
                return None
            try:
                return _declared(self._without_withheld(_load_yaml_file(path), path))
            except ConfigError:
                return None

        def _overlay() -> Any:
            try:
                return _declared(self._overlay_agent())
            except ConfigError:
                return None

        files: list[Optional[Path]] = [
            self._config_dir / "agents" / f"{stem}.yaml",
            self._config_dir / "divisions" / f"{div_name}.yaml" if div_name else None,
            self._config_dir / "org.yaml",
        ]
        for file in files:
            value = _declared_file(file) if file is not None else None
            if value is not None and value != "inherit":
                return value, file, ""
        value = _overlay()
        if value is not None and value != "inherit":
            return value, _resolve_user_overlay_path(self._config_dir), "agent."
        return None, None, ""

    def _global_layer_permission(self, field: str, stem: str, div_name: Optional[str]) -> Any:
        """`permissions.<field>` as the GLOBAL layer alone declares it, or None.

        The same agent > division > org cascade the merged permissions use, read from the global
        dir's files only — the sibling of `_global_layer_kill_file`, generalized because the
        narrow-only union needs the same "what did the operator, not the repo, ask for?" answer
        for `mode` and `workspace_root` (PRD §3.3). Raw YAML for the same reason: this must still
        answer when an unrelated key in the file is invalid.
        """
        def _declared(source: dict) -> Any:
            perms = source.get("permissions") if isinstance(source, dict) else None
            return perms.get(field) if isinstance(perms, dict) else None

        def _declared_file(path: Path) -> Any:
            if not path.exists():
                return None
            try:
                return _declared(self._without_withheld(_load_yaml_file(path), path))
            except ConfigError:
                return None

        global_cfg, global_overlay, _ws_cfg, _ws_overlay = self._raw_config_sources()
        org_val = None
        for source in (global_cfg, global_overlay):
            section = source.get("org") if isinstance(source, dict) else None
            declared = _declared(section) if isinstance(section, dict) else None
            if declared is not None:
                org_val = declared

        # The overlay's `agent:` section is the lowest rung, exactly as step 5b layers it — and it
        # is GLOBAL-only (`_raw_config_sources` strips `agent:` from the workspace overlay), so a
        # value set by `components set agent.permissions.*` is the operator's, not the repo's.
        overlay_agent = self._overlay_agent()
        overlay_val = _declared(overlay_agent) if isinstance(overlay_agent, dict) else None

        return _resolve_scalar(
            field,
            _declared_file(self._config_dir / "agents" / f"{stem}.yaml"),
            _declared_file(self._config_dir / "divisions" / f"{div_name}.yaml") if div_name else None,
            org_val,
            overlay_val,
        )

    def _layer_baseline(
        self, field: str, stem: str, div_name: Optional[str], base_perms: Optional[dict]
    ) -> Any:
        """What `permissions.<field>` is NARROWED AGAINST: the global layer, else the code base.

        The global layer's own declaration always wins (`_global_layer_permission`). `base_perms`
        is the rung under it — the code-defined config a built-in subagent overlay sits on top of,
        which has no file anywhere and so cannot be read by the cascade. None on the config-agent
        path, where a silent global layer means "no baseline" exactly as before.
        """
        value = self._global_layer_permission(field, stem, div_name)
        if value is None and isinstance(base_perms, dict):
            value = base_perms.get(field)
        return value

    def _narrow_project_layer_permissions(
        self,
        merged: dict,
        stem: str,
        div_name: Optional[str],
        *,
        base_perms: Optional[dict] = None,
        declared: Optional[dict] = None,
    ) -> None:
        """Apply the narrow-only union to `mode`, `workspace_root` and the `ask` rule sets.

        PRD §3.3, stated precisely: a repo can only TIGHTEN. It can add deny patterns (the union
        at step 5), register tools and define agents — things that ASK. It cannot hand itself a
        looser session mode, and it cannot move its own confinement leash outward. Both were
        repo-settable before v0.14: `deny_patterns` had the union, `mode` and `workspace_root`
        simply took whatever the highest-priority layer said (PRD §1).

        `permissions.ask.*` was the third loosening surface and the widest (the v0.14 spine
        critic's finding): it reached `AskConfig.to_gate_settings()` untouched, so a repo could
        empty the ungrantable tier and pre-trust an MCP server. See `_narrow_project_layer_ask`.

        Mutates `merged["permissions"]` in place, dropping a project-layer value back to the
        baseline. Every drop warns, naming the value — a silently ignored setting is its own bug
        report. Called only when a workspace layer applies; without one, nothing here can fire.

        `base_perms`: the code-defined base's `permissions`, used as the LOWEST baseline rung
        under the global layer's files. `overlay_builtin_config` passes the built-in subagent's
        base config here, so a repo's `agents/explore.yaml` is narrowed against what the built-in
        actually ships rather than against a generic default. None (the config-agent path) keeps
        the baselines it has always had: `DEFAULT_MODE`, no root, the shipped gate rule sets.

        `declared`: the `permissions` keys the PROJECT layer actually WROTE, when the caller knows
        them. `overlay_builtin_config` merges the base's full `model_dump()`, so every key is
        present in `merged` whether the repo mentioned it or not, and narrowing a key the repo
        never wrote would warn about the operator's own value. None (the config-agent path) means
        every key in `merged` is a candidate, which is what that path needs.
        """
        perms = merged.get("permissions")
        if not isinstance(perms, dict):
            return

        def _wrote(field: str) -> bool:
            """Did the PROJECT layer write this key? (Always yes when the caller cannot tell.)"""
            return declared is None or field in declared

        base_ask = base_perms.get("ask") if isinstance(base_perms, dict) else None
        self._narrow_project_layer_ask(
            perms, stem, div_name,
            base_ask=base_ask,
            declared=declared.get("ask") if isinstance(declared, dict) else declared,
        )

        # --- mode: a project layer may only RAISE strictness.
        global_mode = _normalize_mode(self._layer_baseline("mode", stem, div_name, base_perms))
        baseline = global_mode if global_mode is not None else DEFAULT_MODE
        effective = _normalize_mode(perms.get("mode")) if _wrote("mode") else None
        if effective is not None and MODE_STRICTNESS[effective] < MODE_STRICTNESS[baseline]:
            log.warning(
                "ignoring workspace permissions.mode %r for agent %r: a project layer may only "
                "tighten the session mode, and the global layer asks for %r",
                effective, stem, baseline,
            )
            perms["mode"] = baseline

        # --- workspace_root: a project layer may only name a root INSIDE the boundary the
        # loader can derive from the workspace itself (PRD §3.1: the boundary is derived, never
        # configured). A root equal to the global layer's is the operator's own choice and is
        # left alone wherever it points.
        configured = perms.get("workspace_root") if _wrote("workspace_root") else None
        global_root = self._layer_baseline("workspace_root", stem, div_name, base_perms)
        if not configured or configured == global_root:
            return
        boundary = self._local_dir.parent.resolve()  # type: ignore[union-attr]
        try:
            candidate = Path(configured).expanduser().resolve()
        except (OSError, RuntimeError):  # unresolvable path: treat as outside, fail closed
            candidate = None
        if candidate is None or not candidate.is_relative_to(boundary):
            log.warning(
                "ignoring workspace permissions.workspace_root %r for agent %r: a project layer "
                "may only confine INSIDE the project it sits in (%s)",
                configured, stem, boundary,
            )
            perms["workspace_root"] = global_root

    def _global_layer_ask(self, stem: str, div_name: Optional[str]) -> dict:
        """`permissions.ask` as the GLOBAL layer alone declares it, key by key.

        The dict sibling of `_global_layer_permission`: the same agent > division > org > overlay
        cascade read from the global dir's files only, merged PER KEY because `ask` is a block of
        independent knobs — an operator who trusts an MCP server in the overlay and tunes a rule
        set in `agents/<name>.yaml` declared both, and both keep their full authority (PRD §3.3).
        Raw YAML for the same reason as its sibling: this must still answer when an unrelated key
        in the same file is invalid, because the alternative is a gate that loosens on a typo.
        """
        def _declared(source: Any) -> dict:
            perms = source.get("permissions") if isinstance(source, dict) else None
            ask = perms.get("ask") if isinstance(perms, dict) else None
            return ask if isinstance(ask, dict) else {}

        def _declared_file(path: Path) -> dict:
            if not path.exists():
                return {}
            try:
                return _declared(self._without_withheld(_load_yaml_file(path), path))
            except ConfigError:
                return {}

        global_cfg, global_overlay, _ws_cfg, _ws_overlay = self._raw_config_sources()
        overlay_agent = self._overlay_agent()

        # Lowest rung first: each layer overwrites the keys it declares, exactly as the merge
        # itself resolves them (org < overlay `agent:` < division < agent).
        layers: list[dict] = [
            _declared(source.get("org") if isinstance(source, dict) else None)
            for source in (global_cfg, global_overlay)
        ]
        layers.append(_declared(overlay_agent))
        if div_name:
            layers.append(_declared_file(self._config_dir / "divisions" / f"{div_name}.yaml"))
        layers.append(_declared_file(self._config_dir / "agents" / f"{stem}.yaml"))

        declared: dict = {}
        for layer in layers:
            declared.update({k: v for k, v in layer.items() if v is not None})
        return declared

    def _narrow_project_layer_ask(
        self,
        perms: dict,
        stem: str,
        div_name: Optional[str],
        *,
        base_ask: Optional[dict] = None,
        declared: Optional[dict] = None,
    ) -> None:
        """Narrow the project layer's `permissions.ask.*` to what a repo is allowed to ask for.

        PRD §3.3 again, for the block that actually decides what the gate stops:
        `ASK_TIGHTEN_ONLY_FIELDS` become the UNION of the global layer's list (or the shipped
        `GateSettings` default) and the project's, so a repo can add a dangerous command and can
        never delete one; `ASK_GLOBAL_ONLY_FIELDS` have no tightening direction at all and a
        project-layer value is dropped; `network_hosts` may only be switched ON.

        A value EQUAL to the global layer's is the operator's own and is left alone wherever it
        points — the same test `workspace_root` uses, and what keeps a globally-defined agent
        (no workspace `agents/<name>.yaml`) byte-identical. A key dropped entirely, rather than
        pinned to a restated value, is how the shipped default keeps standing.

        `base_ask` and `declared` carry the same meaning as in `_narrow_project_layer_permissions`:
        the code-defined base's `ask` block as the lowest baseline rung, and the `ask` keys the
        project layer actually wrote.
        """
        ask = perms.get("ask")
        if not isinstance(ask, dict):
            return
        global_ask = self._global_layer_ask(stem, div_name)
        if isinstance(base_ask, dict):
            global_ask = {**{k: v for k, v in base_ask.items() if v is not None}, **global_ask}

        def _restore(field: str, global_val: Any) -> None:
            if global_val is None:
                ask.pop(field, None)
            else:
                ask[field] = global_val

        for field in list(ask):
            if declared is not None and field not in declared:
                continue  # the base's own value, not the repo's
            value = ask[field]
            global_val = global_ask.get(field)
            if value == global_val:  # the operator declared it themselves
                continue

            if field in ASK_GLOBAL_ONLY_FIELDS:
                log.warning(
                    "ignoring workspace permissions.ask.%s %s for agent %r: the key is a "
                    "loosening surface, and a project layer may only tighten the gate "
                    "(the global layer asks for %s)",
                    field, _short_repr(value), stem, _short_repr(global_val),
                )
                _restore(field, global_val)
                continue

            if field == "network_hosts":
                baseline = global_val if isinstance(global_val, bool) else _gate_default(field)
                if value is True or value == baseline:
                    continue  # switching the ask ON is tightening; restating it is a no-op
                log.warning(
                    "ignoring workspace permissions.ask.network_hosts %r for agent %r: a project "
                    "layer may only switch the network-host ask ON (the global layer asks for %r)",
                    value, stem, baseline,
                )
                _restore(field, global_val)
                continue

            if field not in ASK_TIGHTEN_ONLY_FIELDS or not isinstance(value, list):
                continue  # unknown key or wrong shape: AskConfig(extra="forbid") is the answer

            baseline_seq = (
                list(global_val) if isinstance(global_val, list)
                else list(_gate_default(field) or ())
            )
            removed = [item for item in baseline_seq if item not in value]
            extras = [item for item in value if item not in baseline_seq]
            if removed:
                log.warning(
                    "ignoring %d removal(s) from workspace permissions.ask.%s for agent %r: a "
                    "project layer may only ADD to the gate's rule sets, never subtract from "
                    "them (dropped: %s)",
                    len(removed), field, stem, _short_repr(removed),
                )
            if extras:
                ask[field] = [*baseline_seq, *extras]
            else:
                _restore(field, global_val)

    def overlay_builtin_config(self, name: str, base: AgentConfig) -> AgentConfig:
        """Overlay an optional agents/<name>.yaml onto a BUILT-IN subagent's base config.

        Built-in subagents (explore, web-researcher, search-verifier) ship a code-defined base
        config — role + budget + toolset. This lets a user TUNE that base (e.g. a bigger
        web-researcher budget) without restating it: the fields present in agents/<name>.yaml
        deep-merge ON TOP of `base` (base name/role/budget are the defaults; the yaml wins
        per-field), then the result is re-validated as a full AgentConfig.

        Returns `base` UNCHANGED when no agents/<name>.yaml exists — absence is a pure no-op
        (built-in defaults, no behavior change). When one exists it is applied and validated:
        malformed YAML raises ConfigParseError and a value that fails AgentConfig validation
        raises ConfigValidationError — NEVER a silent fallback to the base (explicit-failure rule).

        Read fresh each call (not cached) so a yaml the model just wrote is picked up in the same
        turn, mirroring the dispatch seam's load_agent(bypass_cache=True). The built-in TOOLSET is
        structural (fixed by the dispatcher) and is NOT taken from this overlay — only AgentConfig
        fields (budget, role, temperature, timeout, ...) apply.

        `_find_file` searches the WORKSPACE first, so this is the second door into the same
        loosening surface `_narrow_project_layer_permissions` closes for configured agents — and
        it used to be wide open: `deep_merge` REPLACES lists, so a cloned repo's
        `.localharness/agents/explore.yaml` with `permissions: {deny_patterns: []}` dropped the
        shipped deny list for that subagent outright. When the file that won is the workspace's,
        the same narrow-only union applies, baselined on the built-in's own code-defined config
        (PRD §3.3). A global `agents/<name>.yaml` and a workspace-less session are untouched.
        """
        path = self._find_file("agents", name)
        if path is None:
            return base
        text = path.read_text(encoding="utf-8")
        raw = self._without_withheld(_load_yaml_file(path), path)  # ConfigParseError if malformed
        if self._local_dir is not None and path == self._local_dir / "agents" / f"{name}.yaml":
            raw = self._without_project_mcp(raw, path)
        merged = deep_merge(base.model_dump(), raw)  # base = the defaults; yaml fields win
        if self._local_dir is not None and path == self._local_dir / "agents" / f"{name}.yaml":
            self._narrow_builtin_overlay(merged, base, name, raw)
        return self._validate_dict(AgentConfig, merged, str(path), text)

    def _narrow_builtin_overlay(
        self, merged: dict, base: AgentConfig, name: str, raw: dict
    ) -> None:
        """Narrow a WORKSPACE overlay of a built-in subagent to what a repo may ask for.

        The built-in's code-defined config is the baseline a repo can only tighten from, because
        for these agents there is usually no global file at all: no `agents/explore.yaml` ships,
        so "what did the operator ask for?" is answered by the built-in itself unless the operator
        wrote one (`_layer_baseline`, `_global_layer_ask`, `_global_layer_kill_file` all take that
        rung). Everything else is the shared rule: `mode`, `workspace_root` and the `ask` rule sets
        narrow-only (PRD §3.3), `deny_patterns` UNION (MERG-02 — safety accumulates), `kill_file`
        pinned (F5 — one file stops every agent on the box), `max_actions` and
        `max_duration_minutes` lower-only against the operator's own file else the built-in's
        number (AGENT_CORE_LOWER_ONLY_KEYS), `max_tool_calls` dropped to the base's None.

        Only keys the workspace yaml actually WROTE are considered; the rest of `merged` is the
        base's own dump and is none of the repo's doing.
        """
        # The core sections only the global layer may set come back whole (the global
        # agents/<name>.yaml section, else the built-in's own): `stuck_detector` is restored whole
        # here, as at `load_agent_file`. The overrides' `agent:` section never reaches a built-in.
        # Before the permissions narrowing and its early return.
        for section in sorted(AGENT_CORE_GLOBAL_ONLY_SECTIONS):
            restored = self._global_agent_section(name, section, overlay=False)
            if restored is _UNSET:
                restored = base.model_dump().get(section)
            merged[section] = restored
            # The one predicate both doors use (IN-01): only a declared key that changes what is
            # restored is "tried".
            if _section_tried(section, raw.get(section, _UNSET), restored):
                log.warning(
                    "ignoring %s in %r: only the global config may set it",
                    section, str(self._local_dir / "agents" / f"{name}.yaml"))
        base_perms = base.model_dump().get("permissions") or {}
        declared = raw.get("permissions") if isinstance(raw.get("permissions"), dict) else {}
        self._narrow_project_layer_permissions(
            merged, name, None, base_perms=base_perms, declared=declared
        )
        perms = merged.get("permissions")
        if not isinstance(perms, dict):
            return

        # deny_patterns: the union `_load_agent_yaml` does at step 5, which this path never had.
        if "deny_patterns" in declared:
            base_deny = base_perms.get("deny_patterns") or []
            overlay_deny = perms.get("deny_patterns") or []
            seen: set[str] = set()
            union: list[str] = []
            for pattern in (*base_deny, *overlay_deny):
                if pattern not in seen:
                    seen.add(pattern)
                    union.append(pattern)
            perms["deny_patterns"] = union

        # kill_file: the C2-family invariant (F5). The kill switch is a machine-global CONTROL
        # artifact, so a repo must not be able to point a built-in child at a different one.
        budget = perms.get("budget")
        if isinstance(budget, dict) and "kill_file" in (declared.get("budget") or {}):
            baseline = self._global_layer_kill_file(
                name, None, getattr(base.permissions, "budget", None)
            )
            if budget.get("kill_file") != baseline:
                log.warning(
                    "ignoring workspace kill_file %r for agent %r: the kill switch resolves from "
                    "the global config layer only (%r)",
                    budget.get("kill_file"), name, baseline,
                )
                budget["kill_file"] = baseline

        # The budget: lower-only against the operator's own `agents/<name>.yaml` when it holds a
        # finite value (a global null falls through), else against the built-in's own number — the
        # builder's rung, finite for every built-in. The workspace file shadows the global one
        # (its values are not merged), so an omission or a null restores the baseline here rather
        # than leaving the base's. `max_tool_calls` is never a project's to set: nothing derives
        # it, so the base's value (None) comes back.
        if not isinstance(budget, dict):
            budget = {}
            perms["budget"] = budget
        declared_budget = declared.get("budget") if isinstance(declared.get("budget"), dict) else {}
        base_budget = getattr(base.permissions, "budget", None)
        ws_file = str(self._local_dir / "agents" / f"{name}.yaml")
        for key_path in sorted(AGENT_CORE_LOWER_ONLY_KEYS):
            field = key_path.rsplit(".", 1)[-1]
            finite = ((lambda v: _finite_int(v, **_ACTIONS_BOUNDS)) if field == "max_actions"
                      else (lambda v: _finite_number(v, ge=0.1, le=1440.0)))
            # The operator's value as the model validates it (CR-01: a global `"20"` is 20), else
            # the built-in's own number.
            baseline = self._global_layer_agent_budget(name, field)
            if baseline is not None:
                baseline = _coerce_global(BudgetConfig, field, baseline)
            if baseline is _INVALID or not finite(baseline):
                baseline = getattr(base_budget, field, None)
            value = declared_budget.get(field)
            # A base with no number of its own (none ships today) is no licence: the workspace
            # value is refused and the base's stays (IN-06).
            if value is not None and finite(value) and baseline is not None and value <= baseline:
                budget[field] = value
                continue
            if value is not None:
                # SECURITY.md promises the key AND the file (IN-08), as the stuck_detector note
                # above names it.
                log.warning(
                    "ignoring permissions.budget.%s %r in %r: a project may only lower it (%r)",
                    field, value, ws_file, baseline,
                )
            budget[field] = baseline
        if declared_budget.get("max_tool_calls") is not None:
            log.warning(
                "ignoring permissions.budget.max_tool_calls %r in %r: a project cannot set it",
                declared_budget["max_tool_calls"], ws_file,
            )
        budget["max_tool_calls"] = getattr(base_budget, "max_tool_calls", None)

    def _global_layer_agent_budget(self, stem: str, field: str) -> Any:
        """`permissions.budget.<field>` as the global `agents/<stem>.yaml` alone declares it, or
        None (no file, a file that does not parse, or a key left out or null)."""
        path = self._config_dir / "agents" / f"{stem}.yaml"
        if not path.exists():
            return None
        try:
            raw = self._without_withheld(_load_yaml_file(path), path)
        except ConfigError:
            return None
        perms = raw.get("permissions") if isinstance(raw, dict) else None
        budget = perms.get("budget") if isinstance(perms, dict) else None
        return budget.get(field) if isinstance(budget, dict) else None

    def agent_yaml_paths(self) -> list[Path]:
        """Every agent yaml discovery reads, GLOBAL dir first so the local layer wins by stem.

        The ONE discovery order for the whole harness (#150 phase 38): `agent list`, the start
        menu and this loader all read exactly these files in exactly this order. `self._local_dir`
        is the `local_config_dir` constructor hook: the discovered workspace layer when a caller
        named one, else None (no second layer).
        """
        # reversed(): _search_bases is first-wins order, but discover_agents keys a dict by file
        # stem, so the workspace file must be listed LAST to win by overwrite. Global first.
        return [f for b in reversed(self._search_bases())
                for f in (self._project_layer_files("agents") if b == self._local_dir
                          else layer_files(b, "agents"))]

    def discover_agents(
        self, *, on_error: Optional[Callable[[Path, Exception], None]] = None
    ) -> list[dict]:
        """Raw agent dicts across both layers, local overriding global by file stem.

        An unparseable file is SKIPPED but never SILENTLY: swallowing the parse error made a
        typo'd agents/orchestrator.yaml indistinguishable from a fresh install, and start's mint
        branch then overwrote it with the default template. Warn-and-skip is the ONE malformed-YAML
        behavior (it replaces agent_cmd's `except Exception: pass`).

        `on_error` is the CLI's channel for the user-visible warning: config/ must never import a
        rich console from cli/, so the caller supplies the printer and the default stays a log line.
        """
        agents: dict[str, dict] = {}
        for f in self.agent_yaml_paths():
            try:
                data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
                if "name" not in data:
                    data["name"] = f.stem
                agents[f.stem] = data
            except Exception as exc:  # noqa: BLE001
                if isinstance(exc, yaml.YAMLError):  # its text would quote the broken line (R16)
                    exc = _parse_error(f, exc)
                log.warning("skipping unreadable agent file %s: %s", f, exc)
                if on_error is not None:
                    on_error(f, exc)
        return list(agents.values())

    def list_agents(self) -> list[str]:
        return sorted({f.stem for f in self.agent_yaml_paths()})

    def microagent_paths(self) -> list[Path]:
        """Every `microagents/*.md` file across both layers, GLOBAL dir first.

        Same order contract as `agent_yaml_paths`, for the same reason: `discover_microagents`
        keys by file stem, so the workspace file must be listed LAST to win by overwrite.

        AGNT-02 ships RESOLUTION only. `ContextConfig.microagents` is declared-only today (zero
        readers in src), and the keyword-triggered injection mechanism is deliberately out of
        scope for v0.13 — phase 43's effective-config view is what makes this visible.
        """
        return [
            f
            for d in (b / "microagents" for b in reversed(self._search_bases()))
            if d.exists()
            for f in sorted(d.glob("*.md"))
        ]

    def discover_microagents(self) -> dict[str, Path]:
        """Microagent files by stem, the workspace layer overriding the global one on a collision.

        Wholesale nearest-wins: a workspace `microagents/style.md` REPLACES the global file of the
        same stem rather than being appended to it, matching how a same-named agent resolves.
        Global-only microagents stay resolvable.
        """
        return {f.stem: f for f in self.microagent_paths()}

    def list_divisions(self) -> list[str]:
        return sorted({f.stem for base in self._search_bases()
                       for f in (self._project_layer_files("divisions") if base == self._local_dir
                                 else layer_files(base, "divisions"))})

    def write_agent(self, config: AgentConfig, *, overwrite: bool = False) -> Path:
        dest = self._config_dir / "agents" / f"{config.name}.yaml"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if not overwrite:
                raise FileExistsError(f"Agent config already exists: {dest}")
            bak = dest.with_suffix(".yaml.bak")
            dest.rename(bak)
        yaml_text = to_yaml_str(config)
        from localharness.core.private_files import touch_private
        touch_private(dest)
        dest.write_text(yaml_text, encoding="utf-8")
        # Owner-only, like every other config file this package writes: an agent yaml carries the
        # deny list, the ask rule sets and the kill file — the policy the session is gated by.
        restrict_config_file(dest)
        return dest

    def reload(self) -> None:
        self._agent_cache.clear()
        self._agent_plugin_sections.clear()
        self._division_cache.clear()
        self._harness_cache = None
        self._org_cache = None
        self._raw_sources_cache = None
        self._org_mode_cache = None
        self._plugin_names = None

    def validate_all(self) -> list[tuple[str, Optional[ConfigError]]]:
        """Every config file of every layer, each validated AT ITS OWN PATH.

        The agent/division rows go through `load_agent_file`/`load_division_file` rather than the
        by-name loaders on purpose: resolving a file's own stem back through `_find_file` returns
        the layer WINNER, so a shadowed same-stem file was never opened — its verdict was the
        winner's, filed under its path (both directions proven: a broken global file reported
        valid, a valid global file reported with the workspace file's error).
        """
        results: list[tuple[str, Optional[ConfigError]]] = []

        # harness config
        harness_path = self._config_dir / "config.yaml"
        if harness_path.exists():
            try:
                self.load_harness()
                results.append((str(harness_path), None))
            except ConfigError as e:
                results.append((str(harness_path), e))
        elif self._local_dir is not None:
            # The block above keys the harness row under the GLOBAL config.yaml and is skipped
            # entirely when that file is absent — so on a machine that was never `init`ed, a
            # workspace config.yaml that cannot be parsed produced NO ROW, and `validate` answered
            # "No configuration files found" about the very file it had just been pointed at.
            #
            # Only the ERROR is reported here. A workspace config.yaml is an overlay, partial by
            # design, and `load_harness` still requires the global one — so a row calling it valid
            # would green-light a machine that cannot start a session.
            ws_harness = self._local_dir / "config.yaml"
            if ws_harness.exists():
                try:
                    _load_yaml_file(ws_harness)
                except ConfigError as e:
                    results.append((str(ws_harness), e))

        # org config
        org_path = self._config_dir / "org.yaml"
        if org_path.exists():
            try:
                self.reload()
                self.load_org()
                results.append((str(org_path), None))
            except ConfigError as e:
                results.append((str(org_path), e))

        # divisions, then agents — a project's file that links outside its .localharness/ is
        # ignored here exactly as start ignores it (`ignored_links`, which `validate` prints)
        for subdir, load in (("divisions", self.load_division_file), ("agents", self.load_agent_file)):
            for base in self._search_bases():
                files = (self._project_layer_files(subdir) if base == self._local_dir
                         else layer_files(base, subdir))
                for f in files:
                    try:
                        load(f)
                        results.append((str(f), None))
                    except ConfigError as e:
                        results.append((str(f), e))

        return results
