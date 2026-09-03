"""ConfigLoader: YAML parse, validate, inheritance resolve, write."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Callable, Optional

import yaml
from pydantic import ValidationError
from pydantic_yaml import to_yaml_str

from .models import AgentConfig, DivisionConfig, HarnessConfig, OrgConfig
from localharness.config.overlay import (
    deep_merge,
    load_overlay,
    _resolve_user_overlay_path,
)
from localharness.config.paths import resolve_config_dir

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


class ConfigFieldError:
    """One validation failure for one field."""
    def __init__(
        self,
        field_path: str,
        value: Any,
        message: str,
        yaml_line: Optional[int] = None,
    ) -> None:
        self.field_path = field_path
        self.value = value
        self.message = message
        self.yaml_line = yaml_line

    def __str__(self) -> str:
        loc = f" (line {self.yaml_line})" if self.yaml_line else ""
        return f"{self.field_path}{loc}: {self.message} (got: {self.value!r})"


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


def _load_yaml_file(path: Path) -> dict:
    """Read file, safe_load, return dict (empty dict if file is empty/None)."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ConfigNotFoundError(str(path), [str(path)]) from e
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        mark = getattr(e, "problem_mark", None)
        line = (mark.line + 1) if mark else 0
        column = (mark.column + 1) if mark else 0
        raise ConfigParseError(str(path), line, column, str(e)) from e
    return data or {}


def _pydantic_error_to_field_errors(
    exc: ValidationError,
    path: str,
    line_map: dict[str, int],
) -> list[ConfigFieldError]:
    errors: list[ConfigFieldError] = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"])
        value = err.get("input")
        message = err["msg"]
        yaml_line = line_map.get(loc)
        errors.append(ConfigFieldError(loc, value, message, yaml_line))
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
        self._agent_cache: dict[str, AgentConfig] = {}
        self._division_cache: dict[str, DivisionConfig] = {}
        self._harness_cache: Optional[HarnessConfig] = None
        self._org_cache: Optional[OrgConfig] = None
        self._raw_harness_dict: Optional[dict] = None
        self._raw_sources_cache: Optional[tuple[dict, dict, dict, dict]] = None

    # ---------------------------------------------------------------- #
    # Internal helpers
    # ---------------------------------------------------------------- #

    def _search_bases(self) -> tuple[Path, ...]:
        """Config bases in FIRST-WINS order: the workspace layer (when one applies), then global."""
        if self._local_dir is None:
            return (self._config_dir,)
        return (self._local_dir, self._config_dir)

    def _find_file(self, subdir: str, name: str) -> Optional[Path]:
        """Return path to first existing {name}.yaml in local_dir or config_dir."""
        for base in self._search_bases():
            candidate = base / subdir / f"{name}.yaml"
            if candidate.exists():
                return candidate
        return None

    def _validate_dict(self, model_cls: type, data: dict, path: str, yaml_text: str = "") -> Any:
        """Validate data dict through model_cls, raising ConfigValidationError on failure."""
        line_map = _build_line_map(yaml_text) if yaml_text else {}
        try:
            return model_cls.model_validate(data)
        except ValidationError as exc:
            errors = _pydantic_error_to_field_errors(exc, path, line_map)
            raise ConfigValidationError(path, errors) from exc

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

        global_overlay = load_overlay(_resolve_user_overlay_path(self._config_dir))
        global_overlay = {k: v for k, v in global_overlay.items() if k != "agent"}

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

        # deny_patterns carve-out (MERG-02): see _layered_org_deny. Spliced with deep_merge rather
        # than assigned in place for the same non-aliasing reason. Guarded on non-empty: writing an
        # empty list here would REPLACE PermissionConfig's 25 shipped security defaults with
        # nothing, which is a fail-open regression in a file whose whole job is denying things.
        union_deny = self._layered_org_deny()
        if union_deny:
            merged = deep_merge(merged, {"org": {"permissions": {"deny_patterns": union_deny}}})

        result = self._validate_dict(HarnessConfig, merged, str(cfg_path), text)
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
        """Drop the cached HarnessConfig so the next load_harness() re-reads disk.

        Used by `localharness components set` after writing the overlay.
        """
        self._harness_cache = None
        self._raw_harness_dict = None
        self._raw_sources_cache = None

    def load_org(self) -> OrgConfig:
        if self._org_cache is not None:
            return self._org_cache
        org_path = self._config_dir / "org.yaml"
        if not org_path.exists():
            self._org_cache = OrgConfig()
            return self._org_cache
        text = org_path.read_text(encoding="utf-8")
        data = _load_yaml_file(org_path)
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
        text = path.read_text(encoding="utf-8")
        data = _load_yaml_file(path)
        result = self._validate_dict(DivisionConfig, data, str(path), text)
        self._division_cache[name] = result
        return result

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
            legacy = _load_yaml_file(org_path).get("context")
            return legacy if isinstance(legacy, dict) else {}
        return {}

    def _raw_division_context(self, name: str) -> dict:
        """Explicitly-set `context:` block from a division yaml (empty if absent)."""
        path = self._find_file("divisions", name)
        if path is None:
            return {}
        ctx = _load_yaml_file(path).get("context")
        return ctx if isinstance(ctx, dict) else {}

    def load_agent(self, name: str, *, bypass_cache: bool = False) -> AgentConfig:
        if not bypass_cache and name in self._agent_cache:
            return self._agent_cache[name]

        # 1. Find and load raw YAML
        path = self._find_file("agents", name)
        if path is None:
            searched = [str(b / "agents" / f"{name}.yaml") for b in self._search_bases()]
            raise ConfigNotFoundError(name, searched)
        text = path.read_text(encoding="utf-8")
        raw = _load_yaml_file(path)

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

        # Resolve scalar fields: model, temperature, max_tokens
        agent_model = raw.get("model", "inherit")
        div_model = division.model if division else "inherit"
        org_model = org.default_model if org.default_model else "inherit"
        merged["model"] = _resolve_scalar("model", agent_model, div_model, org_model, "inherit")

        agent_temp = raw.get("temperature")
        div_temp = division.temperature if division else None
        org_temp = org.default_temperature
        merged["temperature"] = _resolve_scalar("temperature", agent_temp, div_temp, org_temp, 0.6)

        agent_mt = raw.get("max_tokens")
        div_mt = division.max_tokens if division else None
        org_mt = org.default_max_tokens
        merged["max_tokens"] = _resolve_scalar("max_tokens", agent_mt, div_mt, org_mt, 4096)

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

        # Build union preserving order, deduplicating
        seen: set[str] = set()
        union_deny: list[str] = []
        for pat in (*org_deny, *div_deny, *agent_deny):
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
            # Remove None values so Pydantic uses its own defaults
            resolved_budget = {k: v for k, v in resolved_budget.items() if v is not None}
            perms_merged["budget"] = resolved_budget

        perms_merged["deny_patterns"] = union_deny
        merged["permissions"] = perms_merged

        # 5b. Overlay the user layer's `agent:` section as a LOW-priority default (issue #22):
        #     the same layer `localharness components set agent.*` writes and `components get`
        #     reads back. Per-agent yaml (and org/division inheritance) WINS — `merged` is
        #     layered ON TOP of the overlay. Resolved through THIS loader's config_dir like
        #     load_harness (#35 — no longer env-only, so --config-dir isolates the agent overlay).
        overlay_agent = load_overlay(_resolve_user_overlay_path(self._config_dir)).get("agent")
        if isinstance(overlay_agent, dict) and overlay_agent:
            merged = deep_merge(overlay_agent, merged)

        # 6. Validate merged dict
        line_map = _build_line_map(text)
        try:
            result = AgentConfig.model_validate(merged)
        except ValidationError as exc:
            errors = _pydantic_error_to_field_errors(exc, str(path), line_map)
            raise ConfigValidationError(str(path), errors) from exc

        self._agent_cache[name] = result
        return result

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
        """
        path = self._find_file("agents", name)
        if path is None:
            return base
        text = path.read_text(encoding="utf-8")
        raw = _load_yaml_file(path)  # raises ConfigParseError on malformed YAML
        merged = deep_merge(base.model_dump(), raw)  # base = the defaults; yaml fields win
        return self._validate_dict(AgentConfig, merged, str(path), text)

    def agent_yaml_paths(self) -> list[Path]:
        """Every agent yaml discovery reads, GLOBAL dir first so the local layer wins by stem.

        The ONE discovery order for the whole harness (#150 phase 38): `agent list`, the start
        menu and this loader all read exactly these files in exactly this order. `self._local_dir`
        is the `local_config_dir` constructor hook: the discovered workspace layer when a caller
        named one, else None (no second layer).
        """
        # reversed(): _search_bases is first-wins order, but discover_agents keys a dict by file
        # stem, so the workspace file must be listed LAST to win by overwrite. Global first.
        return [
            f
            for d in (b / "agents" for b in reversed(self._search_bases()))
            if d.exists()
            for f in sorted(d.glob("*.yaml"))
        ]

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
        names: set[str] = set()
        for base in self._search_bases():
            div_dir = base / "divisions"
            if div_dir.exists():
                for f in div_dir.glob("*.yaml"):
                    names.add(f.stem)
        return sorted(names)

    def write_agent(self, config: AgentConfig, *, overwrite: bool = False) -> Path:
        dest = self._config_dir / "agents" / f"{config.name}.yaml"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if not overwrite:
                raise FileExistsError(f"Agent config already exists: {dest}")
            bak = dest.with_suffix(".yaml.bak")
            dest.rename(bak)
        yaml_text = to_yaml_str(config)
        dest.write_text(yaml_text, encoding="utf-8")
        return dest

    def reload(self) -> None:
        self._agent_cache.clear()
        self._division_cache.clear()
        self._harness_cache = None
        self._org_cache = None
        self._raw_sources_cache = None

    def validate_all(self) -> list[tuple[str, Optional[ConfigError]]]:
        results: list[tuple[str, Optional[ConfigError]]] = []

        # harness config
        harness_path = self._config_dir / "config.yaml"
        if harness_path.exists():
            try:
                self.load_harness()
                results.append((str(harness_path), None))
            except ConfigError as e:
                results.append((str(harness_path), e))

        # org config
        org_path = self._config_dir / "org.yaml"
        if org_path.exists():
            try:
                self.reload()
                self.load_org()
                results.append((str(org_path), None))
            except ConfigError as e:
                results.append((str(org_path), e))

        # divisions
        for base in self._search_bases():
            div_dir = base / "divisions"
            if div_dir.exists():
                for f in sorted(div_dir.glob("*.yaml")):
                    try:
                        self.load_division(f.stem, bypass_cache=True)
                        results.append((str(f), None))
                    except ConfigError as e:
                        results.append((str(f), e))

        # agents
        for base in self._search_bases():
            agents_dir = base / "agents"
            if agents_dir.exists():
                for f in sorted(agents_dir.glob("*.yaml")):
                    try:
                        self.load_agent(f.stem, bypass_cache=True)
                        results.append((str(f), None))
                    except ConfigError as e:
                        results.append((str(f), e))

        return results
