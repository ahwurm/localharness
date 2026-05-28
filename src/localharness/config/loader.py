"""ConfigLoader: YAML parse, validate, inheritance resolve, write."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import ValidationError
from pydantic_yaml import to_yaml_str

from .models import AgentConfig, DivisionConfig, HarnessConfig, OrgConfig
from localharness.config.overlay import (
    deep_merge,
    load_overlay,
    _resolve_user_overlay_path,
)


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
        self._config_dir = Path(config_dir or "~/.localharness").expanduser()
        self._local_dir = Path(local_config_dir or ".localharness")
        self._agent_cache: dict[str, AgentConfig] = {}
        self._division_cache: dict[str, DivisionConfig] = {}
        self._harness_cache: Optional[HarnessConfig] = None
        self._org_cache: Optional[OrgConfig] = None
        self._raw_harness_dict: Optional[dict] = None

    # ---------------------------------------------------------------- #
    # Internal helpers
    # ---------------------------------------------------------------- #

    def _find_file(self, subdir: str, name: str) -> Optional[Path]:
        """Return path to first existing {name}.yaml in local_dir or config_dir."""
        for base in (self._local_dir, self._config_dir):
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

    def load_harness(self) -> HarnessConfig:
        if self._harness_cache is not None:
            return self._harness_cache
        cfg_path = self._config_dir / "config.yaml"
        if not cfg_path.exists():
            raise ConfigNotFoundError("config.yaml", [str(cfg_path)])
        text = cfg_path.read_text(encoding="utf-8")
        project_data = _load_yaml_file(cfg_path)
        # Cache raw project dict so callers (e.g. components_cmd) can rebuild merged config
        self._raw_harness_dict = project_data

        # Apply user overlay cascade (Phase 14)
        overlay_path = _resolve_user_overlay_path()
        user_overlay = load_overlay(overlay_path)
        merged = deep_merge(project_data, user_overlay) if user_overlay else project_data

        result = self._validate_dict(HarnessConfig, merged, str(cfg_path), text)
        self._harness_cache = result
        return result

    def raw_harness_dict(self) -> dict:
        """Return the parsed project YAML dict (NO overlay applied).

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
        return _resolve_user_overlay_path()

    def invalidate_cache(self) -> None:
        """Drop the cached HarnessConfig so the next load_harness() re-reads disk.

        Used by `localharness components set` after writing the overlay.
        """
        self._harness_cache = None
        self._raw_harness_dict = None

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
            searched = [
                str(self._local_dir / "divisions" / f"{name}.yaml"),
                str(self._config_dir / "divisions" / f"{name}.yaml"),
            ]
            raise ConfigNotFoundError(name, searched)
        text = path.read_text(encoding="utf-8")
        data = _load_yaml_file(path)
        result = self._validate_dict(DivisionConfig, data, str(path), text)
        self._division_cache[name] = result
        return result

    def load_agent(self, name: str, *, bypass_cache: bool = False) -> AgentConfig:
        if not bypass_cache and name in self._agent_cache:
            return self._agent_cache[name]

        # 1. Find and load raw YAML
        path = self._find_file("agents", name)
        if path is None:
            searched = [
                str(self._local_dir / "agents" / f"{name}.yaml"),
                str(self._config_dir / "agents" / f"{name}.yaml"),
            ]
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

        # 5. Union deny_patterns: org + division + agent (additive)
        agent_perms = raw.get("permissions") or {}
        agent_deny = agent_perms.get("deny_patterns", []) if isinstance(agent_perms, dict) else []

        div_deny: list[str] = []
        if division:
            div_deny = division.permissions.deny_patterns

        org_deny = org.permissions.deny_patterns

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

        # 6. Validate merged dict
        line_map = _build_line_map(text)
        try:
            result = AgentConfig.model_validate(merged)
        except ValidationError as exc:
            errors = _pydantic_error_to_field_errors(exc, str(path), line_map)
            raise ConfigValidationError(str(path), errors) from exc

        self._agent_cache[name] = result
        return result

    def list_agents(self) -> list[str]:
        names: set[str] = set()
        for base in (self._local_dir, self._config_dir):
            agents_dir = base / "agents"
            if agents_dir.exists():
                for f in agents_dir.glob("*.yaml"):
                    names.add(f.stem)
        return sorted(names)

    def list_divisions(self) -> list[str]:
        names: set[str] = set()
        for base in (self._local_dir, self._config_dir):
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
        for base in (self._local_dir, self._config_dir):
            div_dir = base / "divisions"
            if div_dir.exists():
                for f in sorted(div_dir.glob("*.yaml")):
                    try:
                        self.load_division(f.stem, bypass_cache=True)
                        results.append((str(f), None))
                    except ConfigError as e:
                        results.append((str(f), e))

        # agents
        for base in (self._local_dir, self._config_dir):
            agents_dir = base / "agents"
            if agents_dir.exists():
                for f in sorted(agents_dir.glob("*.yaml")):
                    try:
                        self.load_agent(f.stem, bypass_cache=True)
                        results.append((str(f), None))
                    except ConfigError as e:
                        results.append((str(f), e))

        return results
