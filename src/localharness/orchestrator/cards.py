"""Agent Cards: the capability declarations `/agents` lists."""
from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from localharness.config.defaults import DEFAULT_MAX_CONTEXT_TOKENS
from localharness.config.loader import (
    AgentFile,
    AgentProblem,
    ConfigError,
    ConfigLoader,
    ConfigNotFoundError,
    ConfigParseError,
    ConfigValidationError,
    Origin,
)
from localharness.config.models import AgentConfig
from localharness.tools.builtin.agent_tool import BUILTIN_AGENT_NAMES, INVALID_AGENT_NAME

log = logging.getLogger(__name__)


class AgentCard(BaseModel):
    """
    JSON schema for agent capability declaration.
    Stored at ~/.localharness/agents/{agent_id}/agent_card.json.
    Regenerated automatically when agent config changes.

    Designed to be compact: an orchestrator managing 50 agents should
    be able to hold all 50 cards in context under 8K tokens total.
    """

    model_config = ConfigDict(frozen=True)

    # Identity
    agent_id: str
    name: str
    division_id: str = "default"
    org_id: str = "default"
    version: int = 1

    # Capability metadata
    description: str
    capabilities: list[str] = []
    input_types: list[str] = ["task_description"]
    output_types: list[str] = ["markdown_report"]

    # Operational metadata
    model: str = "inherit"
    avg_duration_seconds: float = 0.0
    avg_action_count: float = 0.0
    last_session_at: str | None = None

    # Constraints declared by the agent
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS
    budget_max_actions: int | None = None
    budget_max_duration_minutes: float | None = None   # None = no turn time limit

    # Status
    origin: Origin | None = None


def _shape_reason(path: Path) -> str | None:
    """Why a file that is not UTF-8 or whose top level is not a mapping cannot load, else None."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return "not valid UTF-8"
    except Exception:  # noqa: BLE001 - unreadable or unparseable files are classified elsewhere
        return None
    return None if data is None or isinstance(data, dict) else "top level is not a mapping"


def _problem(f: AgentFile, exc: ConfigError) -> AgentProblem:
    """The problem record for a file that did not load, classified from the exception alone."""
    from localharness.channels.base import one_line

    if isinstance(exc, ConfigParseError):
        kind, reason, fix = "parse", f"{exc.line}:{exc.column}: {exc.message}", "edit the file"
    elif isinstance(exc, ConfigValidationError):
        kind, fix = "validation", "edit the file"
        reason = "; ".join(f"{e.field_path}: {e.message}" for e in exc.errors)
    elif isinstance(exc, ConfigNotFoundError) and str(exc.__cause__) == INVALID_AGENT_NAME:
        kind, reason, fix = "invalid-name", INVALID_AGENT_NAME, "rename the file"
    else:
        kind, reason, fix = "load", "does not load", "run localharness validate"
    return AgentProblem(f.path, f.origin, kind, one_line(reason)[:200], fix)


class AgentCardRegistry:
    """Manages Agent Cards for all configured agents."""

    def __init__(self, loader: ConfigLoader | None = None) -> None:
        self._cards: dict[str, AgentCard] = {}
        self.loader = loader

    def __iter__(self) -> Iterator[str]:
        rest = sorted(
            (c for c in self._cards.values() if c.agent_id not in (*BUILTIN_AGENT_NAMES, "orchestrator")),
            key=lambda c: (c.origin != "project", c.agent_id),
        )
        return iter([*BUILTIN_AGENT_NAMES, *(c.agent_id for c in rest)])

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def load(self, name: str) -> AgentConfig:
        """Load one agent through the loader's narrowing, register its card under the requested stem."""
        if self.loader is None:
            raise ConfigNotFoundError(name, [])
        try:
            AgentConfig.validate_name_format(name)
        except ValueError:
            raise ConfigNotFoundError(name, self.loader.agent_search_paths(name)) from ValueError(INVALID_AGENT_NAME)
        cfg = self.loader.load_agent(name, bypass_cache=True)
        found = self.loader.agent_file(name)
        card = self.generate_card(cfg.model_copy(update={"name": name}))
        self.register(card.model_copy(update={"origin": found.origin if found else None}))
        return cfg

    def refresh(self) -> list[AgentProblem]:
        """Register every winning agent file not yet registered; return one problem per file that did not load."""
        from localharness.channels.base import one_line

        if self.loader is None:
            return []
        problems: list[AgentProblem] = []
        for stem, f in self.loader.agent_winners().items():
            if stem in self._cards:
                continue
            try:
                cfg = self.load(stem)
            except ConfigError as exc:
                problem = _problem(f, exc)
            except Exception:
                reason = _shape_reason(f.path)
                if reason is None:
                    log.debug("agent file %s was skipped", f.path, exc_info=True)
                    continue
                problem = AgentProblem(f.path, f.origin, "parse", one_line(reason)[:200], "edit the file")
            else:
                if cfg.name == stem:
                    continue
                reason = one_line(f"name: {cfg.name!r} ignored; {stem}.yaml is agent {stem!r}")[:200]
                problem = AgentProblem(f.path, f.origin, "mismatch", reason, "rename the file or the name: to match")
            if stem not in BUILTIN_AGENT_NAMES:
                problems.append(problem)
        return problems

    def generate_card(self, config: "AgentConfig") -> AgentCard:
        """Generate an AgentCard from an AgentConfig."""
        return AgentCard(
            agent_id=config.name,
            name=config.name,
            division_id=config.division or "default",
            description=config.role[:200],
            capabilities=config.capabilities,
            model=config.model,
            max_context_tokens=config.context.max_context_tokens if config.context else DEFAULT_MAX_CONTEXT_TOKENS,
            budget_max_actions=(
                config.permissions.budget.max_actions
                if config.permissions and config.permissions.budget
                else None
            ),
            budget_max_duration_minutes=(
                config.permissions.budget.max_duration_minutes
                if config.permissions and config.permissions.budget
                else None
            ),
        )

    def register(self, card: AgentCard) -> None:
        self._cards[card.agent_id] = card

    def register_from_config(self, config: "AgentConfig") -> AgentCard:
        card = self.generate_card(config)
        self.register(card)
        return card

    def get(self, agent_id: str) -> AgentCard | None:
        return self._cards.get(agent_id)

    def all_cards(self) -> list[AgentCard]:
        return list(self._cards.values())
