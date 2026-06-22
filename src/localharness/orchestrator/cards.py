"""Agent Cards: capability declarations for orchestrator routing."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal

from pydantic import BaseModel, ConfigDict

from localharness.config.defaults import DEFAULT_MAX_CONTEXT_TOKENS

if TYPE_CHECKING:
    from localharness.config.models import AgentConfig


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

    # Routing metadata
    description: str
    capabilities: list[str] = []
    keywords: list[str] = []
    input_types: list[str] = ["task_description"]
    output_types: list[str] = ["markdown_report"]
    example_tasks: list[str] = []

    # Operational metadata
    model: str = "inherit"
    avg_duration_seconds: float = 0.0
    avg_action_count: float = 0.0
    success_rate: float = 1.0
    last_session_at: str | None = None

    # Constraints declared by the agent
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS
    budget_max_actions: int = 100
    budget_max_duration_minutes: float = 30.0

    # Status
    status: Literal["active", "paused", "error"] = "active"


@dataclass(frozen=True)
class RoutingDecision:
    """Result of the routing algorithm for a given task."""

    matched: bool
    agent_id: str | None
    agent_card: AgentCard | None
    confidence: float
    reason: str


def score_card(task: str, card: AgentCard) -> float:
    """
    Score a card against a task string. Returns 0.0-1.0.

    No LLM. No embeddings. Pure string matching.
    Fast enough to score 100 cards in <1ms.
    """
    task_lower = task.lower()
    task_words = set(task_lower.split())

    score = 0.0

    # Keyword overlap: each matching keyword adds 0.15, capped at 0.60
    keyword_hits = sum(1 for kw in card.keywords if kw in task_lower)
    score += min(keyword_hits * 0.15, 0.60)

    # Example task similarity: Jaccard similarity with each example
    max_jaccard = 0.0
    for example in card.example_tasks:
        example_words = set(example.lower().split())
        union = task_words | example_words
        if union:
            sim = len(task_words & example_words) / len(union)
            max_jaccard = max(max_jaccard, sim)
    score += max_jaccard * 0.25

    # Capability phrase match: if any capability phrase appears in task, +0.15
    if any(cap.lower() in task_lower for cap in card.capabilities):
        score += 0.15

    # Agent health penalties
    if card.status == "error":
        score *= 0.5
    if card.success_rate < 0.7:
        score *= 0.8

    return min(score, 1.0)


class AgentCardRegistry:
    """Manages Agent Cards for all configured agents."""

    def __init__(self) -> None:
        self._cards: dict[str, AgentCard] = {}

    def generate_card(self, config: "AgentConfig") -> AgentCard:
        """Generate an AgentCard from an AgentConfig."""
        stopwords = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "that", "this", "and", "or", "for", "to", "of", "in", "on",
            "with", "by", "at", "it", "its", "you", "your", "can", "will",
        }
        text = f"{config.role} {' '.join(config.capabilities)}".lower()
        keywords = [w for w in text.split() if len(w) > 3 and w not in stopwords][:20]
        return AgentCard(
            agent_id=config.name,
            name=config.name,
            division_id=config.division or "default",
            description=config.role[:200],
            capabilities=config.capabilities,
            keywords=keywords,
            model=config.model,
            max_context_tokens=config.context.max_context_tokens if config.context else DEFAULT_MAX_CONTEXT_TOKENS,
            budget_max_actions=(
                config.permissions.budget.max_actions
                if config.permissions and config.permissions.budget
                else 100
            ),
            budget_max_duration_minutes=(
                config.permissions.budget.max_duration_minutes
                if config.permissions and config.permissions.budget
                else 30.0
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

    def route(
        self,
        task: str,
        threshold: float = 0.30,
        tiebreak_fn: "Callable[[str, list[AgentCard]], str] | None" = None,
    ) -> RoutingDecision:
        """Route a task to the best-matching agent.

        Args:
            task: Natural language task description.
            threshold: Minimum score for a match.
            tiebreak_fn: Optional LLM tiebreak callable. Called when the top-2 candidates
                are within 0.10 delta. Signature: (task, [card_a, card_b]) -> agent_id.
                Per user decision: "LLM tiebreak on top-2 within 0.10 delta" (CONTEXT.md).
        """
        scored = [
            (score_card(task, card), card)
            for card in self._cards.values()
            if card.status != "paused"
        ]
        if not scored:
            return RoutingDecision(
                matched=False,
                agent_id=None,
                agent_card=None,
                confidence=0.0,
                reason="No active agents",
            )
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_card = scored[0]

        if best_score < threshold:
            return RoutingDecision(
                matched=False,
                agent_id=None,
                agent_card=None,
                confidence=best_score,
                reason=f"Best score {best_score:.2f} below threshold {threshold}",
            )

        # Check ambiguity: top-2 within 0.10 delta
        ambiguous = len(scored) > 1 and (scored[0][0] - scored[1][0]) < 0.10

        if ambiguous and tiebreak_fn is not None:
            candidates = [scored[0][1], scored[1][1]]
            try:
                winner_id = tiebreak_fn(task, candidates)
                winner_card = next(
                    (c for c in candidates if c.agent_id == winner_id), best_card
                )
                other_card = candidates[1] if winner_card == candidates[0] else candidates[0]
                return RoutingDecision(
                    matched=True,
                    agent_id=winner_card.agent_id,
                    agent_card=winner_card,
                    confidence=best_score,
                    reason=(
                        f"LLM tiebreak chose '{winner_card.agent_id}' over "
                        f"'{other_card.agent_id}' "
                        f"(scores: {scored[0][0]:.2f}, {scored[1][0]:.2f})"
                    ),
                )
            except Exception:
                pass  # Fall through to best-score match on tiebreak failure

        reason = f"Routed to '{best_card.agent_id}' (score: {best_score:.2f})"
        if ambiguous:
            reason += (
                f" — ambiguous with '{scored[1][1].agent_id}' "
                f"(score: {scored[1][0]:.2f}), no tiebreak_fn provided"
            )

        return RoutingDecision(
            matched=True,
            agent_id=best_card.agent_id,
            agent_card=best_card,
            confidence=best_score,
            reason=reason,
        )
