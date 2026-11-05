"""Agent package: AgentLoop, Session, guardrail components, ContextManager, PermissionEvaluator."""
from localharness.agent.loop import (
    AgentLoop,
    Session,
    BudgetTracker,
    BudgetViolation,
    KillWatcher,
    StepResult,
)
from localharness.agent.context import ContextManager
from localharness.agent.permissions import PermissionEvaluator, PermissionResult

__all__ = [
    "AgentLoop",
    "Session",
    "BudgetTracker",
    "BudgetViolation",
    "KillWatcher",
    "StepResult",
    "ContextManager",
    "PermissionEvaluator",
    "PermissionResult",
]
