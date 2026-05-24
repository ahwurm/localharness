# Spec 07: Orchestrator

**Component:** `src/localharness/orchestrator/`
**Requirements:** ORCH-01, ORCH-02, ORCH-03, ORCH-04
**Status:** v1

---

## Purpose

The orchestrator is the user-facing entry point to LocalHarness. It is intentionally thin: it routes, synthesizes, escalates, and manages agent creation workflows. It does not do domain work. It does not load file contents. It does not run multi-step reasoning on domain data.

The single most important invariant: **the orchestrator's context window stays at 10-15% utilization at all times**. This is enforced structurally — the orchestrator receives only summaries and Agent Cards, never file contents or full agent histories.

Architecture provenance: Lean orchestrator / fat subagent pattern from GSD analysis, Agent Cards from the A2A protocol, hub-and-spoke communication from the Orchestrator-Subagent pattern in the Anthropic multi-agent canonical patterns.

---

## Orchestrator Class

```python
# src/localharness/orchestrator/router.py

from dataclasses import dataclass, field
from typing import Any
from localharness.core.events import (
    EventBus, UserMessage, TaskRequest, TaskComplete,
    DelegateResult, Escalation, AgentCreated, SystemReady
)
from localharness.config.models import AgentConfig, OrgConfig
from localharness.orchestrator.cards import AgentCard, AgentCardRegistry
from localharness.orchestrator.workflow import AgentCreationWorkflow
from localharness.provider.client import LLMClient

@dataclass(frozen=True)
class DelegateResult:
    """
    The only object an agent ever returns to the orchestrator.
    Never contains message history. Never contains file contents.
    
    This enforces the lean orchestrator invariant at the type level.
    """
    agent_id: str
    session_id: str
    summary: str                     # ≤500 words, human-readable
    artifact_paths: list[str]        # absolute paths to output files
    exit_reason: str                 # 'complete' | 'budget' | 'stuck' | 'error'
    turn_count: int
    action_count: int
    tokens_used: int
    duration_seconds: float
    error: str | None = None         # populated when exit_reason == 'error'

@dataclass(frozen=True)
class RoutingDecision:
    """Result of the routing algorithm for a given task."""
    matched: bool
    agent_id: str | None
    agent_card: "AgentCard | None"
    confidence: float               # 0.0–1.0 from keyword matching score
    reason: str                     # human-readable routing rationale

@dataclass(frozen=True)
class SynthesisResult:
    """Combined result from multiple delegated agents."""
    summary: str
    agent_results: list[DelegateResult]
    artifact_paths: list[str]       # merged deduplicated artifact paths

class Orchestrator:
    """
    Thin router, synthesizer, and conversation manager.
    
    Lifecycle:
      1. Instantiated with config and LLM client
      2. start() emits SystemReady and begins event bus subscription
      3. Processes events in the asyncio event loop indefinitely
      4. stop() gracefully shuts down

    The orchestrator subscribes to: UserMessage, TaskComplete, Escalation.
    It publishes: TaskRequest, AgentCreated, SystemReady.
    
    All agent communication is mediated through the event bus.
    The orchestrator never holds a direct reference to an agent loop instance.
    """

    def __init__(
        self,
        bus: EventBus,
        llm: LLMClient,
        org_config: OrgConfig,
        card_registry: AgentCardRegistry,
        workflow: "AgentCreationWorkflow",
    ) -> None: ...

    async def start(self) -> None:
        """
        Subscribe to event bus topics, emit SystemReady event.
        Start the conversation loop (orchestrator introduces itself via terminal channel).
        
        The introduction message is sent as a UserMessage reply through the terminal
        channel. It is not an event — it is a formatted string sent directly to the
        channel adapter.
        
        Raises:
            OrchestratorStartError: If event bus subscription fails.
        """
        ...

    async def stop(self) -> None:
        """
        Graceful shutdown. Waits for any in-progress delegation to complete
        (with a 30s timeout), then unsubscribes from event bus.
        """
        ...

    async def handle_user_message(self, event: UserMessage) -> None:
        """
        Main message dispatch. Called by event bus when UserMessage arrives.
        
        Decision tree:
          1. Is this a setup/config command? → handle_setup_command()
          2. Does it match an existing agent? → route() → delegate()
          3. Is it a create-agent request? → workflow.start()
          4. Is it a compound task requiring multiple agents? → multi_delegate()
          5. No match → unknown_intent_response()
        
        Each branch is async and runs to completion before the next UserMessage
        is processed. The orchestrator processes one message at a time (no parallel
        user intents in v1).
        """
        ...

    async def route(self, task: str) -> RoutingDecision:
        """
        Match a task description to the best-fit agent via Agent Card scoring.
        
        Algorithm: see Routing Algorithm section.
        
        Returns a RoutingDecision. If matched=False, the caller must handle
        the no-match case (create new agent, ask for clarification, or escalate).
        """
        ...

    async def delegate(
        self,
        agent_id: str,
        task: str,
        task_file: str | None = None,
        budget_override: dict[str, Any] | None = None,
    ) -> DelegateResult:
        """
        Delegate a task to a specific agent.
        
        Creates a TaskRequest event with:
          - agent_id
          - task_file path (if task was written to a file)
          - budget from agent config (overridable)
          - session_id (new UUID4)
        
        The orchestrator then waits for the corresponding TaskComplete event
        (matched by session_id) with a timeout of budget.max_duration_minutes + 60s.
        
        CRITICAL: The orchestrator passes task_file (a path) to the agent,
        NOT the task contents. The agent reads the file with its own fresh context.
        If task is short (≤200 chars), it is passed inline as task_inline instead.
        
        Args:
            agent_id: Target agent's ID.
            task: Task description (short) or summary (if long task is in task_file).
            task_file: Absolute path to a file containing the full task. Optional.
            budget_override: Override max_actions or max_duration_minutes for this call.
        
        Returns:
            DelegateResult from the agent. Never raises on agent failure —
            agent failures surface as DelegateResult(exit_reason='error').
        
        Raises:
            DelegationTimeoutError: If agent does not respond within timeout.
            AgentNotFoundError: If agent_id is not registered.
        """
        ...

    async def multi_delegate(
        self,
        tasks: list[tuple[str, str]],  # list of (agent_id, task)
    ) -> SynthesisResult:
        """
        Delegate to multiple agents sequentially and synthesize results.
        
        v1: Sequential delegation (no parallelism). Each agent runs to completion
        before the next starts. v2 will add wave-based parallelism.
        
        After all delegations, calls synthesize() to combine results.
        
        Raises:
            DelegationTimeoutError: If any single agent times out.
        """
        ...

    async def synthesize(
        self,
        results: list[DelegateResult],
        original_task: str,
    ) -> SynthesisResult:
        """
        Combine multiple DelegateResult summaries into a coherent response.
        
        The orchestrator uses its LLM client to generate the synthesis.
        Input to the LLM: only the summaries (strings) and artifact_paths lists.
        Never the full agent histories.
        
        The synthesis prompt is kept under 2000 tokens total (enforced by truncating
        individual summaries if needed). This maintains the 10-15% context invariant.
        
        Returns a SynthesisResult with merged summary and deduplicated artifact paths.
        """
        ...

    async def escalate(
        self,
        agent_id: str,
        session_id: str,
        reason: str,
        context: dict[str, Any],
    ) -> None:
        """
        Handle escalation from an agent (stuck, budget, error).
        
        Escalation protocol:
          1. Log the escalation event to audit.
          2. If reason == 'stuck': attempt one retry with a different approach hint.
          3. If reason == 'budget': report to user; offer to extend budget or abandon.
          4. If reason == 'error': surface error to user with recovery options.
          5. If retry also fails: escalate to human via terminal channel message.
        
        The escalation message to the user is sent directly through the terminal
        channel (not through the agent). It includes: which agent, what it was
        doing, what went wrong, what the user can do.
        """
        ...

    async def unknown_intent_response(self, message: str) -> str:
        """
        Generate a helpful response when no agent matches and no workflow triggered.
        
        Uses the LLM to suggest: which existing agents might help, whether a new
        agent should be created, or if the request is out of scope.
        
        Returns the response text (caller sends it to the terminal channel).
        """
        ...
```

---

## Agent Cards

### Schema

Agent Cards are JSON files that describe what an agent can do. They are the only information the orchestrator uses for routing — never the agent's full config or memory.

```python
# src/localharness/orchestrator/cards.py

from pydantic import BaseModel, ConfigDict
from typing import Literal

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
    division_id: str
    org_id: str
    version: int                    # incremented on config change

    # Routing metadata
    description: str                # ≤200 chars: one-line capability summary
    capabilities: list[str]         # list of verb phrases: "search the web", "analyze stocks"
    keywords: list[str]             # trigger words for routing: ["market", "portfolio", "stock"]
    input_types: list[str]          # what the agent accepts: ["task_description", "file_path", "url"]
    output_types: list[str]         # what the agent produces: ["markdown_report", "json_data", "code"]
    example_tasks: list[str]        # 2-3 concrete examples, each ≤100 chars

    # Operational metadata
    model: str                      # model in use (for orchestrator to assess capability)
    avg_duration_seconds: float     # rolling average of recent sessions (default 0.0)
    avg_action_count: float         # rolling average
    success_rate: float             # 0.0–1.0, rolling over last 20 sessions
    last_session_at: int | None     # Unix timestamp of last run, None if never run

    # Constraints declared by the agent
    max_context_tokens: int
    budget_max_actions: int
    budget_max_duration_minutes: int

    # Status
    status: Literal["active", "inactive", "error"]  # "error" = last session failed
```

#### agent_card.json file example

```json
{
  "agent_id": "morning-briefing",
  "name": "Morning Briefing",
  "division_id": "financial",
  "org_id": "default",
  "version": 3,
  "description": "Generates a daily market intelligence report with news, portfolio context, and thesis updates.",
  "capabilities": [
    "search financial news",
    "analyze portfolio positions",
    "summarize market conditions",
    "generate formatted markdown reports"
  ],
  "keywords": ["morning", "briefing", "market", "portfolio", "stocks", "financial", "news", "report"],
  "input_types": ["task_description"],
  "output_types": ["markdown_report"],
  "example_tasks": [
    "Generate today's morning market briefing",
    "What happened in markets this week?",
    "Summarize my portfolio performance"
  ],
  "model": "qwen3.5-122b-a10b",
  "avg_duration_seconds": 47.3,
  "avg_action_count": 8.2,
  "success_rate": 0.95,
  "last_session_at": 1748042400,
  "max_context_tokens": 131072,
  "budget_max_actions": 100,
  "budget_max_duration_minutes": 30,
  "status": "active"
}
```

### AgentCardRegistry

```python
class AgentCardRegistry:
    """
    Manages Agent Cards for all configured agents.
    
    Loaded at orchestrator start from ~/.localharness/agents/*/agent_card.json.
    Updated in-memory when AgentCreated events are received.
    Written to disk when update() is called.
    """

    def __init__(self, base_dir: str) -> None: ...

    async def load_all(self) -> None:
        """
        Scan ~/.localharness/agents/*/agent_card.json and load all valid cards.
        Invalid or missing cards log a warning but do not raise.
        """
        ...

    def get(self, agent_id: str) -> AgentCard | None:
        """Get a card by agent_id. Returns None if not found."""
        ...

    def all(self) -> list[AgentCard]:
        """Return all registered agent cards."""
        ...

    async def update(self, card: AgentCard) -> None:
        """
        Update a card in the registry and write to disk.
        Overwrites ~/.localharness/agents/{card.agent_id}/agent_card.json atomically.
        """
        ...

    async def update_stats(
        self,
        agent_id: str,
        session_result: DelegateResult,
    ) -> None:
        """
        Update rolling averages after a session completes.
        Recomputes avg_duration_seconds, avg_action_count, success_rate.
        Rolling window: last 20 sessions (tracked in agent SQLite, not in the card).
        """
        ...

    def generate_card(self, config: AgentConfig) -> AgentCard:
        """
        Generate an AgentCard from an AgentConfig.
        Called on agent creation and config update.
        
        Extracts keywords from config.role and config.description using
        simple tokenization (split, lowercase, filter stopwords).
        Sets all performance stats to defaults (0.0 / None) for new agents.
        """
        ...
```

---

## Routing Algorithm

The routing algorithm must be fast (no LLM call) and deterministic. It scores each registered agent card against the task string using keyword matching.

### Algorithm

```python
def score_card(task: str, card: AgentCard) -> float:
    """
    Score a card against a task string. Returns 0.0–1.0.
    
    No LLM. No embeddings. Pure string matching.
    Fast enough to score 100 cards in <1ms.
    """
    task_lower = task.lower()
    task_words = set(task_lower.split())

    score = 0.0

    # Keyword overlap: each matching keyword adds 0.15, capped at 0.6
    keyword_hits = sum(1 for kw in card.keywords if kw in task_lower)
    score += min(keyword_hits * 0.15, 0.60)

    # Example task similarity: Jaccard similarity with each example
    # Take max similarity across all examples, add up to 0.25
    max_example_sim = 0.0
    for example in card.example_tasks:
        example_words = set(example.lower().split())
        if task_words | example_words:
            sim = len(task_words & example_words) / len(task_words | example_words)
            max_example_sim = max(max_example_sim, sim)
    score += max_example_sim * 0.25

    # Capability phrase match: if any capability phrase appears in task, +0.15
    for cap in card.capabilities:
        if cap.lower() in task_lower:
            score += 0.15
            break

    # Agent health penalty: degraded agents score lower
    if card.status == "error":
        score *= 0.5
    if card.success_rate < 0.7:
        score *= 0.8

    return min(score, 1.0)

def route(task: str, cards: list[AgentCard]) -> RoutingDecision:
    if not cards:
        return RoutingDecision(matched=False, agent_id=None, agent_card=None,
                               confidence=0.0, reason="No agents registered")

    scored = [(score_card(task, card), card) for card in cards
              if card.status != "inactive"]
    scored.sort(key=lambda x: x[0], reverse=True)

    best_score, best_card = scored[0]

    MATCH_THRESHOLD = 0.30   # below this = no match
    AMBIGUOUS_THRESHOLD = 0.10  # if top two scores within this delta = ambiguous

    if best_score < MATCH_THRESHOLD:
        return RoutingDecision(matched=False, agent_id=None, agent_card=None,
                               confidence=best_score,
                               reason=f"Best match '{best_card.name}' scored {best_score:.2f}, below threshold {MATCH_THRESHOLD}")

    if len(scored) > 1 and (scored[0][0] - scored[1][0]) < AMBIGUOUS_THRESHOLD:
        # Ambiguous: ask the LLM to break the tie using card descriptions only
        # This is the only routing-related LLM call; uses <500 tokens total
        return _llm_tiebreak(task, scored[:3])

    return RoutingDecision(
        matched=True,
        agent_id=best_card.agent_id,
        agent_card=best_card,
        confidence=best_score,
        reason=f"Matched '{best_card.name}' (score={best_score:.2f})"
    )
```

### LLM Tiebreak

When the top two candidates are within 0.10 of each other, the orchestrator makes a single LLM call to resolve the ambiguity. The prompt contains only:
- The task string (≤500 chars)
- Agent names and descriptions (each ≤200 chars)

The LLM returns a single agent name or "none". The orchestrator parses this deterministically (exact string match against agent names). If parsing fails, it returns the highest-scored card.

This LLM call consumes <500 tokens. It is the only LLM call the routing phase makes.

---

## Agent Creation Workflow (discuss → configure → deploy)

```python
# src/localharness/orchestrator/workflow.py

from dataclasses import dataclass, field
from typing import Any
from localharness.config.models import AgentConfig

@dataclass
class WorkflowState:
    """
    Mutable state for a single agent creation workflow.
    Stored in-memory on the orchestrator for the duration of the conversation.
    Persisted to ~/.localharness/workflows/{workflow_id}.json for crash recovery.
    """
    workflow_id: str                # UUID4
    stage: str                      # 'discuss' | 'configure' | 'confirm' | 'deploy'
    user_intent: str                # original user message that triggered creation
    gathered_info: dict[str, Any]   # structured info gathered in 'discuss' stage
    draft_config: AgentConfig | None = None   # populated in 'configure' stage
    confirmed: bool = False
    created_agent_id: str | None = None

class AgentCreationWorkflow:
    """
    Conversational workflow for creating a new agent.
    
    Three stages:
    
    1. DISCUSS: Orchestrator asks focused questions to gather:
       - What should the agent do? (role)
       - What tools does it need? (tool list)
       - Should it run on a schedule? (schedule)
       - Any restrictions on what it can do? (deny_patterns)
       
       The discuss stage asks at most 4 questions. If the user's initial
       message already answered all of them, it skips directly to configure.
    
    2. CONFIGURE: Orchestrator generates a YAML config from gathered_info.
       Presents it to the user as a formatted YAML block.
       User can: approve, request changes (loops back to discuss), or cancel.
    
    3. DEPLOY: On confirmation:
       - Write YAML to ~/.localharness/agents/{agent_id}.yaml
       - Generate Agent Card and register with AgentCardRegistry
       - Emit AgentCreated event
       - Report success to user
    """

    def __init__(
        self,
        bus: EventBus,
        llm: LLMClient,
        card_registry: AgentCardRegistry,
        base_dir: str,
    ) -> None: ...

    def is_creation_intent(self, message: str) -> bool:
        """
        Detect if a user message is requesting agent creation.
        
        Pattern matching (no LLM): checks for phrases like:
        "create an agent", "make an agent", "new agent", "add an agent",
        "build an agent", "set up an agent".
        
        Returns True if any pattern matches (case-insensitive).
        """
        ...

    async def start(self, user_message: str) -> str:
        """
        Begin a new agent creation workflow.
        
        Parses the user_message for already-provided information,
        then generates the first question (or skips to configure if complete).
        
        Returns the orchestrator's first response string.
        """
        ...

    async def continue_conversation(
        self,
        workflow_id: str,
        user_message: str,
    ) -> tuple[str, bool]:
        """
        Continue an in-progress workflow with a new user message.
        
        Returns (response_text, is_complete).
        is_complete=True means the workflow ended (deployed or cancelled).
        
        Stage transitions:
          discuss → configure: when all required info is gathered
          configure → confirm: when YAML is presented to user
          confirm → deploy: when user approves ("yes", "ok", "looks good", "deploy")
          confirm → discuss: when user requests changes ("change", "update", "modify")
          any → cancelled: when user says "cancel", "never mind", "stop"
        """
        ...

    async def _gather_info(
        self,
        state: WorkflowState,
        user_message: str,
    ) -> str:
        """
        Process a user message in the discuss stage.
        
        Updates state.gathered_info with extracted information.
        Returns next question or "COMPLETE" when all required fields are gathered.
        
        Required fields: role (str), tools (list[str]).
        Optional fields: schedule (str | None), deny_patterns (list[str]).
        
        Extraction uses the LLM with a structured extraction prompt.
        The extraction prompt is ≤1000 tokens and returns JSON.
        """
        ...

    async def _generate_config(self, state: WorkflowState) -> AgentConfig:
        """
        Generate a validated AgentConfig from gathered_info.
        
        Uses the LLM to generate the YAML text, then parses it with
        the config loader. If validation fails, retries once with the
        validation error included in the prompt. If retry fails, raises
        ConfigGenerationError.
        
        The agent_id is derived from the name: lowercase, spaces→hyphens, 
        max 32 chars, alphanumeric+hyphens only.
        """
        ...

    async def _deploy(self, state: WorkflowState) -> str:
        """
        Write config to disk, generate card, emit AgentCreated.
        
        File write is atomic (write to .tmp, then os.replace).
        
        Returns success message string for display to user.
        
        Raises:
            ConfigWriteError: On file I/O failure.
        """
        ...

    def get_active_workflow(self, user_message: str) -> WorkflowState | None:
        """
        Check if there's an active workflow expecting input.
        Returns the WorkflowState if yes, None if no active workflow.
        
        Matching: if the orchestrator has exactly one active workflow, any message
        is routed to it. If there are multiple active workflows (edge case), match
        by workflow_id prefix in the message (user can address a specific workflow).
        """
        ...
```

### Workflow LLM Prompt Budget

The orchestrator uses its LLM client in the workflow. Context budget is strictly managed:

| Workflow call | Token budget |
|---|---|
| Initial intent extraction | ≤500 tokens |
| Per-question discussion turn | ≤1000 tokens |
| Config generation | ≤2000 tokens |
| Tiebreak routing | ≤500 tokens |
| Synthesis (per agent result) | ≤200 tokens |
| Synthesis total | ≤2000 tokens |

The orchestrator's system prompt is ≤500 tokens. These limits are enforced by the context manager before each LLM call.

---

## Lean Context Enforcement

The orchestrator enforces its own context budget. This is distinct from agent context management (which handles 80%-window compaction).

```python
class OrchestratorContextGuard:
    """
    Ensures orchestrator stays at 10-15% of model context.
    
    The orchestrator's LLM client has max_context_tokens set to:
        min(org_config.orchestrator_context_cap, llm_model.max_context)
    
    Default: org_config.orchestrator_context_cap = 16000 tokens
    (This is 10-15% of a 128K model; adjust for different model sizes.)
    
    If the orchestrator's accumulated conversation exceeds this cap,
    it summarizes the conversation history (keeping only last 3 turns)
    and continues. The orchestrator never delegates this summarization
    to a subagent — it uses its own LLM call with the truncated context.
    """

    def __init__(self, max_tokens: int) -> None: ...

    def check(self, messages: list[dict]) -> bool:
        """Returns True if adding these messages would exceed the cap."""
        ...

    def trim(self, messages: list[dict]) -> list[dict]:
        """
        Trim orchestrator message history to stay within cap.
        Strategy: keep system prompt + last 3 user/assistant exchanges.
        Older messages are summarized into a single system message prepended
        after the system prompt.
        """
        ...
```

---

## Error Handling

### Error Types

```python
# src/localharness/orchestrator/errors.py

class OrchestratorError(Exception):
    """Base class for orchestrator errors."""

class AgentNotFoundError(OrchestratorError):
    """No agent registered with the given agent_id."""
    def __init__(self, agent_id: str) -> None: ...

class NoMatchingAgentError(OrchestratorError):
    """Routing found no agent with confidence >= MATCH_THRESHOLD."""
    def __init__(self, task: str, best_score: float) -> None: ...

class DelegationTimeoutError(OrchestratorError):
    """Agent did not return TaskComplete within the allowed time."""
    def __init__(self, agent_id: str, session_id: str, timeout_seconds: float) -> None: ...

class OrchestratorStartError(OrchestratorError):
    """Orchestrator failed to start (event bus subscription failure, config error)."""

class ConfigGenerationError(OrchestratorError):
    """LLM failed to generate valid YAML config for a new agent after retries."""
    def __init__(self, validation_error: str) -> None: ...

class ConfigWriteError(OrchestratorError):
    """Failed to write agent YAML config to disk."""
    def __init__(self, path: str, underlying: Exception) -> None: ...
```

### No Matching Agent

When `route()` returns `matched=False`, the orchestrator:
1. Checks if `is_creation_intent()` matches — if yes, starts the workflow.
2. Otherwise, calls `unknown_intent_response()` to generate a helpful message listing:
   - Available agents and their one-line descriptions
   - Suggestion: "Would you like me to create a new agent for this?"
3. The user's message is not lost — it is stored in the orchestrator conversation history so if the user says "yes, create one", the workflow picks up the intent automatically.

### Agent Failure

When a `TaskComplete` event arrives with `exit_reason='error'`, the orchestrator:
1. Calls `escalate()` with reason='error'.
2. The escalation surfaces to the user with the error message and options.
3. Options offered: retry (re-delegates same task), debug (shows the last N tool calls from the agent's history.jsonl), abandon (marks task as failed).

### Delegation Timeout

`DelegationTimeoutError` is raised by `delegate()` if `TaskComplete` does not arrive within `budget_max_duration_minutes * 60 + 60` seconds. The orchestrator:
1. Publishes a kill signal (via a KILL file in the agent's directory).
2. Waits 5 seconds for clean shutdown.
3. Surfaces `DelegationTimeoutError` to the user.
4. Updates the Agent Card's `success_rate` (counts as a failure).

---

## Configuration

```yaml
# In org-level config.yaml, under orchestrator:
orchestrator:
  context_cap_tokens: 16000          # orchestrator max context (10-15% of model window)
  match_threshold: 0.30              # routing score threshold (0.0–1.0)
  ambiguous_delta: 0.10              # threshold for LLM tiebreak
  delegation_timeout_slack_seconds: 60  # extra seconds beyond budget timeout
  max_active_workflows: 3            # max simultaneous agent creation workflows
  introduction_message: |            # shown to user on first start
    Hi! I'm the LocalHarness orchestrator. I manage your agents and route tasks.
    You can ask me to create a new agent, run an existing one, or just describe
    what you need and I'll figure out who to send it to.
    Type 'list agents' to see what's available.
```

---

## Implementation Notes

- The orchestrator is a single asyncio coroutine processing events from a queue. It does not use `asyncio.create_task` to parallelize user messages — one message at a time, in order. This simplifies state management for the workflow conversation.
- Agent Card files are written atomically (tmp + rename) so the registry is never in an inconsistent state on a partial write.
- The `introduce_yourself` message is sent exactly once, on the first `SystemReady` event. Subsequent starts (after `localharness start` re-run) detect that the config already exists and use a shorter greeting: "Welcome back. {N} agents available."
- Workflow state is persisted to `~/.localharness/workflows/{workflow_id}.json` on every stage transition. If the CLI is killed mid-workflow, `localharness start` detects the incomplete workflow file and offers to resume or discard.
- The orchestrator never reads agent `MEMORY.md` or `history.jsonl` directly. It reads only Agent Cards (JSON) and `DelegateResult.summary` (string). This is the structural enforcement of the lean context invariant.
