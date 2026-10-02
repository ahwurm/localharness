"""The memory plugin: the memory slot's occupant — facts recalled into each turn, the three memory
tools, the session row and background consolidation.

On by default. plugins/builtin.py imports this module for every `--help`, `doctor` and `plugins list`,
so it imports only the plugin API at module level; the store (aiosqlite), the router, the resonance
engine (numpy), consolidation, the tools and rich are imported inside the methods that use them.
It lives in memory/ because memory/__init__.py re-exports lazily (PEP 562): importing this module
runs the package __init__, which imports nothing. It never imports cli/start_cmd.py (PAPI-03).

doctor() is fast and offline: each memory.db opened read-only, the embedding model looked up in the
local Hugging Face cache, the sentence_transformers package found — nothing loaded or downloaded.
Behaviour change (locked "missing -> fail"): on an install without the `embeddings` extra, or
without the model in the local cache, doctor now reports a failing `memory-embedding` row."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from localharness.plugins.api import (
    Availability, Check, ContextBudget, ContextContribution, MemorySlotPlugin, PluginContext,
    PluginManifest,
)
from localharness.memory.config import MemoryConfig

if TYPE_CHECKING:
    from localharness.tools.base import ToolProtocol

log = logging.getLogger(__name__)


class MemoryPlugin(MemorySlotPlugin):
    """persistent memory: facts recalled into each turn, memory tools, background consolidation"""

    manifest = PluginManifest(name="memory", version="0.1.0", kind="memory", enabled_by_default=True)
    ConfigModel = None  # resolve() strips enabled; nothing else is harness-level
    AgentConfigModel = MemoryConfig
    wants_artifacts = False

    def __init__(self) -> None:
        self.startup_warnings: list[str] = []
        self._store: Any = None
        self._twin: Any = None
        self._router: Any = None
        self._engine: Any = None
        self._acc: Any = None
        self._sched: Any = None
        self._session_started = False
        self._browse: Any = None

    async def configure(self, ctx: PluginContext) -> Availability:
        """Opens nothing — doctor calls it outside a session, where ctx.llm is None."""
        return "ready"

    async def tools(self, ctx: PluginContext) -> list[ToolProtocol]:
        """CONSTRUCT the store, its global twin, the router and the engine and return the three memory
        tools bound to them. Opens nothing: MemoryStore.__init__ only derives paths (start() opens)."""
        s = ctx.session
        if s is None:
            raise RuntimeError("the memory plugin runs inside a session — ctx.session is None")
        cfg = ctx.agent_config
        from localharness.memory.browse import StoreBrowse
        from localharness.memory.resonance import ResonanceEngine
        from localharness.memory.sqlite import MemoryStore
        from localharness.tools.builtin.memory_tools import (
            MemoryGetTool, MemoryRememberTool, MemorySearchTool,
        )
        division = s.division_id or "default"
        # Agent state follows the work (state_dir); DIVISION.md is always the global layer's.
        self._store = MemoryStore(agent_id=s.agent_id, division_id=division, org_id="default",
                                  base_dir=str(ctx.paths.state_dir),
                                  global_base_dir=str(ctx.paths.global_config_dir), bus=ctx.bus)
        # The global twin: constructed, NOT opened (the router opens it lazily, as a non-owner, only
        # if recall_scope asks). No bus — a subscription would let auto-diary write into the global
        # store. Gated on a workspace: without one state_dir IS the global dir (same file twice).
        if ctx.paths.workspace is not None:
            self._twin = MemoryStore(agent_id=s.agent_id, division_id=division, org_id="default",
                                     base_dir=str(ctx.paths.global_config_dir),
                                     global_base_dir=str(ctx.paths.global_config_dir))
        try:
            from localharness.memory.router import RecallRouter
            self._router = RecallRouter(self._store, self._twin, scope=cfg.recall_scope)
        except Exception:
            log.warning("recall router unavailable — memory reads use the session's store", exc_info=True)
            self._router = None
        self._engine = ResonanceEngine(cfg.embedding_model)
        ws = ctx.paths.workspace
        self._browse = StoreBrowse(self._store, self._router,
                                   workspace_identity=str(ws.resolve().parent) if ws is not None else "")
        read = self._router or self._store  # the scope knob applies to on-demand recall too
        # remember() WRITES — it keeps the session's own store whatever recall_scope says.
        return [MemorySearchTool(read, engine=self._engine), MemoryGetTool(read),
                MemoryRememberTool(self._store, llm=ctx.idle_llm, engine=self._engine)]

    async def start(self, ctx: PluginContext) -> None:
        """Open in today's order: store (hard) -> session row -> accumulator -> scheduler. Each soft
        piece fails alone onto startup_warnings. Core never stops a plugin interrupted inside its own
        start(), so any BaseException here closes what was opened, then re-raises."""
        s, cfg = ctx.session, ctx.agent_config
        try:
            await self._store.open()
            try:
                await self._store.create_session(s.sitting_id, budget=s.budget, model=s.model,
                                                 context_tokens_available=s.context_tokens)
                self._session_started = True
            except Exception as exc:
                self.startup_warnings.append(f"session-start: {exc}")
            try:
                from localharness.cli.session_accumulator import SessionAccumulator
                self._acc = SessionAccumulator(ctx.bus, s.agent_id)
                await self._acc.open()
            except Exception as exc:
                self.startup_warnings.append(f"session-accumulator: {exc}")
            if cfg.consolidation.enabled:
                try:
                    from localharness.memory.consolidation import ConsolidationScheduler
                    self._sched = ConsolidationScheduler(
                        self._store, ctx.bus, s.agent_id, cfg.consolidation, engine=self._engine,
                        llm=ctx.idle_llm, archival=cfg.archival)
                    await self._sched.start()
                except Exception as exc:
                    self.startup_warnings.append(f"memory consolidation: {exc}")
                    self._sched = None
        except BaseException:
            await self._close(ctx, end=False)
            raise

    async def stop(self, ctx: PluginContext) -> None:
        """Today's shutdown tail: scheduler -> accumulator -> end_session -> router -> store."""
        await self._close(ctx, end=True)

    async def _close(self, ctx: PluginContext, *, end: bool) -> None:
        """Each step contained; each handle dropped once closed, so a second call is a no-op."""
        if self._sched is not None:
            try:
                await self._sched.stop()
            except Exception:
                log.debug("consolidation scheduler stop failed", exc_info=True)
            self._sched = None
        if self._acc is not None:
            try:
                await self._acc.close()  # stop counting before the summary reads
            except Exception:
                log.debug("session accumulator close failed", exc_info=True)
        # end_session needs the store OPEN and dreaming STOPPED — hence here, before the store closes.
        if end and self._session_started and self._store is not None:
            s, acc = ctx.session, self._acc
            try:
                from localharness.cli.session_accumulator import derive_session_summary
                await self._store.end_session(
                    s.sitting_id, exit_reason=s.exit_reason, summary=derive_session_summary(acc),
                    turn_count=acc.turn_count if acc else 0, action_count=acc.action_count if acc else 0,
                    tokens_in=acc.tokens_in if acc else 0, tokens_out=acc.tokens_out if acc else 0)
            except Exception as exc:
                # Never silent: a skipped close-out is the amnesia class.
                from rich.console import Console
                Console(stderr=True).print(f"[yellow]⚠ session close-out skipped: {exc}[/yellow]")
        self._acc, self._session_started = None, False
        # The router closes only what IT opened (the global handle); the primary is ours. Both before
        # exit: aiosqlite's worker thread is non-daemon and a leaked handle hangs shutdown (#43).
        if self._router is not None:
            try:
                await self._router.close()
            except Exception:
                log.debug("recall router close failed", exc_info=True)
            self._router = None
        if self._store is not None:
            try:
                await self._store.close()
            except Exception:
                log.debug("memory store close failed", exc_info=True)
            self._store = None

    async def context(self, ctx: PluginContext, turn: str, budget: ContextBudget) -> ContextContribution:
        """This turn's `Division Context` / `Agent Memory` sections, loaded within min(own setting,
        core's ceiling), plus the ambient-injection activation trace. A failure costs this turn its
        memory section, never the turn."""
        if self._store is None:
            return ContextContribution()
        s, cfg = ctx.session, ctx.agent_config
        try:
            self._store.set_current_session(s.sitting_id)  # default provenance for writes (WRITE-04)
            mc = await (self._router or self._store).load_context(
                index_mode=cfg.index_mode,
                max_session_history=min(cfg.max_session_history_entries, budget.max_session_history),
                max_chars=min(cfg.max_notes_chars, budget.max_chars))
            sections = tuple((h, b) for h, b in (("Division Context", mc.division_md),
                                                 ("Agent Memory", mc.agent_memory_md)) if b)
            # #96: an EMPTY shelf still records a row (gate on `is not None`, not truthiness). Own
            # try: a trace failure must not be mislabelled as an injection failure.
            ids = getattr(mc, "injected_fact_ids", None)
            if cfg.trace_ambient_injection and ids is not None:
                try:
                    await self._store.record_injection_trace(stimulus=turn, injected_ids=ids,
                                                             session_id=s.sitting_id)
                except Exception:
                    log.warning("activation-trace write failed (ambient injection)", exc_info=True)
            return ContextContribution(sections=sections)
        except Exception:
            log.warning("memory context load failed — no memory injected this turn", exc_info=True)
            return ContextContribution()

    def doctor(self, ctx: PluginContext) -> list[Check]:
        """`memory-db` per `<state_dir>/agents/*/memory.db` (read-only: schema version + quick_check;
        MemoryStore.open() would create and migrate) and one `memory-embedding` row."""
        import sqlite3

        from localharness.memory.sqlite import CURRENT_SCHEMA_VERSION
        rows: list[Check] = []
        for path in sorted((ctx.paths.state_dir / "agents").glob("*/memory.db")):
            try:
                conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
                try:
                    version = conn.execute("PRAGMA user_version").fetchone()[0]
                    check = conn.execute("PRAGMA quick_check").fetchone()[0]
                finally:
                    conn.close()
            except sqlite3.DatabaseError as exc:
                rows.append(Check(name="memory-db", status="fail",
                                  detail=f"{path} is not a readable memory database ({exc})",
                                  hint="move it aside; a fresh one is created on the next start"))
                continue
            if version > CURRENT_SCHEMA_VERSION:
                rows.append(Check(name="memory-db", status="fail",
                                  detail=f"{path}: schema {version}, this localharness reads up to "
                                         f"{CURRENT_SCHEMA_VERSION}",
                                  hint="a newer localharness wrote it — upgrade localharness"))
            elif check != "ok":
                rows.append(Check(name="memory-db", status="fail", detail=f"{path}: integrity check: {check}",
                                  hint="restore it from a backup, or move it aside to start fresh"))
            else:
                rows.append(Check(name="memory-db", status="pass", detail=f"{path} (schema {version})"))
        if not rows:
            rows.append(Check(name="memory-db", status="skip",
                              detail="no memory database yet — created on first start"))
        return rows + [_embedding_check(getattr(ctx.agent_config, "embedding_model", None)
                                        or "Qwen/Qwen3-Embedding-0.6B")]

    def browse(self) -> Any:
        return self._browse

    def bind_subagent(self, ctx: PluginContext) -> None:
        # G1: no gist persistence exists on this tree (removed in v0.15.0); the seam is wired
        # (agent/subagent.py) and awaits the owner's ruling — persist gists here, or delete the verb.
        return None

    def legacy_handles(self) -> tuple[Any, Any]:
        """Transitional (deleted when /memory becomes the plugin's): the store and router for the REPL's /memory."""
        return (self._store, self._router)


def _embedding_check(model: str) -> Check:
    """Package importable and the model's config.json + modules.json in the local HF cache. Presence
    of those files, not a verified complete weights download; never loads or downloads anything."""
    import importlib.util

    if importlib.util.find_spec("sentence_transformers") is None:
        return Check(name="memory-embedding", status="fail",
                     detail="the sentence_transformers package is not installed — memory search and "
                            "consolidation cannot embed",
                     hint="uv sync --extra embeddings")
    try:
        from huggingface_hub import try_to_load_from_cache
        present = all(isinstance(try_to_load_from_cache(model, f), str)
                      for f in ("config.json", "modules.json"))
    except ImportError:
        present = False
    if not present:
        return Check(name="memory-embedding", status="fail",
                     detail=f"embedding model {model} is not in the local Hugging Face cache",
                     hint=f"download it now: `hf download {model}` — "
                          "the first memory search would otherwise download it")
    return Check(name="memory-embedding", status="pass", detail=f"embedding model {model} is in the local cache")
