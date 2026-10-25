"""Who may launch a model from a channel, the launch confirm, the one ModelSwitched publish and /model as data."""
from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable
from functools import partial
from pathlib import Path

from localharness.cli import model_ops
from localharness.cli.workspace import visible_text
from localharness.config.paths import global_config_dir
from localharness.core.events import ModelSwitched
from localharness.provider.lifecycle import heavy_label

LAUNCH_SLOW = "can take several minutes"
LAUNCH_QUESTION = "Launch {target}? {stops}. A model server starts and " + LAUNCH_SLOW + "{build}; once started it cannot be cancelled."
LAUNCH_STOPS = "This stops {holder} for every client"
LAUNCH_STOPS_NONE = "Nothing the harness started is stopped"
LAUNCH_CHIP = "launching {target}"
LAUNCH_YES = "Launch"
LAUNCH_NO = "Cancel"
LAUNCH_NO_PHONE = "{target} not launched — the app was closed when asked; {current} still serves. Tap it again."
LAUNCH_LOCKED = "Launching a model server from this channel is off — set channels.remote_unattended to true in the machine config to allow it."
LAUNCH_ELSEWHERE = "{target} needs a launch this channel cannot confirm — /model {target} at the terminal launches it{phone}"
LAUNCH_ELSEWHERE_PHONE = " (or in the phone app)"
RESTART_DOWN_ON_PEER = " The managed server is down — you are still on the endpoint you were using; retry /model to relaunch."
ROW_SETUP = "set up at the terminal"
ROW_FOREIGN = "held by another server"
ROW_LOCKED = "launch from the terminal"
ROW_DIGITS = "a bare number picks by list position — type /model list"
ROW_WITHHELD = "[path withheld]"
ROW_CUT = " … — type /model list for more"
ROW_WHY_MAX = 90
MODEL_SETUP_FOOTER = "Terminal: localharness model --setup"
MODEL_FOREIGN_FOOTER = "Stop the other server at this machine; typed /model list shows the command."
MODEL_LOCKED_FOOTER = "Remote launches are off (channels.remote_unattended: false)."
MODEL_NOT_HERE_FOOTER = "Type /model list for the path."
MODEL_MALFORMED = "The bound server's reply wasn't understood — check its base_url at this machine, then reopen."
MODEL_UNAVAILABLE = "Model switching is unavailable in this session."
MODEL_LIST_REUSE_S = 15.0
PATH_TOKEN = re.compile(r"""[A-Za-z][\w+.-]*://[^\s)"',;]*"""
                        r"""|(?<![^\s("'=:,<>\[{`])(?!/proc/|/sys/)(?:\.{1,2}/|[/~]).*?(?= — |[.;,)"']+(?:\s|$)|$)""", re.S)

Stop = tuple[str, str] | None


def short_label(x: str) -> str:
    return Path(x).name if x.startswith(("/", "~", ".")) or x.endswith(".gguf") else x


def _label(x: str) -> str:
    return short_label(visible_text(x))


def launch_holder(active_heavy, harness, fallback=None) -> str | None:
    active = active_heavy or fallback
    return None if active is None else short_label(heavy_label(model_ops.incumbent_heavy(active, harness)))


def _endpoint_name(harness, base_url: str) -> str:
    url = model_ops._norm_url(base_url)
    if url == model_ops._norm_url(harness.provider.base_url):
        return "primary"
    peers = getattr(harness, "extra_endpoints", None) or []
    return next((ep.name for ep in peers if model_ops._norm_url(ep.base_url) == url), "primary")


def _why_text(reason: str) -> str:
    text = PATH_TOKEN.sub(ROW_WITHHELD, reason)
    if len(text) > ROW_WHY_MAX and " — " in text:
        text = text.split(" — ")[0] + ROW_CUT
    return visible_text(text)


def _row(entry, *, foreign: bool, refused: bool, locked: bool, harness, active_heavy, primary_url: str, current_url: str) -> dict:
    peer = entry.endpoint or (entry.record.peer if entry.record else None)
    cold = entry.kind == "cold"
    offered = cold and not foreign and not refused
    holder = None
    if offered:
        holder = launch_holder(active_heavy, harness) if peer else launch_holder(active_heavy, harness, (harness.server, primary_url))
    if entry.kind == "setup" and not refused:
        why = ROW_SETUP
    elif entry.name.isascii() and entry.name.isdigit():
        why = ROW_DIGITS
    elif cold and foreign:
        why = ROW_FOREIGN
    elif refused:
        why = _why_text(model_ops.not_offered_reason(entry.refusal or "", entry.name))
    else:
        why = ROW_LOCKED if cold and locked else None
    return {
        "name": entry.name, "label": _label(entry.name),
        "kind": model_ops.switch_kind(entry, primary_url, current_url),
        "endpoint": peer.name if peer else "primary", "active": entry.active,
        "stops": None if holder is None else visible_text(holder), "why": why,
    }


def list_payload(entries, *, llm, harness, active_heavy, locked: bool, launching: str | None) -> dict:
    primary_url, current_url = harness.provider.base_url, llm.config.base_url
    row = partial(_row, locked=locked, harness=harness, active_heavy=active_heavy, primary_url=primary_url, current_url=current_url)
    rows = [row(e, foreign=not g.numbered, refused=False) for g in entries.groups for e in g.entries]
    rows += [row(e, foreign=False, refused=True) for e in sorted(entries.not_offered, key=lambda e: e.name)]
    whys = [r["why"] or "" for r in rows]
    footer = [text for text, shown in (
        (MODEL_SETUP_FOOTER, ROW_SETUP in whys), (MODEL_FOREIGN_FOOTER, ROW_FOREIGN in whys),
        (MODEL_LOCKED_FOOTER, ROW_LOCKED in whys), (MODEL_NOT_HERE_FOOTER, any(ROW_WITHHELD in w for w in whys))) if shown]
    return {"active": _label(llm.config.model), "endpoint": _endpoint_name(harness, current_url),
            "launching": launching, "models": rows, "footer": footer}


def _phone_enrolled(config_dir) -> bool:
    return config_dir is not None and (global_config_dir(config_dir) / "mobile" / "token").is_file()


class ModelSwitch:
    def __init__(self, agent, bus, channel, harness, gate, config_dir):
        self._agent, self._bus, self._channel = agent, bus, channel
        self._harness, self._gate, self._config_dir = harness, gate, config_dir
        self.launching: str | None = None
        self._launched = False
        self._generation = 0
        self._stored = None
        self._inflight = None

    def _bound(self):
        llm = getattr(self._agent, "_llm", None)
        if llm is None or self._harness is None:
            return None
        return llm.config.model, _endpoint_name(self._harness, llm.config.base_url)

    def refusal(self, target: str) -> str | None:
        if getattr(self._channel, "local_operator", False):
            return None
        gate = self._gate()
        if gate is None or not getattr(gate, "remote_unattended", False):
            return LAUNCH_LOCKED
        if not callable(getattr(self._channel, "can_run_picker", None)):
            phone = LAUNCH_ELSEWHERE_PHONE if _phone_enrolled(self._config_dir) else ""
            return LAUNCH_ELSEWHERE.format(target=visible_text(target), phone=phone)
        return None

    async def confirm(self, target: str, holder: str | None, build: str, recheck: Callable[[], Awaitable[Stop]]) -> Stop:
        if getattr(self._channel, "local_operator", False):
            return None
        current = self._agent._llm.config.model
        can_pick = getattr(self._channel, "can_run_picker", None)
        if not callable(can_pick) or not can_pick():
            return LAUNCH_NO_PHONE.format(target=visible_text(target), current=current), "system.info"
        stops = LAUNCH_STOPS.format(holder=visible_text(holder)) if holder else LAUNCH_STOPS_NONE
        choice = await self._channel.run_picker(
            [LAUNCH_YES, LAUNCH_NO], current,
            prompt=LAUNCH_QUESTION.format(target=_label(target), build=build, stops=stops), confirm=LAUNCH_YES)
        return await recheck() if choice == LAUNCH_YES else ("", "")

    def _box_activity(self, text: str | None) -> None:
        push = getattr(self._channel, "box_activity", None)
        if push is not None:
            push(text)

    def launch(self, target: str) -> None:
        self._launched, self.launching = True, target
        self._box_activity(LAUNCH_CHIP.format(target=target))

    async def run(self, handler, arg: str) -> None:
        """A CancelledError is a shutdown: the bus may be closed and no client reads it, so it publishes nothing."""
        before, self._launched, cancelled = self._bound(), False, False
        self._generation += 1
        try:
            await handler(arg)
        except asyncio.CancelledError:
            cancelled = True
            raise
        finally:
            self._stored, self.launching = None, None
            self._generation += 1
            try:
                after, session_id = self._bound(), self._agent.current_session_id
                if before and after and after != before and not cancelled and session_id:
                    await self._bus.publish(ModelSwitched(
                        agent_id=self._agent._config.name, session_id=session_id,
                        model=_label(after[0]), endpoint=after[1],
                        kind="cold" if self._launched else "hot" if after[1] == before[1] else "cross",
                    ))
            finally:
                if self._launched and not cancelled:
                    self._box_activity(None)

    async def model_list(self, plan, active_heavy) -> dict:
        llm = getattr(self._agent, "_llm", None)
        if llm is None or self._harness is None or self._config_dir is None:
            return {"active": None, "models": [], "footer": [MODEL_UNAVAILABLE]}
        fresh = {"launching": self.launching}
        if self.launching is not None:
            if self._stored:
                return {**self._stored[1], **fresh}
            return {"active": _label(llm.config.model), "endpoint": _endpoint_name(self._harness, llm.config.base_url),
                    "models": [], "footer": [], **fresh}
        if self._stored and time.monotonic() - self._stored[0] < MODEL_LIST_REUSE_S:
            return {**self._stored[1], **fresh}
        if self._inflight is None or self._inflight[0] != self._generation:
            gen = self._generation
            self._inflight = (gen, asyncio.ensure_future(self._planned(plan, llm, active_heavy, gen)))
        task = self._inflight[1]
        return {**await asyncio.shield(task), **fresh}

    async def _planned(self, plan, llm, active_heavy, gen: int) -> dict:
        try:
            entries, _live, malformed = await plan(llm)
            payload = list_payload(entries, llm=llm, harness=self._harness, active_heavy=active_heavy,
                                   locked=self.refusal("") is not None, launching=self.launching)
            if model_ops._norm_url(llm.config.base_url) in malformed:
                payload = {**payload, "models": [r for r in payload["models"] if r["kind"] == "cross"], "footer": [MODEL_MALFORMED]}
            if gen == self._generation:
                self._stored = (time.monotonic(), payload)
            return payload
        finally:
            if self._inflight and self._inflight[1] is asyncio.current_task():
                self._inflight = None
