"""Who may launch a model from a channel, the launch confirm, the one ModelSwitched publish and /model as data."""
from __future__ import annotations

import asyncio
from pathlib import Path

from localharness.cli import model_ops
from localharness.cli.workspace import visible_text
from localharness.config.paths import global_config_dir
from localharness.core.events import ModelSwitched

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


def short_label(x: str) -> str:
    return Path(x).name if x.startswith(("/", "~", ".")) or x.endswith(".gguf") else x


def _label(x: str) -> str:
    return short_label(visible_text(x))


def _endpoint_name(harness, base_url: str) -> str:
    url = model_ops._norm_url(base_url)
    if url == model_ops._norm_url(harness.provider.base_url):
        return "primary"
    peers = getattr(harness, "extra_endpoints", None) or []
    return next((ep.name for ep in peers if model_ops._norm_url(ep.base_url) == url), "primary")


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
        return None if llm is None else (llm.config.model, _endpoint_name(self._harness, llm.config.base_url))

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
                if after != before and not cancelled and session_id:
                    await self._bus.publish(ModelSwitched(
                        agent_id=self._agent._config.name, session_id=session_id,
                        model=_label(after[0]), endpoint=after[1],
                        kind="cold" if self._launched else "hot" if after[1] == before[1] else "cross",
                    ))
            finally:
                if self._launched and not cancelled:
                    self._box_activity(None)
