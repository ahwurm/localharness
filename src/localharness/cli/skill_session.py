"""What the REPL learns about skills: the /name line, body digests, the did-not-load check."""
from __future__ import annotations

import contextlib
import hashlib
import re
from pathlib import Path

from localharness.cli.slash_commands import SlashCommand
from localharness.core.events import Action, AffordanceCued, TurnCompleted, TurnFailed
from localharness.tools.builtin.paths import resolve_user_path

LINE = "The person invoked /{name}{args}. Call skill('{name}') and follow it."
NOT_LOADED = 'the model did not load /{name}; try again, or say "use the {name} skill"'
ADDED = "added since start: {name} ({origin}, {path})"
CHANGED = "changed since first use: {path}; using it"
CUED = "hint to the model: skill {name}{origin}"
FIRST_SENTENCE = re.compile(r".*?[.!?](?=\s+[A-Z]|$)")
ROW_WIDTH = 80


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def skill_row(record) -> SlashCommand:
    sentence = FIRST_SENTENCE.match(record.description)
    description = (sentence.group() if sentence else record.description)[:ROW_WIDTH]
    if record.origin == "project":
        description += " (project)"
    return SlashCommand(f"/{record.name}", description, "_slash_skill", takes_args=True, plugin="skill")


class SkillSession:
    def __init__(self, skill_set, affordances=None):
        self.skill_set = skill_set
        self._affordances = affordances
        self._stash: str | None = None
        self._armed: str | None = None
        self._called: set[str] = set()
        self._say = None
        self._start = frozenset(s.name for s in skill_set.skills)
        self._seen: dict[str, set[str]] = {}
        self._written: set[tuple[Path, str]] = set()

    def find(self, name):
        return self.skill_set.find(name)

    def invoke(self, record, args):
        text = self.skill_set.read_body(record)
        if not isinstance(text, str):
            return text
        digest = _digest(text)
        seen = self._seen.setdefault(record.name, set())
        path = repr(str(record.path))
        notice = None
        if not seen and record.name not in self._start:
            notice = ADDED.format(name=record.name, origin=record.origin, path=path)
        elif seen and digest not in seen:
            notice = CHANGED.format(path=path)
        here = record.path.parent.resolve()
        if any(d == digest and p.is_relative_to(here) for p, d in self._written):
            notice = None
        seen.add(digest)
        self._stash = record.name
        return LINE.format(name=record.name, args=f" with: {args}" if args else ""), (notice,) if notice else ()

    def arm(self, name=None):
        self._armed = name if name is not None else self._stash
        self._stash = None
        self._called = set()
        if self._affordances is not None:
            self._affordances.named = self._armed

    def subscribe(self, bus, agent_id, say):
        self._say = say
        return [
            bus.subscribe(Action, self.on_action, agent_id=agent_id),
            bus.subscribe(TurnCompleted, self.on_turn_completed, agent_id=agent_id),
            bus.subscribe(TurnFailed, self.on_turn_failed, agent_id=agent_id),
            bus.subscribe(AffordanceCued, self.on_affordance_cued, agent_id=agent_id),
        ]

    def on_action(self, event):
        params = event.tool_params
        if event.action_type != "tool_call" or not isinstance(params, dict):
            return
        if event.tool_name == "skill" and isinstance(params.get("name"), str):
            self._called.add(params["name"])
        elif event.tool_name == "write" and params.get("mode", "overwrite") == "overwrite":
            path, content = params.get("path"), params.get("content")
            if isinstance(path, str) and isinstance(content, str):
                with contextlib.suppress(OSError, RuntimeError, ValueError):
                    self._written.add((resolve_user_path(path), _digest(content)))

    async def on_turn_completed(self, event):
        armed, self._armed = self._armed, None
        if armed is not None and armed not in self._called:
            await self._say(NOT_LOADED.format(name=armed))

    def on_turn_failed(self, event):
        self._armed = None

    async def on_affordance_cued(self, event):
        await self._say(CUED.format(name=event.name, origin=" (project)" if event.origin == "project" else ""))
