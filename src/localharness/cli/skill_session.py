"""What the REPL learns about skills: the /name line, body digests, the did-not-load check."""
from __future__ import annotations

from localharness.core.events import Action, TurnCompleted, TurnFailed

LINE = "The person invoked /{name}{args}. Call skill('{name}') and follow it."
NOT_LOADED = 'the model did not load /{name}; try again, or say "use the {name} skill"'


class SkillSession:
    def __init__(self, skill_set):
        self.skill_set = skill_set
        self._stash: str | None = None
        self._armed: str | None = None
        self._called: set[str] = set()
        self._say = None

    def invoke(self, record, args):
        text = self.skill_set.read_body(record)
        if not isinstance(text, str):
            return text
        self._stash = record.name
        return LINE.format(name=record.name, args=f" with: {args}" if args else ""), ()

    def arm(self, name=None):
        self._armed = name if name is not None else self._stash
        self._stash = None
        self._called = set()

    def subscribe(self, bus, agent_id, say):
        self._say = say
        return [
            bus.subscribe(Action, self.on_action, agent_id=agent_id),
            bus.subscribe(TurnCompleted, self.on_turn_completed, agent_id=agent_id),
            bus.subscribe(TurnFailed, self.on_turn_failed, agent_id=agent_id),
        ]

    def on_action(self, event):
        params = event.tool_params
        if event.action_type == "tool_call" and event.tool_name == "skill" and isinstance(params, dict):
            if isinstance(params.get("name"), str):
                self._called.add(params["name"])

    async def on_turn_completed(self, event):
        armed, self._armed = self._armed, None
        if armed is not None and armed not in self._called:
            await self._say(NOT_LOADED.format(name=armed))

    def on_turn_failed(self, event):
        self._armed = None
