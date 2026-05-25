"""OrchestratorREPL — interactive prompt_toolkit loop for LocalHarness."""
from __future__ import annotations

from typing import Any


class OrchestratorREPL:
    """Interactive REPL for orchestrator conversation.

    Reads user input via TerminalChannel, dispatches to AgentLoop, streams output.
    Exits on 'exit', 'quit', or EOFError (Ctrl-D).
    """

    def __init__(self, agent_loop: Any, channel: Any, bus: Any) -> None:
        self._agent = agent_loop
        self._channel = channel
        self._bus = bus

    async def run(self) -> None:
        """Main REPL loop. Reads user input, dispatches to agent, streams output."""
        await self._channel.start()
        try:
            await self._channel.send_message(
                "LocalHarness ready. Type a task for the agent, or 'exit' to quit.",
                metadata={"style": "system.info"},
            )
            while True:
                try:
                    user_input = await self._channel.read_input("you> ")
                except EOFError:
                    break
                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit"):
                    break
                summary = await self._agent.run_turn(
                    task=user_input,
                    on_token=None,
                )
                await self._channel.send_message(
                    summary,
                    agent_id=self._agent._config.name,
                )
        finally:
            await self._channel.stop()
