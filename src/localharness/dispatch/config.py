"""The dispatch plugin's settings — `dispatch.discord.*` (its ConfigModel). Pydantic + stdlib only.

Precedence, per field: a setting wins; an env source fills only a field still at its default. The
env fallback (`env_fallback`) is one release of grace for the pre-settings variables, and
tests/unit/test_dispatch_env_fallback_expiry.py fails at 0.17.0 until it is deleted (DISP-02).
Caveat: a field explicitly set to its default (`allow: []`, `ack: "✅"`) is indistinguishable from
unset, so the env still fills it.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, SecretStr

from localharness.plugins.api import GLOBAL_ONLY

DEFAULT_ACK = "✅"
ENV_FILE_RELATIVE = ".claude/channels/discord/.env"
"""The OpenClaw bot-token file, joined onto the home dir passed to env_fallback (deleted at 0.17.0)."""


def _ids(value: Any) -> list[str]:
    """Ids as digit strings: "1, 2,,3" (a `--set` / `components set` raw string), a YAML int
    snowflake, or a list of either. A non-digit id is refused."""
    if value is None:
        return []
    if isinstance(value, bool):
        raise ValueError(f"{value!r} is not a Discord id")
    if isinstance(value, int):
        value = [value]
    elif isinstance(value, str):
        value = value.split(",")
    elif not isinstance(value, (list, tuple)):
        raise ValueError(f"{value!r} is not a list of Discord ids")
    out = [str(v).strip() for v in value if str(v).strip()]
    bad = [v for v in out if not v.isdigit()]
    if bad:
        raise ValueError(f"not a Discord id (digits only): {', '.join(map(repr, bad))}")
    return out


IdList = Annotated[list[str], BeforeValidator(_ids)]


class DiscordSettings(BaseModel):
    """The Discord adapter. token/allow/channels are global-only: a project cannot change who may
    drive the agent or which bot it is; ack (the received-message emoji, "" = none) may be layered."""
    model_config = ConfigDict(extra="forbid")

    token: SecretStr = Field(SecretStr(""), json_schema_extra=GLOBAL_ONLY,
                             description="Bot token. Never printed.")
    allow: IdList = Field(default_factory=list, json_schema_extra=GLOBAL_ONLY,
                          description="User ids allowed to drive the agent (required to start).")
    channels: IdList = Field(default_factory=list, json_schema_extra=GLOBAL_ONLY,
                             description="Channel ids to listen in; empty = any channel the bot sees.")
    ack: str = Field(DEFAULT_ACK, description='Reaction on a received message; "" disables it.')


class DispatchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    discord: DiscordSettings = Field(default_factory=DiscordSettings)
