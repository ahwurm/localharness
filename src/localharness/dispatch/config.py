"""The dispatch plugin's settings — `dispatch.discord.*` (its ConfigModel). Pydantic + stdlib only.

Precedence, per field: a setting wins; an env source fills only a field still at its default. The
env fallback (`env_fallback`) is one release of grace for the pre-settings variables, and
tests/unit/test_dispatch_env_fallback_expiry.py fails at 0.17.0 until it is deleted (DISP-02).
Caveat: a field explicitly set to its default (`allow: []`, `ack: "✅"`) is indistinguishable from
unset, so the env still fills it. Claude Code's Discord token file is never read: it belongs to
another program, and a token is this machine's to give this harness, not to borrow.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, SecretStr

from localharness.plugins.api import GLOBAL_ONLY

DEFAULT_ACK = "✅"


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


def _deprecated(source: str, field: str) -> str:
    key = f"dispatch.discord.{field}"
    return (f"dispatch: {source} is deprecated and stops working in 0.17.0 — set {key} "
            f"(localharness components set {key} …)")


def _split(raw: str) -> list[str]:
    return [p.strip() for p in raw.split(",") if p.strip()]


def env_fallback(s: DiscordSettings, environ: Mapping[str, str]) -> tuple[DiscordSettings, list[str]]:
    """The effective settings and one deprecation line per env source that decided a field. Pure:
    `environ` is passed in and no file is read. Sources, in the pre-settings reader's order: token
    from LOCALHARNESS_DISCORD_TOKEN, else DISCORD_BOT_TOKEN; allow/channels from the comma lists;
    ack from LOCALHARNESS_DISCORD_ACK when SET (empty disables the ack). Each fills only a field at
    its default. Claude Code's Discord token file is never a source: it belongs to another
    program. Deleted at 0.17.0 (DISP-02)."""
    update: dict[str, Any] = {}
    lines: list[str] = []
    if not s.token.get_secret_value():
        for source in ("LOCALHARNESS_DISCORD_TOKEN", "DISCORD_BOT_TOKEN"):
            if token := environ.get(source, ""):
                update["token"] = token
                lines.append(_deprecated(source, "token"))
                break
    for field in ("allow", "channels"):
        source = f"LOCALHARNESS_DISCORD_{field.upper()}"
        if not getattr(s, field) and (ids := _split(environ.get(source, ""))):
            update[field] = ids
            lines.append(_deprecated(source, field))
    if s.ack == DEFAULT_ACK and "LOCALHARNESS_DISCORD_ACK" in environ:
        update["ack"] = environ["LOCALHARNESS_DISCORD_ACK"]
        lines.append(_deprecated("LOCALHARNESS_DISCORD_ACK", "ack"))
    if not update:
        return s, lines
    return DiscordSettings.model_validate({**s.model_dump(), **update}), lines
