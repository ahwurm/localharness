"""Plugin CLI commands mounted LAZILY (PAPI-06). The root command group lists and finds core commands
as registered, plus each ON plugin's CliDescriptors — from the manifest, without importing the
command's module; the module is imported only when the command runs. Plugins are resolved only when
Click asks for the command LIST (`--help`) or for a name no core command has, so `localharness start`
pays nothing for this. An off plugin contributes nothing; a core command always wins a name clash.

Resolving reads the default config layers (the root has no --config-dir) and never asks about a
workspace. Like every resolve, it imports an enabled plugin you installed — its manifest is in its
code — never one that is not enabled, and never the module a command runs until it runs.

Every Click class here is read off typer's own classes, so it is the copy of click that typer runs,
whichever that is.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import typer
from rich.markup import escape
from typer.core import TyperCommand, TyperGroup

if TYPE_CHECKING:
    import click  # annotations only: never evaluated (`from __future__ import annotations`)

    from localharness.plugins.api import CliDescriptor

log = logging.getLogger(__name__)
_CACHE = "localharness.plugin_commands"  # key in the root context's meta: one resolve per invocation

# typer < 0.26 builds on the `click` package; typer >= 0.26 vendors its own copy (typer._click), and a
# class from one copy is not the other's. On a fresh PyPI install `import click` named the copy typer
# does not run, and every plugin command failed while the locked suite passed (QA-01). So the Click
# classes the mount checks and lets through are typer's own.
_COMMAND: type[Any] = TyperCommand.__bases__[0]  # click.core.Command, or typer._click.core.Command
_PASS_THROUGH: tuple[type[BaseException], ...] = (
    typer.Exit, typer.Abort,  # Click's control flow: the root handles it, as for a core command
    next(c for c in typer.BadParameter.__mro__ if c.__name__ == "ClickException"),  # usage errors
)


class LazyPluginCommand(TyperCommand):
    """One plugin command, known by its descriptor alone until it runs."""

    def __init__(self, plugin: str, desc: CliDescriptor) -> None:
        # The help line is the plugin's text as written: escaped, rich markup in it can neither
        # style nor break `localharness --help`.
        super().__init__(desc.name, help=escape(desc.help), add_help_option=False)
        self.plugin, self.target = plugin, desc.target

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        ctx.args = list(args)  # all of it — `--help` and `--` included — is the plugin command's
        return ctx.args

    def invoke(self, ctx: click.Context) -> Any:
        from localharness.plugins.discovery import import_target

        failed = "could not be imported:"
        try:
            target = import_target(self.target)
            command = typer.main.get_command(target) if isinstance(target, typer.Typer) else target
            if not isinstance(command, _COMMAND):
                raise TypeError(f"{self.target} is not a Typer app or a click command")
            failed = "raised"
            # Run as a child of the root context, as a core command runs: its usage line reads
            # `localharness <name>`, and its exit code and Click's own errors are the root's to
            # handle. (main(standalone_mode=False) would turn a plugin's typer.Exit(3) into a
            # return value, and the process would exit 0.)
            with command.make_context(ctx.info_name, list(ctx.args), parent=ctx.parent) as sub:
                return command.invoke(sub)
        except _PASS_THROUGH:
            raise
        except Exception as exc:  # noqa: BLE001 — a plugin's command is named, never a traceback
            log.debug("plugin %s: command %s %s", self.plugin, self.name, failed, exc_info=True)
            typer.echo(f"plugin {self.plugin}: command {self.name} {failed} "
                       f"{type(exc).__name__}: {exc}", err=True)
            raise typer.Exit(1) from exc


class PluginCommandGroup(TyperGroup):
    """The root group: the core commands as registered, then the ON plugins' commands."""

    def list_commands(self, ctx: click.Context) -> list[str]:
        return [*super().list_commands(ctx),
                *sorted(name for name in _plugin_commands(ctx) if name not in self.commands)]

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        command = super().get_command(ctx, cmd_name)  # a core command first: it wins, and is free
        return command if command is not None else _plugin_commands(ctx).get(cmd_name)


def _plugin_commands(ctx: click.Context) -> dict[str, click.Command]:
    """{name: LazyPluginCommand} for the ON plugins, resolved once per invocation."""
    meta = ctx.find_root().meta
    if _CACHE not in meta:
        meta[_CACHE] = _resolve_commands()
    return meta[_CACHE]


def _resolve_commands() -> dict[str, click.Command]:
    from localharness.cli.workspace import resolve_workspace_layer
    from localharness.config.loader import ConfigLoader
    from localharness.plugins.resolve import resolve

    try:
        plan = resolve(ConfigLoader(local_config_dir=resolve_workspace_layer(
            None, interactive=False))).plan
    except Exception:  # noqa: BLE001 — never breaks the CLI: no plugin commands, core as before
        log.debug("plugin commands unavailable", exc_info=True)
        return {}
    commands: dict[str, click.Command] = {}
    for name in plan.order:
        entry = plan.entry(name)
        for desc in entry.manifest.cli if entry is not None and entry.manifest is not None else ():
            commands.setdefault(desc.name, LazyPluginCommand(name, desc))  # first in start order wins
    return commands
