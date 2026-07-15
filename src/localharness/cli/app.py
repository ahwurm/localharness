"""LocalHarness CLI entry point."""
import sys
from importlib.metadata import PackageNotFoundError, version as _pkg_version

import typer

from localharness.cli.agent_cmd import agent_app
from localharness.cli.autoresearch_cmd import autoresearch_app
from localharness.cli.bench_cmd import bench_app
from localharness.cli.components_cmd import components_app
from localharness.cli.config_cmd import config_app
from localharness.cli.doctor_cmd import doctor
from localharness.cli.experiment_cmd import experiment_app
from localharness.cli.init_cmd import init_app
from localharness.cli.model_cmd import model
from localharness.cli.propose_cmd import propose
# report_cmd registers `report`/`sentinel` on autoresearch_app at import time (sibling commands).
from localharness.cli import report_cmd as _report_cmd  # noqa: F401
from localharness.cli.start_cmd import start_app
from localharness.cli.validate_cmd import validate

app = typer.Typer(
    name="localharness",
    help="Model-agnostic hierarchical agent harness for local LLMs.",
    add_completion=True,
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)

app.command("init")(init_app)
app.command("start")(start_app)
app.command("doctor")(doctor)
app.command("validate")(validate)
app.command("model")(model)
app.command("propose")(propose)
app.add_typer(agent_app, name="agent")
app.add_typer(bench_app, name="bench")
app.add_typer(components_app, name="components")
app.add_typer(config_app, name="config")
app.add_typer(autoresearch_app, name="autoresearch")
app.add_typer(experiment_app, name="experiment")


def _version_callback(value: bool) -> None:
    """`localharness --version` — a user's reflexive first command. Reads the installed
    package version, falling back to 'unknown' when metadata isn't found (raw checkout)."""
    if value:
        try:
            v = _pkg_version("localharness")
        except PackageNotFoundError:
            v = "unknown"
        typer.echo(f"localharness {v}")
        raise typer.Exit()


@app.callback()
def _root(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    """Model-agnostic hierarchical agent harness for local LLMs."""


def main() -> None:
    """Entry point registered in pyproject.toml."""
    # Windows consoles commonly default to a legacy codepage even though our output (doctor's
    # checkmarks, rich's box-drawing) is UTF-8 — reconfigure so it doesn't UnicodeEncodeError.
    # Best-effort: some stream wrappers (pytest capture, certain redirects) lack reconfigure.
    try:
        for s in (sys.stdout, sys.stderr):
            if hasattr(s, "reconfigure"):
                s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    app()
