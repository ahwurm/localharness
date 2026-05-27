"""LocalHarness CLI entry point."""
import typer

from localharness.cli.agent_cmd import agent_app
from localharness.cli.bench_cmd import bench_app
from localharness.cli.doctor_cmd import doctor
from localharness.cli.init_cmd import init_app
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
app.add_typer(agent_app, name="agent")
app.add_typer(bench_app, name="bench")


def main() -> None:
    """Entry point registered in pyproject.toml."""
    app()
