"""LocalHarness CLI entry point."""
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


def main() -> None:
    """Entry point registered in pyproject.toml."""
    app()
