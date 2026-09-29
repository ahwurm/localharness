"""`localharness example` — imported only when the command runs. `localharness --help` shows the
command from the manifest's CliDescriptor without importing this module."""
import typer

from localharness_plugin_example.plugin import __version__

app = typer.Typer(help="The example plugin's command.")


@app.command()
def show() -> None:
    """Say what the example plugin does. The app has one command, so `localharness example` runs it."""
    typer.echo(f"example plugin {__version__}: the example_swatch tool renders a solid-color swatch "
               "PNG into the artifact directory LocalHarness assigns it.")
