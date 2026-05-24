"""LocalHarness CLI entry point."""
import typer

app = typer.Typer(name="localharness", help="Model-agnostic hierarchical agent harness for local LLMs.")


def main() -> None:
    app()
