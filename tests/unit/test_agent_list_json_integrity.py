"""F1 — `agent list --json` must not corrupt its own payload (v0.13 dogfood, release-gating).

`agent_list` was the ONE machine-output emitter in the whole `cli/` package that went through
`console.print(json.dumps(...))` instead of `typer.echo(...)`; the other eight sites across five
files already did it right. Rich is a RENDERER, so handing it JSON does two things to the data:

* it hard-wraps at the terminal width, injecting a newline into the middle of the payload — the
  post-42 dogfood measured a break at col 79 at width 80 and at col 201 at width 200, i.e. at
  EVERY width, because a roster is eventually longer than any terminal;
* it parses `[...]` in the DATA as markup and silently DROPS what it recognises, so a description
  reading `Reviews [bold]pull requests[/bold]` arrives as `Reviews pull requests`.

The second one is the nastier bug: the output still parses as JSON, so a script sees a clean
success and a quietly different string.

**Why the pre-existing `test_agent_list_json_output` never caught this, which is the whole lesson.**
It already called `json.loads(result.output)` and it PASSED against the broken emitter — not because
the emitter was safe, but because its fixture role is `"JSON role"`: short enough never to reach the
wrap width, and free of brackets for Rich to eat. A test's fixture is what it actually grades. That
test is not wrong, just weak, and it is left exactly as it was.

So this file's fixture IS the test: a description over 80 characters that carries markup-shaped
tokens, driven at two widths. Measured against the pre-fix emitter, the two widths fail for the two
DIFFERENT reasons above — the wrap at 80, the eaten markup at 200 — which is why one width would not
have been enough.

Assertion order is deliberate (41-06's lesson): the description equality comes before the
newline-count check, because a payload can be perfectly single-line and still have had its brackets
eaten. The weakest assertion goes last.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from localharness.cli.agent_cmd import agent_app

runner = CliRunner()

# Two markup-shaped tokens, on purpose, and they are not equivalent. Rich's markup tag must start
# with a lowercase letter, `#`, `/` or `@`, so `[bold]`/`[/bold]` are EATEN while `[P1]` survives —
# a fixture carrying only `[P1]` would prove nothing. 137 characters, comfortably past width 80.
_LONG_BRACKETED = (
    "Reviews [bold]pull requests[/bold] for the payments service and files "
    "[P1] regressions against the on-call rota, then summarises them nightly."
)


def _write_agent_yaml(agents_dir: Path, name: str) -> None:
    """Written by hand rather than through `agent create`, which has no --description flag.

    `ConfigLoader.discover_agents` yaml-loads each file into a RAW dict with no validation, so any
    extra key flows straight through into the --json payload — which is exactly the surface a
    scripting user relies on and exactly what Rich was mangling.
    """
    agents_dir.mkdir(parents=True, exist_ok=True)
    (agents_dir / f"{name}.yaml").write_text(
        yaml.dump(
            {"name": name, "role": "reviewer", "description": _LONG_BRACKETED},
            default_flow_style=False,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("width", [80, 200], ids=["width80", "width200"])
def test_agent_list_json_survives_a_long_bracketed_description(tmp_path, monkeypatch, width):
    """One agent, one long bracket-bearing description, in and back out unchanged."""
    _write_agent_yaml(tmp_path / "agents", "reviewer")
    monkeypatch.setenv("COLUMNS", str(width))

    result = runner.invoke(agent_app, ["list", "--json", "--config-dir", str(tmp_path)])

    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data[0]["description"] == _LONG_BRACKETED, "the payload's text did not survive the emitter"
    assert result.stdout.strip().count("\n") == 0, "a newline was injected inside the JSON payload"
