"""The autoresearch plugin: the experiment loop — `propose`, `experiment` and `autoresearch`
(run / review / adopt / report / sentinel / archive).

On by default, kind `dev`, no install extra. plugins/builtin.py imports this module for every
`--help`, `doctor` and `plugins list`, so it imports only the plugin API at module level; the
package __init__ re-exports lazily (PEP 562), so the archive (aiosqlite), scipy and the command
modules load only when a command runs or doctor() reads the config. It owns the pre-existing core
settings `proposer:` and `sentinel:` under their old names (`sections`, bundled only). Nothing runs
in a session: tools, start and stop are the base no-ops. `plugins enable autoresearch` asks the
proposer's address, model and API key (empty for a local server; never echoed) and checks once that
it answers (setup_action), reading the address and key from the machine-level config only, as
`propose` does; doctor never contacts it. The proposer may be the main model again at a local
address, or a cloud model with its key."""
from __future__ import annotations

from typing import Any

from localharness.plugins.api import (
    Check, CliDescriptor, Plugin, PluginContext, PluginManifest, SetupField,
)

NO_PROPOSER = ("no proposer configured — `propose` and `autoresearch run` need one (a local server "
               "or a cloud API); `experiment`, `report` and `archive` do not")

_TRANSPORT: Any = None
"""httpx transport for the step's one GET (tests swap in httpx.MockTransport); None = the network."""


class AutoresearchPlugin(Plugin):
    """experiment loop"""

    manifest = PluginManifest(
        name="autoresearch", version="0.1.0", kind="dev", enabled_by_default=True,
        sections=("proposer", "sentinel"),
        cli=(
            # report/sentinel register on this group when report_cmd is imported (cli/report_cmd.py
            # bottom), so the target is the module that finishes assembling it.
            CliDescriptor(name="autoresearch", help="Autoresearch loop tools.",
                          target="localharness.cli.report_cmd:autoresearch_app"),
            CliDescriptor(name="experiment",
                          help="Run a proposal through the promotion gate (train Welch -> holdout Bonferroni).",
                          target="localharness.cli.experiment_cmd:experiment_app"),
            CliDescriptor(name="propose",
                          help="Generate ONE typed mutation {diff, rationale} for ONE component from failed "
                               "TRAIN traces.",
                          target="localharness.cli.propose_cmd:propose_app")),
        setup=(SetupField(key="proposer.base_url", prompt="Proposer address (an OpenAI-compatible "
                                                          "base URL — a local server or a cloud API)"),
               SetupField(key="proposer.model", prompt="Proposer model id"),
               SetupField(key="proposer.api_key", prompt="Proposer API key (leave empty for a local server)",
                          secret=True)),
        next_steps="Then: `localharness propose --help` shows how to write the first proposal.",
        agent_prompt=(
            "Set up the LocalHarness autoresearch plugin on this machine. It needs a proposer behind an\n"
            "OpenAI-compatible endpoint: a second local endpoint, where the model LocalHarness already\n"
            "runs on is fine, or a cloud API with its API key. {machine} For a local proposer, serve the\n"
            "main model again (or another model this hardware can run alongside it). Then run\n"
            "`localharness plugins enable autoresearch` and give it the proposer's address and model id.\n"
            "Leave the key empty for a local server; for a cloud API, let me type the key at its hidden\n"
            "prompt myself: never print the key or paste it into this chat. You are done when\n"
            "`localharness doctor` shows the proposer row passing."))
    ConfigModel = None  # proposer:/sentinel: stay core HarnessConfig fields (sections)
    AgentConfigModel = None

    def doctor(self, ctx: PluginContext) -> list[Check]:
        """One offline row: which proposer is configured. ctx.config is None (no ConfigModel), so it
        reads the harness config itself. Never the api key; never a probe of the endpoint (it may be
        a paid API — a doctor run must not spend)."""
        from localharness.config.loader import ConfigLoader
        try:
            p = ConfigLoader(config_dir=ctx.paths.global_config_dir,
                             local_config_dir=ctx.paths.workspace).load_harness().proposer
        except Exception as exc:  # noqa: BLE001 — a broken config is core doctor's row to report
            return [Check(name="autoresearch", status="skip", detail=f"config not readable: {type(exc).__name__}")]
        if p is None:
            return [Check(name="autoresearch", status="skip", detail=NO_PROPOSER,
                          hint="set proposer.base_url and proposer.model (your main model is fine at a "
                               "local address), and proposer.api_key for a cloud API")]
        return [Check(name="autoresearch", status="pass", detail=f"proposer: {p.model} at {p.base_url}")]

    def setup_action(self, ctx: PluginContext) -> list[Check]:
        """The step's check that the proposer answers: one GET <base_url>/models (3 s), which spends
        nothing on an OpenAI-compatible server. Run only by `plugins enable` — doctor stays offline
        (it may be a paid API, and a doctor run must not spend). The key is sent, never shown.

        The proposer is read from the machine-level config only — the layers `propose` and
        `autoresearch run` read — never through the project's .localharness/, which loads without a
        prompt inside it: a cloned repo must not choose where the key is sent. A different address
        the project sets is named, and ignored."""
        from localharness.config.loader import ConfigLoader
        try:
            p = ConfigLoader(config_dir=ctx.paths.global_config_dir).load_harness().proposer
        except Exception:  # noqa: BLE001 — an unreadable config is doctor's row to report
            return []
        return ([] if p is None else [_answers(p)]) + _project_address(ctx, None if p is None else p.base_url)


def _answers(p: Any) -> Check:
    """One GET <base_url>/models, the host it contacts printed before the request leaves."""
    import httpx
    url, key = p.base_url.rstrip("/"), p.api_key.get_secret_value()
    headers = {"Authorization": f"Bearer {key}"} if key and key != "none" else {}
    try:
        target = httpx.URL(f"{url}/models")
        if target.host:  # parsed as httpx sends it: the real host, never a user:password@ part
            print(f"Contacting the proposer at {target.scheme}://{target.netloc.decode()}"
                  + (" (sending proposer.api_key)" if headers else "") + " …", flush=True)
        with httpx.Client(transport=_TRANSPORT, timeout=3.0) as client:
            resp = client.get(target, headers=headers)
    except httpx.HTTPError as exc:
        return Check(name="autoresearch-proposer", status="fail",
                     detail=f"no answer from {url}/models ({type(exc).__name__})",
                     hint="start the proposer's server, or fix proposer.base_url")
    if resp.status_code >= 400:
        return Check(name="autoresearch-proposer", status="fail",
                     detail=f"the proposer at {url} answered {resp.status_code}",
                     hint="check proposer.base_url and proposer.api_key")
    try:
        served = [m.get("id") for m in resp.json().get("data", []) if isinstance(m, dict)]
    except (ValueError, AttributeError, TypeError):  # not JSON, or JSON that is not a model list
        served = []
    if p.model in served:
        return Check(name="autoresearch-proposer", status="pass",
                     detail=f"the proposer answers at {url} and serves {p.model}")
    return Check(name="autoresearch-proposer", status="warn",
                 detail=f"the proposer answers at {url}, but its model list does not name {p.model}",
                 hint="check proposer.model against the server's model list")


def _project_address(ctx: PluginContext, machine: str | None) -> list[Check]:
    """One warn row when the project's own config sets a proposer.base_url other than the machine's.
    The value is the project's text: shown escaped and cut short, never as markup or terminal codes."""
    if ctx.paths.workspace is None:
        return []
    import reprlib

    from localharness.config.loader import ConfigLoader
    try:
        asked = ConfigLoader(config_dir=ctx.paths.global_config_dir,
                             local_config_dir=ctx.paths.workspace).workspace_value("proposer.base_url")
    except Exception:  # noqa: BLE001 — an unreadable project file is start's and doctor's to report
        return []
    if asked is None or asked == machine:
        return []
    return [Check(name="autoresearch-proposer", status="warn",
                  detail=f"a project file sets proposer.base_url to {reprlib.Repr(maxstring=120).repr(asked)}; "
                         "it is ignored here and by `propose`")]
