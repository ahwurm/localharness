"""The autoresearch plugin: the experiment loop — `propose`, `experiment` and `autoresearch`
(run / review / adopt / report / sentinel / archive).

On by default, kind `dev`, no install extra. plugins/builtin.py imports this module for every
`--help`, `doctor` and `plugins list`, so it imports only the plugin API at module level; the
package __init__ re-exports lazily (PEP 562), so the archive (aiosqlite), scipy and the command
modules load only when a command runs or doctor() reads the config. It owns the pre-existing core
settings `proposer:` and `sentinel:` under their old names (`sections`, bundled only). Nothing runs
in a session: tools, start and stop are the base no-ops. `plugins enable autoresearch` asks the
proposer's address and model and checks once that it answers (setup_action); doctor never does."""
from __future__ import annotations

from typing import Any

from localharness.plugins.api import (
    Check, CliDescriptor, Plugin, PluginContext, PluginManifest, SetupField,
)

NO_PROPOSER = ("no proposer configured — `propose` and `autoresearch run` need one; "
               "`experiment`, `report` and `archive` do not")

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
        setup=(SetupField(key="proposer.base_url", prompt="Proposer address (an OpenAI-compatible base URL)"),
               SetupField(key="proposer.model", prompt="Proposer model id (not your main model)")),
        next_steps="Then: `localharness propose --help` shows how to write the first proposal.",
        agent_prompt=(
            "Set up the LocalHarness autoresearch plugin on this machine. It needs a proposer: a second\n"
            "model, different from the one LocalHarness already runs on, served behind an\n"
            "OpenAI-compatible endpoint. {machine} Pick the strongest model this hardware can serve\n"
            "alongside the main one and serve it locally. Then run\n"
            "`localharness plugins enable autoresearch` and give it that address and model id. Leave\n"
            "proposer.api_key unset for a local server. You are done when `localharness doctor` shows\n"
            "the proposer row passing."))
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
                          hint="set proposer.base_url / proposer.model (a model distinct from "
                               "provider.default_model)")]
        return [Check(name="autoresearch", status="pass", detail=f"proposer: {p.model} at {p.base_url}")]

    def setup_action(self, ctx: PluginContext) -> list[Check]:
        """The step's check that the proposer answers: one GET <base_url>/models (3 s), which spends
        nothing on an OpenAI-compatible server. Run only by `plugins enable` — doctor stays offline
        (it may be a paid API, and a doctor run must not spend). The key is sent, never shown."""
        from localharness.config.loader import ConfigLoader
        try:
            p = ConfigLoader(config_dir=ctx.paths.global_config_dir,
                             local_config_dir=ctx.paths.workspace).load_harness().proposer
        except Exception:  # noqa: BLE001 — an unreadable config is doctor's row to report
            return []
        if p is None:
            return []
        import httpx
        url, key = p.base_url.rstrip("/"), p.api_key.get_secret_value()
        headers = {"Authorization": f"Bearer {key}"} if key and key != "none" else {}
        try:
            with httpx.Client(transport=_TRANSPORT, timeout=3.0) as client:
                resp = client.get(f"{url}/models", headers=headers)
        except httpx.HTTPError as exc:
            return [Check(name="autoresearch-proposer", status="fail",
                          detail=f"no answer from {url}/models ({type(exc).__name__})",
                          hint="start the proposer's server, or fix proposer.base_url")]
        if resp.status_code >= 400:
            return [Check(name="autoresearch-proposer", status="fail",
                          detail=f"the proposer at {url} answered {resp.status_code}",
                          hint="check proposer.base_url and proposer.api_key")]
        try:
            served = [m.get("id") for m in resp.json().get("data", []) if isinstance(m, dict)]
        except (ValueError, AttributeError, TypeError):  # not JSON, or JSON that is not a model list
            served = []
        if p.model in served:
            return [Check(name="autoresearch-proposer", status="pass",
                          detail=f"the proposer answers at {url} and serves {p.model}")]
        return [Check(name="autoresearch-proposer", status="warn",
                      detail=f"the proposer answers at {url}, but its model list does not name {p.model}",
                      hint="check proposer.model against the server's model list")]
