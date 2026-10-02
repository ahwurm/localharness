"""The autoresearch plugin: the experiment loop — `propose`, `experiment` and `autoresearch`
(run / review / adopt / report / sentinel / archive).

On by default, kind `dev`, no install extra. plugins/builtin.py imports this module for every
`--help`, `doctor` and `plugins list`, so it imports only the plugin API at module level; the
package __init__ re-exports lazily (PEP 562), so the archive (aiosqlite), scipy and the command
modules load only when a command runs or doctor() reads the config. It owns the pre-existing core
settings `proposer:` and `sentinel:` under their old names (`sections`, bundled only). Nothing runs
in a session: tools, start and stop are the base no-ops."""
from __future__ import annotations

from localharness.plugins.api import Check, CliDescriptor, Plugin, PluginContext, PluginManifest

NO_PROPOSER = ("no proposer configured — `propose` and `autoresearch run` need one; "
               "`experiment`, `report` and `archive` do not")


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
                          target="localharness.cli.propose_cmd:propose_app")))
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
