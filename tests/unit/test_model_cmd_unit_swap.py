"""`localharness model <name>` swaps at once when autostart runs, and otherwise only selects
(spec 20 §8.1 "`localharness model <name>` from the CLI", §12, §14; D-23; START-06).

The command decides before it asks: the harness's own unit is active, the machine has a trust record and
`LOCALHARNESS_COMMIT` is absent (`will_launch`), all judged BEFORE the trust question, which is gated on the
terminal while the action is not. A swap runs `start_serving.run_swap` under the launch lock with the sensed
unit; everything else is Phase 13's selection, with a receipt that says when it takes effect.

Nothing real runs. The world is `unit_strategy_support.unit_world` (a faked service manager, a `FakeServe`
thread for the unit's `serve`, `fake_proc` for `/proc`, the HTTP wait through `server._TRANSPORT`) with the
incumbents of `test_lifecycle_swap_unit`; `model_ops.list_live_models` is the faked sensor and
`server.start_server` fails. No service manager, unit or server is touched.
"""
from __future__ import annotations

import pytest
import yaml
from typer.testing import CliRunner

from localharness.cli import model_cmd, model_ops, start_serving
from localharness.cli.app import app
from localharness.cli.model_ops import RUNG_UNIT
from localharness.cli.workspace import MachineTrust
from localharness.config import session_presence, trust
from localharness.config.loader import ConfigLoader
from localharness.config.overlay import load_overlay
from localharness.provider import lifecycle, server, speed_stats, units
from tests.unit.procfs_support import fake_proc  # noqa: F401 — the fixture
from tests.unit.test_lifecycle_swap_unit import flash_ready, spawned_incumbent, unit_incumbent
from tests.unit.unit_strategy_support import PEER_URLS, PRIMARY, PRIMARY_URL, UNIT, unit_world

fcntl = pytest.importorskip("fcntl")
runner = CliRunner()
DOCKER_URL = "http://127.0.0.1:8090/v1"
RUNS_OUTSIDE = "(runs outside autostart — the next boot serves nothing)"


class Cli:
    """One `localharness model` invocation's surroundings on top of a `unit_world`."""

    def __init__(self, world, monkeypatch) -> None:
        self.world = world
        self.monkeypatch = monkeypatch
        self.answering: set[str] = {PRIMARY_URL}
        self.decided: list[dict] = []
        self.plans: list[tuple[object, str, str]] = []
        self.sessions: list = []
        self.operator = False
        monkeypatch.delenv("LOCALHARNESS_COMMIT", raising=False)
        monkeypatch.setattr(model_cmd, "local_operator", lambda no_input: self.operator)
        monkeypatch.setattr(model_ops, "list_live_models", self._live)
        monkeypatch.setattr(session_presence, "live_sessions", lambda *a, **k: list(self.sessions))
        real_decide = model_cmd.decide_machine_trust

        def spy_decide(global_dir, **kw):
            self.decided.append(dict(kw))
            return real_decide(global_dir, **kw)

        monkeypatch.setattr(model_cmd, "decide_machine_trust", spy_decide)
        real_clause = start_serving.plan_clause

        def spy_clause(plan, *, undo_field="server.speculative"):
            got = real_clause(plan, undo_field=undo_field)
            self.plans.append((plan, undo_field, got))
            return got

        monkeypatch.setattr(start_serving, "plan_clause", spy_clause)
        monkeypatch.setattr(lifecycle, "GPU_FREE_SETTLE_SECONDS", 0.0)
        monkeypatch.setattr(server, "start_server", lambda *a, **kw: pytest.fail("the harness started a server"))

    def _live(self, base_url, timeout=3.0, **kw):
        return (["m"], True) if base_url in self.answering else ([], False)

    def record_machine(self) -> None:
        """The machine has adopted its files (a trust record exists)."""
        trust.record_machine(self.world.g, trust.machine_snapshot(self.world.g))

    def run(self, *args: str, input: str | None = None):
        return runner.invoke(app, ["model", *args, "--config-dir", str(self.world.g)], input=input)

    @property
    def overlay(self) -> dict:
        path = self.world.g / "overrides.yaml"
        return load_overlay(path) if path.exists() else {}

    @property
    def verbs(self) -> list[tuple[str, ...]]:
        return self.world.manager.verbs

    @property
    def last_served(self) -> str | None:
        return speed_stats.last_served(speed_stats.default_speed_stats_path(self.world.g))


@pytest.fixture
def cli(tmp_path, monkeypatch, fake_proc):  # noqa: F811
    world = unit_world(tmp_path, monkeypatch, fake_proc)
    made = Cli(world, monkeypatch)
    world.manager.answer_prefix("systemctl", "--user", "stop", out="")
    world.events = []  # the incumbents of test_lifecycle_swap_unit record into these
    world.stop_servers_called = []
    made.record_machine()
    yield made
    world.stop_servers()


def lines_of(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.strip()]


def idle_unit(cli: Cli, **state) -> None:
    """The unit is `state` (default: active, enabled), serving nothing the harness launched."""
    cli.world.manager.base.update(state)


# ======================================================================================== the tracer


def test_a_script_run_swaps_the_units_server_to_flash_next_and_says_so_before_and_after(cli):
    w = cli.world
    unit_incumbent(w)
    flash = flash_ready(w)

    result = cli.run("flash-next")

    assert result.exit_code == 0, result.output
    flash.join()
    assert flash.error is None and flash.took_lock.is_set()
    assert [(c["ask"], c["launching"]) for c in cli.decided] == [(False, True)]
    assert len(cli.plans) >= 1
    plan, undo_field, clause = cli.plans[0]
    assert undo_field == "extra_endpoints.flash-next.lifecycle.speculative"
    assert clause                                                # flash-next derives a clause (MTP drafting)
    assert clause == start_serving.plan_clause(plan, undo_field=undo_field)
    launch = (f"stopping {PRIMARY}…, starting flash-next ({clause}; model load can take several minutes)...")
    receipt = "Serving flash-next — autostart restarted"
    out = lines_of(result.output)
    assert launch in out and receipt in out
    assert out.index(launch) < out.index(receipt)                # said BEFORE the wait, the receipt after
    assert [v for v in cli.verbs if v[0] in ("reset-failed", "stop", "restart")] == [
        ("reset-failed", UNIT), ("stop", UNIT), ("reset-failed", UNIT), ("restart", UNIT)]
    assert units.read_target(w.g) == "flash-next"
    assert cli.overlay["active_endpoint"]["name"] == "flash-next"
    assert "pending_preset" not in cli.overlay                   # persist_landed(swap)
    assert cli.last_served == "flash-next"


def test_a_script_run_with_an_autostart_env_the_record_does_not_cover_leaves_the_incumbent_serving(cli):
    w = cli.world
    unit_incumbent(w)
    trust.forget_entries(w.g, [trust.AUTOSTART_KEY])             # `serve` would exit 78 on the restart

    result = cli.run("flash-next")

    assert result.exit_code == 1
    refusal = lifecycle.UNIT_ENV_LINE.format(why=trust.AUTOSTART_ENV_MISSING)
    assert lifecycle.SWAP_REFUSED.format(name="flash-next", why=refusal) in " ".join(result.output.split())
    assert "localharness model --autostart on" in result.output
    assert cli.verbs == [] and w.events == []                    # nothing reset, stopped or restarted
    assert units.read_target(w.g) is None and "active_endpoint" not in cli.overlay
    assert cli.last_served is None


def test_start_and_the_cli_build_the_launch_clause_with_the_same_function():
    import inspect

    from localharness.cli import start_cmd

    assert callable(start_serving.plan_clause)
    assert not hasattr(start_cmd, "_plan_clause")
    source = inspect.getsource(start_cmd)
    assert source.count("start_serving.plan_clause(") == 2       # the launch line and the swap's
    assert inspect.signature(start_serving.plan_clause).parameters["undo_field"].default == "server.speculative"


# ================================================================ helpers for the matrix


def edit_config(cli: Cli, change) -> None:
    path = cli.world.g / "config.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def peer_of(data: dict, name: str) -> dict:
    return next(e for e in data["extra_endpoints"] if e["name"] == name)


def add_docker_peer(cli: Cli) -> None:
    """A vLLM docker peer `vllm-box` (a record the unit cannot serve), adopted by the machine record."""
    edit_config(cli, lambda data: data["extra_endpoints"].append({
        "name": "vllm-box", "base_url": DOCKER_URL, "provider_type": "vllm", "gpu": True,
        "lifecycle": {"gpu": True, "runtime": "vllm", "launch": "docker", "docker_image": "vllm/vllm-openai:x",
                      "model": "/m/ckpt", "port": 8090}}))
    cli.record_machine()


def flash_ready_choosing(w, choose):
    """`flash_ready` whose `serve` picks its record through `choose` (the real ladder), as the unit's does."""
    w.fast_polls()
    w.fast_loan()
    w.http(served="flash-next")
    w.probe()
    serving = w.serve_on_restart(w.peer("flash-next"), main_pid=False, choose=choose)
    w.manager.on_restart.append(lambda: w.manager.base.update(MainPID=serving.pid))
    return serving


def ladder_choice(w, chosen: list):
    def choose():
        got = model_ops.resolve_serving(
            ConfigLoader(config_dir=w.g).load_harness(), MachineTrust(), None, config_dir=w.g)
        chosen.append((got.record.name, got.rung))
        return got.record.spec

    return choose


def fail_flash_next(cli: Cli, secret: str = "") -> list[str]:
    """The unit cannot bring flash-next up: its `activate` raises (carrying `secret` in the text, as a log tail
    would). Any other record activates as before. Returns the specs it was asked for."""
    asked: list[str] = []
    cli.world.fast_polls()
    cli.world.fast_loan()
    real = lifecycle.UnitStrategy.activate
    flash = cli.world.peer("flash-next")

    async def activate(self, spec, config_dir, base_url, **kw):
        asked.append(str(spec.model))
        if str(spec.model) == str(flash.model):   # the command loads its own copy of the config
            raise RuntimeError(f"the unit exited. Log tail: Bearer sk-planted {secret} done")
        return await real(self, spec, config_dir, base_url, **kw)

    cli.monkeypatch.setattr(lifecycle.UnitStrategy, "activate", activate)
    return asked


def no_verbs(cli: Cli) -> bool:
    return [v for v in cli.verbs if v[0] in ("reset-failed", "stop", "restart", "start")] == []


# ================================================================ who is stopped


def test_a_spawned_incumbent_is_freed_by_its_own_verified_stop_then_the_unit_restarts(cli):
    w = cli.world
    spawned_incumbent(w)
    idle_unit(cli, MainPID=0)
    flash = flash_ready(w)

    result = cli.run("flash-next")

    flash.join()
    assert result.exit_code == 0, result.output
    assert len(w.stop_servers_called) == 1 and w.events == ["stop_server"]       # server.stop_server, not the unit
    assert [v for v in cli.verbs if v[0] in ("reset-failed", "stop", "restart")] == [
        ("reset-failed", UNIT), ("restart", UNIT)]
    assert any(ln.startswith(f"stopping {PRIMARY}…, starting flash-next (") for ln in lines_of(result.output))


def test_with_nothing_running_the_line_stops_nothing_and_the_unit_is_restarted(cli):
    flash = flash_ready(cli.world)

    result = cli.run("flash-next")

    flash.join()
    assert result.exit_code == 0, result.output
    out = lines_of(result.output)
    assert any(ln.startswith("starting flash-next (") for ln in out)
    assert not any("stopping" in ln for ln in out)
    assert "Serving flash-next — autostart restarted" in out
    assert [v for v in cli.verbs if v[0] in ("stop",)] == []


def test_a_pidfile_naming_an_unrelated_process_is_no_incumbent_and_nothing_is_stopped(cli):
    from tests.unit.unit_strategy_support import install_process

    w = cli.world
    install_process(w.proc, 3000, argv=("/usr/bin/sleep", "600"), exe="/usr/bin/sleep", start_time=4000,
                    cgroup="/user.slice/app.slice/run-r1.scope", comm="sleep")
    server.pid_path(w.g).parent.mkdir(parents=True, exist_ok=True)
    server.pid_path(w.g).write_text("3000 4000 8080", encoding="utf-8")        # a claim, not a launch
    flash = flash_ready(w)

    result = cli.run("flash-next")

    flash.join()
    assert result.exit_code == 0, result.output
    assert not any("stopping" in ln for ln in lines_of(result.output))
    assert [v for v in cli.verbs if v[0] == "stop"] == []
    assert w.stop_servers_called == []


# ================================================================ when it only selects


def test_commit_marker_selects_and_never_swaps_even_with_the_unit_active_and_a_record(cli, monkeypatch):
    monkeypatch.setenv("LOCALHARNESS_COMMIT", "1")
    unit_incumbent(cli.world)

    result = cli.run("flash-next")

    assert result.exit_code == 0, result.output
    assert no_verbs(cli)
    assert cli.overlay["pending_preset"] == "flash-next"
    assert "Selected flash-next — the next start or boot serves it" in lines_of(result.stdout)
    assert [(c["ask"], c["launching"]) for c in cli.decided] == [(False, False)]
    assert units.read_target(cli.world.g) is None


def test_an_inactive_own_unit_that_is_enabled_selects_and_says_the_next_start_or_boot(cli):
    idle_unit(cli, ActiveState="inactive")

    result = cli.run("flash-next")

    assert result.exit_code == 0, result.output
    assert no_verbs(cli)
    assert cli.overlay["pending_preset"] == "flash-next"
    assert "Selected flash-next — the next start or boot serves it" in lines_of(result.stdout)
    assert [c["launching"] for c in cli.decided] == [False]


def test_an_own_unit_that_is_not_enabled_selects_with_the_plain_receipt(cli):
    idle_unit(cli, ActiveState="inactive", UnitFileState="disabled")

    result = cli.run("flash-next")

    assert result.exit_code == 0, result.output
    assert "Selected flash-next — the next start serves it" in lines_of(result.stdout)
    assert "or boot" not in result.stdout


def test_enabled_runtime_also_names_the_boot(cli):
    idle_unit(cli, ActiveState="inactive", UnitFileState="enabled-runtime")
    result = cli.run("flash-next")
    assert "Selected flash-next — the next start or boot serves it" in lines_of(result.stdout)


def test_no_unit_at_all_selects_with_the_plain_receipt(cli, monkeypatch):
    monkeypatch.setattr(units, "_run_manager", lambda argv, *, timeout, cap, **kw: None)  # the manager cannot be asked

    result = cli.run("flash-next")

    assert result.exit_code == 0, result.output
    assert "Selected flash-next — the next start serves it" in lines_of(result.stdout)
    assert cli.overlay["pending_preset"] == "flash-next"
    assert [c["launching"] for c in cli.decided] == [False]


def test_a_foreign_unit_with_a_drop_in_selects_and_never_swaps(cli):
    idle_unit(cli, DropInPaths="/etc/systemd/user/localharness-llm.service.d/override.conf")

    result = cli.run("flash-next")

    assert result.exit_code == 0, result.output
    assert no_verbs(cli)
    assert "Selected flash-next — the next start serves it" in lines_of(result.stdout)
    assert [c["launching"] for c in cli.decided] == [False]


def test_a_machine_with_no_record_selects_even_with_the_unit_active_and_the_adoption_enables_no_swap(cli):
    trust.trust_store_path().unlink()                                   # the machine has no record yet
    assert trust.recorded_machine(cli.world.g) is None
    unit_incumbent(cli.world)

    result = cli.run("flash-next")

    assert result.exit_code == 0, result.output
    assert [(c["ask"], c["launching"]) for c in cli.decided] == [(False, False)]
    assert trust.recorded_machine(cli.world.g) is not None              # first-start adoption happened ...
    assert no_verbs(cli)                                                # ... and nothing launched on its back
    assert cli.overlay["pending_preset"] == "flash-next"


def test_a_withheld_target_with_the_unit_active_is_refused_before_any_verb_and_leaves_the_selection(cli):
    model_ops.write_pending_preset(cli.world.harness, "qwen-27b", config_dir=cli.world.g)
    edit_config(cli, lambda data: peer_of(data, "flash-next")["lifecycle"].update(binary="/x/other-server"))
    held = (cli.world.g / "overrides.yaml").read_bytes()
    unit_incumbent(cli.world)

    result = cli.run("flash-next")

    assert result.exit_code == 2, result.output
    assert no_verbs(cli)
    assert (cli.world.g / "overrides.yaml").read_bytes() == held       # pending_preset untouched
    assert cli.overlay["pending_preset"] == "qwen-27b"
    assert [c["launching"] for c in cli.decided] == [True]             # the decision was made for a launch


def test_a_name_the_ladder_does_not_know_goes_through_phase_13s_resolution_unchanged(cli):
    unit_incumbent(cli.world)
    cli.answering.add(PRIMARY_URL)

    result = cli.run("unknown")

    assert result.exit_code == 2, result.output
    assert "unknown model 'unknown'" in result.output
    assert no_verbs(cli)
    assert cli.overlay == {}


# ================================================================ the question and the action


def test_on_a_terminal_the_question_is_asked_with_launching_true_while_a_script_asks_nothing(cli):
    unit_incumbent(cli.world)
    edit_config(cli, lambda data: peer_of(data, "flash-next")["lifecycle"].update(bind_all=True))
    flash = flash_ready(cli.world)
    cli.operator = True

    result = cli.run("flash-next", input="y\n")

    assert result.exit_code == 0, result.output
    flash.join()
    assert [(c["ask"], c["launching"]) for c in cli.decided] == [(True, True)]   # `ask` is the gate's value


def test_a_script_with_a_bind_all_target_is_not_asked_and_adopts_nothing_but_still_swaps(cli):
    unit_incumbent(cli.world)
    edit_config(cli, lambda data: peer_of(data, "flash-next")["lifecycle"].update(bind_all=True))
    before = trust.recorded_machine(cli.world.g)
    flash = flash_ready(cli.world)

    result = cli.run("flash-next", input="y\n")                       # a piped yes answers nothing

    assert result.exit_code == 0, result.output
    flash.join()
    assert [(c["ask"], c["launching"]) for c in cli.decided] == [(False, True)]
    assert "Serving flash-next — autostart restarted" in lines_of(result.output)   # the action is not gated
    assert trust.recorded_machine(cli.world.g) == before               # the piped yes adopted nothing


# ================================================================ lines and receipts


def test_a_live_repl_session_adds_the_clause_to_the_receipt(cli):
    unit_incumbent(cli.world)
    flash = flash_ready(cli.world)
    cli.sessions = [object()]

    result = cli.run("flash-next")

    flash.join()
    assert result.exit_code == 0, result.output
    assert ("Serving flash-next — autostart restarted; a REPL session on this machine now talks to it"
            in lines_of(result.output))


def test_no_live_session_leaves_the_receipt_without_the_clause(cli):
    unit_incumbent(cli.world)
    flash = flash_ready(cli.world)

    result = cli.run("flash-next")

    flash.join()
    assert "Serving flash-next — autostart restarted" in lines_of(result.output)
    assert "REPL session" not in result.output


def test_a_target_the_unit_cannot_serve_says_it_runs_outside_autostart_and_leaves_the_unit_target(cli):
    add_docker_peer(cli)
    units.write_target(cli.world.g, units.PRIMARY_TARGET)
    seen: list[dict] = []

    async def activate(self, spec, config_dir, base_url, **kw):
        seen.append({"target": units.read_target(config_dir)})
        on_plan = kw.get("on_plan")
        if on_plan is not None:
            on_plan(None)
        return lifecycle.LiveEndpoint(base_url=base_url, served_models=["m"], handle=5555)

    cli.monkeypatch.setattr(lifecycle.ManagedVllmStrategy, "activate", activate)

    result = cli.run("vllm-box")

    assert result.exit_code == 0, result.output
    out = lines_of(result.output)
    launch = [ln for ln in out if ln.startswith("starting vllm-box (")]
    assert len(launch) == 1 and launch[0].endswith(RUNS_OUTSIDE)
    assert "Serving vllm-box" in out
    assert "autostart restarted" not in result.output
    assert seen == [{"target": None}]                                   # unit-target removed before the launch
    assert units.read_target(cli.world.g) is None
    assert no_verbs(cli)                                                # the unit was never asked to restart


def test_a_target_already_answering_is_not_restarted(cli):
    cli.answering.add(PEER_URLS["flash-next"])
    unit_incumbent(cli.world)

    result = cli.run("flash-next")

    assert result.exit_code == 0, result.output
    assert "flash-next is already serving" in lines_of(result.output)
    assert no_verbs(cli)
    assert cli.last_served == "flash-next"                              # persist_landed(attach)
    assert "active_endpoint" not in cli.overlay                         # an attach persists no selection


def test_an_already_serving_target_clears_a_stale_pending_preset_that_names_a_third_record(cli):
    cli.answering.add(PEER_URLS["flash-next"])
    model_ops.write_pending_preset(cli.world.harness, "qwen-27b", config_dir=cli.world.g)

    result = cli.run("flash-next")

    assert result.exit_code == 0, result.output
    # the receipt says what the landing did to the older selection, in one line
    assert lines_of(result.output) == [
        "flash-next is already serving — the older selection qwen-27b was cleared"]
    assert start_serving.ALREADY_SERVING_CLEARED_LINE.format(name="flash-next", older="qwen-27b") in result.output
    assert "pending_preset" not in cli.overlay                          # an explicit choice outlives no older one
    assert no_verbs(cli)
    assert cli.last_served == "flash-next"


def test_an_already_serving_target_whose_own_selection_was_stored_says_only_that_it_is_serving(cli):
    cli.answering.add(PEER_URLS["flash-next"])
    model_ops.write_pending_preset(cli.world.harness, "flash-next", config_dir=cli.world.g)

    result = cli.run("flash-next")

    assert result.exit_code == 0, result.output
    assert lines_of(result.output) == ["flash-next is already serving"]      # not "the older selection flash-next was cleared"
    assert "cleared" not in result.output
    assert "pending_preset" not in cli.overlay                          # and the stored selection is still removed
    assert cli.last_served == "flash-next"


def test_an_already_serving_target_with_no_older_selection_says_only_that_it_is_serving(cli):
    cli.answering.add(PEER_URLS["flash-next"])

    result = cli.run("flash-next")

    assert result.exit_code == 0, result.output
    assert lines_of(result.output) == ["flash-next is already serving"]
    assert "cleared" not in result.output


def test_the_cleared_line_is_not_said_when_the_landing_could_not_be_written(cli):
    cli.answering.add(PEER_URLS["flash-next"])
    model_ops.write_pending_preset(cli.world.harness, "qwen-27b", config_dir=cli.world.g)

    async def broken(*a, **kw):
        raise OSError("disk full")

    cli.monkeypatch.setattr(model_ops, "persist_landed", broken)

    result = cli.run("flash-next")

    assert result.exit_code == 2, result.output
    assert "was cleared" not in result.output


def test_a_failed_swap_exits_1_after_a_redacted_line_and_records_nothing(cli):
    secret = "k-" + "S3CR3T" * 4
    edit_config(cli, lambda data: data["provider"].update(api_key=secret))   # the command loads its own config
    fail_flash_next(cli, secret)
    idle_unit(cli)

    result = cli.run("flash-next")

    assert result.exit_code == 1, result.output
    assert "Launch of flash-next" in result.output                      # the reporter said why ...
    assert "sk-planted" not in result.output and secret not in result.output   # ... with the secrets masked
    assert "Serving flash-next" not in result.output
    assert "active_endpoint" not in cli.overlay and "pending_preset" not in cli.overlay
    assert cli.last_served is None


# ================================================================ a stale pending_preset


def test_a_stale_pending_preset_does_not_redirect_the_swap_and_is_removed_by_the_landing(cli):
    w = cli.world
    model_ops.write_pending_preset(w.harness, "qwen-27b", config_dir=w.g)   # an agent's earlier selection
    unit_incumbent(w)
    chosen: list = []
    flash = flash_ready_choosing(w, ladder_choice(w, chosen))

    result = cli.run("flash-next")

    flash.join()
    assert result.exit_code == 0, result.output
    assert flash.error is None
    assert chosen == [("flash-next", RUNG_UNIT)]                         # served what it asked for, never qwen-27b
    assert "Serving flash-next — autostart restarted" in lines_of(result.output)
    assert "pending_preset" not in cli.overlay                           # persist_landed(swap) removed it
    assert cli.overlay["active_endpoint"]["name"] == "flash-next"


def test_a_stale_pending_preset_stays_when_the_immediate_swap_fails(cli):
    w = cli.world
    model_ops.write_pending_preset(w.harness, "qwen-27b", config_dir=w.g)
    fail_flash_next(cli)
    idle_unit(cli)

    result = cli.run("flash-next")

    assert result.exit_code == 1, result.output
    assert cli.overlay["pending_preset"] == "qwen-27b"                   # a failed swap persists nothing
