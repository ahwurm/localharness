"""`localharness model --script [NAME]` and `--script --all` (spec 20 §10): the launch `start` runs, as sh.

The world is `test_serve_cmd.serve_world` (a recorded machine, a Swift llama.cpp primary, the box faked at the
autosense seams). Nothing is started: the command senses, plans and prints, and writes nothing.
"""
from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from localharness.cli import model_cmd, workspace
from localharness.cli.app import app
from localharness.config import trust
from localharness.config.loader import ConfigLoader
from localharness.provider import lifecycle, server, speed_stats
from tests.unit.launch_support import record_opens
from tests.unit.test_serve_cmd import attach_peer, docker_peer, llama_peer, ollama_peer, serve_world

WITHHELD = "change(s)"
GIB = 2**30


def _run(world, *args: str):
    return CliRunner().invoke(app, ["model", *args, "--config-dir", str(world.g)])


def _exec_tokens(script: str) -> list[str]:
    body = [ln for ln in script.splitlines() if not ln.startswith(("#", "export "))]
    return shlex.split("\n".join(body).replace("\\\n", " "))[1:]


def _files(root: Path) -> dict[str, bytes]:
    return {str(p): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def _plan_of(world, **sensing):
    harness = ConfigLoader(config_dir=world.g, local_config_dir=None).load_harness()
    spec = harness.server
    sensed = lifecycle.sense_launch(spec, world.g, harness.autosense, **sensing)
    decision = server.launch_decision(spec, False)
    return lifecycle.plan_launch(spec, sensed, decision, autosense=harness.autosense, api_key_held=False)


@pytest.fixture
def world(tmp_path, monkeypatch):
    return serve_world(tmp_path, monkeypatch)


def test_script_prints_the_swift_launch_with_the_fitted_window_left_to_fit(world):
    result = _run(world, "--script", "swift")

    assert result.exit_code == 0, result.stderr
    lines = result.stdout.splitlines()
    plan = _plan_of(world)
    assert lines[0].startswith("# swift — the launch `localharness start` / `/model` runs (")
    assert "derived: kv q8_0 (autosense.kv_cache_type) · draft-mtp (nextn_predict_layers=1)" in lines[0]
    assert f"-c left to llama-server --fit (the harness fitted {plan.full_fit} today)" in lines[1]
    assert "LLAMA_ARG_*/LLAMA_API_KEY/MTMD_BACKEND_DEVICE dropped" in lines[1] and "XDG_CONFIG_HOME →" in lines[1]
    assert lines[2] == f"exec {world.binary} \\"
    argv = list(plan.argv)
    at = argv.index("-c")
    assert argv[at + 1] == str(plan.full_fit)
    assert _exec_tokens(result.stdout) == argv[:at] + argv[at + 2:]


def test_script_decides_nothing_and_writes_nothing(world, monkeypatch):
    def raises(*a, **kw):
        raise AssertionError("a script render must not call this")

    monkeypatch.setattr(server, "prepare_config_ini_home", raises)
    monkeypatch.setattr(speed_stats, "record_retry", raises)
    monkeypatch.setattr(workspace, "decide_machine_trust", raises)
    monkeypatch.setattr(model_cmd, "decide_machine_trust", raises)
    before = _files(world.g)

    result = _run(world, "--script", "swift")

    assert result.exit_code == 0, result.stderr
    assert _files(world.g) == before
    assert not (world.g / "xdg").exists()


def test_script_with_no_name_renders_what_the_ladder_resolves(world):
    result = _run(world, "--script")

    assert result.exit_code == 0, result.stderr
    assert result.stdout.splitlines()[0].startswith("# swift — the launch")


def test_a_record_changed_since_it_was_confirmed_is_refused_with_the_withheld_line(world):
    world.edit_config(lambda d: d["server"].update(binary="/x/other-server"))

    result = _run(world, "--script", "swift")

    assert result.exit_code == 2
    assert result.stdout == "" and WITHHELD in result.stderr


def test_a_machine_with_no_record_withholds_every_launch_entry(world):
    trust.trust_store_path().unlink()
    assert trust.recorded_machine(world.g) is None

    result = _run(world, "--script", "swift")

    assert result.exit_code == 2 and result.stdout == "" and "server.binary (config.yaml)" in result.stderr
    assert trust.recorded_machine(world.g) is None


def test_all_emits_one_function_per_preset_primary_first_and_sh_accepts_it(world, tmp_path):
    world.edit_config(lambda d: d["extra_endpoints"].append(llama_peer("flash-next", 8082)))
    world.rerecord()

    result = _run(world, "--script", "--all")

    assert result.exit_code == 0, result.stderr
    functions = [ln for ln in result.stdout.splitlines() if ln.endswith("() {")]
    assert functions == ["serve_swift() {", "serve_flash_next() {"]
    assert result.stdout.count("    exec ") == 2
    path = tmp_path / "all.sh"
    path.write_text(result.stdout + "\n")
    assert subprocess.run(["sh", "-n", str(path)], capture_output=True).returncode == 0


def test_all_is_in_the_same_order_every_run(world):
    world.edit_config(lambda d: d["extra_endpoints"].extend([llama_peer("zz", 8082), llama_peer("aa", 8083)]))
    world.rerecord()

    first = _run(world, "--script", "--all").stdout

    assert [ln for ln in first.splitlines() if ln.endswith("() {")] == [
        "serve_swift() {", "serve_zz() {", "serve_aa() {"]
    assert _run(world, "--script", "--all").stdout == first


def test_two_names_that_make_one_function_are_numbered(world):
    world.edit_config(lambda d: d["extra_endpoints"].extend([llama_peer("a-b", 8082), llama_peer("a_b", 8083)]))
    world.rerecord()

    result = _run(world, "--script", "--all")

    assert [ln for ln in result.stdout.splitlines() if ln.endswith("() {")] == [
        "serve_swift() {", "serve_a_b() {", "serve_a_b_2() {"]


def test_all_says_why_a_record_is_not_rendered(world):
    world.edit_config(lambda d: d["extra_endpoints"].extend(
        [attach_peer("attached"), ollama_peer("oll"), llama_peer("moved", 8085)]))
    world.rerecord()
    world.edit_config(lambda d: d["extra_endpoints"][-1]["lifecycle"].update(model="/m/other.gguf"))

    result = _run(world, "--script", "--all")

    assert result.exit_code == 0, result.stderr
    out = result.stdout.splitlines()
    assert "# attached: not rendered — it launches nothing" in out
    assert "# oll: ollama — not rendered" in out
    assert ("# moved: not rendered — its record has changes you have not confirmed "
            "(run localharness start on a terminal)") in out
    assert [ln for ln in out if ln.endswith("() {")] == ["serve_swift() {"]


def _refusing_port(monkeypatch, port: int) -> None:
    real = lifecycle.plan_launch

    def plan(spec, *a, **kw):
        if spec.port == port:
            raise server.LaunchRefused("The model does not fit.")
        return real(spec, *a, **kw)

    monkeypatch.setattr(lifecycle, "plan_launch", plan)


def test_all_reports_the_refusing_record_and_renders_the_rest(world, monkeypatch):
    world.edit_config(lambda d: d["extra_endpoints"].append(llama_peer("flash-next", 8082)))
    world.rerecord()
    _refusing_port(monkeypatch, 8082)

    result = _run(world, "--script", "--all")

    assert result.exit_code == 0, result.stderr
    out = result.stdout.splitlines()
    assert "# flash-next: The model does not fit." in out
    assert [ln for ln in out if ln.endswith("() {")] == ["serve_swift() {"]


def test_a_named_record_that_refuses_is_its_sentence_and_exit_two(world, monkeypatch):
    _refusing_port(monkeypatch, 8080)

    result = _run(world, "--script", "swift")

    assert result.exit_code == 2 and result.stdout == ""
    assert "The model does not fit." in result.stderr


def test_a_record_window_below_the_cached_floor_is_the_floor_sentence_and_exit_two(world):
    world.edit_config(lambda d: d["server"].update(context=4096))
    world.rerecord()
    lifecycle.record_floor(world.g, 20_000, {})

    result = _run(world, "--script", "swift")

    assert result.exit_code == 2 and result.stdout == ""
    assert "is set to a 4,096-token window (context:) and the harness needs at least 20,000" in result.stderr


def test_all_on_attach_only_records_prints_only_comments_and_exits_zero(world):
    world.edit_config(lambda d: (d.pop("server"), d["extra_endpoints"].append(attach_peer("attached"))))
    world.rerecord()

    result = _run(world, "--script", "--all")

    assert result.exit_code == 0, result.stderr
    assert set(ln[:1] for ln in result.stdout.splitlines() if ln) == {"#"}
    assert "# attached: not rendered — it launches nothing" in result.stdout


def test_a_docker_peer_renders_as_a_function_with_its_run_line(world):
    world.edit_config(lambda d: d["extra_endpoints"].append(docker_peer("dock", 8082)))
    world.rerecord()

    result = _run(world, "--script", "--all")

    assert "serve_dock() {" in result.stdout and "    exec docker \\" in result.stdout


def test_a_vllm_primary_renders_serve_with_the_vllm_key_variable_named(tmp_path, monkeypatch):
    world = serve_world(tmp_path, monkeypatch, runtime="vllm", provider_key="sk-secret")

    result = _run(world, "--script")

    assert result.exit_code == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[1] == "# environment: VLLM_API_KEY dropped"
    assert "# VLLM_API_KEY=<provider.api_key>" in lines
    at = lines.index(f"exec {world.binary} \\")
    assert lines[at + 1].startswith("  serve ")
    assert "sk-secret" not in result.stdout


def test_an_inline_key_and_an_extra_args_key_never_reach_the_page(tmp_path, monkeypatch):
    world = serve_world(
        tmp_path, monkeypatch, provider_key="sk-secret",
        server_extra={"extra_args": ["--api-key", "sk-x"]})

    result = _run(world, "--script", "swift")

    assert result.exit_code == 0, result.stderr
    assert "# LLAMA_API_KEY=<provider.api_key>" in result.stdout.splitlines()
    assert "--api-key '<key>'" in result.stdout
    assert "sk-secret" not in result.stdout and "sk-x" not in result.stdout


def test_a_peer_key_is_named_by_its_own_placeholder(tmp_path, monkeypatch):
    peer = llama_peer("flash", 8082)
    peer["api_key"] = "sk-peer"
    peer["lifecycle"]["require_api_key"] = True
    world = serve_world(tmp_path, monkeypatch, peers=[peer])

    result = _run(world, "--script", "flash")

    assert result.exit_code == 0, result.stderr
    assert "# LLAMA_API_KEY=<extra_endpoints.flash.api_key>" in result.stdout.splitlines()
    assert "sk-peer" not in result.stdout


def test_a_key_file_is_printed_as_its_path_and_never_opened(tmp_path, monkeypatch):
    key = tmp_path / "the.key"
    key.write_text("sk-file\n")
    key.chmod(0o600)
    world = serve_world(
        tmp_path, monkeypatch, provider_key="sk-secret",
        server_extra={"api_key_file": str(key)})
    world.rerecord()

    with record_opens() as opened:
        result = _run(world, "--script", "swift")

    assert result.exit_code == 0, result.stderr
    assert f"--api-key-file {key}" in result.stdout
    assert str(key) not in opened and "sk-file" not in result.stdout


def test_no_launchable_record_is_said_in_one_line_and_exits_two(world):
    world.edit_config(lambda d: (d.pop("server"), d["extra_endpoints"].append(attach_peer("attached"))))
    world.rerecord()

    result = _run(world, "--script")

    assert result.exit_code == 2
    assert result.stderr.strip() == model_cmd.SCRIPT_NOTHING_LINE and result.stdout == ""


def test_a_name_that_launches_nothing_is_refused_with_its_kind(world):
    world.edit_config(lambda d: d["extra_endpoints"].append(ollama_peer("oll")))
    world.rerecord()

    result = _run(world, "--script", "oll")

    assert result.exit_code == 2
    assert result.stderr.strip() == model_cmd.SCRIPT_KIND_LINE.format(name="oll", kind="Ollama")


def test_an_attach_only_record_the_ladder_resolves_is_refused_while_another_record_launches(world):
    world.edit_config(lambda d: (d.pop("server"), d["extra_endpoints"].append(llama_peer("flash", 8082))))
    world.rerecord()

    result = _run(world, "--script")

    assert result.exit_code == 2 and "is attach-only" in result.stderr


def test_an_unknown_name_is_the_ladders_refusal(world):
    result = _run(world, "--script", "nope")

    assert result.exit_code == 2 and "unknown preset 'nope'" in result.stderr


@pytest.mark.parametrize("args", [
    ["--script", "--download", "x/y"],
    ["--script", "--autostart", "status"],
    ["--script", "--all", "swift"],
    ["--all"],
    ["--all", "swift"],
])
def test_script_takes_a_name_or_all_and_nothing_else(world, args):
    result = _run(world, *args)

    assert result.exit_code == 2
    assert result.stderr.strip() == model_cmd.SCRIPT_ALONE_LINE and result.stdout == ""


def test_a_config_that_will_not_load_is_one_line_and_exit_two(world):
    (world.g / "config.yaml").write_text(yaml.safe_dump({"version": "1", "server": {"runtime": "nope"}}))

    result = _run(world, "--script")

    assert result.exit_code == 2 and "failed to load config" in result.stderr


def test_a_verified_launch_of_the_harnesss_own_lends_the_script_the_memory_it_holds(world, monkeypatch):
    from localharness.provider import autosense, procfs

    probes: list = []
    monkeypatch.setattr(autosense, "free_memory",
                        lambda ratios: probes.append(ratios) or autosense.FreeMemory(5 * GIB, 48 * GIB, "nvidia-smi"))
    monkeypatch.setattr(autosense, "held_by", lambda pid: 30 * GIB if pid == 4321 else None)
    monkeypatch.setattr(procfs, "read_launch", lambda cfg, **kw: None)
    world.edit_config(lambda d: d["extra_endpoints"].append(llama_peer("flash-next", 8082)))
    world.rerecord()
    refused = _run(world, "--script", "--all")
    assert refused.exit_code == 0 and "# swift: swift cannot run here" in refused.stdout
    assert "serve_swift() {" not in refused.stdout

    monkeypatch.setattr(procfs, "read_launch",
                        lambda cfg, *, expect=None, **kw: procfs.Launch(4321, None, 8080, "spawned", expect is not None))
    probes.clear()
    result = _run(world, "--script", "--all")

    assert result.exit_code == 0, result.stderr
    assert "cannot run here" not in result.stdout and len(probes) == 1
    assert len([ln for ln in result.stdout.splitlines() if ln.endswith("() {")]) == 2
    fitted = _plan_of(world, memory=autosense.FreeMemory(35 * GIB, 48 * GIB, "nvidia-smi")).full_fit
    assert f"-c left to llama-server --fit (the harness fitted {fitted} today)" in result.stdout
