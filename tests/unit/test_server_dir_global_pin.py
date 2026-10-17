"""The vLLM lifecycle family resolves GLOBAL, always (v0.13 Standing Invariant, phase 39).

``<global>/vllm/{server.pid,serve.log,venv/}`` is machine-wide: two workspaces must never each
believe they own the one GPU daemon. These sites are value-identical today (nothing feeds them a
workspace path yet), so a pure behavior test cannot discriminate a pinned site from an unpinned
one — same as ``tests/unit/test_bench_config_pinning.py``'s amendment-#3 half. Two kinds of proof:

1. BEHAVIORAL where a seam exists: ``OrchestratorREPL._server_config_dir`` is a real function of
   its input and is tested as one, including the None case; and the whole ``server.py`` path
   family is proven to derive from ``server_dir``, so pinning one input pins all four paths.
2. STRUCTURAL for the rest: the ``start_cmd`` and ``init_cmd`` call sites are asserted to name a
   pinned symbol. Mutation-check before trusting these (38-03's standing rule): revert one site,
   watch the test fail, restore. Done for every structural assertion below when written.

What this file does NOT claim: it fixes no live bug. It is the regression guard for the moment
``resolve_config_dir`` learns to discover a workspace ``.localharness/`` — if a later change makes
a lifecycle site follow that layer, it fails here rather than as two harnesses fighting over one
pidfile.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from localharness.config.paths import global_config_dir

SRC = Path(__file__).resolve().parents[2] / "src" / "localharness" / "cli"
PROVIDER = SRC.parent / "provider"


def _where(text: str, needle: str) -> str:
    """Name the methods `needle` occurs in, so a failing count says what moved, not just that it did.

    The counts below stay exact (that is the assertion); this only makes the failure readable:
    "self._config_dir: _announce_window x1, persist ... x2" instead of "16 != 15"."""
    import ast
    import collections

    tree = ast.parse(text)
    spans = sorted(
        (node.lineno, node.end_lineno or node.lineno, node.name)
        for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    seen: collections.Counter[str] = collections.Counter()
    for lineno, line in enumerate(text.splitlines(), start=1):
        for _ in range(line.count(needle)):
            inner = [n for start, end, n in spans if start <= lineno <= end]
            seen[inner[-1] if inner else "<module>"] += 1
    return f"{needle!r} occurs in: " + ", ".join(f"{n} x{c}" for n, c in sorted(seen.items()))


def _repl(config_dir):
    """The minimal OrchestratorREPL the property needs — it only reads self._config_dir."""
    from localharness.cli.repl import OrchestratorREPL

    return OrchestratorREPL(
        orchestrator=None,
        agent_loop=SimpleNamespace(),
        channel=SimpleNamespace(),
        bus=None,
        config_dir=config_dir,
        harness_config=SimpleNamespace(server=None),
    )


# --- Behavioral: the one real seam -------------------------------------------------------- #


def test_repl_server_config_dir_is_global(tmp_path):
    """The property answers with the GLOBAL layer for whatever config dir the session holds."""
    session = _repl(tmp_path)
    assert session._server_config_dir == global_config_dir(tmp_path)


def test_repl_server_config_dir_none_stays_none():
    """None must stay None. ``global_config_dir(None)`` would invent ``~/.localharness`` where
    today an unset config dir reaches the lifecycle call as None — that is a real state here (the
    /model guard checks for it), and the strategies handle it themselves."""
    assert _repl(None)._server_config_dir is None


def test_server_paths_all_derive_from_server_dir(tmp_path):
    """pid/log/venv all hang off ``server_dir``, so pinning its INPUT pins the whole family.

    This is what makes 16 call sites a single guarantee rather than 16 separate ones.
    """
    from localharness.provider import server as managed_server

    root = managed_server.server_dir(tmp_path)
    assert root == Path(tmp_path) / "vllm"
    for path in (
        managed_server.pid_path(tmp_path),
        managed_server.log_path(tmp_path),
        managed_server.venv_vllm_bin(tmp_path),
        managed_server.lock_path(tmp_path),
    ):
        assert root in path.parents, f"{path} escaped server_dir({tmp_path})"


# --- Structural: the sites name the pin (mutation-checked) --------------------------------- #


def test_start_cmd_lifecycle_sites_name_the_global_pin():
    """start's three lifecycle calls read a name derived from the RAW --config-dir."""
    text = (SRC / "start_cmd.py").read_text(encoding="utf-8")
    assert "server_cfg_path = global_config_dir(config_dir)" in text
    # The down-server branch hands the PINNED dir to the one launch-or-wait helper (launch path 1),
    # and the helper's own liveness, launch and wait calls all name the dir it was handed. Named in
    # full so reverting the call site to the raw cfg_path (or the helper's liveness alone) fails.
    # (the record the ladder chose: `_serving_spec` is `harness.server` for the primary, a peer's lifecycle else)
    assert "_serving_spec, server_cfg_path, provider.base_url, launch_key," in text
    assert "_serving_spec, cfg_path, provider.base_url" not in text
    assert "harness.server, cfg_path, provider.base_url" not in text
    assert "_managed_server_running(strategy, srv, global_dir)" in text
    assert "_managed_server_running(strategy, harness.server, cfg_path)" not in text
    # The two launch calls carry the key the launched server requires (`**launch_key`, which is {}
    # unless server.require_api_key is on) — the pin is unchanged: both still name the handed dir.
    # (`**launch_kw` is the launch line's `on_plan` for a planning strategy, {} for any other.)
    # (`peer` rides only for a peer's launch, so the primary's call is the one it always was)
    assert "activate(srv, global_dir, base_url, autosense=autosense," in text  # launch
    assert "**launch_kw, **launch_key)" in text
    assert "**launch_key)" in text
    assert "wait_ready(base_url, config_dir=strategy.recovery_config_dir(global_dir)," in text
    assert "wait_ready(provider.base_url, config_dir=cfg_path)" not in text
    # The helper states which layer it expects, so a caller cannot hand it a workspace by accident.
    assert "def _managed_server_running(strategy: Any, srv: Any, global_dir: Path)" in text


def test_repl_lifecycle_sites_name_the_global_pin():
    """Nine lifecycle calls go through the property; the non-lifecycle uses stay put.

    The counts are the assertion: moving an eighth site onto the property (or leaving a lifecycle
    call on the raw attribute) changes one of them.
    """
    text = (SRC / "repl.py").read_text(encoding="utf-8")
    # 8 = the 7 that stop, launch or restore a server + 1 from Phase 5's `_foreign_incumbent`, which
    # asks the primary's strategy liveness (read-only) whether a listener on its port is the
    # harness's own (D-33), + 1 from the accelerator-lock seed in `__init__`, which asks the same
    # liveness (D-33) instead of trusting `server:` alone, + 1 from `_pidfile_process_holds` (IN-54),
    # which reads the same pidfile's pid to see whether the process on the port is that one,
    # + 1 from Phase 7's `_launch_lock`, which reads the global dir to lock it (the launch lock
    # lives beside the pidfile; the relaunch closure gets the dir as an argument instead),
    # + 1 from Phase 7's `server_recovery`, the reconnect supplier, which reads the global dir to
    # give a launched server's pair its strategy's config dir (read once, passed down as a local).
    # + 1 from the warm restart's pre-flight (`unit_launch_refusal`), which reads the unit file and the
    # trust store to refuse a restart `serve` would exit 78 on, before anything is stopped.
    # + 1 from the managed restart's swap_refusal, which plans the target against the memory the restart will have,
    # + 1 from `_plan_model_entries`, which prices the /model list against the global dir (read-only).
    # None of them starts or stops a server.
    assert text.count("self._server_config_dir") == 15, _where(text, "self._server_config_dir")
    # assignment + the property's own two reads on one line + 6 deliberately-untouched uses,
    # + 1 from phase 41's `_audit_base_dir` property, which reads `self._config_dir` as its
    # FALLBACK when the session has no workspace layer, + 1 from Phase 2's `_announce_window`,
    # which opens the speed ledger under the session config dir, + 4 from Phase 3's /effort
    # (`persist_effort`'s overlay dir, `agents_with_effort`'s agent dir, and the rejected-key ledger
    # opened by `_announce_rejected_effort_keys` and `_rebind_effort_ledger`): none starts or stops
    # a server; + 2 from plan 13-08: the cold swap's `resolve_serving(..., config_dir=...)` (it reads
    # the machine's selections, never launches) and `_persist_landed`'s `persist_landed(...)` (an
    # overlay + ledger write; `_persist_active_endpoint` swapped its own use of the same name for
    # the same call). MEASURED at 17, not assumed.
    assert text.count("self._config_dir") == 17, _where(text, "self._config_dir")
    for raw in (
        "free_accelerator(self._active_heavy, ep.lifecycle, self._config_dir)",
        "strategy.stop(managed, self._config_dir)",
        "strategy_for(spec).activate(spec, self._config_dir, base_url)",
    ):
        assert raw not in text, f"lifecycle call still on the raw attribute: {raw}"
    # ...and the non-lifecycle writes are untouched: agent creation, endpoint + default-model
    # persistence keep handing the OVERLAY the session's config dir. Still 3 after phase 41: the
    # audit log moved to its own `audit_base_dir=` argument rather than re-pointing this one, and
    # `self._server_config_dir` is still 7 because the GPU daemon did not move either. 4 after Phase 3:
    # `/effort`'s `persist_effort` is a fourth overlay write and takes the session's config dir the
    # same way `persist_default_model` does. 6 after plan 13-08: `resolve_serving` (the cold swap's
    # trust-resolved target) and `persist_landed` (the landing's overlay + ledger writer) take it too.
    assert text.count("config_dir=self._config_dir") == 6, _where(text, "config_dir=self._config_dir")


def test_init_guided_setup_names_the_global_pin():
    """The guided vLLM install names the global layer at its install and lookup sites, and init
    neither launches a server nor waits for one (D-39): the record is written, `start` launches."""
    text = (SRC / "init_cmd.py").read_text(encoding="utf-8")
    assert "server_config_path = global_config_dir(config_path)" in text
    for pinned in (
        "managed_server.find_vllm(server_config_path)",
        "managed_server.install_vllm_venv(server_config_path",
    ):
        assert pinned in text, f"guided setup lost its pin: {pinned}"
    assert "managed_server.start_server(" not in text
    assert "wait_ready(" not in text
    assert "managed_server.find_vllm(config_path)" not in text
    offer = (SRC / "library.py").read_text(encoding="utf-8")
    assert "global_dir = global_config_dir(config_path)" in offer and "managed_server.server_dir(global_dir)" in offer


def test_the_llamacpp_activation_prepares_its_config_dir_under_the_global_dir_it_was_handed():
    """The harness-owned `XDG_CONFIG_HOME` (D-30, D-59) hangs off the `config_dir` `activate`
    receives, which is the GLOBAL layer on every path, and not off `server_dir` (`vllm/`).

    Mutation-checked: replacing `config_dir` in the call (or the helper's own `Path(os.path.abspath(config_dir))`)
    makes this fail."""
    text = (PROVIDER / "lifecycle.py").read_text(encoding="utf-8")
    assert "prepare_config_ini_home(config_dir, spec.port" in text
    assert "xdg_config_home=str(home)" in text
    helper = (PROVIDER / "server.py").read_text(encoding="utf-8")
    assert 'return Path(os.path.abspath(config_dir)) / "xdg" / str(port)' in helper
