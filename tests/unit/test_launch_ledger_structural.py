"""The launch ledger has one writer, the boot marker one remover, and sensing never writes (spec 20 §6.0,
§14, D-18, D-29, D-30; SEC-05 structural guarantees).

A runtime test can only prove the callers that exist today, so these are structural, over the source:

- every call of `record_retry`, `clear_retry` or `remove_marker` anywhere under `src/localharness/` sits
  inside `confirm_launch` in `provider/lifecycle.py` (the definitions in speed_stats.py and units.py are
  not calls);
- `confirm_launch` never senses: it calls none of the sensing functions, nor reads the ledger, nor
  reaches `subprocess` or `httpx` (after a load the free memory says nothing true about the fit);
- `sense_launch` and `plan_launch` write nothing: no ledger or marker call, no file write, no `mkdir`,
  no config.ini preparation, no server start.

Each assertion has a "bites" test: the same checker run over a small source string that plants the
forbidden call must report it, so a checker that stopped looking would fail here.
"""
from __future__ import annotations

import ast
import textwrap
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "localharness"
LIFECYCLE = "provider/lifecycle.py"

WRITERS = {"record_retry", "clear_retry", "remove_marker"}
SENSING = {
    "sense_launch", "compute_fit", "free_memory", "_read_meminfo", "weights_bytes", "read_header",
    "read_model_config", "probe_window", "last_retry",
}
SENSING_MODULES = {"subprocess", "httpx"}
NO_WRITES = WRITERS | {
    "_write", "_update_entry", "prepare_config_ini_home", "start_server", "write_text", "write_bytes",
    "unlink", "mkdir",
}
_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


def _name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _chain(func: ast.expr) -> list[str]:
    """Every name of a dotted call, so `server.subprocess.Popen` is `server`, `subprocess`, `Popen`."""
    names = []
    while isinstance(func, ast.Attribute):
        names.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        names.append(func.id)
    return names


def _root(func: ast.expr) -> str | None:
    """The leftmost name of a dotted call: `subprocess` for `subprocess.run`."""
    chain = _chain(func)
    return chain[-1] if chain else None


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}


def _enclosing(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> list[str]:
    names = []
    while node in parents:
        node = parents[node]
        if isinstance(node, _FUNCTIONS):
            names.append(node.name)
    return names


def _calls(tree: ast.AST) -> list[ast.Call]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)]


def _writes_a_file(call: ast.Call) -> bool:
    """`open(path, "w")` and its relatives: a mode string with w, a, x or + (positional or keyword)."""
    if _name(call.func) != "open":
        return False
    modes = [a.value for a in call.args[1:2] if isinstance(a, ast.Constant)]
    modes += [k.value.value for k in call.keywords if k.arg == "mode" and isinstance(k.value, ast.Constant)]
    return any(isinstance(m, str) and set(m) & set("wax+") for m in modes)


def stray_writer_calls(source: str, module: str) -> list[tuple[str, int, str]]:
    """Calls of a ledger writer / marker remover that are not inside `confirm_launch` of provider/lifecycle.py."""
    tree = ast.parse(source)
    parents = _parents(tree)
    out = []
    for call in _calls(tree):
        name = _name(call.func)
        if name in WRITERS and not (module == LIFECYCLE and "confirm_launch" in _enclosing(call, parents)):
            out.append((module, call.lineno, name))
    return out


def _function(source: str, name: str) -> ast.AST:
    found = [n for n in ast.walk(ast.parse(source)) if isinstance(n, _FUNCTIONS) and n.name == name]
    assert len(found) == 1, f"{name} must be defined exactly once, found {len(found)}"
    return found[0]


def sensing_calls(source: str, function: str = "confirm_launch") -> list[tuple[int, str]]:
    """Sensing calls inside the body of `function`: the sensing functions and anything on `subprocess` / `httpx`."""
    out = []
    for call in _calls(_function(source, function)):
        name = _name(call.func)
        if name in SENSING or SENSING_MODULES & set(_chain(call.func)):
            out.append((call.lineno, name or "?"))
    return out


def writing_calls(source: str, function: str) -> list[tuple[int, str]]:
    """Write-shaped calls inside the body of `function`: the ledger and marker writers, config.ini
    preparation, a server start, a file write, an atomic replace, an unlink, a mkdir."""
    out = []
    for call in _calls(_function(source, function)):
        name = _name(call.func)
        os_write = _root(call.func) == "os" and name in {"replace", "unlink", "rename", "remove"}
        if name in NO_WRITES or _writes_a_file(call) or os_write:
            out.append((call.lineno, name or "?"))
    return out


def _lifecycle_source() -> str:
    return (SRC / LIFECYCLE).read_text()


# ------------------------------------------------------------------- the live source


def test_confirm_launch_is_the_only_caller_of_the_ledger_writer_and_the_marker_remover():
    found: list[tuple[str, int, str]] = []
    seen: set[str] = set()
    for path in sorted(SRC.rglob("*.py")):
        module = path.relative_to(SRC).as_posix()
        source = path.read_text()
        found += stray_writer_calls(source, module)
        seen |= {_name(c.func) for c in _calls(ast.parse(source)) if _name(c.func) in WRITERS}
    assert found == []
    assert seen == WRITERS  # the scan is not vacuous: confirm_launch really does call all three


def test_confirm_launch_never_senses():
    assert sensing_calls(_lifecycle_source()) == []


def test_sense_launch_and_plan_launch_write_nothing():
    source = _lifecycle_source()
    assert writing_calls(source, "sense_launch") == []
    assert writing_calls(source, "plan_launch") == []
    assert writing_calls(source, "boot_prior_death") == []


# ------------------------------------------------------------------- the scans bite

INSIDE = '''
def confirm_launch(config_dir, key):
    speed_stats.record_retry(path, key)
    speed_stats.clear_retry(path, key)
    units.remove_marker(config_dir)
'''


def test_the_writer_scan_accepts_confirm_launch_in_lifecycle_and_nothing_else():
    assert stray_writer_calls(INSIDE, LIFECYCLE) == []
    assert [n for _, _, n in stray_writer_calls(INSIDE, "cli/start_cmd.py")] == [
        "record_retry", "clear_retry", "remove_marker"]


def test_the_writer_scan_bites_on_a_second_writer_in_lifecycle():
    other = INSIDE + '''
async def activate(spec):
    speed_stats.record_retry(path, key)
    units.remove_marker(spec)
    clear_retry(path, key)
'''
    assert [n for _, _, n in stray_writer_calls(other, LIFECYCLE)] == ["record_retry", "remove_marker", "clear_retry"]


def test_the_writer_scan_bites_on_a_call_nested_in_a_lambda_or_helper_outside_confirm():
    assert stray_writer_calls("f = lambda: speed_stats.record_retry(p, k)\n", LIFECYCLE) != []
    helper = textwrap.dedent('''
        def helper():
            def inner():
                units.remove_marker(d)
    ''')
    assert stray_writer_calls(helper, LIFECYCLE) != []


def test_the_sensing_scan_bites_on_each_forbidden_call():
    for call in ("sense_launch(spec, d, a)", "_auto.compute_fit(spec, a)", "autosense.free_memory(a)",
                 "_read_meminfo()", "weights_bytes(m)", "read_header(m, K)", "tuning.read_model_config(spec)",
                 "probe_window(u, m)", "speed_stats.last_retry(p, k)", "subprocess.run(['x'])",
                 "httpx.get(u)", "server.subprocess.Popen(c)"):
        source = f"def confirm_launch(spec):\n    {call}\n"
        assert sensing_calls(source) != [], call


def test_the_sensing_scan_does_not_flag_what_confirm_launch_legitimately_calls():
    assert sensing_calls(INSIDE + "\n") == []
    assert sensing_calls("def confirm_launch(s):\n    tuning.derive(s, {}, a)\n    min(1, 2)\n") == []


def test_the_write_scan_bites_on_each_forbidden_call():
    for call in ("speed_stats.record_retry(p, k)", "clear_retry(p, k)", "units.remove_marker(d)",
                 "_write(p, data)", "_update_entry(p, k)", "server.prepare_config_ini_home(d, 1, None)",
                 "server.start_server(d, c)", "p.write_text('x')", "p.write_bytes(b'x')",
                 "open(p, 'w')", "open(p, mode='a')", "open(p, 'r+')", "os.replace(a, b)",
                 "os.unlink(p)", "p.unlink()", "p.mkdir()", "os.rename(a, b)"):
        source = f"def sense_launch(spec):\n    {call}\n"
        assert writing_calls(source, "sense_launch") != [], call


def test_the_write_scan_does_not_flag_a_read():
    source = "def plan_launch(s):\n    open(p)\n    open(p, 'rb')\n    x = server.config_ini_home(d, 1)\n"
    assert writing_calls(source, "plan_launch") == []
