"""`install_route` / `installed_extras` / `install_pass`: one spec, six installer rows, nothing spawned."""
from __future__ import annotations

import ast
import asyncio
import importlib.metadata
import importlib.util
import io
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import click
import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from rich.console import Console

from localharness.cli import plugins_cmd, update_cmd
from localharness.cli.plugins_cmd import PassProgress, PluginSession, extra_for, install_pass, install_route_for
from localharness.cli.update_cmd import Manual, install_route
from localharness.plugins import resolve as resolve_mod
from localharness.plugins.api import PluginManifest
from localharness.plugins.plan import PlanEntry, installed_extras

EXTRAS = ["browser", "dev", "Exact_Tokenizer", "mobile"]
SPEC = "localharness[browser,dev,mobile]==1.2.3"
PIP_ARGV = [sys.executable, "-I", "-m", "pip", "install", SPEC]
UV_PREREQ = "Install uv first (https://docs.astral.sh/uv/)."


class _Metadata:
    def __init__(self, extras: list[str]) -> None:
        self._extras = extras

    def get_all(self, key: str):
        return list(self._extras) if key == "Provides-Extra" else None


@pytest.fixture(autouse=True)
def nothing_spawns(monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError(f"spawned {a}")
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)


@pytest.fixture
def dist(monkeypatch):
    """localharness 1.2.3 declaring EXTRAS with browser+dev installed, as a pip venv on linux."""
    def distribution(name: str):
        if name == "localharness":
            return SimpleNamespace(metadata=_Metadata(EXTRAS), version="1.2.3")
        raise importlib.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(importlib.metadata, "distribution", distribution)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: distribution(name).version)
    monkeypatch.setattr(update_cmd, "installed_extras", lambda: frozenset({"dev", "browser"}))
    monkeypatch.setattr(update_cmd, "_is_source_install", lambda: False)
    monkeypatch.setattr(update_cmd, "_installer_kind", lambda: ("pip", None))
    monkeypatch.setattr(update_cmd, "_is_windows", lambda: False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())


def _checkout(monkeypatch, tmp_path) -> Path:
    root = tmp_path / "my checkout"
    pkg = root / "src" / "localharness"
    pkg.mkdir(parents=True)
    (root / "pyproject.toml").write_text("")
    monkeypatch.setattr(update_cmd.localharness, "__file__", str(pkg / "__init__.py"))
    monkeypatch.setattr(update_cmd, "_is_source_install", lambda: True)
    return root.resolve()


SYNC = "uv sync --extra browser --extra dev --extra mobile"


# --------------------------------------------------------------------------- the spec

def test_spec_parses_and_names_every_installed_extra_plus_the_new_one(dist):
    route = install_route("mobile")
    req = Requirement(route.spec)
    assert route.spec == SPEC
    assert (req.name, req.extras, str(req.specifier)) == ("localharness", {"browser", "dev", "mobile"}, "==1.2.3")


def test_spec_carries_only_the_one_extra_when_none_is_installed(dist, monkeypatch):
    monkeypatch.setattr(update_cmd, "installed_extras", lambda: frozenset())
    assert install_route("mobile").spec == "localharness[mobile]==1.2.3"


def test_spec_and_command_do_not_depend_on_install_order(dist, monkeypatch):
    routes = []
    for order in (["dev", "browser"], ["browser", "dev"]):
        monkeypatch.setattr(update_cmd, "installed_extras", lambda order=order: frozenset(order))
        routes.append(install_route("mobile"))
    assert routes[0] == routes[1]
    assert routes[0].argv[-1] == SPEC


@pytest.mark.parametrize("extra, spec", [
    ("BROWSER", "localharness[browser,dev]==1.2.3"),
    ("exact-tokenizer", "localharness[browser,dev,exact-tokenizer]==1.2.3"),
    ("Exact_Tokenizer", "localharness[browser,dev,exact-tokenizer]==1.2.3"),
])
def test_members_are_canonicalized(dist, extra, spec):
    assert install_route(extra).spec == spec


@pytest.mark.parametrize("extra", ["browser; rm -rf /", "nope", "", "localharness[browser]"])
def test_unknown_or_hostile_extra_has_no_route(dist, extra):
    assert install_route(extra) is None


def test_no_route_without_an_installed_distribution(dist, monkeypatch):
    monkeypatch.setattr(importlib.metadata, "distribution",
                        lambda name: (_ for _ in ()).throw(importlib.metadata.PackageNotFoundError(name)))
    assert install_route("mobile") is None


# --------------------------------------------------------------------------- the six rows

def test_source_checkout_syncs_the_union_with_dev_surviving(dist, monkeypatch, tmp_path):
    root = _checkout(monkeypatch, tmp_path)
    route = install_route("mobile")
    assert route.argv is None
    assert route.manual == Manual(None, False, f"cd '{root}' && {SYNC}")


def test_source_checkout_on_windows_quotes_with_list2cmdline(dist, monkeypatch, tmp_path):
    root = _checkout(monkeypatch, tmp_path)
    monkeypatch.setattr(update_cmd, "_is_windows", lambda: True)
    assert install_route("mobile").manual == Manual(None, False, f'cd "{root}" && {SYNC}')


def test_uv_tool_install_runs_uv_here(dist, monkeypatch):
    monkeypatch.setattr(update_cmd, "_installer_kind", lambda: ("uv-tool", "/usr/bin/uv"))
    route = install_route("mobile")
    assert (route.argv, route.manual) == (["/usr/bin/uv", "tool", "install", SPEC], None)


@pytest.mark.parametrize("windows, command", [
    (False, f"uv tool install {shlex.quote(SPEC)}"),
    (True, f"uv tool install {SPEC}"),
])
def test_uv_tool_install_without_uv_asks_for_uv_first(dist, monkeypatch, windows, command):
    monkeypatch.setattr(update_cmd, "_installer_kind", lambda: ("uv-tool", None))
    monkeypatch.setattr(update_cmd, "_is_windows", lambda: windows)
    route = install_route("mobile")
    assert route.argv is None
    assert route.manual == Manual(UV_PREREQ, windows, command)


def test_uv_tool_install_on_windows_quits_first(dist, monkeypatch):
    monkeypatch.setattr(update_cmd, "_installer_kind", lambda: ("uv-tool", "C:\\uv\\uv.exe"))
    monkeypatch.setattr(update_cmd, "_is_windows", lambda: True)
    route = install_route("mobile")
    assert route.argv is None
    assert route.manual == Manual(None, True, f"uv tool install {SPEC}")


def test_pip_install_runs_isolated_pip_here(dist):
    route = install_route("mobile")
    assert (route.argv, route.manual) == (PIP_ARGV, None)


def test_pip_install_without_pip_asks_for_ensurepip_first(dist, monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    route = install_route("mobile")
    assert route.argv is None
    assert route.manual == Manual("Run python -m ensurepip --upgrade first.", False, shlex.join(PIP_ARGV))


# --------------------------------------------------------------------------- _installer_kind

@pytest.mark.parametrize("prefix, uv, expected", [
    ("uv/tools/localharness", "/usr/bin/uv", ("uv-tool", "/usr/bin/uv")),
    ("uv/tools/localharness", None, ("uv-tool", None)),
    ("some/venv", "/usr/bin/uv", ("pip", None)),
])
def test_installer_kind(monkeypatch, tmp_path, prefix, uv, expected):
    (tmp_path / prefix).mkdir(parents=True)
    monkeypatch.setattr(update_cmd.sys, "prefix", str(tmp_path / prefix))
    monkeypatch.setattr(update_cmd, "which_outside_cwd", lambda _n: uv)
    assert update_cmd._installer_kind() == expected


@pytest.mark.parametrize("kind, expected", [
    (("uv-tool", "/usr/bin/uv"), ["/usr/bin/uv", "tool", "upgrade", "localharness"]),
    (("uv-tool", None), None),
    (("pip", None), [sys.executable, "-m", "pip", "install", "--upgrade", "localharness"]),
])
def test_upgrade_command_is_byte_identical(monkeypatch, kind, expected):
    monkeypatch.setattr(update_cmd, "_installer_kind", lambda: kind)
    assert update_cmd._upgrade_command() == expected


# --------------------------------------------------------------------------- installed_extras

_REQUIRES = ["pydantic>=2", "starlette<1,>=0.40; extra == 'mobile'", "uvicorn>=0.30; extra == 'mobile'",
             "discord.py>=2.3; extra == 'dispatch'"]


def _fake_dists(monkeypatch, *, extras: list[str], installed: dict[str, str]) -> None:
    def distribution(name: str):
        if name == "localharness":
            return SimpleNamespace(requires=_REQUIRES, metadata=_Metadata(extras))
        if name in installed:
            return SimpleNamespace(version=installed[name])
        raise importlib.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(importlib.metadata, "distribution", distribution)


@pytest.mark.parametrize("extras, installed, expected", [
    (["mobile", "Dispatch"], {"starlette": "0.46.0", "uvicorn": "0.34.0"}, {"mobile"}),
    (["mobile", "Dispatch"], {"starlette": "0.46.0", "uvicorn": "0.34.0", "discord.py": "2.4"}, {"mobile", "dispatch"}),
    (["mobile", "Dispatch"], {"starlette": "0.46.0"}, set()),
    ([], {}, set()),
])
def test_installed_extras_are_the_canonical_names_extra_installed_accepts(monkeypatch, extras, installed, expected):
    _fake_dists(monkeypatch, extras=extras, installed=installed)
    assert installed_extras() == frozenset(expected)


def test_installed_extras_is_empty_without_an_installed_localharness(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "distribution",
                        lambda name: (_ for _ in ()).throw(importlib.metadata.PackageNotFoundError(name)))
    assert installed_extras() == frozenset()


def test_installed_extras_on_this_install():
    """Unmocked: this suite runs under the dev extra (`uv sync --extra dev`)."""
    found = installed_extras()
    assert "dev" in found
    assert all(e == canonicalize_name(e) for e in found)


# --------------------------------------------------------------------------- several extras

def test_several_extras_make_one_spec_over_every_installed_extra(dist):
    route = install_route("mobile", "exact-tokenizer")
    req = Requirement(route.spec)
    assert route.spec == "localharness[browser,dev,exact-tokenizer,mobile]==1.2.3"
    assert req.extras == {"browser", "dev", "exact-tokenizer", "mobile"}
    assert route.argv[-1] == route.spec
    assert install_route("mobile") == install_route("mobile", "mobile")
    assert install_route("mobile").spec == SPEC


@pytest.mark.parametrize("wanted", [("mobile", "nope"), ("nope", "mobile"), ()])
def test_several_extras_with_one_undeclared_or_none_have_no_route(dist, wanted):
    assert install_route(*wanted) is None


# --------------------------------------------------------------------------- the installer guard

def test_installer_guard_refuses_an_install_argv():
    with pytest.raises(AssertionError, match="spawned an installer"):
        asyncio.run(asyncio.create_subprocess_exec(sys.executable, "-I", "-m", "pip", "install", "x"))


def test_installer_guard_stops_install_pass_on_an_argv_route_with_no_fake(bare):
    with pytest.raises(AssertionError, match="spawned an installer"):
        bare.run()
    assert list(bare.cfg.iterdir()) == []


# --------------------------------------------------------------------------- extra_for

def _entry(extra, *, bundled=True, manifest=True) -> PlanEntry:
    m = PluginManifest(name="plug", version="0.1", kind="tools", requires_extra=extra) if manifest else None
    return PlanEntry("plug", bundled, "off", "built in", "", "", m)


@pytest.fixture
def seam(monkeypatch):
    monkeypatch.setitem(resolve_mod.resolve.__kwdefaults__, "extra_installed", lambda e: e in {"browser", "dev"})


@pytest.mark.parametrize("entry, expected", [
    (_entry("mobile"), "mobile"),
    (_entry("mobile", bundled=False), None),
    (_entry("x;rm -rf ~"), None),
    (_entry("nope"), None),
    (_entry("browser"), None),
    (_entry(None), None),
    (_entry("mobile", manifest=False), None),
])
def test_extra_for_gives_only_a_missing_provided_extra_of_a_bundled_plugin(dist, seam, entry, expected):
    assert extra_for(entry) == expected
    route = install_route_for(entry)
    assert (route is not None) == (expected is not None)
    assert route is None or route.spec == SPEC


# --------------------------------------------------------------------------- install_pass on a faked spawn

class _Proc:
    def __init__(self, chunks=(), code=0, *, blocked=False, hangs=False, kill_raises=False):
        self.chunks, self.code, self.blocked, self.hangs, self.kill_raises = list(chunks), code, blocked, hangs, kill_raises
        self.returncode, self.calls, self.reading = None, [], asyncio.Event()
        self.stdout = SimpleNamespace(read=self._read)

    async def _read(self, n):
        if self.blocked:
            self.reading.set()
            await asyncio.Event().wait()
        return self.chunks.pop(0) if self.chunks else b""

    async def wait(self):
        self.calls.append("wait")
        if self.hangs and "kill" not in self.calls:
            await asyncio.Event().wait()
        self.returncode = -9 if "kill" in self.calls else -15 if "terminate" in self.calls else self.code
        return self.returncode

    def terminate(self):
        self.calls.append("terminate")

    def kill(self):
        self.calls.append("kill")
        if self.kill_raises:
            raise ProcessLookupError


class Harness:
    def __init__(self, monkeypatch, tmp_path):
        self.cfg = tmp_path / "cfg"
        self.cfg.mkdir(mode=0o700)
        self.session = PluginSession(self.cfg, None, frozenset(), {}, {})
        self.progress, self.proc, self.answer = PassProgress(), _Proc(), True
        self.buf = io.StringIO()
        self.console = Console(file=self.buf, width=400)
        self.spawns, self.phases, self.asked = [], [], []
        self.questions = {"a": "", "b": ""}
        monkeypatch.setattr(resolve_mod, "resolve", lambda loader: SimpleNamespace(plan=SimpleNamespace(
            entry=lambda n: SimpleNamespace(manifest=SimpleNamespace(setup_action=self.questions[n])))))

    async def _spawn(self, *argv, **kw):
        self.spawns.append((argv, kw))
        self.phases.append(self.progress.phase)
        return self.proc

    async def _confirm(self, text, *, default, tail=""):
        self.asked.append((text, default, tail))
        return self.answer

    def call(self, names=("a",), extras=("mobile",), **kw):
        return install_pass(self.session, set(extras), list(names), progress=self.progress,
                            confirm=self._confirm, console=self.console, **kw)

    def run(self, *a, **kw):
        return asyncio.run(self.call(*a, **kw))

    @property
    def lines(self) -> list[str]:
        return self.buf.getvalue().splitlines()


@pytest.fixture
def bare(dist, monkeypatch, tmp_path):
    return Harness(monkeypatch, tmp_path)


@pytest.fixture
def h(bare, monkeypatch):
    monkeypatch.setattr(asyncio, "create_subprocess_exec", bare._spawn)
    return bare


def test_install_pass_consents_installs_and_returns_the_names_on_uv(h, monkeypatch):
    monkeypatch.setattr(update_cmd, "_installer_kind", lambda: ("uv-tool", "/usr/bin/uv"))
    h.questions["a"] = "Download the model (1.2 GB)?"
    spec = "localharness[browser,dev,exact-tokenizer,mobile]==1.2.3"
    h.proc = _Proc([b"all good\n"])
    got = h.run(("a", "b"), ("mobile", "exact-tokenizer"))
    assert got == frozenset({"a", "b"})
    assert h.lines == [
        "Enable a, b",
        f"uv installs {spec} from your package index and runs its build steps."
        " Packages added with uv tool install --with are dropped.",
        "a: Download the model (1.2 GB)?",
        "/usr/bin/uv tool install 'localharness[browser,dev,exact-tokenizer,mobile]==1.2.3'",
        f"uv installing {spec} — output is shown only if it fails"]
    [(text, default, tail)] = h.asked
    assert (text, default) == ("", True)
    assert tail == "n = enable without installing or downloading · Ctrl-C cancels this enable — nothing is written"
    [(argv, kw)] = h.spawns
    assert list(argv) == ["/usr/bin/uv", "tool", "install", spec]
    assert Path(kw["cwd"]).parent == h.cfg and Path(kw["cwd"]) != Path.cwd()
    assert (kw["stdin"], kw["stdout"], kw["stderr"]) == (
        asyncio.subprocess.DEVNULL, asyncio.subprocess.PIPE, asyncio.subprocess.STDOUT)
    assert "env" not in kw and "shell" not in kw
    assert h.phases == ["install"]
    assert list(h.cfg.iterdir()) == []


def test_install_pass_on_pip_runs_isolated_pip_and_adds_no_with_sentence(h):
    assert h.run() == frozenset({"a"})
    assert h.lines[1] == f"pip installs {SPEC} from your package index and runs its build steps."
    assert h.lines[-1] == f"pip installing {SPEC} — output is shown only if it fails"
    [(argv, _)] = h.spawns
    assert list(argv) == PIP_ARGV


def test_install_pass_in_a_batch_names_the_batch_stop(h):
    h.run(in_batch=True)
    assert h.asked[0][2].endswith("Ctrl-C stops the batch")


def test_install_pass_a_listed_plugin_with_no_question_adds_no_download_paragraph(h):
    h.questions["b"] = "Fetch b?"
    h.run(("a", "b"))
    assert [ln for ln in h.lines if ln.startswith(("a:", "b:"))] == ["b: Fetch b?"]


def test_the_install_strings_are_the_spec_text():
    p = plugins_cmd
    assert p.CTRLC.format(does=p.CTRLC_DOES["prompt"]) == "Ctrl-C cancels this enable — nothing is written"
    assert p.CTRLC.format(does=p.CTRLC_DOES["batch"]) == "Ctrl-C stops the batch"
    assert p.CTRLC_IGNORED.format(step="install") == (
        "Ctrl-C is ignored in this session; this install cannot be stopped from the keyboard")
    assert p.INSTALLING.format(installer="pip", spec="s") == "pip installing s — output is shown only if it fails"
    assert p.CONSENT == ("Enable {names}",
                         "{installer} installs {spec} from your package index and runs its build steps.{with}",
                         "{name}: {question}", "{command}")
    assert p.CONSENT_WITH == " Packages added with uv tool install --with are dropped."
    assert p.CONSENT_ASK.format(ctrlc="C") == "n = enable without installing or downloading · C"
    assert p.INSTALL_SKIPPED == ("{names}: enabled in config, package not installed{why}",
                                 "To finish later, in another terminal run:", "{command}", "then {route}")
    assert dict(p.INSTALL_SKIPPED_WHY) == {
        "declined": "", "failed": " — the install failed (exit {code}; see the output above)",
        "not_writable": " — {dir} is not writable",
        "open": " — {dir} is open to other accounts and could not be made private",
        "no_temp": " — no temp dir under {dir}", "no_start": " — the installer could not start: {error}"}
    assert p.ROUTE_BACK == "{rerun} under /plugins → {names}"
    assert p.ENABLE_MANUAL == "{name}: enabled in config; this install must be done by hand."
    assert p.TYPED_NEEDS_EXTRA == "{name}: needs {extra}."
    assert p.TYPED_ROUTE == "/plugins enable {name} again"


# --------------------------------------------------------------------------- decline, abort, failure

SKIP = ["To finish later, in another terminal run:", shlex.join(PIP_ARGV), "then Re-run setup under /plugins → a"]


def test_install_pass_declined_prints_one_skipped_block_and_runs_nothing(h):
    h.answer = False
    assert h.run() == frozenset()
    assert h.lines[-4:] == ["a: enabled in config, package not installed", *SKIP]
    assert h.buf.getvalue().count("package not installed") == 1
    assert "installing" not in h.buf.getvalue() and h.spawns == []
    assert h.progress.phase == "consent"


def test_install_pass_aborted_at_the_consent_spawns_nothing(h):
    h.answer = None
    with pytest.raises(click.exceptions.Abort):
        h.run()
    assert h.spawns == [] and "package not installed" not in h.buf.getvalue()


def test_install_pass_exit_code_shows_the_last_thirty_lines_then_one_skipped_block(h):
    h.proc = _Proc([b"".join(b"line %d\n" % i for i in range(40))], code=3)
    assert h.run() == frozenset()
    at = h.lines.index(f"pip installing {SPEC} — output is shown only if it fails")
    assert h.lines[at + 1:] == [f"line {i}" for i in range(10, 40)] + [
        "a: enabled in config, package not installed — the install failed (exit 3; see the output above)", *SKIP]
    assert h.buf.getvalue().count("installing") == 1 and h.proc.calls == ["wait"]


def test_install_pass_failure_output_never_shows_a_url_credential(h):
    h.proc = _Proc([b"Looking in indexes: https://user:pw@idx.test/simple\nboom\n"], code=1)
    assert h.run() == frozenset()
    assert "Looking in indexes: https://idx.test/simple" in h.lines and "pw" not in h.buf.getvalue()


def test_install_pass_silent_success_prints_installing_once_and_nothing_else(h):
    assert h.run() == frozenset({"a"})
    assert h.lines[-1] == f"pip installing {SPEC} — output is shown only if it fails"
    assert h.proc.calls == ["wait"]


def test_install_pass_silent_failure_prints_the_exit_code_and_no_output_lines(h):
    h.proc = _Proc(code=7)
    h.run()
    at = h.lines.index(f"pip installing {SPEC} — output is shown only if it fails")
    assert h.lines[at + 1:] == [
        "a: enabled in config, package not installed — the install failed (exit 7; see the output above)", *SKIP]


def test_install_pass_a_success_refreshes_the_import_caches(h, monkeypatch):
    seen = []
    monkeypatch.setattr(plugins_cmd.importlib, "invalidate_caches", lambda: seen.append(h.proc.calls[:]))
    h.run()
    assert seen == [["wait"]]


def test_install_pass_a_manual_route_asks_nothing_and_prints_its_block(h, monkeypatch):
    monkeypatch.setattr(update_cmd, "_installer_kind", lambda: ("uv-tool", None))
    assert h.run(("a", "b")) == frozenset()
    assert h.lines == [
        "a, b: enabled in config, package not installed", "Install uv first (https://docs.astral.sh/uv/).",
        "In another terminal run:", f"uv tool install {shlex.quote(SPEC)}",
        "then Re-run setup under /plugins → a, b"]
    assert h.asked == [] and h.spawns == []


# --------------------------------------------------------------------------- the private directory

def _only_skipped(h, why: str) -> None:
    assert h.lines[-4:] == [f"a: enabled in config, package not installed{why}", *SKIP]
    assert "installing" not in h.buf.getvalue() and h.spawns == []
    assert list(h.cfg.iterdir()) == []


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write anywhere")
def test_install_pass_not_writable_config_dir_aborts_with_its_reason(h):
    h.cfg.chmod(0o500)
    try:
        assert h.run() == frozenset()
    finally:
        h.cfg.chmod(0o700)
    _only_skipped(h, f" — {h.cfg} is not writable")


def test_install_pass_open_to_others_config_dir_aborts_with_its_reason(h, monkeypatch):
    monkeypatch.setattr(plugins_cmd, "ensure_private_dir", lambda path: None)
    h.cfg.chmod(0o755)
    assert h.run() == frozenset()
    _only_skipped(h, f" — {h.cfg} is open to other accounts and could not be made private")


def test_install_pass_open_to_others_is_not_checked_on_windows(h, monkeypatch):
    monkeypatch.setattr(plugins_cmd, "ensure_private_dir", lambda path: None)
    monkeypatch.setattr(update_cmd, "_is_windows", lambda: True)
    h.cfg.chmod(0o755)
    assert h.run() == frozenset({"a"}) and len(h.spawns) == 1


def test_install_pass_tightens_an_open_config_dir_it_can_and_goes_on(h):
    h.cfg.chmod(0o755)
    assert h.run() == frozenset({"a"})
    assert h.cfg.stat().st_mode & 0o077 == 0


def test_install_pass_no_temp_dir_aborts_with_its_reason(h, monkeypatch):
    def refuse(**kw):
        raise OSError("disk full")
    monkeypatch.setattr(plugins_cmd.tempfile, "mkdtemp", refuse)
    assert h.run() == frozenset()
    _only_skipped(h, f" — no temp dir under {h.cfg}")


def test_install_pass_a_spawn_that_raises_takes_the_decline_path_and_leaves_no_directory(h, monkeypatch):
    async def refuse(*argv, **kw):
        raise FileNotFoundError(2, "No such file https://user:pw@idx.test/simple")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", refuse)
    assert h.run() == frozenset()
    why = " — the installer could not start: [Errno 2] No such file https://idx.test/simple"
    assert h.lines[-4:] == [f"a: enabled in config, package not installed{why}", *SKIP]
    assert h.buf.getvalue().count("package not installed") == 1 and "pw" not in h.buf.getvalue()
    assert list(h.cfg.iterdir()) == []


# --------------------------------------------------------------------------- the output reader

def _stream(chunks):
    queue = list(chunks)

    async def read(n):
        return queue.pop(0) if queue else b""
    return SimpleNamespace(read=read)


@pytest.mark.parametrize("chunks, expected", [
    ([b"x" * 4096], ["x" * 4096]),
    ([b"x" * 4097], ["x" * 4096]),
    ([b"x" * 70_000], ["x" * 4096]),
    ([b"x" * 4096] * 17 + [b"x" * 1000], ["x" * 4096]),
    ([b"x" * 5000 + b"\nnext\n"], ["x" * 4096, "next"]),
    ([b"abc\rdef\n"], ["abcdef"]),
    ([b"ab", b"c\r", b"\ndef"], ["abc", "def"]),
    ([b"\xff\xfe"], ["\ufffd\ufffd"]),
    ([b"\x1b]8;;http://evil.test\x07link\x1b]8;;\x07\n"], ["link"]),
    ([b"", b"never"], []),
])
def test_output_lines_are_truncated_sanitized_and_never_raise(chunks, expected):
    assert asyncio.run(plugins_cmd._read_lines(_stream(chunks))) == expected


def test_output_keeps_only_the_last_thirty_lines():
    out = asyncio.run(plugins_cmd._read_lines(_stream([b"".join(b"%d\n" % i for i in range(100))])))
    assert out == [str(i) for i in range(70, 100)]


# --------------------------------------------------------------------------- stopping the run

def _cancel_during_read(h) -> None:
    async def go():
        task = asyncio.ensure_future(h.call())
        await h.proc.reading.wait()
        task.cancel()
        await task

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(go())


def test_install_pass_cancel_terminates_the_child_and_removes_the_directory(h):
    h.proc = _Proc(blocked=True)
    _cancel_during_read(h)
    assert h.proc.calls == ["terminate", "wait"] and h.progress.phase == "install"
    assert list(h.cfg.iterdir()) == []


def test_install_pass_cancel_timeout_kills_the_child_and_still_raises_cancelled(h, monkeypatch):
    monkeypatch.setattr(plugins_cmd, "_TERMINATE_GRACE", 0.01)
    h.proc = _Proc(blocked=True, hangs=True)
    _cancel_during_read(h)
    assert h.proc.calls == ["terminate", "wait", "kill", "wait"]
    assert list(h.cfg.iterdir()) == []


def test_install_pass_cancel_kill_on_a_reaped_child_does_not_raise(h, monkeypatch):
    monkeypatch.setattr(plugins_cmd, "_TERMINATE_GRACE", 0.01)
    h.proc = _Proc(blocked=True, hangs=True, kill_raises=True)
    _cancel_during_read(h)
    assert h.proc.calls == ["terminate", "wait", "kill", "wait"]


def test_install_pass_a_child_already_exited_is_not_signalled(h):
    h.run()
    assert not {"terminate", "kill"} & set(h.proc.calls)


def test_install_pass_ignored_sigint_says_so_before_installing(h, monkeypatch):
    monkeypatch.setattr(signal, "getsignal", lambda s: signal.SIG_IGN)
    h.run()
    said = "Ctrl-C is ignored in this session; this install cannot be stopped from the keyboard"
    assert h.lines[-2:] == [said, f"pip installing {SPEC} — output is shown only if it fails"]
    assert h.buf.getvalue().count(said) == 1


def test_install_pass_a_live_sigint_handler_says_nothing_about_ignoring(h):
    h.run()
    assert "ignored" not in h.buf.getvalue()


# --------------------------------------------------------------------------- rendering

OSC = "\x1b]8;;http://evil.test\x07"
OSC_END = "\x1b]8;;\x07"


def test_install_pass_prints_every_text_verbatim_and_never_as_markup(h):
    h.questions["a"] = f"Fetch [bold]x[/bold] and [/x] {OSC}now{OSC_END}?"
    h.proc = _Proc([f"[red]boom[/red] [/x] {OSC}bad{OSC_END}\r\n".encode()], code=2)
    h.run(("a",))
    out = h.buf.getvalue()
    assert "a: Fetch [bold]x[/bold] and [/x] now?" in out
    assert "[red]boom[/red] [/x] bad" in out
    assert "localharness[browser,dev,mobile]==1.2.3" in out
    assert "\x1b" not in out and "\x07" not in out and "\r" not in out


def test_install_pass_prints_a_manual_route_command_with_brackets_intact(h, monkeypatch):
    monkeypatch.setattr(update_cmd, "_installer_kind", lambda: ("uv-tool", None))
    h.run()
    assert f"uv tool install {shlex.quote(SPEC)}" in h.lines


def test_manual_block_with_a_prereq_and_quit_first():
    manual = Manual("Install uv first (https://docs.astral.sh/uv/).", True, "uv tool install x")
    assert plugins_cmd.manual_block(["a"], manual, plugins_cmd.ROUTE_BACK) == (
        "Install uv first (https://docs.astral.sh/uv/).", "Quit localharness first, then run:",
        "uv tool install x", "then start localharness and Re-run setup under /plugins → a")


def test_manual_block_without_a_prereq_or_quit_first_takes_the_typed_route():
    assert plugins_cmd.manual_block(["a"], Manual(None, False, "uv tool install x"), plugins_cmd.TYPED_ROUTE) == (
        "In another terminal run:", "uv tool install x", "then /plugins enable a again")


def test_manual_block_does_not_reparse_a_name_as_a_template():
    block = plugins_cmd.manual_block(["{rerun}", "b"], Manual(None, False, "c"), plugins_cmd.ROUTE_BACK)
    assert block[-1] == "then Re-run setup under /plugins → {rerun}, b"


# --------------------------------------------------------------------------- the strings, once each

_STRINGS = ("CTRLC", "CTRLC_DOES", "CTRLC_IGNORED", "INSTALLING", "CONSENT", "CONSENT_WITH", "CONSENT_ASK",
            "INSTALL_SKIPPED", "INSTALL_SKIPPED_WHY", "ROUTE_BACK", "ENABLE_MANUAL", "TYPED_NEEDS_EXTRA",
            "TYPED_ROUTE")
_SOURCE = Path(plugins_cmd.__file__).read_text()
_TREE = ast.parse(_SOURCE)


@pytest.mark.parametrize("name", _STRINGS)
def test_each_install_string_is_assigned_once(name):
    stores = [n for n in ast.walk(_TREE) if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Store)]
    assert len(stores) == 1


@pytest.mark.parametrize("fragment", [
    "from your package index", "output is shown only if it fails", "package not installed",
    "To finish later, in another terminal run:", "this install must be done by hand",
    "Quit localharness first, then run:", "In another terminal run:", "cancels this enable",
    "stops the batch", "cannot be stopped from the keyboard", "dropped.", "under /plugins",
])
def test_each_install_text_appears_once_in_the_module(fragment):
    assert _SOURCE.count(fragment) == 1


def test_the_keyed_install_strings_cannot_be_changed_at_runtime():
    for table in (plugins_cmd.CTRLC_DOES, plugins_cmd.INSTALL_SKIPPED_WHY):
        with pytest.raises(TypeError):
            table["x"] = "y"


def test_the_third_party_confirm_is_unchanged():
    assert plugins_cmd.THIRD_PARTY_CONFIRM == (
        "{name} is a plugin you installed, not one that ships with LocalHarness: turning it on lets its code "
        "run with your permissions on this machine, in every session. Turn it on?")


def test_plugins_cmd_imports_nothing_from_the_flow_module():
    modules = {a.name for n in ast.walk(_TREE) if isinstance(n, ast.Import) for a in n.names}
    modules |= {(n.module or "") + "." + a.name for n in ast.walk(_TREE) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert not [m for m in modules if "plugins_interview" in m]
