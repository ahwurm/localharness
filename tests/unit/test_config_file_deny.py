"""The harness's own settings files are out of the agent tools' reach, in every mode.

Orchestrator ruling R13: the agent's `write` and `edit` tools, and every shell command the gate can
read as writing or deleting a file, cannot change `config.yaml` or `overrides.yaml` in the
machine's config folder or in any project's `.localharness/` — in every mode, `unattended`
included. The model gets one line naming `localharness components set`; nobody is asked. The rule
sits in `PermissionGate.check` AHEAD of the verdict, so no ticket, staged approval or mode turns it
into an allow, and `evaluate()` keeps answering the protected-path ask the 2026-09-11 pins state.

Self-extension is untouched: agent files, tool scripts, audit.jsonl and memory.db get exactly
today's outcome. Code run inline through an interpreter is not read (the named residual).

The machine's config folder is conftest's hermetic LOCALHARNESS_HOME.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from localharness.agent import gate as gate_mod
from localharness.agent import gate_types, verdict
from localharness.agent.gate import PermissionGate, call_identity
from localharness.agent.gate_types import MODE_STRICTNESS, Decision, GateSettings, ToolMeta, Verdict
from localharness.agent.verdict import GateContext, evaluate
from localharness.config.grants import GrantStore

SETTINGS = GateSettings()
MODES = sorted(MODE_STRICTNESS)
WRITE = ToolMeta(group="fs.write", destructive=True)
SHELL = ToolMeta(group="shell", destructive=True)


@pytest.fixture
def g(monkeypatch) -> Path:
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    home = Path(os.environ["LOCALHARNESS_HOME"])
    (home / "overrides.yaml").write_text("{}\n", encoding="utf-8")
    return home


@pytest.fixture
def proj(tmp_path) -> Path:
    root = (tmp_path / "proj").resolve()
    (root / ".localharness" / "agents").mkdir(parents=True)
    for name in ("config.yaml", "overrides.yaml"):
        (root / ".localharness" / name).write_text("{}\n", encoding="utf-8")
    return root


def _gate(tmp_path: Path, proj: Path, mode: str, asker) -> PermissionGate:
    return PermissionGate(boundary=proj, workspace=proj, grants=GrantStore(tmp_path / "grants.yaml"),
                          channel_name="test", mode=mode, asker=asker)  # type: ignore[arg-type]


async def _never(request):
    raise AssertionError(f"asked: {request.display}")


async def _once(request):
    return Decision(kind="allow_once")


async def _check(gate: PermissionGate, tool: str, params: dict):
    return await gate.check(tool, params, SHELL if tool == "bash_exec" else WRITE,
                            agent_id="a", session_id="s")


def _refused(path: Path) -> str:
    return gate_types.HARNESS_CONFIG_FILE_REASON.format(path=path)


BLOCKED = {
    "write-global-config": lambda g, p: ("write", {"path": str(g / "config.yaml"), "content": "x"},
                                         g / "config.yaml"),
    "edit-global-overrides": lambda g, p: ("edit", {"path": str(g / "overrides.yaml"),
                                                    "old_string": "{}", "new_string": "x"},
                                           g / "overrides.yaml"),
    "write-project-config": lambda g, p: ("write", {"path": str(p / ".localharness" / "config.yaml"),
                                                    "content": "x"}, p / ".localharness/config.yaml"),
    "write-project-overrides": lambda g, p: (
        "write", {"path": str(p / ".localharness" / "overrides.yaml"), "content": "x"},
        p / ".localharness/overrides.yaml"),
    "write-relative-to-the-workspace": lambda g, p: (
        "write", {"path": ".localharness/config.yaml", "content": "x"}, p / ".localharness/config.yaml"),
    "shell-redirect-in-the-project": lambda g, p: (
        "bash_exec", {"command": "echo x > .localharness/overrides.yaml", "working_dir": str(p)},
        p / ".localharness/overrides.yaml"),
    "shell-rm": lambda g, p: ("bash_exec", {"command": f"rm {g / 'config.yaml'}"}, g / "config.yaml"),
    "shell-tee": lambda g, p: ("bash_exec", {"command": f"tee {g / 'config.yaml'}"}, g / "config.yaml"),
    "shell-truncate": lambda g, p: ("bash_exec", {"command": f"truncate -s0 {g / 'overrides.yaml'}"},
                                    g / "overrides.yaml"),
    "shell-sed-in-place": lambda g, p: ("bash_exec", {"command": f"sed -i s/a/b/ {g / 'overrides.yaml'}"},
                                        g / "overrides.yaml"),
    "shell-cp-after-cd": lambda g, p: ("bash_exec", {"command": f"cd {g} && cp /tmp/x overrides.yaml"},
                                       g / "overrides.yaml"),
    "write-through-a-symlink": lambda g, p: ("write", {"path": str(p / "link.yaml"), "content": "x"},
                                             g / "config.yaml"),
    # macOS and Windows filesystems ignore case: there this IS the settings file.
    "another-case-spelling": lambda g, p: ("write", {"path": str(g / "Config.YAML"), "content": "x"},
                                           g / "Config.YAML"),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", sorted(BLOCKED))
async def test_the_four_files_are_refused_in_every_mode_and_nobody_is_asked(case, mode, g, proj,
                                                                             tmp_path):
    (proj / "link.yaml").symlink_to(g / "config.yaml")
    tool, params, target = BLOCKED[case](g, proj)
    gate = _gate(tmp_path, proj, mode, _never)

    outcome = await _check(gate, tool, params)

    assert not outcome.allowed
    assert outcome.reason == _refused(target.resolve())
    assert outcome.pending is None and not gate.pending, "nothing is staged for a human either"


@pytest.mark.asyncio
async def test_an_approved_ticket_never_turns_it_into_an_allow(g, proj, tmp_path):
    """Before this rule a config write in `auto` was parked, and `/approve` bought one run of it."""
    gate = _gate(tmp_path, proj, "auto", _never)
    params = {"path": str(g / "config.yaml"), "content": "x"}
    gate._approved_once[call_identity("write", params)] = 1

    outcome = await _check(gate, "write", params)

    assert not outcome.allowed and outcome.reason == _refused((g / "config.yaml").resolve())
    assert not gate._approved_once, "the ticket is spent on sight, never stockpiled"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_a_config_file_kept_elsewhere_through_a_symlink_is_still_refused(mode, g, proj,
                                                                               tmp_path):
    """A dotfiles setup: <global>/config.yaml is a link to a file in another folder. The file the
    harness reads is still the one named config.yaml in its config folder."""
    kept = tmp_path / "dotfiles" / "lh-config.yaml"
    kept.parent.mkdir()
    kept.write_text((g / "config.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    (g / "config.yaml").unlink()
    (g / "config.yaml").symlink_to(kept)
    gate = _gate(tmp_path, proj, mode, _never)

    for tool, params in (("write", {"path": str(g / "config.yaml"), "content": "x"}),
                         ("bash_exec", {"command": f"echo x > {g / 'config.yaml'}"})):
        outcome = await _check(gate, tool, params)
        assert not outcome.allowed and "localharness components set" in outcome.reason, tool


# The re-review's F-B and the fix-commit review: shapes the rule used to walk past. A copy, move or
# link INTO the config folder is a settings write when a source is named like a settings file (the
# refusal names that file); an unpack, a copy of a folder's CONTENTS, or a download whose name the
# server chooses, into the folder is refused as the folder (its contents are not known in advance).
# `$HOME` paths are expanded. Each is refused in every mode, `unattended` included.
INTO_THE_FOLDER = {
    "cp-into-the-folder": lambda g, p, h: (f"cp /tmp/config.yaml {g}/", g / "config.yaml"),
    "cp-into-the-folder-no-slash": lambda g, p, h: (f"cp /tmp/config.yaml {g}", g / "config.yaml"),
    "mv-into-the-folder": lambda g, p, h: (f"mv /tmp/overrides.yaml {g}/", g / "overrides.yaml"),
    "install-into-the-folder": lambda g, p, h: (f"install /tmp/config.yaml {g}", g / "config.yaml"),
    "ln-into-the-folder": lambda g, p, h: (f"ln -sf /tmp/config.yaml {g}/", g / "config.yaml"),
    "ln-target-directory": lambda g, p, h: (f"ln -sf -t {g} /tmp/config.yaml", g / "config.yaml"),
    "ln-target-directory-in-a-cluster": lambda g, p, h: (f"ln -sft {g} /tmp/config.yaml",
                                                         g / "config.yaml"),
    "cp-target-directory-in-a-cluster": lambda g, p, h: (f"cp -rt {g} /tmp/overrides.yaml",
                                                         g / "overrides.yaml"),
    "cp-into-a-projects-folder": lambda g, p, h: (f"cp /tmp/config.yaml {p / '.localharness'}/",
                                                  p / ".localharness" / "config.yaml"),
    "wget-into-the-folder": lambda g, p, h: (f"wget -P {g} https://x.test/config.yaml",
                                             g / "config.yaml"),
    "curl-into-the-folder": lambda g, p, h: (f"curl --output-dir {g} -O https://x.test/overrides.yaml",
                                             g / "overrides.yaml"),
    "cp-to-home-dollar-path": lambda g, p, h: ("cp /tmp/x $HOME/.localharness/config.yaml",
                                               h / ".localharness" / "config.yaml"),
    "redirect-to-home-dollar-path": lambda g, p, h: ("echo x > $HOME/.localharness/overrides.yaml",
                                                     h / ".localharness" / "overrides.yaml"),
    "redirect-to-braced-home": lambda g, p, h: ("echo x > ${HOME}/.localharness/overrides.yaml",
                                                h / ".localharness" / "overrides.yaml"),
    # the folder itself: what lands there is not on the command line
    "rsync-a-folders-contents": lambda g, p, h: (f"rsync -a /tmp/c/ {g}/", g),
    "cp-a-folders-contents": lambda g, p, h: (f"cp -r /tmp/c/. {g}/", g),
    "cp-no-target-directory": lambda g, p, h: (f"cp -rT /tmp/c {g}", g),
    "tar-extract-into-the-folder": lambda g, p, h: (f"tar -xf a.tar -C {g}", g),
    "tar-attached-directory": lambda g, p, h: (f"tar -xf a.tar -C{g}", g),
    "tar-directory-long-form": lambda g, p, h: (f"tar --directory={g} -xf a.tar", g),
    "tar-old-style-extract": lambda g, p, h: (f"tar xf a.tar -C {g}", g),
    "bsdtar-extract": lambda g, p, h: (f"bsdtar -xf a.tar -C {g}", g),
    "gtar-extract": lambda g, p, h: (f"gtar -xf a.tar --directory={g}", g),
    "unzip-into-the-folder": lambda g, p, h: (f"unzip -o a.zip -d {g}", g),
    "unzip-attached-directory": lambda g, p, h: (f"unzip -o a.zip -d{g}", g),
    "7z-into-the-folder": lambda g, p, h: (f"7z x a.7z -o{g}", g),
    "wget-server-named": lambda g, p, h: (f"wget --content-disposition -P {g} https://x.test/d", g),
    "curl-server-named": lambda g, p, h: (f"curl --output-dir {g} -OJ https://x.test/d", g),
    "unzip-into-home-dollar-folder": lambda g, p, h: ("unzip a.zip -d $HOME/.localharness",
                                                      h / ".localharness"),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", sorted(INTO_THE_FOLDER))
async def test_into_the_folder_and_home_shapes_are_refused_in_every_mode(case, mode, g, proj,
                                                                        tmp_path, monkeypatch):
    home = (tmp_path / "home").resolve()
    monkeypatch.setenv("HOME", str(home))
    command, target = INTO_THE_FOLDER[case](g, proj, home)
    gate = _gate(tmp_path, proj, mode, _never)

    outcome = await _check(gate, "bash_exec", {"command": command})

    assert not outcome.allowed
    reason = (gate_types.HARNESS_CONFIG_FILE_REASON if target.name in ("config.yaml", "overrides.yaml")
              else gate_types.HARNESS_CONFIG_FOLDER_REASON)
    assert outcome.reason == reason.format(path=target.resolve())
    assert outcome.pending is None and not gate.pending


def test_an_unpack_into_the_folder_says_to_unpack_elsewhere_in_one_line(g):
    reason = gate_types.HARNESS_CONFIG_FOLDER_REASON.format(path=g)
    assert "\n" not in reason and "unpack or fetch it elsewhere" in reason
    assert "localharness components set" in reason


@pytest.mark.parametrize("command", [
    "cp /tmp/cert.pem {g}/",
    "mv /tmp/notes.txt {g}",
    "install -m 600 /tmp/key.pem {g}/",
    "ln -s /tmp/x.yaml {g}/",
    "cp -t {g} /tmp/cert.pem",
    "rsync -a /tmp/stuff {g}/",
    "cp -r /tmp/stuff {g}/",
    "wget -P {g} https://x.test/cert.pem",
    "curl --output-dir {g} -O https://x.test/cert.pem",
], ids=["cp-cert", "mv-notes", "install-key", "ln-other-yaml", "cp-t-cert", "rsync-a-folder",
        "cp-a-folder", "wget-cert", "curl-cert"])
def test_a_copy_into_the_folder_of_anything_but_a_settings_file_is_left_to_the_verdict(
        g, proj, tmp_path, command):
    """The fix-commit review's precision point: refusing `cp cert.pem ~/.localharness/` sent the
    model to `components set`, which cannot copy a file. Only a settings file's name is refused."""
    ctx = _gate(tmp_path, proj, "auto", _once).context()

    assert verdict.harness_config_file_target("bash_exec", {"command": command.format(g=g)}, ctx,
                                              SETTINGS) is None


@pytest.mark.parametrize("command", [
    "tar -czf /tmp/backup.tgz -C {g} .",
    "tar czf /tmp/backup.tgz -C {g} config.yaml",
    "chmod -R go-rwx {g}",
    "chmod 700 {g}",
    "mkdir -p {g}",
    "touch {g}",
    "cp {g}/config.yaml /tmp/",
    "cp a.yaml {g}/agents/",
    "mv {g}/notes.txt {g}/agents/",
    "rsync -a {g}/ /tmp/backup/",
], ids=["tar-create-from-the-folder", "tar-create-one-file", "chmod-recursive", "chmod", "mkdir",
        "touch", "copy-out", "copy-into-agents", "move-into-agents", "rsync-out"])
def test_reading_or_tidying_the_folder_is_not_a_settings_write(g, proj, tmp_path, command):
    """The folder rule reads only a copy, move, link or unpack INTO the folder: an archive made
    FROM it (`tar -c -C`), doctor's own `chmod -R go-rwx` advice, `mkdir`, and copies out of it or
    into its agents/ folder stay with the verdict, exactly as before."""
    ctx = _gate(tmp_path, proj, "auto", _once).context()

    assert verdict.harness_config_file_target("bash_exec", {"command": command.format(g=g)}, ctx,
                                              SETTINGS) is None


UNTOUCHED = {
    "global-agent-file": lambda g, p: ("write", {"path": str(g / "agents" / "x.yaml"), "content": "x"}),
    "global-tool-script": lambda g, p: ("write", {"path": str(g / "tools" / "x.sh"), "content": "x"}),
    "audit-log": lambda g, p: ("write", {"path": str(g / "audit.jsonl"), "content": "x"}),
    "memory-db": lambda g, p: ("write", {"path": str(g / "agents" / "orchestrator" / "memory.db"),
                                         "content": "x"}),
    "an-agents-config-yaml": lambda g, p: ("write", {"path": str(p / ".localharness" / "agents" /
                                                                  "config.yaml"), "content": "x"}),
    "a-backup": lambda g, p: ("write", {"path": str(p / ".localharness" / "config.yaml.bak"),
                                        "content": "x"}),
    "a-project-source-config": lambda g, p: ("write", {"path": str(p / "src" / "config.yaml"),
                                                       "content": "x"}),
    "shell-read": lambda g, p: ("bash_exec", {"command": f"cat {g / 'config.yaml'}"}),
    "shell-find-read": lambda g, p: ("bash_exec", {"command": f"find {g / 'config.yaml'} -maxdepth 0"}),
    "interpreter-inline": lambda g, p: (
        "bash_exec", {"command": f"python3 -c \"open('{g / 'config.yaml'}','w')\""}),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("case", sorted(UNTOUCHED))
async def test_what_the_rule_never_touches_gets_exactly_todays_outcome(case, g, proj, tmp_path,
                                                                      monkeypatch):
    tool, params = UNTOUCHED[case](g, proj)
    ctx = _gate(tmp_path, proj, "auto", _once).context()
    assert verdict.harness_config_file_target(tool, params, ctx, SETTINGS) is None

    for mode in MODES:
        now = await _check(_gate(tmp_path / mode / "now", proj, mode, _once), tool, params)
        with monkeypatch.context() as m:
            m.setattr(gate_mod, "harness_config_file_target", lambda *a: None)
            today = await _check(_gate(tmp_path / mode / "today", proj, mode, _once), tool, params)
        assert (now.allowed, now.reason) == (today.allowed, today.reason), mode


def test_the_reason_is_one_line_naming_the_sanctioned_command(g):
    reason = _refused((g / "config.yaml").resolve())

    assert "\n" not in reason
    assert "localharness components set" in reason and str((g / "config.yaml").resolve()) in reason


@pytest.mark.parametrize("tool,params", [
    ("write", {"path": 42, "content": "x"}),
    ("write", {"path": "$HOME/.localharness/config.yaml", "content": "x"}),
    ("write", {"content": "x"}),
    ("write", "not a dict"),
    ("read", {"path": "<g>/config.yaml"}),
    ("python_exec", {"code": "open('<g>/config.yaml', 'w')"}),
    ("bash_exec", {"command": ["rm", "<g>/config.yaml"]}),
    ("bash_exec", {"command": "echo x > config.yaml", "working_dir": 7}),
], ids=["non-string-path", "unresolvable", "no-path", "params-not-a-dict", "read-tool",
        "python-exec", "non-string-command", "unplaceable-working-dir"])
def test_a_call_this_rule_cannot_read_is_left_to_the_verdict(g, proj, tmp_path, tool, params):
    if isinstance(params, dict):
        params = {k: v.replace("<g>", str(g)) if isinstance(v, str) else v for k, v in params.items()}
    ctx = _gate(tmp_path, proj, "auto", _once).context()

    assert verdict.harness_config_file_target(tool, params, ctx, SETTINGS) is None


def test_the_verdict_itself_still_asks_about_these_files(g, proj):
    """The 2026-09-11 pins: evaluate() answers a protected-path ask; the deny is the gate's."""
    ctx = GateContext(boundary=proj, workspace=proj, grants=lambda *a: None, mode="auto",
                      has_review_surface=True)
    result = evaluate("write", {"path": str(g / "config.yaml"), "content": "x"}, WRITE, ctx, SETTINGS)

    assert result.verdict is Verdict.ASK and result.request.klass == "protected-path"


@pytest.mark.parametrize("command,caught", [
    ("sh -c \"echo x > {g}/config.yaml\"", True),
    ("bash -c \"rm -f {g}/config.yaml\"", True),
    ("eval \"echo x > {g}/overrides.yaml\"", True),
    ("cd {g} && echo x > config.yaml", True),
    ("mv {g}/config.yaml /tmp/away.yaml", False),
    ("unlink {g}/config.yaml", False),
    ("shred -u {g}/config.yaml", False),
    ("find {g} -name config.yaml -delete", False),
    ("ln {g}/config.yaml hard.yaml", False),
    ("cd {g} && rm config.yaml", False),
    ("localharness components set provider.base_url http://elsewhere", False),
    ("cp /tmp/config.yaml {g}/", True),
    ("tar -xf a.tar -C {g}", True),
    ("cp /tmp/x $LOCALHARNESS_DIR/config.yaml", False),
    ("sudo cp /tmp/config.yaml {g}/", False),
    ("tar -xf evil.tar -C {g}/..", False),
    ("chmod 777 {g}/config.yaml", True),
    ("touch {g}/config.yaml", True),
    ("mkdir {g}/config.yaml", True),
    ("rsync /tmp/x {g}/config.yaml", True),
    ("dd if=/tmp/x of={g}/config.yaml", True),
    ("ln -sft {g} /tmp/config.yaml", True),
    ("curl -o {g}/config.yaml https://x.test/a", True),
    ("wget -P {g} https://x.test/config.yaml", True),
    ("unzip -o a.zip -d{g}", True),
    ("cd {g} && curl -OJ https://x.test/d", True),
    ("cp /tmp/cert.pem {g}/", False),
    ("export HOME={g} && cp /tmp/x ~/config.yaml", False),
    ("ln -s {g} {g}-l && cp /tmp/config.yaml {g}-l/", False),
], ids=["sh-c", "bash-c", "eval", "redirect-after-cd", "mv-away", "unlink", "shred", "find-by-name",
        "hard-link", "plain-rm-after-cd", "the-cli", "copy-into-the-folder", "unpack-into-the-folder",
        "another-variable", "through-sudo", "unpacked-above-the-folder", "chmod", "touch", "mkdir",
        "rsync-onto-the-file", "dd-of",
        "target-directory-in-a-cluster", "curl-o", "wget-P", "unzip-d-attached",
        "server-named-download", "another-name-into-the-folder", "home-reassigned-first",
        "a-link-made-by-the-same-command"])
def test_what_security_md_says_the_rule_reads_and_does_not(g, proj, tmp_path, command, caught):
    """SECURITY.md, "What the agent may change about its own setup": a shell payload is read; a
    command the gate does not read as writing or deleting that file is not caught, and the
    harness's own CLI changes settings by design. A change here is a change to that paragraph."""
    ctx = _gate(tmp_path, proj, "auto", _once).context()
    hit = verdict.harness_config_file_target("bash_exec", {"command": command.format(g=g)}, ctx,
                                             SETTINGS)

    assert (hit is not None) is caught
