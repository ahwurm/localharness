"""Nothing found: one question over this platform's reference rows, a verified fetch, and a record built from that file alone.
The hub is never reached: `download_file` is a stub that returns a real tiny GGUF, whose size and sha256 sit on the row."""
from __future__ import annotations

import hashlib
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
import yaml
from rich.console import Console

import localharness.cli.init_cmd as init_cmd
from localharness.cli import library
from localharness.cli.workspace import shown_path
from localharness.config import trust
from localharness.provider import autosense
from localharness.provider import server as managed_server
from localharness.provider.refarch import RTX_4060
from tests.unit.interview_support import UNREACHED, scripted
from tests.unit.test_init_core_setup import _detect, _exited, _flat, _hermetic, _init, _terminal  # noqa: F401
from tests.unit.test_init_discovery_records import _model, _not_found, _projector

SMI = "24564, 32607\n"
LINUX_LINE = library.INSTALL_LINES["linux"]


def _row(hf, **over):
    data = (hf / "model.gguf").read_bytes()
    return replace(RTX_4060, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), **over)


class Box(SimpleNamespace):
    def at(self, *, platform="linux", machine="x86_64", smi=None, ram=(16e9, 12e9), llama=False, vllm=False, rows=None):
        self.monkeypatch.setattr(sys, "platform", platform)
        self.monkeypatch.setattr(library.platform, "machine", lambda: machine)
        self.monkeypatch.setattr(autosense, "_run_nvidia_smi", lambda: smi)
        self.monkeypatch.setattr(autosense, "_read_meminfo",
                                 lambda: None if ram is None else {"MemTotal": int(ram[0]), "MemAvailable": int(ram[1])})
        self.monkeypatch.setattr(library, "REF_ARCHS", rows if rows is not None else (self.row,))
        self.monkeypatch.setattr(library, "which_outside_cwd", lambda n: f"/opt/{n}" if (
            n == "llama-server" and llama) else None)
        self.monkeypatch.setattr(managed_server, "vllm_binary", lambda _d: ("/opt/vllm", "on PATH") if vllm else None)
        return self


@pytest.fixture
def box(tmp_path, monkeypatch, fake_home):
    fake_home(tmp_path / "home", clear_overrides=False)
    hf = tmp_path / "hf"
    _model(hf / "model.gguf")
    _model(hf / "other.gguf")
    _projector(hf / "mmproj-x.gguf")
    row = _row(hf)
    calls: list[dict] = []

    def download_file(repo, file, *, revision=None):
        calls.append({"repo": repo, "file": file, "revision": revision})
        return str(hf / "model.gguf")

    monkeypatch.setattr(managed_server, "download_file", download_file)
    monkeypatch.setattr(managed_server, "download_model", lambda r: pytest.fail("a whole-repo download"))
    detect = _not_found()
    monkeypatch.setattr(init_cmd, "detect_provider", detect)
    monkeypatch.setattr(library, "detect_provider", detect)
    _terminal(monkeypatch)
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    return Box(monkeypatch=monkeypatch, hf=hf, row=row, calls=calls, cfg=cfg, tmp=tmp_path).at()


def test_enter_downloads_the_small_row_and_records_exactly_that_file(box, monkeypatch, capsys):
    script = scripted(monkeypatch, "download")

    recs = library.offer(box.cfg, person=True)

    [asked] = script.asked
    size = f"{box.row.bytes / 1e9:.1f} GB"
    assert asked.question == f"Download Qwen3.8-9B-Q4_K_M.gguf ({size}, fits in your 10 GB)?"
    assert [o.label for o in asked.options] == ["Download it", "Show the other rows", "Not now"]
    assert asked.options[0].hint == library.SMALL_ROW_HINT and asked.default == 0 and asked.cancel == "not-now"
    assert box.calls == [{"repo": box.row.repo, "file": box.row.file, "revision": box.row.revision}]
    path = box.hf / "model.gguf"
    assert recs.server.model == str(path) and recs.server.port == 8081 and recs.server.mmproj is None
    assert recs.peers == [] and recs.confirmed and recs.block_in_receipt and recs.served == "model"
    out = capsys.readouterr().out
    assert library.record_line("model", "llamacpp", 8081, None, path) in out
    assert recs.lines == [LINUX_LINE + library.THEN_START]


def test_the_install_line_comes_before_the_question_when_there_is_no_llama_server(box, monkeypatch, capsys):
    order: list[str] = []
    monkeypatch.setattr(library, "console", Console(file=sys.stdout, width=300))

    def ask(question, options, **_k):
        order.append(capsys.readouterr().out)
        return library.interview.Answer(("not-now",), False, True)

    monkeypatch.setattr(library.interview, "ask", ask)

    library.offer(box.cfg, person=True)

    assert LINUX_LINE in order[0]


def test_a_found_llama_server_prints_no_install_line_and_is_the_binary(box, monkeypatch, capsys):
    box.at(llama=True)
    scripted(monkeypatch, "download")

    recs = library.offer(box.cfg, person=True)

    assert recs.server.binary == "/opt/llama-server" and recs.lines == []
    assert LINUX_LINE not in capsys.readouterr().out


def test_a_planted_sibling_is_not_swept_up_and_the_first_run_writes_and_adopts_that_record(box, monkeypatch):
    scripted(monkeypatch, "download")
    _detect(monkeypatch, found=False)
    monkeypatch.setattr(init_cmd, "detect_provider", _not_found())

    result = _init(box.cfg)

    _exited(result, 0)
    written = yaml.safe_load((box.cfg / "config.yaml").read_text(encoding="utf-8"))
    assert written["server"]["model"] == str(box.hf / "model.gguf") and written["server"]["port"] == 8081
    assert "mmproj" not in written["server"] and written["extra_endpoints"] == []
    assert "other.gguf" not in (box.cfg / "config.yaml").read_text() and not (box.cfg / "overrides.yaml").exists()
    flat = _flat(result)
    assert flat.index(shown_path(box.hf / "model.gguf") + " · port 8081") < flat.index("LocalHarness configured")
    assert "Later:" in flat and LINUX_LINE.split(",")[0] in flat
    entries = trust.recorded_machine(box.cfg)["entries"]
    assert any(e["name"] == "server.model" for e in entries)


def test_without_a_person_nothing_is_asked(box, monkeypatch):
    script = scripted(monkeypatch)

    assert library.offer(box.cfg, person=False) is None
    assert script.asked == [] and box.calls == []


def test_a_question_nobody_could_answer_is_none_and_fetches_nothing(box, monkeypatch):
    scripted(monkeypatch, UNREACHED)

    assert library.offer(box.cfg, person=True) is None
    assert box.calls == []
