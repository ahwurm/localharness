"""Nothing found: one question over this platform's reference rows, a verified fetch, and a record built from that file alone.
The hub is never reached: `download_file` is a stub that returns a real tiny GGUF, whose size and sha256 sit on the row."""
from __future__ import annotations

import hashlib
import os
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
from localharness.provider.refarch import DGX_SPARK, MAC_MINI, RTX_4060
from tests.unit.interview_support import CANCEL, UNREACHED, scripted
from tests.unit.test_init_core_setup import _Asked, _detect, _exited, _flat, _hermetic, _init, _terminal  # noqa: F401
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
    assert asked.question == f"Download Qwen3.8-9B-Q4_K_M.gguf ({size} of weights, 9.6 GB of free memory)?"
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


# ------------------------------------------------------------------ the default is the box's


def _question(script, at=0):
    return script.asked[at].question


def test_a_linux_nvidia_box_with_vllm_is_offered_the_spark_row_sized_by_the_gpu_less_the_margin(box, monkeypatch):
    box.at(smi="100000, 131072\n", vllm=True, rows=(box.row, DGX_SPARK, MAC_MINI))
    script = scripted(monkeypatch, "not-now")

    library.offer(box.cfg, person=True)

    assert _question(script) == "Download Qwen/Qwen3.6-27B-FP8 (30.9 GB of weights, 94.4 GB of free memory)?"
    assert script.asked[0].options[0].hint == "download Qwen/Qwen3.6-27B-FP8"


def test_a_cpu_only_linux_box_is_never_offered_the_spark_row(box, monkeypatch):
    box.at(vllm=True, rows=(box.row, DGX_SPARK, MAC_MINI))
    script = scripted(monkeypatch, "rows", CANCEL)

    library.offer(box.cfg, person=True)

    assert _question(script).startswith("Download Qwen3.8-9B-Q4_K_M.gguf")
    assert [o.key for o in script.asked[1].options] == [box.row.key]


def test_darwin_arm64_with_llama_server_is_offered_the_gguf_row_and_no_install_line(box, monkeypatch, capsys):
    box.at(platform="darwin", machine="arm64", llama=True, rows=(box.row, DGX_SPARK, MAC_MINI))
    monkeypatch.setattr(os, "sysconf", lambda name: 4096 if name == "SC_PAGE_SIZE" else 4_000_000)
    script = scripted(monkeypatch, "rows", CANCEL)

    library.offer(box.cfg, person=True)

    assert _question(script).startswith("Download Qwen3.8-9B-Q4_K_M.gguf") and "13.9 GB of free memory" in _question(script)
    assert [o.key for o in script.asked[1].options] == [box.row.key, MAC_MINI.key]
    assert library.INSTALL_LINES["darwin"] not in capsys.readouterr().out


def test_memory_that_cannot_be_read_offers_the_smallest_row_and_says_what_it_needs(box, monkeypatch):
    def no_sysconf(_name):
        raise ValueError

    box.at(ram=None, vllm=True, smi=None, rows=(box.row, replace(DGX_SPARK, needs=None)))
    monkeypatch.setattr(os, "sysconf", no_sysconf)
    script = scripted(monkeypatch, "not-now")

    library.offer(box.cfg, person=True)

    assert _question(script) == "Download Qwen/Qwen3.6-27B-FP8 (30.9 GB, needs about 30.9 GB)?"


def test_a_row_the_memory_cannot_hold_is_not_said_to_fit(box, monkeypatch):
    box.at(ram=(8e9, 6e9), rows=(replace(box.row, bytes=9_000_000_000),))
    script = scripted(monkeypatch, "not-now")

    library.offer(box.cfg, person=True)

    assert "needs about" in _question(script) and "free memory" not in _question(script)


def test_win32_has_no_row_and_nothing_is_asked(box, monkeypatch):
    box.at(platform="win32")
    script = scripted(monkeypatch)

    assert library.offer(box.cfg, person=True) is None and script.asked == []


# ------------------------------------------------------------------ the other rows, and the way back


def test_show_the_other_rows_prints_each_with_its_page_and_asks_the_download_question_for_the_one_chosen(box, monkeypatch, capsys):
    box.at(smi="100000, 131072\n", vllm=True, rows=(box.row, DGX_SPARK))
    script = scripted(monkeypatch, "rows", DGX_SPARK.key, "not-now")

    result = library.offer(box.cfg, person=True)

    out = capsys.readouterr().out
    assert f"{box.row.name} — {box.row.repo}/{box.row.file} · " in out and box.row.page in out
    assert f"{DGX_SPARK.name} — {DGX_SPARK.repo} · 30.9 GB · {DGX_SPARK.page}" in out
    which = script.asked[1]
    assert which.question == "Download which?" and [o.key for o in which.options] == [box.row.key, DGX_SPARK.key]
    assert which.options[1].hint == f"{DGX_SPARK.repo} · 30.9 GB · {DGX_SPARK.page}" and which.default == 1
    assert _question(script, 2).startswith("Download Qwen/Qwen3.6-27B-FP8") and isinstance(result, library.NotNow)
    assert DGX_SPARK.page in out


def test_a_cancel_on_the_row_list_is_not_now(box, monkeypatch, capsys):
    scripted(monkeypatch, "rows", CANCEL)

    assert isinstance(library.offer(box.cfg, person=True), library.NotNow)
    assert box.calls == [] and library.RUN_AGAIN_LINE in capsys.readouterr().out


def test_not_now_prints_the_page_the_command_and_the_way_back_and_init_exits_0_without_a_config(box, monkeypatch):
    scripted(monkeypatch, "not-now")
    monkeypatch.setattr(init_cmd, "detect_provider", _not_found())

    result = _init(box.cfg)

    _exited(result, 0)
    flat = _flat(result)
    assert box.row.page in flat
    assert f"localharness model --download {box.row.repo} --file {box.row.file}" in flat
    assert library.RUN_AGAIN_LINE in flat and "No local LLM detected" not in flat
    assert not (box.cfg / "config.yaml").exists() and box.calls == []


def test_a_cancelled_offer_is_not_now(box, monkeypatch):
    scripted(monkeypatch, CANCEL)

    assert isinstance(library.offer(box.cfg, person=True), library.NotNow) and box.calls == []


# ------------------------------------------------------------------ nothing unverified is recorded


def _serve(box, monkeypatch, path):
    monkeypatch.setattr(managed_server, "download_file", lambda *_a, **_k: str(path))
    scripted(monkeypatch, "download")


def _expect_nothing_recorded(box, monkeypatch, capsys, what):
    result = library.offer(box.cfg, person=True)
    out = capsys.readouterr().out
    assert isinstance(result, library.NotNow) and what in out and "nothing recorded" in out
    assert library.RUN_AGAIN_LINE in out and not (box.cfg / "config.yaml").exists()


def test_a_file_one_byte_short_is_named_and_not_recorded(box, monkeypatch, capsys):
    short = box.hf / "short.gguf"
    short.write_bytes((box.hf / "model.gguf").read_bytes()[:-1])
    _serve(box, monkeypatch, short)

    _expect_nothing_recorded(box, monkeypatch, capsys, f"{box.row.bytes - 1} bytes, expected {box.row.bytes}")


def test_a_file_of_the_right_size_with_one_byte_flipped_fails_the_hash(box, monkeypatch, capsys):
    data = bytearray((box.hf / "model.gguf").read_bytes())
    data[-1] ^= 1
    flipped = box.hf / "flipped.gguf"
    flipped.write_bytes(bytes(data))
    _serve(box, monkeypatch, flipped)

    _expect_nothing_recorded(box, monkeypatch, capsys, "sha256")


def test_a_longer_file_is_named_without_hashing_the_rest(box, monkeypatch, capsys):
    longer = box.hf / "longer.gguf"
    longer.write_bytes((box.hf / "model.gguf").read_bytes() + b"x" * 4096)
    _serve(box, monkeypatch, longer)

    _expect_nothing_recorded(box, monkeypatch, capsys, f"more than {box.row.bytes} bytes")


def test_a_fifo_at_the_returned_path_is_refused_without_blocking(box, monkeypatch, capsys):
    fifo = box.hf / "pipe.gguf"
    os.mkfifo(fifo)
    _serve(box, monkeypatch, fifo)

    _expect_nothing_recorded(box, monkeypatch, capsys, "not a regular file")


def test_a_link_to_a_device_is_refused(box, monkeypatch, capsys):
    link = box.hf / "zero.gguf"
    link.symlink_to("/dev/zero")
    _serve(box, monkeypatch, link)

    _expect_nothing_recorded(box, monkeypatch, capsys, "not a regular file")


def test_a_projector_with_the_right_hash_is_not_a_model(box, monkeypatch, capsys):
    proj = box.hf / "mmproj-x.gguf"
    data = proj.read_bytes()
    monkeypatch.setattr(library, "REF_ARCHS", (replace(box.row, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()),))
    _serve(box, monkeypatch, proj)

    _expect_nothing_recorded(box, monkeypatch, capsys, "not a model GGUF")


def test_a_missing_file_is_named(box, monkeypatch, capsys):
    _serve(box, monkeypatch, box.hf / "gone.gguf")

    _expect_nothing_recorded(box, monkeypatch, capsys, "cannot be read")


def test_an_interrupted_download_records_nothing_and_exits_with_the_error_line(box, monkeypatch):
    def interrupted(*_a, **_k):
        raise OSError("connection reset")

    monkeypatch.setattr(managed_server, "download_file", interrupted)
    scripted(monkeypatch, "download")
    monkeypatch.setattr(init_cmd, "detect_provider", _not_found())

    result = _init(box.cfg)

    assert result.exit_code == 1 and "download failed: connection reset" in _flat(result)
    assert not (box.cfg / "config.yaml").exists()


def test_a_swap_after_the_hash_cannot_change_what_the_header_says(box, monkeypatch):
    path = box.hf / "model.gguf"
    _model(box.hf / "other.gguf", embedding=4096)
    real = library.read_header_from

    def swap_then_read(f, want):
        os.replace(box.hf / "other.gguf", path)
        return real(f, want)

    monkeypatch.setattr(library, "read_header_from", swap_then_read)

    checked = library.verify_download(path, size=box.row.bytes, sha256=box.row.sha256)

    assert checked.embedding == 5120


def test_a_file_larger_than_the_row_says_more_than_the_expected_size_and_a_short_one_says_its_size(tmp_path):
    path = tmp_path / "m.gguf"
    path.write_bytes(b"x" * 5000)

    assert library.verify_download(path, size=100, sha256="0" * 64) == "size is more than 100 bytes"
    assert library.verify_download(path, size=6000, sha256="0" * 64) == "size is 5000 bytes, expected 6000"


# ------------------------------------------------------------------ a vLLM row carries its own consent


def test_a_vllm_row_without_a_runtime_carries_the_install_in_its_description_and_fetches_nothing_itself(box, monkeypatch):
    box.at(platform="darwin", machine="arm64", rows=(MAC_MINI,))
    monkeypatch.setattr(os, "sysconf", lambda name: 4096 if name == "SC_PAGE_SIZE" else 4_000_000)
    monkeypatch.setattr(library, "REF_ARCHS", (box.row, MAC_MINI))
    script = scripted(monkeypatch, "rows", MAC_MINI.key, "download")

    result = library.offer(box.cfg, person=True)

    venv = shown_path(managed_server.server_dir(library.global_config_dir(box.cfg)) / "venv")
    assert script.asked[2].options[0].hint == f"download {MAC_MINI.repo} and install vllm-metal into {venv}"
    assert result == library.OfferVllm(MAC_MINI) and box.calls == []


def test_the_docker_row_carries_the_image_pull(box, monkeypatch):
    box.at(smi="100000, 131072\n", rows=(box.row, DGX_SPARK))
    script = scripted(monkeypatch, "rows", DGX_SPARK.key, "download")

    result = library.offer(box.cfg, person=True)

    assert script.asked[2].options[0].hint.endswith(f" and pull {DGX_SPARK.docker_image}")
    assert result == library.OfferVllm(DGX_SPARK)


def test_a_chosen_vllm_row_installs_and_fetches_through_the_reduced_guided_setup(box, monkeypatch):
    box.at(platform="darwin", machine="arm64", rows=(box.row, MAC_MINI))
    monkeypatch.setattr(os, "sysconf", lambda name: 4096 if name == "SC_PAGE_SIZE" else 4_000_000)
    installed: list[str] = []
    fetched: list[str] = []
    monkeypatch.setattr(managed_server, "find_vllm", lambda _d: None)
    monkeypatch.setattr(managed_server, "install_vllm_venv", lambda _d, pkg: installed.append(pkg) or "/v/vllm")
    monkeypatch.setattr(managed_server, "is_model_cached", lambda _r: False)
    monkeypatch.setattr(managed_server, "download_model", lambda repo: fetched.append(repo) or "/cache")
    for name in ("Confirm", "IntPrompt", "Prompt"):
        monkeypatch.setattr(init_cmd, name, _Asked)
    scripted(monkeypatch, "rows", MAC_MINI.key, "download")
    monkeypatch.setattr(init_cmd, "detect_provider", _not_found())

    result = _init(box.cfg)

    _exited(result, 0)
    assert installed == ["vllm-metal"] and fetched == [MAC_MINI.repo]
    written = yaml.safe_load((box.cfg / "config.yaml").read_text(encoding="utf-8"))["server"]
    assert written["refarch"] == "mac-mini" and written["model"] == MAC_MINI.repo and written["port"] == 8081
    assert written["binary"] == "/v/vllm"


def test_win32_with_a_person_reaches_skip_for_now_and_without_one_says_so_and_exits_1(box, monkeypatch):
    box.at(platform="win32")
    scripted(monkeypatch)
    monkeypatch.setattr(init_cmd, "detect_provider", _not_found())
    monkeypatch.setattr(init_cmd, "Prompt", type("P", (), {"ask": staticmethod(lambda *a, **k: "")}))

    result = _init(box.cfg)

    _exited(result, 0)
    assert "Skip for now" in _flat(result) and "No local LLM detected" in _flat(result)
    assert _init(box.cfg, "--no-input").exit_code == 1
