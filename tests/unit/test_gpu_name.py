"""52-03: gpu_name() — this machine's GPU as the machine reports it, for the {machine} sentence of
a coding-agent prompt. nvidia-smi first (the only source that names a GB10), then the PCI vendor
in sysfs, then a Mac's chip; None when nothing answers, so the prompt prints no sentence.

Every source is faked (setup.shutil, setup._lines, setup._SYSFS_DRM, setup.sys): no test runs
nvidia-smi or reads the real /sys. `_lines` itself runs only this Python interpreter."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from localharness.plugins import setup

SMI = "/usr/bin/nvidia-smi"


def _machine(monkeypatch, tmp_path: Path, *, smi=None, vendors=(), platform="linux", sysctl=None):
    """`smi`: nvidia-smi's output lines (None: not installed); `vendors`: one sysfs card per PCI
    vendor id; `sysctl`: the Mac's chip line (None: nothing)."""
    monkeypatch.setattr(setup, "shutil", SimpleNamespace(
        which=lambda exe: SMI if smi is not None and exe == "nvidia-smi" else None))

    def lines(argv):
        if argv[0] == SMI:
            assert argv[1:] == ["--query-gpu=name", "--format=csv,noheader"], argv
            return list(smi)
        if argv[0] == "sysctl":
            assert argv[1:] == ["-n", "machdep.cpu.brand_string"], argv
            return [sysctl] if sysctl else []
        raise AssertionError(f"unexpected command {argv}")

    monkeypatch.setattr(setup, "_lines", lines)
    drm = tmp_path / "drm"
    drm.mkdir()
    for i, vendor in enumerate(vendors):
        (drm / f"card{i}" / "device").mkdir(parents=True)
        (drm / f"card{i}" / "device" / "vendor").write_text(f"{vendor}\n", encoding="utf-8")
    monkeypatch.setattr(setup, "_SYSFS_DRM", drm)
    monkeypatch.setattr(setup, "sys", SimpleNamespace(platform=platform))


def test_nvidia_smi_names_the_gpu(monkeypatch, tmp_path) -> None:
    _machine(monkeypatch, tmp_path, smi=["NVIDIA GB10"], vendors=["0x10de"])
    assert setup.gpu_name() == "NVIDIA GB10"


def test_two_identical_gpus_are_counted(monkeypatch, tmp_path) -> None:
    _machine(monkeypatch, tmp_path, smi=["NVIDIA RTX 4090", "NVIDIA RTX 4090"])
    assert setup.gpu_name() == "2x NVIDIA RTX 4090"


def test_without_nvidia_smi_the_sysfs_vendor_says_nvidia(monkeypatch, tmp_path) -> None:
    _machine(monkeypatch, tmp_path, vendors=["0x10de"])
    assert setup.gpu_name() == "an NVIDIA GPU"


def test_an_amd_card_in_sysfs_says_amd(monkeypatch, tmp_path) -> None:
    _machine(monkeypatch, tmp_path, vendors=["0x1002"])
    assert setup.gpu_name() == "an AMD GPU"


def test_a_mac_reports_its_chip(monkeypatch, tmp_path) -> None:
    _machine(monkeypatch, tmp_path, platform="darwin", sysctl="Apple M4 Pro")
    assert setup.gpu_name() == "Apple M4 Pro"


@pytest.mark.parametrize("platform, vendors", [("linux", []), ("linux", ["0x8086"]), ("win32", ["0x10de"])])
def test_nothing_that_answers_is_none(monkeypatch, tmp_path, platform, vendors) -> None:
    _machine(monkeypatch, tmp_path, platform=platform, vendors=vendors)
    assert setup.gpu_name() is None


def test_lines_returns_the_non_empty_output_lines() -> None:
    assert setup._lines([sys.executable, "-c", "print('a'); print(); print(' b ')"]) == ["a", "b"]


def test_lines_is_empty_for_a_missing_binary() -> None:
    assert setup._lines(["/nonexistent/no-such-binary-52"]) == []


def test_lines_is_empty_for_a_non_zero_exit() -> None:
    assert setup._lines([sys.executable, "-c", "print('x'); raise SystemExit(3)"]) == []


def test_lines_is_empty_after_a_timeout(monkeypatch) -> None:
    monkeypatch.setattr(setup, "_TIMEOUT_S", 0.2)
    assert setup._lines([sys.executable, "-c", "import time; time.sleep(5); print('late')"]) == []
