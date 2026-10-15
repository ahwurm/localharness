"""`python -m localharness` is the unit's `-I -m localharness` entry (spec 20 §9.2)."""
from __future__ import annotations

import ast
import runpy
import sys
from pathlib import Path

import pytest

import localharness

MAIN = Path(localharness.__file__).parent / "__main__.py"


def test_the_module_entry_runs_the_cli(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["localharness", "--help"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("localharness", run_name="__main__")
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "model" in out and "start" in out


def test_the_module_entry_imports_only_the_cli_main():
    tree = ast.parse(MAIN.read_text(encoding="utf-8"))
    imported = [(n.module, [a.name for a in n.names]) for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert imported == [("localharness.cli.app", ["main"])]
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Import)]


def test_the_module_entry_runs_main_only_as_the_main_module():
    src = MAIN.read_text(encoding="utf-8")
    assert 'if __name__ == "__main__":' in src
    assert src.count("main()") == 1
