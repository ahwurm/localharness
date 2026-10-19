"""The browser plugin: browser_fetch, the Chromium download and four doctor rows.

Off by default. plugins/builtin.py imports this module for every `--help`, `doctor` and `plugins list`,
so it imports only pydantic and the plugin API; playwright, the tool and the runtime are imported
when a session or a check asks for them."""
from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from localharness.plugins.api import GLOBAL_ONLY, Availability, Check, Plugin, PluginContext, PluginManifest

if TYPE_CHECKING:
    from localharness.tools.base import ToolProtocol


class BrowserConfig(BaseModel):
    """The `browser:` section. executable_path and sandbox are machine-level only: they choose the program
    the harness starts and whether it is sandboxed, so a project folder must never set them."""

    model_config = ConfigDict(extra="forbid")
    idle_close_s: int = Field(120, ge=1, le=3600,
                              description="Seconds without a call before the browser process is closed.")
    executable_path: str = Field("", json_schema_extra=GLOBAL_ONLY,
                                 description="Your own Chromium; empty uses the one Playwright installed.")
    sandbox: bool = Field(True, json_schema_extra=GLOBAL_ONLY,
                          description="Launch Chromium sandboxed; false accepts an unsandboxed browser.")


BROWSER_SETUP_HELP = f"""\
Browser needs Playwright's headless Chromium on this machine, from Playwright's CDN:
  {sys.executable} -m playwright install --only-shell          (about 100 MB, into ~/.cache/ms-playwright)
  sudo {sys.executable} -m playwright install-deps chromium    (Linux libraries; apt only — else set browser.executable_path)"""

BROWSER_AGENT_PROMPT = (
    "Set up LocalHarness browser_fetch on this machine. Install LocalHarness with its browser extra,\n"
    "keeping the extras I already use (a reinstall keeps only the extras it names). Then run\n"
    "`localharness plugins enable browser` and answer yes to downloading Playwright's headless\n"
    "Chromium (about 100 MB). On Linux, if `localharness doctor` says system libraries are missing,\n"
    "run the deps install it names (apt only; otherwise set browser.executable_path to a system\n"
    "Chromium). If doctor says the sandbox is unavailable, add an AppArmor profile for the downloaded\n"
    "headless shell or, accepting the residual, set browser.sandbox: false in the machine-level\n"
    "config. You are done when `localharness doctor` shows browser-package, browser-binary,\n"
    "browser-launch and browser-sandbox all passing.")

_DOWNLOAD_HINT = "run `localharness plugins enable browser` to download it"
_INSTALL_HINT = ("uv tool install 'localharness[browser]', naming the other extras you use too "
                 "(a reinstall keeps only the extras it names)")


def _cache_root() -> Path:
    path = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if path == "0":
        spec = importlib.util.find_spec("playwright")
        if spec and spec.origin:
            return Path(spec.origin).parent / "driver" / "package" / ".local-browsers"
    elif path:
        return Path(path)
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    elif sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "ms-playwright"


def _revision(folder: Path) -> int:
    n = folder.name.rpartition("-")[2]
    return int(n) if n.isdigit() else 0


def _binary(cfg: BrowserConfig) -> str | None:
    """The installed Chromium's label, found offline without starting anything; None when there is none."""
    if cfg.executable_path:
        return cfg.executable_path if Path(cfg.executable_path).is_file() else None
    shells = [d for d in _cache_root().glob("chromium_headless_shell-*") if d.is_dir()]
    return max(shells, key=_revision).name if shells else None


def _missing(cfg: BrowserConfig) -> str:
    return (f"browser.executable_path {cfg.executable_path} is not a file" if cfg.executable_path
            else f"no chromium_headless_shell-* folder in {_cache_root()}")


def _package_row() -> Check:
    try:
        return Check(name="browser-package", status="pass",
                     detail=f"playwright {importlib.metadata.version('playwright')}")
    except importlib.metadata.PackageNotFoundError:
        return Check(name="browser-package", status="fail", detail="the playwright package is not installed",
                     hint=_INSTALL_HINT)


class BrowserPlugin(Plugin):
    """reads JavaScript-rendered pages through a sandboxed headless Chromium"""

    manifest = PluginManifest(
        name="browser", version="0.1.0", kind="tools", enabled_by_default=False, requires_extra="browser",
        setup_action="Download Playwright's headless Chromium now? About 100 MB from Playwright's CDN; "
                     "it runs as a program on this machine (sandboxed by default).",
        setup_help=BROWSER_SETUP_HELP, agent_prompt=BROWSER_AGENT_PROMPT,
        next_steps="browser_fetch (pages that need JavaScript) works once Chromium is downloaded and "
                   "`localharness doctor` shows every browser row passing; `localharness plugins enable "
                   "browser` does the download.",
    )
    ConfigModel = BrowserConfig
    _runtime = None

    async def configure(self, ctx: PluginContext) -> Availability:
        """Finds the package and nothing else; the binary is a doctor row, so no start ever warns about it."""
        if importlib.util.find_spec("playwright") is None:
            raise ImportError("the playwright package is not installed — install localharness[browser]")
        return "ready"

    async def tools(self, ctx: PluginContext) -> list[ToolProtocol]:
        return []

    async def doctor(self, ctx: PluginContext) -> list[Check]:
        rows = [_package_row()]
        if (label := _binary(ctx.config)) is not None:
            return [*rows, Check(name="browser-binary", status="pass", detail=label)]
        return [*rows,
                Check(name="browser-binary", status="fail", detail=_missing(ctx.config), hint=_DOWNLOAD_HINT),
                Check(name="browser-launch", status="skip", detail="needs the binary"),
                Check(name="browser-sandbox", status="skip", detail="needs the binary")]
