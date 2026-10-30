"""The browser plugin: browser_fetch and the browser_open, click, type and scroll session tools, the Chromium download and four doctor rows.

Off by default. plugins/builtin.py imports this module for every `--help`, `doctor` and `plugins list`,
so it imports only pydantic and the plugin API; playwright, the tool and the runtime are imported
when a session or a check asks for them."""
from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import subprocess
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
_SANDBOX_HINT = ("add an AppArmor profile for the downloaded headless shell (Ubuntu 23.10 and later restrict "
                 "unprivileged user namespaces), or accept the residual with browser.sandbox: false in the "
                 "machine-level config")
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


def setup_help() -> str:
    try:
        cache = _cache_root()
    except RuntimeError:  # no home directory to anchor the default cache
        cache = "Playwright's cache"
    return f"""\
Browser needs Playwright's headless Chromium on this machine, from Playwright's CDN:
  {sys.executable} -m playwright install --only-shell chromium        (about 100 MB, into {cache})
  sudo {sys.executable} -m playwright install-deps chromium    (Linux libraries; apt only — else set browser.executable_path)"""


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


async def _launch_rows(cfg: BrowserConfig) -> list[Check]:
    """The launch and sandbox rows, from one real launch on a runtime of their own, closed again."""
    from localharness.tools.builtin.browser_tool import (
        MISSING_LIBS, NO_PROCESS_LIST, NOT_INSTALLED, REACH_FAILED, SANDBOX_FAILED, SANDBOX_OFF, BrowserRuntime, RenderError)

    runtime, reason = BrowserRuntime(cfg), None
    try:
        await runtime.ensure()
    except RenderError as e:
        reason = e.detail
    finally:
        await runtime.close()
    if reason is None:
        return [Check(name="browser-launch", status="pass", detail="reach check ok"),
                Check(name="browser-sandbox", status="pass") if cfg.sandbox else
                Check(name="browser-sandbox", status="warn",
                      detail="off: browser.sandbox is false in the machine-level config, so Chromium runs unsandboxed")]
    if reason in (SANDBOX_FAILED, SANDBOX_OFF):
        return [Check(name="browser-launch", status="skip", detail="needs the sandbox"),
                Check(name="browser-sandbox", status="fail", detail=reason, hint=_SANDBOX_HINT)]
    detail, hint = {
        MISSING_LIBS: ("Host system is missing dependencies to run browsers",
                       f"sudo {sys.executable} -m playwright install-deps chromium (apt only), "
                       "or set browser.executable_path to a system Chromium"),
        NOT_INSTALLED: (reason, _DOWNLOAD_HINT),
        NO_PROCESS_LIST: (reason, "browser_fetch stays off on a platform where the launch check cannot read the "
                                  "browser's command line; it is checked on Linux and macOS"),
        REACH_FAILED: (reason, "browser_fetch stays off until a launch passes; this Chromium did not go "
                               "through its checking proxy"),
    }.get(reason, (reason, "this is Chromium's own error; set browser.executable_path to another Chromium, "
                           "or run `localharness doctor` again"))
    return [Check(name="browser-launch", status="fail", detail=detail, hint=hint),
            Check(name="browser-sandbox", status="skip", detail="needs a working launch")]


class BrowserPlugin(Plugin):
    """reads JavaScript-rendered pages through a sandboxed headless Chromium"""

    manifest = PluginManifest(
        name="browser", version="0.1.0", kind="tools", enabled_by_default=False, requires_extra="browser",
        setup_action="Download Playwright's headless Chromium now? About 100 MB from Playwright's CDN; "
                     "it runs as a program on this machine (sandboxed by default).",
        setup_help=setup_help(), agent_prompt=BROWSER_AGENT_PROMPT,
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
        """Nothing without Chromium: a tool that cannot work never reaches a prompt or the model's budget."""
        if _binary(ctx.config) is None:
            return []
        from localharness.tools.builtin.browser_session import (
            BrowserClickTool, BrowserOpenTool, BrowserScrollTool, BrowserTypeTool)
        from localharness.tools.builtin.browser_tool import BrowserFetchTool, BrowserRuntime

        self._runtime = BrowserRuntime(ctx.config)
        return [cls(self._runtime) for cls in (
            BrowserFetchTool, BrowserOpenTool, BrowserClickTool, BrowserTypeTool, BrowserScrollTool)]

    async def stop(self, ctx: PluginContext) -> None:
        if self._runtime is not None:
            await self._runtime.close()

    async def doctor(self, ctx: PluginContext) -> list[Check]:
        rows = [_package_row()]
        if (label := _binary(ctx.config)) is not None:
            return [*rows, Check(name="browser-binary", status="pass", detail=label), *await _launch_rows(ctx.config)]
        return [*rows,
                Check(name="browser-binary", status="fail", detail=_missing(ctx.config), hint=_DOWNLOAD_HINT),
                Check(name="browser-launch", status="skip", detail="needs the binary"),
                Check(name="browser-sandbox", status="skip", detail="needs the binary")]

    def setup_action(self, ctx: PluginContext) -> list[Check]:
        """Playwright's own installer, inheriting this terminal so its progress shows."""
        done = subprocess.run([sys.executable, "-m", "playwright", "install", "--only-shell", "chromium"])
        hint = "the download is Playwright's own; check the network, then run `localharness plugins enable browser` again"
        if done.returncode:
            return [Check(name="browser-binary", status="fail",
                          detail=f"the download exited with code {done.returncode}", hint=hint)]
        if (label := _binary(ctx.config)) is None:
            return [Check(name="browser-binary", status="fail", detail=_missing(ctx.config), hint=hint)]
        return [Check(name="browser-binary", status="pass", detail=f"{label} installed")]
