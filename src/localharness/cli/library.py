"""The model library's side that talks and writes: the shown lines, the questions and the one peer writer."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import sys
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import typer
from pydantic import SecretStr
from rich.console import Console

from localharness.provider.gguf_header import (
    DISCOVERY_KEYS,
    DiscoveredModel,
    FileKind,
    GgufHeaderError,
    model_of,
    preset_name,
    rank_models,
    read_header,
    read_header_from,
    unique_preset_name,
)
from localharness.cli import interview, workspace
from localharness.cli.interview import Option
from localharness.cli.workspace import shown_path, visible_text
from localharness.config import trust
from localharness.config.loader import ConfigLoader
from localharness.config.models import AutosenseConfig, EndpointRef, ManagedServerConfig
from localharness.config.overlay import load_overlay
from localharness.config.paths import global_config_dir
from localharness.config.redact import reveal
from localharness.core.private_files import NotRegularFile, open_regular
from localharness.core.which import which_outside_cwd
from localharness.provider import autosense, model_library, speed_stats
from localharness.provider import server as managed_server
from localharness.provider.detector import DetectorResult, detect_provider
from localharness.provider.refarch import REF_ARCHS, RefArch, default_row, needs_met
from localharness.provider.runtimes import RUNTIMES, row_of
from localharness.provider.server import RUNTIME_LABELS
from localharness.provider.tuning import mtp_layers

console = Console()
err_console = Console(stderr=True)

OVERLAY_BUSY_LINE = "overrides.yaml is busy — nothing written"
PEERS_COPIED_COUNTS = " — {copied} copied, {kept} still confirmed"
PEERS_DETAIL = "localharness: peers added to overrides.yaml"
PRIMARY_PORT = 8081
FIRST_RUN_QUESTION = "Set up these {n} models?"
ALL_LABEL = "All, serving {best} first"
ALL_HINT = "every file above becomes a preset; {best} serves on this start"
ALL_HINT_EXCEPT_DECLINED = "every file above not declined earlier becomes a preset; {best} serves on this start"
CHOOSE_LABEL = "Choose which, and which first"
CHOOSE_HINT = "pick the files, then the one to serve"
SKIP_LABEL = "Skip for now"
SKIP_HINT = "nothing is set up; run localharness start again later"
ONE_QUESTION = "Set up {name}?"
SET_IT_UP = "Set it up"
SET_IT_UP_HINT = "{name} becomes a preset"
NOT_NOW = "Not now"
NOT_NOW_HINT = "nothing is set up; localharness model --setup lists it later"
SET_UP_WHICH = "Set up which?"
SERVE_WHICH_FIRST = "Serve which first?"
SKIP_LINE = "Nothing set up; run localharness start again later."
DECLINED_ALL_LINE = "Nothing set up — {n} declined, remembered in {library}; run localharness start again to choose"
NEXT_START_LINE = "Written unconfirmed — the next localharness start on a terminal asks about them"
PEERS_NOT_WRITTEN_LINE = "The other {n} models were not written to overrides.yaml — localharness model --setup lists them"
LATER_LINE = "Later: just run localharness start — it brings back your last choice."
LATER_MODEL_CLAUSE = " /model {name} switches in a session."
BULLET = "• {name} ({clause}) ← {path} · {size} · first seen {date}"
DECLINED_TAG = " (declined earlier)"
NEW_ONE_QUESTION = "New model {name} — set it up?"
NEW_MANY_QUESTION = "{n} new models — set them up?"
SET_THEM_UP = "Set them up"
SET_THEM_UP_HINT = "every new model becomes a preset"
CHOOSE_WHICH = "Choose which"
CHOOSE_WHICH_HINT = "pick the files to set up"
SET_UP_LINE = "/model {name} switches in a session; localharness start --preset {name} starts on it"
NOT_SET_UP_LINE = "Not set up: {names} — localharness model --setup lists them"
ADDED_KEPT_LINE = "Added {n}; kept config.yaml."
NOTHING_TO_SET_UP_LINE = "Nothing to set up — every model found has a preset"
LIBRARY_FAILED_LINE = "model library: {why} — start goes on"
LIBRARY_NOT_SAVED_LINE = "model library: not saved, it is busy or unreadable — the same question comes back next start"
NOT_WRITTEN_LINE = "Nothing was written to overrides.yaml — localharness model --setup lists the models"
OFFER_QUESTION = "Download {what} ({size} of weights, {free} of free memory)?"
OFFER_QUESTION_UNKNOWN = "Download {what} ({size}, needs about {size})?"
DOWNLOAD_IT = "Download it"
SMALL_ROW_HINT = "a small model that runs anywhere; larger ones under Show the other rows"
SHOW_ROWS = "Show the other rows"
SHOW_ROWS_HINT = "each row with its page, then pick one"
DOWNLOAD_WHICH = "Download which?"
NOT_NOW_OFFER_HINT = "nothing is fetched; the page and the command are printed"
RUN_AGAIN_LINE = "Nothing set up; run localharness start again later."
THEN_START = ", then run localharness start"
ROW_LINE = "{name} — {where} · {size} · {page}"
NOT_NOW_PAGE_LINE = "Reference: {page}"
NOT_NOW_COMMAND_LINE = "Fetch it yourself: {command}"
DOWNLOAD_FAILED_LINE = "download failed: {why}"
RECORD_LINE = "{name} ({runtime}) ← {path} · port {port} · binary: {binary}"
NO_BINARY_TEXT = "none — set server.binary"
VERIFY_MISMATCH_LINE = "{path}: {what} — nothing recorded"
VERIFIED_LINE = "Verified {file} at {commit} (sha256 {sha256})"
START_FINDS_IT_LINE = "Not recorded ({why}) — the next localharness start finds it"
INSTALL_LINES = {
    "darwin": "brew install llama.cpp",
    "linux": "a llama.cpp release build for your GPU from github.com/ggml-org/llama.cpp/releases, "
             "llama-server on PATH — docs/runtimes/llamacpp.md",
}


class _NothingToAppend(ValueError):
    """Raised inside the write so a call that would change nothing writes nothing."""


def _say(line: str) -> None:
    console.print(line, markup=False, soft_wrap=True)


def audit_entries(prefix: str, mapping: dict) -> list[tuple[str, object, object]]:
    """The audit triples of one record: its launch entries and posture keys, named, never the values of a key."""
    found = [(e["name"], None, e["shown"]) for e in trust._launch_entries("overrides.yaml", prefix, mapping)]
    return found + [(prefix + key, None, json.dumps(mapping[key])) for key in trust._POSTURE_KEYS if key in mapping]


def _port_of(url: object) -> int | None:
    try:
        return urlparse(str(url)).port
    except ValueError:
        return None


def _recorded(server: dict, peers: list) -> tuple[set[str], set[str], set[int]]:
    """The model realpaths, names and ports the merged view already uses."""
    lives = [p.get("lifecycle") for p in peers if isinstance(p, dict)]
    records = [r for r in (server, *lives) if isinstance(r, dict)]
    models = {os.path.realpath(os.path.expanduser(str(r["model"]))) for r in records if r.get("model")}
    names = {str(p["name"]) for p in peers if isinstance(p, dict) and "name" in p}
    names |= {str(server["served_name"])} if server.get("served_name") else set()
    ports = {int(r["port"]) for r in records if isinstance(r.get("port"), int)}
    ports |= {n for n in (_port_of(p.get("base_url")) for p in peers if isinstance(p, dict)) if n}
    return models, names, ports


class PeerFragment:
    """The peers an append adds, as a `mutate` over the overlay: what it wrote and the audit triples of that."""

    def __init__(self, global_dir: Path, peers: Sequence[EndpointRef]) -> None:
        self.global_dir = Path(global_dir)
        self.peers = list(peers)
        self.written: list[dict] = []
        self.audit: list[tuple[str, object, object]] = []
        self.copied = 0

    def mutate(self, overlay: dict) -> None:
        self.written.clear()
        self.audit.clear()
        config_raw = workspace.raw_settings(self.global_dir)[0]
        server = {**(config_raw.get("server") or {}), **(overlay.get("server") or {})}
        self.copied = workspace.copy_peer_list(overlay, config_raw)
        listed = overlay["extra_endpoints"]
        models, names, ports = _recorded(server, listed)
        for peer in self.peers:
            life = peer.lifecycle
            real = os.path.realpath(os.path.expanduser(life.model)) if life is not None and life.model else None
            if real in models:
                continue
            name = unique_preset_name(peer.name, names)
            port = _port_of(peer.base_url) if life is None else life.port
            if port is not None:
                free = model_library.free_ports(port, 1, taken=ports)
                if not free:
                    continue
                port = free[0]
            update: dict = {"name": name}
            if port is not None:
                update["base_url"] = peer.base_url.replace(f":{_port_of(peer.base_url)}", f":{port}", 1)
            if life is not None:
                update["lifecycle"] = life.model_copy(update={
                    "port": port, "served_name": name if RUNTIMES[life.runtime].gguf and life.served_name in (None, peer.name)
                    else life.served_name})
            record = _record_dicts_of(None, [peer.model_copy(update=update)])[1][0]
            listed.append(record)
            self.written.append(record)
            models |= {real} if real else set()
            names.add(name)
            ports |= {port} if port is not None else set()
            self.audit.extend(audit_entries(f"extra_endpoints.{name}.lifecycle.", record.get("lifecycle") or {}))


def append_peers(config_path: Path, peers: Sequence[EndpointRef], *, adopt: bool,
                 say: Callable[[str], None] | None = None) -> list[dict] | None:
    """Append `peers` to overrides.yaml's `extra_endpoints` under the overlay lock (adopted when `adopt`); the dicts written, None when none."""
    say = say or _say
    global_dir = global_config_dir(config_path)
    fragment = PeerFragment(global_dir, peers)

    def mutate(overlay: dict) -> None:
        fragment.mutate(overlay)
        if not fragment.written:
            raise _NothingToAppend

    if not workspace.write_overlay_validated(global_dir, mutate, fragment.audit, actor_detail=PEERS_DETAIL,
                                             on_busy=lambda: say(OVERLAY_BUSY_LINE)):
        return None
    finish_peers(global_dir, None, fragment, adopt=adopt, say=say)
    return fragment.written


def finish_peers(global_dir: Path, server: dict | None, fragment: PeerFragment, *, adopt: bool,
                 say: Callable[[str], None] | None = None) -> None:
    """After a write: adopt what it wrote and the copied peers whose config.yaml twin was covered (when `adopt`), and say the copied line once."""
    counts = ""
    if adopt:
        recorded = trust.recorded_machine(global_dir)
        covered = {trust._canonical(e) for e in recorded["entries"]} if recorded else set()
        _adopt_written(global_dir, "overrides.yaml", server, fragment.written)
        counts = _readopt_copied(global_dir, fragment.copied, covered) if fragment.copied else ""
    if fragment.copied:
        (say or _say)(workspace.PEER_LIST_COPIED + counts)


def _readopt_copied(global_dir: Path, copied: int, covered: set[str]) -> str:
    """Adopt the copied peers' entries whose config.yaml twin was covered; the counts to say."""
    raws = dict(zip(("config.yaml", "overrides.yaml"), workspace.raw_settings(global_dir)))
    twins = workspace._copied_peer_entries(raws, covered, skip="")
    if twins:
        trust.adopt_entries(global_dir, twins)
    adopted = {e["name"] for e in twins}
    kept = sum({e["name"] for e in trust._launch_entries(
        "overrides.yaml", f"extra_endpoints.{p['name']}.lifecycle.", p.get("lifecycle"))} <= adopted
        for p in raws["overrides.yaml"]["extra_endpoints"][:copied] if isinstance(p, dict) and "name" in p)
    return PEERS_COPIED_COUNTS.format(copied=copied, kept=kept)


@dataclass
class InitRecords:
    """A model server init writes as a record: the primary launch record, its peers and the block shown first."""
    server: ManagedServerConfig
    served: str
    peers: list[EndpointRef] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)
    api_key: SecretStr | None = field(default=None, repr=False)
    deferred: list[Callable[[], None]] = field(default_factory=list)
    base_url: str | None = None
    serving: bool = False
    label: str = ""
    block_in_receipt: bool = False
    confirmed: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.api_key, str):
            self.api_key = SecretStr(self.api_key)

    @property
    def launchable(self) -> bool:
        return bool(self.server.binary or self.server.docker_image)


def _machine_autosense(config_path: Path) -> AutosenseConfig:
    """The `autosense:` section the machine's files set, else the defaults; an unreadable config never stops discovery."""
    try:
        return ConfigLoader(config_dir=config_path).load_harness().autosense
    except Exception:  # noqa: BLE001 — no config yet, or an unreadable one: never a traceback
        pass
    try:  # a fresh machine has no config.yaml, but may already have an overrides.yaml saying where
        raw = load_overlay(config_path / "overrides.yaml").get("autosense")
        return AutosenseConfig.model_validate(raw) if isinstance(raw, dict) else AutosenseConfig()
    except Exception:  # noqa: BLE001 — whatever stops that read leaves the defaults
        return AutosenseConfig()


def model_mtp(path: str | os.PathLike[str]) -> bool:
    """Does the model's header declare valid MTP layers? An unreadable file says no."""
    try:
        return mtp_layers(read_header(path, {"block_count", "nextn_predict_layers"})) is not None
    except Exception:  # noqa: BLE001 — no header, no clause
        return False


def _clause(runtime: str, *, mtp: bool, mmproj: str | os.PathLike[str] | None) -> str:
    clause = RUNTIME_LABELS.get(runtime, runtime)
    if mtp:
        clause += ", mtp"
    if mmproj:
        clause += ", +" + visible_text(os.path.basename(mmproj))
    return clause


def preset_clause(server: ManagedServerConfig) -> str:
    """`(llama.cpp, mtp, +<projector basename>)`: the runtime, `mtp` when the model has it, the projector."""
    mtp = bool(RUNTIMES[server.runtime].gguf and server.model and model_mtp(server.model))
    return f"({_clause(server.runtime, mtp=mtp, mmproj=server.mmproj)})"


def _model_text(server: ManagedServerConfig) -> str:
    return visible_text(server.model) if RUNTIMES[server.runtime].checkpoint else shown_path(server.model)


def vllm_peer(checkpoint: model_library.Checkpoint, *, name: str, port: int, binary: str) -> EndpointRef:
    """A launchable vLLM peer for a cached checkpoint: the repo id, served by the shown binary."""
    return EndpointRef(
        name=name, base_url=f"http://127.0.0.1:{port}/v1", provider_type="vllm", gpu=True,
        lifecycle=ManagedServerConfig(runtime="vllm", launch="binary", binary=binary, model=checkpoint.repo_id,
                                      port=port, gpu=True))


def preset_lines(server: ManagedServerConfig, served: str, peers: Sequence[EndpointRef], *,
                 binary_from: str, extras: Sequence[str] = (), vllm_from: str = "on PATH") -> list[str]:
    """The primary's line with its binary and where it came from, the escaped ``extras``, then one line per peer."""
    primary = f"serving {visible_text(served)} {preset_clause(server)} ← {_model_text(server)}"
    if server.binary:
        primary += f" · binary: {shown_path(server.binary)} ({binary_from})"
    lines = [primary, *extras]
    for peer in peers:
        life = peer.lifecycle
        if life is not None:
            line = f"/model switches to {visible_text(peer.name)} {preset_clause(life)} ← {_model_text(life)}"
            if RUNTIMES[life.runtime].checkpoint and life.binary:
                line += f" · binary: {shown_path(life.binary)} ({vllm_from})"
            lines.append(line)
    return lines


def _daemon_lines() -> list[str]:
    """An Ollama or LM Studio that answers holds models of its own: how many, the first three, and that no preset was written."""
    lines: list[str] = []
    for ports in ([11434], [1234]):
        try:
            found = asyncio.run(detect_provider(timeout_seconds=1.0, ports=ports))
        except Exception:  # noqa: BLE001, S112 — a probe that fails says nothing
            continue
        if not found.found or found.provider_type not in ("ollama", "lmstudio") or not found.models:
            continue
        label = RUNTIME_LABELS[found.provider_type]
        names = ", ".join(visible_text(m) for m in found.models[:3])
        if len(found.models) > 3:
            names += ", …"
        count = f"{len(found.models)} model{'s' if len(found.models) != 1 else ''}"
        lines.append(f"{label} holds {count} ({names}) — left in {label}, no preset written")
    return lines


def record_block(recs: InitRecords) -> list[str]:
    """The lines that show a record: its own, then what a daemon on this machine holds."""
    return [*recs.lines, *(_daemon_lines() if RUNTIMES[recs.server.runtime].gguf else [])]


def show_records(recs: InitRecords) -> None:
    """Print the block a record path shows before it writes; a name from a file is text, never markup."""
    for line in record_block(recs):
        console.print(line, markup=False, soft_wrap=True)


def _record_dicts_of(server: ManagedServerConfig | None,
                     peers: Sequence[EndpointRef]) -> tuple[dict | None, list[dict]]:
    """The server record and every peer as written to a file: no `null` preset field, no invented `effort`."""
    return (reveal(server.model_dump(mode="python", exclude_none=True)) if server is not None else None,
            [reveal(peer.model_dump(mode="python", exclude_none=True)) for peer in peers])


def _adopt_written(global_dir: Path, file: str, server: dict | None, peers: Sequence[dict]) -> None:
    """Record the launch entries just written and shown as approved; with no machine record the first is made, of the launch kinds only."""
    written: dict[str, dict] = {}  # entry-name prefix -> the dict written under it
    if server is not None:
        written["server."] = server
    for peer in peers:
        if isinstance(peer.get("lifecycle"), dict):
            written[f"extra_endpoints.{peer['name']}.lifecycle."] = peer["lifecycle"]
    entries = [e for prefix, mapping in written.items()
               for e in trust._launch_entries(file, prefix, mapping)]
    posture = {prefix + key: json.dumps(mapping.get(key))
               for prefix, mapping in written.items() for key in trust._POSTURE_KEYS}
    # `shown` is `<runtime>:<true|false>:<keyed|open>`: the middle part is the value in effect.
    entries += [e for e in trust.machine_snapshot(global_dir)
                if e["kind"] == trust.LAUNCH_KIND and e["name"] in posture
                and str(e.get("shown", "")).split(":")[1:2] == [posture[e["name"]]]]
    if trust.recorded_machine(global_dir) is not None:
        trust.adopt_entries(global_dir, entries)
    else:
        trust.record_machine(global_dir, sorted(entries, key=trust._entry_key),
                             kinds=frozenset({trust.LAUNCH_KIND, trust.LAUNCH_MODEL_KIND}))


def _result_from_record(recs: InitRecords) -> DetectorResult:
    """The detection-shaped answer for a record: the address it will serve on and the name it will serve under."""
    return DetectorResult(
        found=True,
        provider_type=recs.server.runtime,
        base_url=recs.base_url or f"http://127.0.0.1:{recs.server.port}/v1",
        models=[recs.served],
        suggested_model=recs.served,
        probe_duration_ms=0.0,
    )


def recorded_paths(global_dir: Path) -> frozenset[str]:
    """The realpaths of every model and projector the machine's two files name in a record, and the raw model of each vLLM one."""
    records: list = []
    for raw in workspace.raw_settings(global_dir):
        peers = raw.get("extra_endpoints")
        records += [raw.get("server"), *(p.get("lifecycle") for p in peers if isinstance(p, dict))] \
            if isinstance(peers, list) else [raw.get("server")]
    live = [r for r in records if isinstance(r, dict)]
    return frozenset({os.path.realpath(os.path.expanduser(str(r[key]))) for r in live
                      for key in ("model", "mmproj") if r.get(key)}
                     | {str(r["model"]) for r in live if getattr(row_of(r.get("runtime", "vllm")), "checkpoint", False) and r.get("model")})


def prune_recorded(config_path: Path) -> bool:
    """Drop from an existing library file every entry a record now names; True when it was rewritten."""
    global_dir = global_config_dir(config_path)
    path = model_library.library_path(global_dir)
    lib = model_library.Library.load(path)
    if lib.missing or lib.corrupt:
        return False
    kept = lib.unrecorded(recorded_paths(global_dir))
    return len(kept.entries) != len(lib.entries) and kept.save(path, merge=True)


def _merged(global_dir: Path) -> tuple[dict, list]:
    """The merged view's `server:` record and peer list: the overlay's keys win, its list replaces."""
    config_raw, overlay = workspace.raw_settings(global_dir)
    peers = overlay["extra_endpoints"] if isinstance(overlay.get("extra_endpoints"), list) \
        else config_raw.get("extra_endpoints")
    return {**(config_raw.get("server") or {}), **(overlay.get("server") or {})}, peers if isinstance(peers, list) else []


def recorded_names_ports(global_dir: Path) -> tuple[frozenset[str], frozenset[int]]:
    """The preset names and ports the merged config already uses."""
    _, names, ports = _recorded(*_merged(global_dir))
    return frozenset(names), frozenset(ports)


@dataclass(frozen=True)
class Counts:
    set_up: int
    found: int
    declined: int


def counts(config_path: Path) -> Counts:
    """What the box holds, from the merged config and the library file alone: no walk, no header read."""
    global_dir = global_config_dir(config_path)
    recorded = recorded_paths(global_dir)
    lib = model_library.Library.load(model_library.library_path(global_dir)).unrecorded(recorded)
    vllm = managed_server.vllm_binary(global_dir) is not None
    here = [e for key, e in lib.entries.items()
            if e.verdict is FileKind.MODEL and os.path.lexists(key) and (vllm or e.repo is None)]
    server, peers = _merged(global_dir)
    lives = [p.get("lifecycle") for p in peers if isinstance(p, dict)]
    set_up = sum(isinstance(r, dict) and bool(r.get("model")) for r in (server, *lives))
    return Counts(set_up, sum(not e.declined for e in here), sum(e.declined for e in here))


def format_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def no_binary_line(models: Sequence[DiscoveredModel], roots: Sequence[str]) -> str:
    """The models were found and there is no llama-server to launch them with: the roots that hold one."""
    holding: list[str] = []
    for root in roots:
        real = os.path.realpath(os.path.expanduser(root))
        if any(m.real == real or m.real.startswith(real.rstrip(os.sep) + os.sep) for m in models):
            holding.append(shown_path(os.path.expanduser(root)))
    n = len(models)
    return (f"found {n} GGUF{'s' if n != 1 else ''} under {', '.join(holding)}; "
            "no llama-server — set server.binary")


def _ranker(config_path: Path, auto: AutosenseConfig, unit_model: str | None = None):
    """Best first: free memory is probed once, on the first call, and only a call with models reaches it."""
    probed: list = []

    def rank(models: Sequence[DiscoveredModel]) -> list[DiscoveredModel]:
        if not probed:
            probed.append(autosense.free_memory(auto))
        fits: dict[str, int | None] = {}
        for model in models:
            try:
                spec = ManagedServerConfig(runtime="llamacpp", model=str(model.path),
                                           mmproj=str(model.mmproj) if model.mmproj else None)
                fits[str(model.path)] = autosense.compute_fit(spec, auto, memory=probed[0]).fits
            except Exception:  # noqa: BLE001 — one model that cannot be sized is ranked without a fit
                fits[str(model.path)] = None
        ledger = speed_stats.model_paths(speed_stats.default_speed_stats_path(config_path))
        return rank_models(models, unit_model=unit_model, ledger_paths=ledger, fits=fits)

    return rank


def _save_library(found: model_library.Found, path: Path, *, declined: Collection[str] = (),
                  dropped: Collection[str] = ()) -> None:
    """Save the scan's library with these marks, merged over the file as it is now; say one line when it was not."""
    if not found.library.marked(declined=declined, dropped=dropped).save(path, merge=True):
        _say(LIBRARY_NOT_SAVED_LINE)


def baseline(config_path: Path, *, person: bool) -> None:
    """A first run that recorded no model: every file under the roots is seen, so a later run asks only about a new one."""
    if not person:
        return
    global_dir = global_config_dir(config_path)
    path = model_library.library_path(global_dir)
    found = model_library.scan(_machine_autosense(config_path).model_roots, recorded_paths(global_dir), path,
                               scope="new", rank=list, checkpoints=_vllm_world(global_dir)[0])
    if found.changed:
        _save_library(found, path)


def import_peer_models(config_path: Path, *, exclude: str, unit_model: str | None) -> list[DiscoveredModel]:
    """The models an import offers as peers, best first: new or seen, never declined, none already recorded."""
    global_dir = global_config_dir(config_path)
    auto = _machine_autosense(config_path)
    found = model_library.scan(auto.model_roots, recorded_paths(global_dir) | {exclude},
                               model_library.library_path(global_dir), scope="all", rank=list, first_run=True)
    models = [s.model for s in (*found.new, *found.seen) if s.model is not None]
    return _ranker(config_path, auto, unit_model)(models) if models else []


@dataclass(frozen=True)
class Sync:
    records: InitRecords | None = None
    lines: tuple[str, ...] = ()
    appended: int = 0
    declined: tuple[str, ...] = ()
    shown: int = 0


@dataclass(frozen=True)
class _Choice:
    """What the person chose: the files in order (the first serves), the ones declined, or nothing at all."""
    order: list[model_library.Shown]
    declined: list[model_library.Shown]
    skipped: bool = False


def _vllm_world(global_dir: Path) -> tuple[tuple[model_library.Checkpoint, ...], tuple[str, str] | None]:
    """The vLLM binary and its source, and the cached checkpoints it could serve: none without a binary."""
    vllm = managed_server.vllm_binary(global_dir)
    return (model_library.checkpoints() if vllm else ()), vllm


def _listable(shown: Sequence[model_library.Shown]) -> list[model_library.Shown]:
    return [s for s in shown if s.model is not None or s.checkpoint is not None]


def _label(s: model_library.Shown) -> str:
    return s.model.name if s.model else preset_name(s.checkpoint.repo_id.rsplit("/", 1)[-1])


def _origin(s: model_library.Shown) -> str:
    return visible_text(s.checkpoint.repo_id) if s.checkpoint else shown_path(s.path)


def _ranked(config_path: Path, auto: AutosenseConfig, shown: Sequence[model_library.Shown]) -> list[model_library.Shown]:
    """The GGUFs best first, then the checkpoints in repo-id order."""
    ggufs = [s for s in shown if s.model]
    by_path = {str(s.model.path): s for s in ggufs}
    ranked = [by_path[str(m.path)] for m in _ranker(config_path, auto)([s.model for s in ggufs])] if ggufs else []
    return ranked + sorted((s for s in shown if s.checkpoint), key=lambda s: s.checkpoint.repo_id)


def _bullets(listed: Sequence[model_library.Shown], names: dict[str, str],
             vllm: tuple[str, str] | None = None) -> None:
    for s in listed:
        if s.checkpoint:
            line = BULLET.format(name=visible_text(names[s.real]), clause=_clause("vllm", mtp=False, mmproj=None),
                                 path=_origin(s), size=format_size(s.size), date=s.first_seen)
            line += f" · binary: {shown_path(vllm[0])} ({vllm[1]})"
        else:
            clause = _clause("llamacpp", mtp=bool(s.model.mtp), mmproj=s.model.mmproj)
            line = BULLET.format(name=visible_text(names[s.real]), clause=clause, path=shown_path(s.path),
                                 size=format_size(s.size), date=s.first_seen)
        console.print(line + (DECLINED_TAG if s.declined else ""), markup=False, soft_wrap=True)


def _peers(chosen: Sequence[model_library.Shown], names: dict[str, str], *, start: int, taken_names: Collection[str],
           taken_ports: Collection[int], vllm: tuple[str, str] | None) -> list[EndpointRef]:
    """One launchable peer per chosen file: the GGUFs first, then the checkpoints on the next free ports."""
    peers = model_library.peers_for([replace(s.model, name=names[s.real]) for s in chosen if s.model],
                                    start=start, taken_names=taken_names, taken_ports=taken_ports)
    ports = {*taken_ports, *(p.lifecycle.port for p in peers)}
    used = {*taken_names, *(p.name for p in peers)}
    for s in chosen:
        if s.checkpoint is None or vllm is None or not (free := model_library.free_ports(start, 1, taken=ports)):
            continue
        name = unique_preset_name(names[s.real], used)
        peers.append(vllm_peer(s.checkpoint, name=name, port=free[0], binary=vllm[0]))
        ports.add(free[0])
        used.add(name)
    return peers


def _first_of(files: Sequence[model_library.Shown], named: model_library.Shown | None) -> list[model_library.Shown]:
    return [named, *(s for s in files if s is not named)] if named in files else list(files)


def _interview(listed: Sequence[model_library.Shown], names: dict[str, str],
               named: model_library.Shown | None, *, person: bool) -> _Choice | None:
    """The first-run question and what each answer sets up; None when nobody could be asked."""
    if not person:
        return None
    skip = _Choice([], [], skipped=True)
    eligible = [s for s in listed if not s.declined]
    if len(listed) == 1:
        [only] = listed
        name = visible_text(names[only.real])
        answer = interview.ask(ONE_QUESTION.format(name=name), [
            Option("set-up", SET_IT_UP, SET_IT_UP_HINT.format(name=name)),
            Option("not-now", NOT_NOW, NOT_NOW_HINT)], default=1 if only.declined else 0, cancel="not-now")
        if not answer.reached:
            return None
        return _Choice([only], []) if answer.keys == ("set-up",) and not answer.cancelled else skip
    options = [Option("choose", CHOOSE_LABEL, CHOOSE_HINT), Option("skip", SKIP_LABEL, SKIP_HINT)]
    if eligible:
        best = visible_text(names[_first_of(eligible, named)[0].real])
        hint = ALL_HINT_EXCEPT_DECLINED if len(eligible) < len(listed) else ALL_HINT
        options.insert(0, Option("all", ALL_LABEL.format(best=best), hint.format(best=best)))
    answer = interview.ask(FIRST_RUN_QUESTION.format(n=len(listed)), options, default=0, cancel="skip")
    if not answer.reached:
        return None
    if answer.cancelled or answer.keys == ("skip",):
        return skip
    if answer.keys == ("all",):
        return _Choice(_first_of(eligible, named), [])
    picked = interview.ask(SET_UP_WHICH, [Option(s.real, visible_text(names[s.real]), _origin(s))
                                          for s in listed],
                           default={s.real for s in eligible}, cancel="cancel", multi=True)
    if not picked.reached:
        return None
    if picked.cancelled:
        return skip
    chosen = [s for s in listed if s.real in picked.keys]
    declined = [s for s in listed if s.real not in picked.keys]
    if len(chosen) > 1 and named not in chosen:
        first = interview.ask(SERVE_WHICH_FIRST, [Option(s.real, visible_text(names[s.real]), _origin(s))
                                                  for s in chosen], default=0, cancel="cancel")
        if not first.reached:
            return None
        if first.cancelled:
            return skip
        chosen = _first_of(chosen, next(s for s in chosen if s.real == first.keys[0]))
    return _Choice(_first_of(chosen, named), declined)


def _named(listed: Sequence[model_library.Shown], names: dict[str, str],
           preset: str | None) -> model_library.Shown | None:
    """The listed file whose proposed preset name is `preset`: exactly, else ignoring case."""
    if preset is None:
        return None
    return (next((s for s in listed if names[s.real] == preset), None)
            or next((s for s in listed if names[s.real].casefold() == preset.casefold()), None))


def _records(config_path: Path, order: Sequence[model_library.Shown], names: dict[str, str], *,
             confirmed: bool, vllm: tuple[str, str] | None = None) -> InitRecords:
    """The first file serves on `PRIMARY_PORT`, the rest are launchable peers above it."""
    first = order[0]
    if first.checkpoint:
        served = first.checkpoint.repo_id
        server = ManagedServerConfig(runtime="vllm", launch="binary", binary=vllm[0], model=served,
                                     port=PRIMARY_PORT, gpu=True)
    else:
        served = names[first.real]
        server = ManagedServerConfig(runtime="llamacpp", model=str(first.model.path),
                                     mmproj=str(first.model.mmproj) if first.model.mmproj else None,
                                     served_name=served, port=PRIMARY_PORT, binary=which_outside_cwd("llama-server"))
    peers = _peers(order[1:], names, start=PRIMARY_PORT + 1, taken_names={names[first.real]},
                   taken_ports={PRIMARY_PORT}, vllm=vllm)
    lines = preset_lines(server, served, peers, binary_from=vllm[1] if first.checkpoint else "on PATH",
                         vllm_from=vllm[1] if vllm else "on PATH")
    if server.binary is None:
        lines.append(no_binary_line([s.model for s in order if s.model], _machine_autosense(config_path).model_roots))
    return InitRecords(server=server, served=served, peers=peers, lines=lines, block_in_receipt=True, confirmed=confirmed)


def _first_run(config_path: Path, *, scope: Literal["new", "all"], person: bool, preset: str | None) -> Sync:
    """Every unrecorded model under the roots, shown best first and set up as the person says."""
    global_dir = global_config_dir(config_path)
    auto = _machine_autosense(config_path)
    lib_path = model_library.library_path(global_dir)
    checkpoints, vllm = _vllm_world(global_dir)
    found = model_library.scan(auto.model_roots, frozenset(), lib_path, scope=scope, rank=list, first_run=True,
                               checkpoints=checkpoints)
    shown = _listable((*found.new, *found.seen, *found.declined))
    if not shown:
        return Sync()
    listed = _ranked(config_path, auto, shown)
    names: dict[str, str] = {}
    for s in listed:
        names[s.real] = unique_preset_name(_label(s), names.values())
    wanted = _named(listed, names, preset)
    if person:
        _bullets(listed, names, vllm)
    choice = _interview(listed, names, wanted, person=person)
    if choice is None:
        if person:
            _say(NEXT_START_LINE)
        order = _first_of([s for s in listed if not s.declined], wanted)
        if order:
            return Sync(_records(config_path, order, names, confirmed=False, vllm=vllm), shown=len(listed))
        choice = _Choice([], [s for s in listed if s.declined])
    elif choice.skipped:
        _say(SKIP_LINE)
        return Sync(shown=len(listed))
    else:
        _save_library(found, lib_path, declined=[s.real for s in choice.declined])
    if choice.order:
        records = _records(config_path, choice.order, names, confirmed=True, vllm=vllm)
        return Sync(records, tuple(records.lines), 0, tuple(s.real for s in choice.declined), len(listed))
    _say(DECLINED_ALL_LINE.format(n=len(choice.declined), library=shown_path(lib_path)))
    return Sync(declined=tuple(s.real for s in choice.declined), shown=len(listed))


def _later_interview(shown: Sequence[model_library.Shown], new: Sequence[model_library.Shown],
                     names: dict[str, str], wanted: model_library.Shown | None, *, choose: bool,
                     person: bool) -> _Choice | None:
    """The later-run question and what each answer sets up; Not now and a cancel are the same; None when nobody could be asked."""
    if not person:
        return None
    not_now = _Choice([], [], skipped=True)
    if not choose:
        if len(new) == 1:
            name = visible_text(names[new[0].real])
            question, yes = NEW_ONE_QUESTION.format(name=name), Option("set-up", SET_IT_UP, SET_IT_UP_HINT.format(name=name))
        else:
            question, yes = NEW_MANY_QUESTION.format(n=len(new)), Option("set-up", SET_THEM_UP, SET_THEM_UP_HINT)
        options = [yes, Option("not-now", NOT_NOW, NOT_NOW_HINT)]
        if len(shown) > 1:
            options.append(Option("choose", CHOOSE_WHICH, CHOOSE_WHICH_HINT))
        answer = interview.ask(question, options, default=0, cancel="not-now")
        if not answer.reached:
            return None
        if answer.cancelled or answer.keys == ("not-now",):
            return not_now
        if answer.keys == ("set-up",):
            return _Choice(list(new), [])
    picked = interview.ask(
        SET_UP_WHICH,
        [Option(s.real, visible_text(names[s.real]) + (DECLINED_TAG if s.declined else ""), _origin(s))
         for s in shown],
        default={s.real for s in new} | ({wanted.real} if wanted else set()), cancel="cancel", multi=True)
    if not picked.reached:
        return None
    if picked.cancelled:
        return not_now
    return _Choice([s for s in shown if s.real in picked.keys], [s for s in shown if s.real not in picked.keys])


def _later_run(config_path: Path, *, scope: Literal["new", "all"], person: bool, preset: str | None,
               choose: bool) -> Sync:
    """A model that arrives after the first run: asked about once, set up as a peer, never asked twice."""
    if not person:
        return Sync()
    global_dir = global_config_dir(config_path)
    auto = _machine_autosense(config_path)
    lib_path = model_library.library_path(global_dir)
    checkpoints, vllm = _vllm_world(global_dir)
    found = model_library.scan(auto.model_roots, recorded_paths(global_dir), lib_path, scope=scope,
                               rank=_ranker(config_path, auto), checkpoints=checkpoints)
    shown = _listable((*found.new, *found.seen, *found.declined))
    new = _listable(found.new)
    if not (shown if choose else new):
        if found.changed:
            _save_library(found, lib_path)
        if choose:
            _say(NOTHING_TO_SET_UP_LINE)
        return Sync(shown=len(shown))
    taken_names, taken_ports = recorded_names_ports(global_dir)
    names: dict[str, str] = {}
    for s in shown:
        names[s.real] = unique_preset_name(_label(s), {*taken_names, *names.values()})
    _bullets(shown, names, vllm)
    choice = _later_interview(shown, new, names, _named(shown, names, preset), choose=choose, person=person)
    if choice is None:
        return Sync(shown=len(shown))
    written: list[dict] = []
    if choice.order:
        peers = _peers(choice.order, names, start=PRIMARY_PORT + 1, taken_names=taken_names, taken_ports=taken_ports,
                       vllm=vllm)
        if (written := append_peers(config_path, peers, adopt=True)) is None:
            _say(NOT_WRITTEN_LINE)
            _save_library(found, lib_path)
            return Sync(shown=len(shown))
        for peer in written:
            _say(SET_UP_LINE.format(name=visible_text(peer["name"])))
    left = new if choice.skipped else choice.declined
    if left:
        _say(NOT_SET_UP_LINE.format(names=", ".join(visible_text(names[s.real]) for s in left)))
    _save_library(found, lib_path, declined=[s.real for s in choice.declined], dropped=[s.real for s in choice.order])
    return Sync(None, (), len(written), tuple(s.real for s in choice.declined), len(shown))


def sync(config_path: Path, *, scope: Literal["new", "all"], person: bool, preset: str | None = None,
         first_run: bool = False, choose: bool = False) -> Sync:
    """Every unrecorded model the roots hold, asked about and set up as the person says; nothing without a person."""
    if first_run:
        return _first_run(config_path, scope=scope, person=person, preset=preset)
    return _later_run(config_path, scope=scope, person=person, preset=preset, choose=choose)


@dataclass(frozen=True)
class OfferVllm:
    """A vLLM row chosen with its consent: the caller installs the runtime and fetches the checkpoint."""
    row: RefArch


@dataclass(frozen=True)
class NotNow:
    """The offer was declined, or a download did not verify; the way back was said."""


def install_line(platform_name: str | None = None) -> str | None:
    name = platform_name or sys.platform
    return next((line for key, line in INSTALL_LINES.items() if name.startswith(key)), None)


def record_line(name: str, runtime: str, port: int | None, binary: str | None, path: str | os.PathLike[str]) -> str:
    return RECORD_LINE.format(name=visible_text(name), runtime=RUNTIME_LABELS.get(runtime, runtime), path=shown_path(path),
                              port=port, binary=shown_path(binary) if binary else NO_BINARY_TEXT)


def verify_download(path: str | os.PathLike[str], *, size: int, sha256: str) -> DiscoveredModel | str:
    """The model a downloaded file is (bytes counted and hashed, header read through one descriptor), else what is wrong with it."""
    try:
        with os.fdopen(open_regular(path), "rb") as f:
            digest, count = hashlib.sha256(), 0
            while count <= size and (chunk := f.read(min(1 << 20, size + 1 - count))):
                count += len(chunk)
                digest.update(chunk)
            if count != size:
                return f"size is more than {size} bytes" if count > size else f"size is {count} bytes, expected {size}"
            if digest.hexdigest() != sha256:
                return f"sha256 {digest.hexdigest()} differs from {sha256}"
            f.seek(0)
            header = read_header_from(f, DISCOVERY_KEYS | autosense.FIT_KEYS)
            model = model_of(path, header, mtime=os.fstat(f.fileno()).st_mtime)
    except NotRegularFile:
        return "not a regular file"
    except GgufHeaderError as exc:
        return f"not a GGUF ({exc})"
    except OSError as exc:
        return f"cannot be read ({exc.strerror or type(exc).__name__})"
    return model if model is not None else "not a model GGUF"


def record_path(config_path: Path, model: DiscoveredModel, *, first_run: bool, adopt: bool,
                say: Callable[[str], None] | None = None) -> InitRecords | list[dict] | None:
    """Record one verified file as it is, its record line said first: `server:` on a first run, else a peer in overrides.yaml."""
    say = say or _say
    binary = which_outside_cwd("llama-server")
    if first_run:
        server = ManagedServerConfig(runtime="llamacpp", model=str(model.path), served_name=model.name,
                                     port=PRIMARY_PORT, binary=binary)
        say(record_line(model.name, "llamacpp", PRIMARY_PORT, binary, model.path))
        return InitRecords(server=server, served=model.name, block_in_receipt=True, confirmed=adopt)
    names, ports = recorded_names_ports(global_config_dir(config_path))
    [peer] = model_library.peers_for([model], start=PRIMARY_PORT + 1, taken_names=names, taken_ports=ports)
    say(record_line(peer.name, "llamacpp", peer.lifecycle.port, binary, model.path))
    return append_peers(config_path, [peer], adopt=adopt, say=say)


def _ram(auto: AutosenseConfig) -> float | None:
    """The box's RAM less the unified reserve; None when neither /proc/meminfo nor sysconf says."""
    if sys.platform.startswith("linux") and (info := autosense._read_meminfo()) is not None:
        return max(0.0, info["MemAvailable"] - auto.unified_reserve_ratio * info["MemTotal"])
    try:
        total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        return None
    return total * (1 - auto.unified_reserve_ratio)


def _on_path(runtime: str, *, global_dir: Path) -> bool:
    binary = RUNTIMES[runtime].path_binary
    found = which_outside_cwd(binary) if binary else managed_server.vllm_binary(global_dir)
    return found is not None


def _sense(auto: AutosenseConfig) -> tuple[bool, float | None]:
    """Whether nvidia-smi answers, and the memory a model may use."""
    free = autosense.free_memory(auto)
    if free is None:
        return False, _ram(auto)
    return True, free.free_bytes * (1 - auto.compute_margin_ratio if free.source == "nvidia-smi" else 1)


def _gb(n: float) -> str:
    return f"{n / 1e9:.1f} GB"


def _where(row: RefArch) -> str:
    return f"{row.repo}/{row.file}" if row.file else row.repo


def _ask(question: str, options: Sequence[Option], *, default: int, cancel: str, person: bool) -> interview.Answer:
    if not person:
        return interview.Answer((cancel,), True, reached=False)
    return interview.ask(question, options, default=default, cancel=cancel)


def _consent(row: RefArch, global_dir: Path) -> str:
    """What `Download it` also does on a vLLM row that has no runtime yet."""
    if managed_server.vllm_binary(global_dir) is not None:
        return ""
    if row.launch == "docker":
        return f" and pull {row.docker_image}"
    return f" and install {row.pip_package} into {shown_path(managed_server.server_dir(global_dir) / 'venv')}"


def _download_question(row: RefArch, memory: float | None, global_dir: Path, *, rows: bool, person: bool) -> interview.Answer:
    size = _gb(row.bytes)
    what = visible_text(row.file or row.repo)
    question = (OFFER_QUESTION.format(what=what, size=size, free=_gb(memory))
                if memory is not None and row.bytes <= memory else OFFER_QUESTION_UNKNOWN.format(what=what, size=size))
    hint = SMALL_ROW_HINT if RUNTIMES[row.runtime].gguf else f"download {visible_text(row.repo)}" + _consent(row, global_dir)
    options = [Option("download", DOWNLOAD_IT, hint),
               *([Option("rows", SHOW_ROWS, SHOW_ROWS_HINT)] if rows else []),
               Option("not-now", NOT_NOW, NOT_NOW_OFFER_HINT)]
    return _ask(question, options, default=0, cancel="not-now", person=person)


def _pick_row(rows: Sequence[RefArch], sensed: RefArch, *, person: bool) -> interview.Answer:
    for row in rows:
        _say(ROW_LINE.format(name=visible_text(row.name), where=visible_text(_where(row)), size=_gb(row.bytes), page=row.page))
    options = [Option(r.key, visible_text(r.name), f"{visible_text(_where(r))} · {_gb(r.bytes)} · {r.page}") for r in rows]
    return _ask(DOWNLOAD_WHICH, options, default=rows.index(sensed), cancel="not-now", person=person)


def _not_now(row: RefArch) -> NotNow:
    command = f"localharness model --download {row.repo}" + (f" --file {row.file}" if row.file else "")
    for line in (NOT_NOW_PAGE_LINE.format(page=row.page), NOT_NOW_COMMAND_LINE.format(command=command), RUN_AGAIN_LINE):
        _say(line)
    return NotNow()


def _tell_install(row: RefArch) -> None:
    if (binary := RUNTIMES[row.runtime].path_binary) and which_outside_cwd(binary) is None and (line := install_line()):
        _say(line)


def _fetch(config_path: Path, row: RefArch, *, person: bool) -> InitRecords | NotNow:
    try:
        path = managed_server.download_file(row.repo, row.file, revision=row.revision)
    except Exception as exc:  # noqa: BLE001 — whatever the hub client raises is one line
        err_console.print(DOWNLOAD_FAILED_LINE.format(why=visible_text(exc)), markup=False, soft_wrap=True)
        raise typer.Exit(1) from exc
    checked = verify_download(path, size=row.bytes, sha256=row.sha256)
    if isinstance(checked, str):
        _say(VERIFY_MISMATCH_LINE.format(path=shown_path(path), what=checked))
        _say(RUN_AGAIN_LINE)
        return NotNow()
    recs = record_path(config_path, checked, first_run=True, adopt=person)
    if recs.server.binary is None and (line := install_line()):
        recs.lines.append(line + THEN_START)
    return recs


def offer(config_path: Path, *, person: bool) -> InitRecords | OfferVllm | NotNow | None:
    """One question over this platform's reference rows, the sensed one first; None when nobody could be asked or no row fits."""
    if not person:
        return None
    auto = _machine_autosense(config_path)
    global_dir = global_config_dir(config_path)
    has_nvidia, memory = _sense(auto)
    plat, machine = sys.platform, platform.machine()
    rows = tuple(r for r in REF_ARCHS if any(plat.startswith(p) for p in r.platforms)
                 and needs_met(r, platform=plat, machine=machine, has_nvidia=has_nvidia))
    sensed = row = default_row(rows, platform=plat, machine=machine, has_nvidia=has_nvidia,
                               on_path=partial(_on_path, global_dir=global_dir), memory=memory)
    if row is None:
        return None
    _tell_install(row)
    answer = _download_question(row, memory, global_dir, rows=True, person=person)
    if not answer.reached:
        return None
    if answer.keys == ("rows",) and not answer.cancelled:
        picked = _pick_row(rows, sensed, person=person)
        if not picked.reached:
            return None
        if picked.cancelled:
            return _not_now(row)
        row = next(r for r in rows if r.key == picked.keys[0])
        if row is not sensed:
            _tell_install(row)
        answer = _download_question(row, memory, global_dir, rows=False, person=person)
        if not answer.reached:
            return None
    if answer.cancelled or answer.keys != ("download",):
        return _not_now(row)
    return _fetch(config_path, row, person=person) if RUNTIMES[row.runtime].gguf else OfferVllm(row)
