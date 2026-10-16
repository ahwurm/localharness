"""The model library's side that talks and writes: the shown lines, the questions and the one peer writer."""
from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from pydantic import SecretStr
from rich.console import Console

from localharness.agent.gguf_header import read_header, unique_preset_name
from localharness.cli import workspace
from localharness.cli.workspace import shown_path, visible_text
from localharness.config import trust
from localharness.config.loader import ConfigLoader
from localharness.config.models import AutosenseConfig, EndpointRef, ManagedServerConfig
from localharness.config.overlay import load_overlay
from localharness.config.paths import global_config_dir
from localharness.config.redact import reveal
from localharness.provider import model_library
from localharness.provider.detector import DetectorResult, detect_provider
from localharness.provider.server import RUNTIME_LABELS
from localharness.provider.tuning import mtp_layers

console = Console()

OVERLAY_BUSY_LINE = "overrides.yaml is busy — nothing written"
PEERS_COPIED_COUNTS = " — {copied} copied, {kept} still confirmed"
PEERS_DETAIL = "localharness: peers added to overrides.yaml"


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
                    "port": port, "served_name": name if life.served_name in (None, peer.name) else life.served_name})
            record = _record_dicts_of(None, [peer.model_copy(update=update)])[1][0]
            listed.append(record)
            self.written.append(record)
            models |= {real} if real else set()
            names.add(name)
            ports |= {port} if port is not None else set()
            self.audit.extend(audit_entries(f"extra_endpoints.{name}.lifecycle.", record.get("lifecycle") or {}))


def append_peers(config_path: Path, peers: Sequence[EndpointRef], *, adopt: bool,
                 say: Callable[[str], None] | None = None) -> list[dict] | None:
    """Append ``peers`` to overrides.yaml's `extra_endpoints` under the overlay lock; `server:` is untouched.
    Returns the dicts written, None when nothing was. ``adopt`` records them as approved."""
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
    """After a write: adopt what it wrote beside ``server`` and the copied peers whose config.yaml twin was
    covered (when ``adopt``), and say the copied line once."""
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


def preset_clause(server: ManagedServerConfig) -> str:
    """`(llama.cpp, mtp, +<projector basename>)`: the runtime, `mtp` when the model has it, the projector."""
    clause = RUNTIME_LABELS.get(server.runtime, server.runtime)
    if server.model and model_mtp(server.model):
        clause += ", mtp"
    if server.mmproj:
        clause += ", +" + visible_text(os.path.basename(server.mmproj))
    return f"({clause})"


def preset_lines(server: ManagedServerConfig, served: str, peers: Sequence[EndpointRef], *,
                 binary_from: str, extras: Sequence[str] = ()) -> list[str]:
    """The primary's line with its binary and where it came from, the escaped ``extras``, then one line per peer."""
    primary = f"serving {visible_text(served)} {preset_clause(server)} ← {shown_path(server.model)}"
    if server.binary:
        primary += f" · binary: {shown_path(server.binary)} ({binary_from})"
    lines = [primary, *extras]
    for peer in peers:
        life = peer.lifecycle
        if life is not None:
            lines.append(f"/model switches to {visible_text(peer.name)} {preset_clause(life)} "
                         f"← {shown_path(life.model)}")
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


def show_records(recs: InitRecords) -> None:
    """Print the block a record path shows before it writes; a name from a file is text, never markup."""
    lines = list(recs.lines)
    if recs.server.runtime == "llamacpp":
        lines += _daemon_lines()
    for line in lines:
        console.print(line, markup=False, soft_wrap=True)


def _record_dicts_of(server: ManagedServerConfig | None,
                     peers: Sequence[EndpointRef]) -> tuple[dict | None, list[dict]]:
    """The server record and every peer as written to a file: no `null` preset field, no invented `effort`."""
    return (reveal(server.model_dump(mode="python", exclude_none=True)) if server is not None else None,
            [reveal(peer.model_dump(mode="python", exclude_none=True)) for peer in peers])


def _adopt_written(global_dir: Path, file: str, server: dict | None, peers: Sequence[dict]) -> None:
    """Record, as already approved, the launch entries just written and shown: the primary's `server.` keys, each
    peer's `lifecycle.` keys and the posture entries under those prefixes that still say what was written. With no
    machine record the first is made, of the launch kinds only; with one these entries are added or replaced and
    nothing else changes. No key is ever an entry."""
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
