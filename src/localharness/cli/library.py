"""The model library's side that talks and writes: the shown lines, the questions and the one peer writer."""
from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from urllib.parse import urlparse

from rich.console import Console

from localharness.cli import workspace
from localharness.config import trust
from localharness.config.models import EndpointRef
from localharness.config.paths import global_config_dir
from localharness.agent.gguf_header import unique_preset_name
from localharness.provider import model_library

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
        from localharness.cli.init_cmd import _record_dicts_of

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
    from localharness.cli.init_cmd import _adopt_written

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
    if adopt:
        recorded = trust.recorded_machine(global_dir)
        covered = {trust._canonical(e) for e in recorded["entries"]} if recorded else set()
        _adopt_written(global_dir, "overrides.yaml", None, fragment.written)
    if fragment.copied:
        say(workspace.PEER_LIST_COPIED + (_readopt_copied(global_dir, fragment.copied, covered) if adopt else ""))
    return fragment.written


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
