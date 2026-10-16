"""The model library's side that talks and writes: the shown lines, the questions and the one peer writer."""
from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import SecretStr
from rich.console import Console

from localharness.agent.gguf_header import DiscoveredModel, rank_models, read_header, unique_preset_name
from localharness.cli import interview, workspace
from localharness.cli.interview import Option
from localharness.cli.workspace import shown_path, visible_text
from localharness.config import trust
from localharness.config.loader import ConfigLoader
from localharness.config.models import AutosenseConfig, EndpointRef, ManagedServerConfig
from localharness.config.overlay import load_overlay
from localharness.config.paths import global_config_dir
from localharness.config.redact import reveal
from localharness.core.which import which_outside_cwd
from localharness.provider import autosense, model_library, speed_stats
from localharness.provider.detector import DetectorResult, detect_provider
from localharness.provider.server import RUNTIME_LABELS
from localharness.provider.tuning import mtp_layers

console = Console()

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
    return f"({_clause(server.runtime, mtp=bool(server.model and model_mtp(server.model)), mmproj=server.mmproj)})"


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


def record_block(recs: InitRecords) -> list[str]:
    """The lines that show a record: its own, then what a daemon on this machine holds."""
    return [*recs.lines, *(_daemon_lines() if recs.server.runtime == "llamacpp" else [])]


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


def recorded_paths(global_dir: Path) -> frozenset[str]:
    """The realpaths of every model and projector the machine's two files name in a record."""
    records: list = []
    for raw in workspace.raw_settings(global_dir):
        peers = raw.get("extra_endpoints")
        records += [raw.get("server"), *(p.get("lifecycle") for p in peers if isinstance(p, dict))] \
            if isinstance(peers, list) else [raw.get("server")]
    return frozenset(os.path.realpath(os.path.expanduser(str(r[key])))
                     for r in records if isinstance(r, dict) for key in ("model", "mmproj") if r.get(key))


def prune_recorded(config_path: Path) -> bool:
    """Drop from an existing library file every entry a record now names; True when it was rewritten."""
    global_dir = global_config_dir(config_path)
    path = model_library.library_path(global_dir)
    lib = model_library.Library.load(path)
    if lib.missing or lib.corrupt:
        return False
    kept = lib.unrecorded(recorded_paths(global_dir))
    return len(kept.entries) != len(lib.entries) and kept.save(path)


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


def _bullets(listed: Sequence[model_library.Shown], names: dict[str, str]) -> None:
    for s in listed:
        clause = _clause("llamacpp", mtp=bool(s.model.mtp), mmproj=s.model.mmproj)
        line = BULLET.format(name=visible_text(names[s.real]), clause=clause, path=shown_path(s.path),
                             size=format_size(s.size), date=s.first_seen)
        console.print(line + (DECLINED_TAG if s.declined else ""), markup=False, soft_wrap=True)


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
    picked = interview.ask(SET_UP_WHICH, [Option(s.real, visible_text(names[s.real]), shown_path(s.path))
                                          for s in listed],
                           default={s.real for s in eligible}, cancel="cancel", multi=True)
    if not picked.reached:
        return None
    if picked.cancelled:
        return skip
    chosen = [s for s in listed if s.real in picked.keys]
    declined = [s for s in listed if s.real not in picked.keys]
    if len(chosen) > 1 and named not in chosen:
        first = interview.ask(SERVE_WHICH_FIRST, [Option(s.real, visible_text(names[s.real]), shown_path(s.path))
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
             confirmed: bool) -> InitRecords:
    """The first file serves on `PRIMARY_PORT`, the rest are launchable peers above it."""
    first, rest = order[0].model, [s.model for s in order[1:]]
    served = names[order[0].real]
    server = ManagedServerConfig(runtime="llamacpp", model=str(first.path),
                                 mmproj=str(first.mmproj) if first.mmproj else None, served_name=served,
                                 port=PRIMARY_PORT, binary=which_outside_cwd("llama-server"))
    peers = model_library.peers_for(rest, start=PRIMARY_PORT + 1, taken_names={served}, taken_ports={PRIMARY_PORT})
    lines = preset_lines(server, served, peers, binary_from="on PATH")
    if server.binary is None:
        lines.append(no_binary_line([s.model for s in order], _machine_autosense(config_path).model_roots))
    return InitRecords(server=server, served=served, peers=peers, lines=lines, block_in_receipt=True, confirmed=confirmed)


def sync(config_path: Path, *, scope: Literal["new", "all"], person: bool, preset: str | None = None,
         first_run: bool = False) -> Sync:
    """The first run: every unrecorded model under the roots, shown best first and set up as the person says."""
    if not first_run:
        raise ValueError("sync: only the first run is wired")
    global_dir = global_config_dir(config_path)
    auto = _machine_autosense(config_path)
    lib_path = model_library.library_path(global_dir)
    found = model_library.scan(auto.model_roots, frozenset(), lib_path, scope=scope, rank=list, first_run=True)
    shown = [s for s in (*found.new, *found.seen, *found.declined) if s.model is not None]
    if not shown:
        return Sync()
    by_path = {str(s.model.path): s for s in shown}
    listed = [by_path[str(m.path)] for m in _ranker(config_path, auto)([s.model for s in shown])]
    names: dict[str, str] = {}
    for s in listed:
        names[s.real] = unique_preset_name(s.model.name, names.values())
    wanted = _named(listed, names, preset)
    if person:
        _bullets(listed, names)
    choice = _interview(listed, names, wanted, person=person)
    if choice is None:
        if person:
            _say(NEXT_START_LINE)
        order = _first_of([s for s in listed if not s.declined], wanted)
        if order:
            return Sync(_records(config_path, order, names, confirmed=False), shown=len(listed))
        choice = _Choice([], [s for s in listed if s.declined])
    elif choice.skipped:
        _say(SKIP_LINE)
        return Sync(shown=len(listed))
    else:
        found.library.marked(declined=[s.real for s in choice.declined]).save(lib_path)
    if choice.order:
        records = _records(config_path, choice.order, names, confirmed=True)
        return Sync(records, tuple(records.lines), 0, tuple(s.real for s in choice.declined), len(listed))
    _say(DECLINED_ALL_LINE.format(n=len(choice.declined), library=shown_path(lib_path)))
    return Sync(declined=tuple(s.real for s in choice.declined), shown=len(listed))
