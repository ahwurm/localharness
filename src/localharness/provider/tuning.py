"""What a launch derives from the model's own header and the machine's knobs (spec 20 §6.0-6.2).

Pure but for one read: `derive` touches nothing but its arguments, and the only file this module
opens is a vLLM checkpoint's `config.json`, through `read_model_config` (bounded, `open_regular`,
never raises). It runs no process and executes nothing from any file. `derive` decides the K/V cache
type and MTP drafting of a llama.cpp launch and the reasoning / tool-call parsers of a vLLM launch;
it never writes a record and never raises on a hostile header or config. Every derived field carries one reason (`Tuned.reasons`, keyed by field name) so
the launch can say why.

Precedence, for every derived flag: the record field > `extra_args` / `env:` / `config_ini` in any
spelling > the derivation. A derivation is the fallback for a person who said nothing.
"""
from __future__ import annotations

import json
import os
import re
import shlex
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from localharness.core.private_files import open_regular
from localharness.provider.autosense import (
    extra_cache_types,
    flash_attn_off,
    hand_set_options,
    has_cache_type_flag,
    has_spec_flag,
    spec_type_flag,
)
from localharness.provider.server import KEY_ENV

# llama-server's own default; deriving it would change the 0.18 argv, so it derives nothing (D-08).
_LLAMA_DEFAULT_CACHE_TYPE = "f16"


# vLLM parsers by the checkpoint's config.json `model_type` (spec 20 §6.3): (reasoning, tool-call),
# None where the family has no reasoning parser. The ONLY place a parser name is chosen: a
# `model_type` read from a file is a lookup key here and is never emitted (D-12).
MODEL_TYPE_PARSERS: Mapping[str, tuple[str | None, str | None]] = MappingProxyType({
    "qwen3": ("qwen3", "qwen3_coder"),
    "qwen3_moe": ("qwen3", "qwen3_coder"),
    "qwen3_next": ("qwen3", "qwen3_coder"),
    "qwen3_5": ("qwen3", "qwen3_coder"),
    "qwen3_5_moe": ("qwen3", "qwen3_coder"),
    "gemma4": ("gemma4", "gemma4"),
    "glm4_moe": ("glm45", "glm45"),
    "glm4v_moe": ("glm45", "glm45"),
    "deepseek_v3": ("deepseek_r1", "deepseek_v3"),
    "gpt_oss": ("openai_gptoss", "openai"),
    "llama": (None, "llama3_json"),
    "llama4": (None, "llama4_json"),
    "mistral": (None, "mistral"),
    "mistral3": (None, "mistral"),
    "kimi_k2": (None, "kimi_k2"),
})

MODEL_CONFIG_CAP = 1 << 20  # a config.json larger than this derives nothing

_PARSER_FLAGS = ("--reasoning-parser", "--tool-call-parser", "--enable-auto-tool-choice")


def _empty_reasons() -> Mapping[str, str]:
    return MappingProxyType({})


@dataclass(frozen=True)
class Tuned:
    """What `derive` decided for one launch. Every field is None when nothing was derived for it;
    `reasons` holds one phrase per derived field, keyed by the field's name."""

    cache_type: str | None = None
    spec_type: str | None = None
    draft_p_min: float | None = None
    reasoning_parser: str | None = None
    tool_call_parser: str | None = None
    mtp_layers: int | None = None
    reasons: Mapping[str, str] = field(default_factory=_empty_reasons)

    def __post_init__(self) -> None:
        # A frozen record holds a read-only copy, whatever mapping the caller handed in.
        object.__setattr__(self, "reasons", MappingProxyType(dict(self.reasons)))


NOTHING_DERIVED = Tuned()


def _int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def mtp_layers(header: Mapping[str, Any]) -> int | None:
    """The model's MTP layer count: `nextn_predict_layers`, only when `block_count` is an int above
    zero and the count is an int with 1 <= n <= block_count. A bool, a string, a negative, zero,
    a count above the block count or a missing key gives None; a hostile header never raises."""
    try:
        blocks, nextn = _int(header.get("block_count")), _int(header.get("nextn_predict_layers"))
    except AttributeError:
        return None
    if blocks is None or blocks <= 0 or nextn is None or not 1 <= nextn <= blocks:
        return None
    return nextn


def draft_share(header: Mapping[str, Any]) -> float:
    """The multiple of the main context's KV a drafting launch holds: the MTP draft context keeps
    its own KV for nextn_predict_layers layers, so `1 + nextn / block_count`; 1.0 when the header
    declares no valid MTP layers."""
    n = mtp_layers(header)
    return 1.0 if n is None else 1 + n / header["block_count"]


def has_parser_flag(extra_args: Sequence[str]) -> bool:
    """True when `extra_args` carries `--reasoning-parser`, `--tool-call-parser` or
    `--enable-auto-tool-choice`, as a bare flag or in `--flag=value` form."""
    return any(isinstance(a, str) and a.split("=", 1)[0] in _PARSER_FLAGS for a in extra_args)


def _hf_cached_config(repo_id: str) -> str | None:
    """The cached snapshot's config.json for a Hugging Face repo id: a local cache lookup, never the
    network. A non-path answer (the library's cached-as-missing sentinel) is None. Its own function so
    the suite can make the lookup hermetic at one seam (tests/conftest.py)."""
    from huggingface_hub import try_to_load_from_cache

    found = try_to_load_from_cache(repo_id=repo_id, filename="config.json")
    return found if isinstance(found, str) else None


def _checkpoint_config_path(spec: Any) -> str | None:
    """Where the checkpoint's config.json would be: the registry entry's directory, a model that is
    itself a directory (an absolute path after `~`: what is derived never depends on the working
    directory, IN-04), else the Hugging Face cache snapshot (a local cache lookup, no network)."""
    entry = spec.entry_for(spec.model)
    if entry is not None:
        return os.path.join(os.path.expanduser(entry.path), "config.json")
    model = os.path.expanduser(spec.model)
    if os.path.isabs(model) and os.path.isdir(model):
        return os.path.join(model, "config.json")
    return _hf_cached_config(spec.model)


def read_model_config(spec: Any) -> Mapping[str, Any] | None:
    """The parsed config.json of a vLLM record's checkpoint, or None.

    Bounded and failure-tolerant (D-12, D-22): `open_regular` (a FIFO or a directory is refused
    without blocking), at most `MODEL_CONFIG_CAP` bytes, `json.loads` only (no import, nothing run),
    and the value kept only when it is an object. A missing, unreadable, oversized, malformed or
    deeply nested file, or any other runtime, gives None; this never raises."""
    try:
        if getattr(spec, "runtime", None) != "vllm":
            return None
        path = _checkpoint_config_path(spec)
        if path is None:
            return None
        fd = open_regular(path)
        with os.fdopen(fd, "rb") as fh:
            raw = fh.read(MODEL_CONFIG_CAP + 1)
        if len(raw) > MODEL_CONFIG_CAP:
            return None
        value = json.loads(raw)
    except (OSError, ValueError, RecursionError):  # missing, not a file, unreadable, malformed, too deep
        return None
    except Exception:  # the hub library on an odd repo id, an odd spec: derive nothing, never raise
        return None
    return value if isinstance(value, dict) else None


def _derive_vllm(spec: Any, model_config: Mapping[str, Any] | None) -> Tuned:
    if not isinstance(model_config, dict):
        return NOTHING_DERIVED
    model_type = model_config.get("model_type")
    row = MODEL_TYPE_PARSERS.get(model_type) if isinstance(model_type, str) else None
    if row is None:
        return NOTHING_DERIVED
    entry = spec.entry_for(spec.model)
    extra = [*(getattr(spec, "extra_args", None) or []), *(entry.extra_args if entry is not None else [])]
    if has_parser_flag(extra):
        return NOTHING_DERIVED
    reasoning = row[0] if spec.reasoning_parser is None else None
    tool = row[1] if spec.tool_call_parser is None else None
    fields: dict[str, Any] = {}
    reasons: dict[str, str] = {}
    if reasoning is not None:
        fields["reasoning_parser"] = reasoning
        reasons["reasoning_parser"] = f"model_type={model_type}"
    if tool is not None:
        fields["tool_call_parser"] = tool
        reasons["tool_call_parser"] = f"model_type={model_type}"
    return Tuned(reasons=reasons, **fields) if fields else NOTHING_DERIVED


def derive(spec: Any, header: Mapping[str, Any], autosense: Any, *,
           model_config: Mapping[str, Any] | None = None) -> Tuned:
    """The flags a launch of `spec` derives (llama.cpp side; `model_config` is the vLLM side's input).

    `cache_type` is `autosense.kv_cache_type` unless the record sets a cache type, `extra_args` /
    `env` / `config_ini` pin one in any spelling, flash attention is switched off in any of them, or
    the knob is `f16`.
    `spec_type` is `draft-mtp` (with `autosense.draft_p_min`) only when `autosense.draft_mtp` is on,
    the record's `speculative` is unset, no speculative flag is set by hand, and the header declares
    MTP layers. A vLLM record derives its reasoning / tool-call parsers from `model_config`'s
    `model_type` (the table above) unless the record's field or a parser flag in `extra_args` says
    otherwise. Any other runtime derives nothing."""
    if getattr(spec, "runtime", None) == "vllm":
        return _derive_vllm(spec, model_config)
    if getattr(spec, "runtime", None) != "llamacpp":
        return NOTHING_DERIVED
    # `extra_args`, `env:` and the record's `config_ini` in one argv / env view: a cache type, a
    # flash-attention off or a speculative option set in any of them is the person's choice (CR-01).
    extra, env = hand_set_options(spec)
    fields: dict[str, Any] = {}
    reasons: dict[str, str] = {}

    kv = autosense.kv_cache_type
    if (kv != _LLAMA_DEFAULT_CACHE_TYPE
            and spec.cache_type_k is None and spec.cache_type_v is None
            and not has_cache_type_flag(extra, env) and not flash_attn_off(extra, env)):
        fields["cache_type"] = kv
        reasons["cache_type"] = "autosense.kv_cache_type"

    n = mtp_layers(header)
    if (autosense.draft_mtp and spec.speculative is None and not has_spec_flag(extra, env)
            and n is not None):
        fields.update(spec_type="draft-mtp", draft_p_min=autosense.draft_p_min, mtp_layers=n)
        reasons["spec_type"] = f"nextn_predict_layers={n}"

    return Tuned(reasons=reasons, **fields) if fields else NOTHING_DERIVED


def launch_pricing(spec: Any, header: Mapping[str, Any], tuned: Tuned) -> tuple[str | None, str | None, float]:
    """What a llama.cpp launch of `spec` is priced at: `(cache_type_k, cache_type_v, draft_share)`. The KV
    types are the record's, else `extra_args` `-ctk` / `-ctv`, else a `config_ini` cache-type key (as a
    flag name), else env `LLAMA_ARG_CACHE_TYPE_K` / `_V` (the record's `env:`, then a `config_ini` key
    spelled as the env name), else `tuned`'s derived type (R-08e, CR-01). The share is the MTP draft
    context's (`draft_share`) when the launch drafts with MTP (derived, `speculative: draft-mtp` or a
    hand-set `--spec-type draft-mtp`), else 1.0. The one pricing rule for the launch, the attach report
    and doctor's row (WR-06). Pure."""
    args, env = hand_set_options(spec)
    extra_k, extra_v = extra_cache_types(args)
    k = spec.cache_type_k or extra_k or env.get("LLAMA_ARG_CACHE_TYPE_K") or tuned.cache_type
    v = spec.cache_type_v or extra_v or env.get("LLAMA_ARG_CACHE_TYPE_V") or tuned.cache_type
    drafting = (tuned.spec_type == "draft-mtp" or getattr(spec, "speculative", None) == "draft-mtp"
                or spec_type_flag(args, env) == "draft-mtp")
    return k, v, draft_share(header) if drafting else 1.0


_RETRIED_DEATH = "(retried: the full shape died)"


def _full_parts(tuned: Tuned) -> list[str]:
    """The words for what a full-shape launch derived, in the order the launch line reads them."""
    parts: list[str] = []
    if tuned.cache_type is not None:
        parts.append(f"kv {tuned.cache_type}")
    if tuned.spec_type == "draft-mtp":
        n = tuned.mtp_layers
        parts.append("mtp" if n is None else f"mtp ({n} layer{'' if n == 1 else 's'})")
    if tuned.reasoning_parser is not None:
        parts.append(f"reasoning {tuned.reasoning_parser}")
    if tuned.tool_call_parser is not None:
        parts.append(f"tools {tuned.tool_call_parser}")
    return parts


def derived_clause(
    tuned: Tuned, *, shape: str = "full", cause: str | None = None, dropped: Sequence[str] = (),
    remembered_at: str | None = None, context: int | None = None, halved: bool = True,
    undo_field: str = "server.speculative", parser_fields: Sequence[str] = (),
) -> str:
    """What a launch derived, as the clause of `start`'s launch line (spec 20 §6.5, D-23). The one place
    these words are composed. Pure string building from closed values: the cache type, `draft-mtp`, a
    table parser, ints and the first ten printable ASCII characters of the ledger's ISO `at`.

    A full shape reads `kv q8_0 · mtp (1 layer)` (vLLM: `reasoning qwen3 · tools qwen3_coder`) and never
    carries the fitted `-c`: the window line prints that. A retry shape says what it dropped and, when
    the draft was dropped, the `undo_field: off` that pins drafting off. `context` is a death retry's
    halved `-c` and `halved` says whether a boot retry halved one; only a retry that halved a window
    says `off` keeps the full window (a record `context:` is never halved, so its retry only pins
    drafting off, WR-05). A vLLM retry that dropped its derived parsers (`dropped` holds "parsers",
    D-37) names the record fields that pin them off, `parser_fields` (both when not given), under
    `undo_field`'s section. Nothing derived gives an empty string."""
    if shape != "retry":
        return " · ".join(_full_parts(tuned))
    if "parsers" in dropped:
        # D-37: a vLLM retry without its derived parsers; the pin is the record's own fields
        prefix = undo_field.rsplit(".", 1)[0]
        pins = " / ".join(f"{prefix}.{f}: off" for f in (parser_fields or ("reasoning_parser", "tool_call_parser")))
        return f"parsers off {_RETRIED_DEATH} · {pins} keeps them pinned off"
    drafted_off = "draft-mtp" in dropped

    def undo(window_halved: bool) -> str:
        return f"{undo_field}: off " + ("keeps the full window" if window_halved else "keeps drafting pinned off")

    if cause == "boot":
        dropping = (["mtp off"] if drafted_off else []) + (["-c halved"] if halved else [])
        text = f"retried at boot: {', '.join(dropping)}" if dropping else "retried at boot"
        return f"{text} · {undo(halved)}" if drafted_off else text
    if cause == "remembered":
        # The ledger sits where agent-run code can write: only printable ASCII of its `at` reaches the
        # terminal, so no escape sequence rides the line (WR-04, T-12-35).
        shown = "".join(c for c in remembered_at or "" if " " <= c <= "~")[:10].strip()
        date = f" {shown}" if shown else ""
        what = "mtp off" if drafted_off else "window halved"
        kv = [f"kv {tuned.cache_type}"] if tuned.cache_type is not None else []
        return " · ".join([*kv, f"{what} (remembered{date})"])
    dropping = (["mtp off"] if drafted_off else []) + ([f"-c {context}"] if context is not None else [])
    text = f"{', '.join(dropping)} {_RETRIED_DEATH}" if dropping else _RETRIED_DEATH
    return f"{text} · {undo(context is not None)}" if drafted_off else text


SCRUBBED_SHOWN = MappingProxyType({
    "llamacpp": "LLAMA_ARG_*/LLAMA_API_KEY/MTMD_BACKEND_DEVICE", "vllm": "VLLM_API_KEY"})

_DERIVED_WORDS = (
    ("cache_type", lambda t: f"kv {t.cache_type}"),
    ("spec_type", lambda t: "draft-mtp"),
    ("reasoning_parser", lambda t: f"reasoning {t.reasoning_parser}"),
    ("tool_call_parser", lambda t: f"tools {t.tool_call_parser}"),
)


def script_function_name(name: str, taken: Collection[str]) -> str:
    """`serve_<name>` with every character outside `[A-Za-z0-9_]` made `_`, numbered `_2`, `_3` ... when `taken` has it."""
    base = "serve_" + re.sub(r"[^A-Za-z0-9_]", "_", name)
    candidate, n = base, 1
    while candidate in taken:
        n += 1
        candidate = f"{base}_{n}"
    return candidate


def _window_clause(plan: Any) -> str | None:
    report = plan.report
    if report is None:
        return None
    if report.mode == "fit":
        return f"-c left to llama-server --fit (the harness fitted {plan.full_fit} today)"
    if report.mode == "fit-by-server":
        return "-c left to llama-server --fit"
    if report.mode == "record":
        return f"-c {report.asked} from the record"
    return "-c from extra_args"


def _script_argv(plan: Any) -> list[str]:
    argv = list(plan.argv)
    if plan.report is not None and plan.report.mode == "fit":
        pair = ["-c", str(plan.serve_kwargs["context"])]
        at = next((i for i in range(len(argv) - 1) if argv[i : i + 2] == pair), None)
        if at is not None:
            del argv[at : at + 2]
    shown, hide = [], False
    for tok in argv:
        if hide:
            tok, hide = "<key>", False
        elif tok == "--api-key":
            hide = True
        elif tok.startswith("--api-key="):
            tok = "--api-key=<key>"
        shown.append(tok)
    return shown


def _exec_lines(tokens: Sequence[str], quote: Callable[[str], str]) -> list[str]:
    groups: list[list[str]] = []
    for tok in tokens[1:]:
        if tok.startswith("-") or not groups:
            groups.append([tok])
        else:
            groups[-1].append(tok)
    lines = [f"exec {quote(tokens[0])} \\"]
    lines += [f"  {' '.join(map(quote, g))}" + (" \\" if g is not groups[-1] else "") for g in groups]
    return lines


def render_script(
    plan: Any, spec: Any, *, name: str, version: str, key_placeholder: str, home_shown: str,
    visible: Callable[[str], str], function: str | None = None,
) -> str:
    """The launch `plan` as POSIX sh, every token and comment through `visible`, no key value printed; pure."""
    derived = [f"{word(plan.tuned)} ({plan.tuned.reasons[field_]})"
               for field_, word in _DERIVED_WORDS if field_ in plan.tuned.reasons]
    if plan.shape == "retry":
        derived.append(derived_clause(
            NOTHING_DERIVED, shape="retry", cause=plan.cause, dropped=plan.dropped,
            remembered_at=plan.remembered_at))
    environment = f"environment: {SCRUBBED_SHOWN[spec.runtime]} dropped"
    if plan.config_ini_home is not None:
        environment += f", XDG_CONFIG_HOME → {home_shown}"
    joined = " · ".join(derived) or "nothing"
    comments = [
        f"# {name} — the launch `localharness start` / `/model` runs ({version}); "
        f"derived: {joined}",
        "# " + "; ".join(filter(None, (_window_clause(plan), environment))),
    ]
    key_env = KEY_ENV.get(spec.runtime)
    if key_env in plan.env_names:
        comments.append(f"# {key_env}=<{key_placeholder}>")
    quote = lambda tok: visible(shlex.quote(tok))  # noqa: E731
    body = [visible(c) for c in comments]
    body += [f"export {quote(k)}={quote(v)}" for k, v in sorted((getattr(spec, "env", None) or {}).items())]
    body += _exec_lines(_script_argv(plan), quote)
    if function is None:
        return "\n".join(body)
    return "\n".join([f"{function}() {{", "  (", *(f"    {line}" for line in body), "  )", "}"])
