"""One row per runtime: every answer the harness used to read off a runtime-name if-chain.

Imports nothing from the package at module level: lifecycle, server, tuning, config.models and agent.context all
import this module, so each callable column resolves its implementation inside the function.
"""
from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class Runtime:
    name: str
    strategy: Callable[[Any], Any]
    unit_servable: Callable[[Any], bool]
    label: Callable[[Any], str]
    key_env: str | None
    probe: Callable[..., Any]
    price: Callable[..., Any] | None
    argv: Callable[..., list[str]]
    facts: Callable[[], Mapping] | None
    record_fields: tuple[str, ...]
    gguf: bool
    checkpoint: bool
    path_binary: str | None
    exposed_by_default: bool
    scrub: tuple[str, ...]
    window_note: str | None
    derived_args: Mapping[str, str]


def _strategy(cls_name: str, module: str = "lifecycle") -> Callable[[Any], Any]:
    def build(spec: Any) -> Any:
        return getattr(importlib.import_module(f"localharness.provider.{module}"), cls_name)()
    return build


def _probe(name: str) -> Callable[..., Any]:
    def probe(root: str, model: str, hk: dict[str, Any]) -> Any:
        from localharness.agent import context
        return getattr(context, name)(root, model, hk)
    return probe


def _price(name: str) -> Callable[..., Any]:
    def price(spec: Any, sensed: Any, tuned: Any, autosense: Any, floor: int | None = None) -> Any:
        from localharness.provider import autosense as _auto
        return getattr(_auto, name)(spec, sensed, tuned, autosense, floor)
    return price


def _argv(name: str, module: str = "server") -> Callable[..., list[str]]:
    def build(srv: Any, **kw: Any) -> list[str]:
        return getattr(importlib.import_module(f"localharness.provider.{module}"), name)(srv, **kw)
    return build


def _parsers() -> Mapping:
    from localharness.provider import tuning
    return tuning.MODEL_TYPE_PARSERS


def _no_parsers() -> Mapping:
    return MappingProxyType({})


_LMSTUDIO_GPU = "it is derived from `gpu` (--gpu max when gpu=True, --gpu off when False)"

RUNTIMES: Mapping[str, Runtime] = MappingProxyType({
    "llamacpp": Runtime(
        name="llama.cpp", strategy=_strategy("SpawnedProcessStrategy"),
        unit_servable=lambda spec: True, label=lambda spec: "llama.cpp",
        key_env="LLAMA_API_KEY", probe=_probe("_probe_llamacpp"), price=_price("price_llamacpp"),
        argv=_argv("_llamacpp_argv"), facts=None,
        record_fields=("mmproj", "served_name", "cache_type_k", "cache_type_v", "cache_ram",
                       "api_key_file", "env", "config_ini", "speculative"),
        gguf=True, checkpoint=False, path_binary="llama-server", exposed_by_default=False,
        scrub=("LLAMA_ARG_*", "LLAMA_API_KEY", "MTMD_BACKEND_DEVICE"), window_note=None,
        derived_args=MappingProxyType({})),
    "vllm": Runtime(
        name="vLLM", strategy=_strategy("ManagedVllmStrategy"),
        unit_servable=lambda spec: spec.launch == "binary",
        label=lambda spec: "vLLM binary" if spec.launch == "binary" else "docker",
        key_env="VLLM_API_KEY", probe=_probe("_probe_openai"), price=_price("price_vllm"), argv=_argv("_vllm_argv"), facts=_parsers,
        record_fields=("reasoning_parser", "tool_call_parser"),
        gguf=False, checkpoint=True, path_binary=None, exposed_by_default=True,
        scrub=("VLLM_API_KEY",), window_note=None, derived_args=MappingProxyType({})),
    "ollama": Runtime(
        name="Ollama", strategy=_strategy("DaemonStrategy"),
        unit_servable=lambda spec: False, label=lambda spec: "Ollama",
        key_env=None, probe=_probe("_probe_ollama"), price=None, argv=_argv("_ollama_argv"), facts=None, record_fields=(),
        gguf=False, checkpoint=False, path_binary=None, exposed_by_default=False,
        scrub=(), window_note=" · Ollama decides the window", derived_args=MappingProxyType({})),
    "lmstudio": Runtime(
        name="LM Studio", strategy=_strategy("LmsStrategy"),
        unit_servable=lambda spec: False, label=lambda spec: "LM Studio",
        key_env=None, probe=_probe("_probe_lmstudio"), price=None, argv=_argv("_vllm_argv"), facts=None, record_fields=(),
        gguf=False, checkpoint=False, path_binary=None, exposed_by_default=False,
        scrub=(), window_note="", derived_args=MappingProxyType({"--gpu": _LMSTUDIO_GPU})),
    "strata": Runtime(
        name="Strata", strategy=_strategy("StrataStrategy", "strata"),
        unit_servable=lambda spec: True, label=lambda spec: "Strata",
        key_env="STRATA_API_KEY", probe=_probe("_probe_strata"), price=None,
        argv=_argv("argv", "strata"), facts=_no_parsers,
        record_fields=("mmproj", "served_name", "engine_dir", "engine_config"),
        gguf=True, checkpoint=False, path_binary=None, exposed_by_default=False,
        scrub=("STRATA_API_KEY", "STRATA_SKIP_SHA256", "STRATA_PREBUILT_URL", "STRATA_SOURCE", "STRATA_EXECV",
               "HF_ENDPOINT"),
        window_note=None, derived_args=MappingProxyType({})),
})


def row_of(name: object) -> Runtime | None:
    return RUNTIMES.get(name) if isinstance(name, str) else None
