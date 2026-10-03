"""Secret values out of validation error text (R14, R16): a SecretStr setting's value never reaches
what a command prints, in text or --json, not even when the settings around it are refused.

pydantic keeps the rejected input in every error: str(ValidationError) prints it as input_value, and
the config loader as `(got: …)`. For an error on a whole model or section (a cross-field rule, or a
missing field, whose input is the section around it), that input is the section, secrets included.
pydantic's own str() shortens it to its head and tail, so a long key leaks its last characters,
which no whole-string scrub can catch. So printed error text carries each error's location and
message (`validation_text`), an input only as the loader masks it, and a message scrubbed of every
secret value the validated data holds (a validator's own message can quote its input).

Which values are secret is read from the model: a SecretStr field, or a field that holds one in a
list, a dict or a union arm (masked whole). Core code: pydantic and the standard library only."""
from __future__ import annotations

import functools
import types
import typing
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, SecretStr, ValidationError

SECRET_MASK = "**********"
_WALK_DEPTH = 8  # a secret field's value is a string; a container held there is walked this deep


def is_secret(annotation: Any) -> bool:
    """A SecretStr leaf (Optional included): its value is never printed, only SECRET_MASK."""
    return annotation is SecretStr or SecretStr in getattr(annotation, "__args__", ())


def scrub(text: str, secrets: Any) -> str:
    """Replace every non-empty secret string in `text` with SECRET_MASK (error texts may echo input)."""
    for secret in secrets:
        if isinstance(secret, str) and secret:
            text = text.replace(secret, SECRET_MASK)
    return text


def validation_text(exc: BaseException, secrets: Iterable[str] = ()) -> str:
    """Why a value was refused, without echoing any input: each pydantic error's location and
    message, scrubbed of `secrets` — never its input_value, which can hold a whole section, keys
    included. Anything else as `<type>: <text>`, scrubbed the same way."""
    if isinstance(exc, ValidationError):
        text = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" if e["loc"] else e["msg"]
                         for e in exc.errors())
    else:
        text = f"{type(exc).__name__}: {exc}"
    return scrub(text, secrets)


def _model_of(annotation: Any) -> type[BaseModel] | None:
    """The model a field holds directly, Optional unwrapped; None for anything else."""
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        arms = [a for a in typing.get_args(annotation) if a is not type(None)]
        annotation = arms[0] if len(arms) == 1 else None
    return annotation if isinstance(annotation, type) and issubclass(annotation, BaseModel) else None


def _holds_secret(annotation: Any, seen: tuple[type, ...] = ()) -> bool:
    if annotation is SecretStr:
        return True
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation not in seen and any(
            _holds_secret(f.annotation, (*seen, annotation)) for f in annotation.model_fields.values())
    return any(_holds_secret(a, seen) for a in typing.get_args(annotation))


@functools.lru_cache(maxsize=None)
def secret_paths(model: type[BaseModel]) -> tuple[tuple[str, ...], ...]:
    """Every key path under `model` whose value holds a secret, by field name and by alias."""
    out: list[tuple[str, ...]] = []

    def walk(m: type[BaseModel], prefix: tuple[str, ...], seen: tuple[type, ...]) -> None:
        for name, field in m.model_fields.items():
            keys = {name} | {a for a in (field.alias, field.validation_alias) if isinstance(a, str)}
            sub = _model_of(field.annotation)
            for key in keys:
                if sub is not None and sub not in seen:
                    walk(sub, (*prefix, key), (*seen, sub))
                elif _holds_secret(field.annotation):
                    out.append((*prefix, key))

    walk(model, (), (model,))
    return tuple(out)


def at_secret(model: type[BaseModel] | None, loc: Iterable[Any]) -> bool:
    """Is a pydantic error at `loc` (relative to `model`) on a secret field, or inside one?"""
    parts = tuple(map(str, loc))
    return model is not None and any(parts[:len(p)] == p for p in secret_paths(model))


def secret_values(model: type[BaseModel] | None, data: Any) -> frozenset[str]:
    """Every value `data`, the input `model` validated, holds at a secret field — as text: a string,
    or a number's repr (a key typed as a number). A container held there gives everything inside.
    Never the literal "none", a proposer's "no key": masking it would hide every "none" around it.
    Linear in the distinct objects walked, so an alias-amplified YAML value stays cheap."""
    found: set[str] = set()
    seen: set[int] = set()

    def collect(node: Any, depth: int) -> None:
        if isinstance(node, str):
            found.add(node)
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            found.add(repr(node))
        elif isinstance(node, (Mapping, list, tuple)) and id(node) not in seen and depth < _WALK_DEPTH:
            seen.add(id(node))
            for child in node.values() if isinstance(node, Mapping) else node:
                collect(child, depth + 1)

    for path in secret_paths(model) if model is not None else ():
        node: Any = data
        for key in path:
            node = node.get(key) if isinstance(node, Mapping) else None
        collect(node, 0)
    return frozenset(v for v in found if v and v != "none")
