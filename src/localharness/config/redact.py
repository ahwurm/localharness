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
list, a dict or a union arm (masked whole). Inside a list or dict of models only those models' own
secret fields are (R14): an endpoint's name or an MCP server's transport is never masked. Core
code: pydantic and the standard library only."""
from __future__ import annotations

import collections.abc
import functools
import re
import types
import typing
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import AliasChoices, AliasPath, BaseModel, SecretStr, ValidationError

SECRET_MASK = "**********"
_WALK_DEPTH = 8  # a secret field's value is a string; a container held there is walked this deep
_ANY = "*"  # in a secret path: any one item of a list, or any one value of a dict


def is_secret(annotation: Any) -> bool:
    """A SecretStr leaf, or a container (Optional, list, dict, tuple, union) whose arms hold one —
    masked whole. A model is not a leaf: its own fields decide (model_dump(mode="json") masks them)."""
    if annotation is SecretStr:
        return True
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return False
    return any(is_secret(a) for a in typing.get_args(annotation))


def reveal(node: Any) -> Any:
    """The raw values under `node` — SecretStr unwrapped, dicts and lists walked. Only for writing a
    machine's own config file (0600); never for anything shown."""
    if isinstance(node, SecretStr):
        return node.get_secret_value()
    if isinstance(node, Mapping):
        return {k: reveal(v) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [reveal(v) for v in node]
    return node


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


def _element_model(annotation: Any) -> type[BaseModel] | None:
    """The model each element holds for list[M], tuple[M, ...], Sequence[M] or dict[str, M]
    (Optional unwrapped); None for anything else."""
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        arms = [a for a in typing.get_args(annotation) if a is not type(None)]
        annotation = arms[0] if len(arms) == 1 else None
    origin, args = typing.get_origin(annotation), typing.get_args(annotation)
    if origin in (list, collections.abc.Sequence) and len(args) == 1:
        return _model_of(args[0])
    if origin is tuple and len(args) == 2 and args[1] is Ellipsis:
        return _model_of(args[0])
    if origin in (dict, collections.abc.Mapping) and len(args) == 2:
        return _model_of(args[1])
    return None


def holds_secret(annotation: Any, seen: tuple[type, ...] = ()) -> bool:
    """Does a value of this type hold a secret anywhere — a SecretStr, or a model with one, at any
    depth? A value typed as text for such a field cannot be told apart: it is shown whole as the mask."""
    if annotation is SecretStr:
        return True
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation not in seen and any(
            holds_secret(f.annotation, (*seen, annotation)) for f in annotation.model_fields.values())
    return any(holds_secret(a, seen) for a in typing.get_args(annotation))


def _alias_keys(alias: Any) -> set[str]:
    """The input keys an alias names: a str, each choice of an AliasChoices, and the first key of an
    AliasPath (the whole value under it is then treated as the secret)."""
    if isinstance(alias, str):
        return {alias}
    if isinstance(alias, AliasPath):
        return {alias.path[0]} if alias.path and isinstance(alias.path[0], str) else set()
    if isinstance(alias, AliasChoices):
        return set().union(*(_alias_keys(choice) for choice in alias.choices))
    return set()


@functools.lru_cache(maxsize=None)
def secret_paths(model: type[BaseModel]) -> tuple[tuple[str, ...], ...]:
    """Every key path under `model` whose value holds a secret, by field name and by alias. A list
    or dict of models is walked into (`_ANY` stands for any one of its items), so only its models'
    own secret fields are paths, never the whole list (R14)."""
    out: list[tuple[str, ...]] = []

    def walk(m: type[BaseModel], prefix: tuple[str, ...], seen: tuple[type, ...]) -> None:
        for name, field in m.model_fields.items():
            keys = {name} | _alias_keys(field.alias) | _alias_keys(field.validation_alias)
            sub, element = _model_of(field.annotation), _element_model(field.annotation)
            for key in keys:
                if sub is not None and sub not in seen:
                    walk(sub, (*prefix, key), (*seen, sub))
                elif element is not None and element not in seen:
                    walk(element, (*prefix, key, _ANY), (*seen, element))
                elif holds_secret(field.annotation):
                    out.append((*prefix, key))

    walk(model, (), (model,))
    return tuple(out)


def at_secret(model: type[BaseModel] | None, loc: Iterable[Any]) -> bool:
    """Is a pydantic error at `loc` (relative to `model`) on a secret field, or inside one? An `_ANY`
    in a path matches any one part of `loc` (a list index or a dict key)."""
    parts = tuple(map(str, loc))
    return model is not None and any(
        len(parts) >= len(p) and all(want in (_ANY, got) for want, got in zip(p, parts))
        for p in secret_paths(model))


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
        elif isinstance(node, SecretStr):  # data dumped from a model (mode="python") holds these
            found.add(node.get_secret_value())
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            found.add(repr(node))
        elif isinstance(node, (Mapping, list, tuple)) and id(node) not in seen and depth < _WALK_DEPTH:
            seen.add(id(node))
            for child in node.values() if isinstance(node, Mapping) else node:
                collect(child, depth + 1)

    def follow(node: Any, path: tuple[str, ...]) -> None:
        if not path:
            collect(node, 0)
        elif path[0] == _ANY:  # every item of a list, every value of a dict; anything else: none
            items = node.values() if isinstance(node, Mapping) else node if isinstance(node, (list, tuple)) else ()
            for item in items:
                follow(item, path[1:])
        elif isinstance(node, Mapping):
            follow(node.get(path[0]), path[1:])

    for path in secret_paths(model) if model is not None else ():
        follow(data, path)
    return frozenset(v for v in found if v and v != "none")


def yaml_problem(exc: BaseException) -> str:
    """What a YAML parse error says, without the source it quotes: the parser's context, where that
    began, and the problem ("while scanning a quoted scalar at line 9, column 12: found unexpected
    end of stream"), never the snippet and caret its marks print, which can show the head or tail
    of a key on the broken line (R16). The position matters when the problem is only noticed later,
    as an unclosed quote is at the end of the file."""
    context, problem = getattr(exc, "context", None), getattr(exc, "problem", None)
    mark = getattr(exc, "context_mark", None)
    if context and mark is not None:
        context = f"{context} at line {mark.line + 1}, column {mark.column + 1}:"
    text = " ".join(str(part) for part in (context, problem) if part)
    # An alias, anchor or tag name can be text copied from a secret: never quoted back.
    text = re.sub(r"((?:alias|anchor|tag(?: handle)?) )(?:'[^']*'|\"[^\"]*\")", r"\1'<name>'", text)
    return text or type(exc).__name__


def yaml_where(exc: BaseException) -> tuple[int, int]:
    """(line, column), 1-based, where a YAML parse error's problem is; (0, 0) when it has no mark."""
    mark = getattr(exc, "problem_mark", None)
    return (mark.line + 1, mark.column + 1) if mark else (0, 0)
