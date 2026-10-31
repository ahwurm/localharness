"""The task flow's renderer: a LINES row filled with a channel's levers and printed through notice()."""
from __future__ import annotations


def progress(flow, subject, state, commands, tail, levers):
    raise NotImplementedError
