"""The three error texts state the fact and the one sanctioned next step, pinned verbatim.

* ``bash_exec`` timeout: the retry sentence appears (on ``error``, which is what the model
  reads for a failed call, and on ``output``) only when a longer timeout is possible.
* ``agent`` not-wired refusal: two lines, the closest roster names then the searched directories.
* ingest-via-exec refusal: one line that still blocks.
"""
from __future__ import annotations

import pytest

from localharness.tools.builtin.bash_tool import (
    BASH_TIMEOUT_MAX_S,
    BashExecTool,
    _timeout_result,
)

_RETRY = "retry with timeout_s up to 300 if the command needs longer."


# --- bash_exec timeout ----------------------------------------------------

def test_timeout_below_the_maximum_carries_the_retry_sentence_on_both_fields():
    result = _timeout_result(10, "sleep 99")
    assert result.output == f"Command timed out after 10s: sleep 99\n{_RETRY}"
    assert result.error == f"Timeout after 10s\n{_RETRY}"
    assert result.success is False
    assert result.error_type == "timeout_error"
    assert result.output.count(_RETRY) == 1
    assert result.error.count(_RETRY) == 1
    assert result.error.count("\n") == 1


def test_timeout_at_the_maximum_carries_the_fact_alone():
    result = _timeout_result(300.0, "x")
    assert result.error == "Timeout after 300.0s"
    assert result.output == "Command timed out after 300.0s: x"
    assert result.error_type == "timeout_error"


def test_timeout_s_500_clamps_to_the_maximum_and_gets_the_fact_alone():
    assert _timeout_result(min(500, BASH_TIMEOUT_MAX_S), "x") == _timeout_result(300.0, "x")


def test_the_schema_maximum_is_the_one_constant():
    props = BashExecTool().info().parameters["properties"]
    assert props["timeout_s"]["maximum"] == BASH_TIMEOUT_MAX_S == 300.0


async def test_a_real_timeout_tells_the_model_it_may_retry_longer():
    result = await BashExecTool().run(command="sleep 5", timeout_s=1)
    assert result.success is False
    assert result.error == f"Timeout after 1s\n{_RETRY}"
    assert result.output.endswith(_RETRY)
