"""serve_command places each derived flag at the record field's own slot (spec 20 §6.0, D-08)."""
from __future__ import annotations

from localharness.agent.gguf_header import read_header
from localharness.config.models import AutosenseConfig, ManagedServerConfig
from localharness.provider import server
from localharness.provider.autosense import FIT_KEYS, fit_header
from localharness.provider.tuning import derive
from tests.unit.gguf_support import write_gguf
from tests.unit.test_tuning_derive import SWIFT

TUNED_FLAGS = ["--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--spec-type", "draft-mtp",
               "--spec-draft-p-min", "0.6"]


def _llama(**over) -> ManagedServerConfig:
    return ManagedServerConfig(**{"runtime": "llamacpp", "binary": "/x/llama-server",
                                  "model": "/x/m.gguf", "port": 8080, **over})


def test_the_derived_flags_sit_after_mmproj_and_before_the_window():
    cmd = server.serve_command(_llama(mmproj="/x/p.gguf"), cache_type="q8_0", spec_type="draft-mtp",
                               draft_p_min=0.6, context=100000)
    at = cmd.index("--mmproj") + 2
    assert cmd[at:at + 8] == TUNED_FLAGS and cmd[at + 8:at + 10] == ["-c", "100000"]


def test_without_mmproj_they_follow_the_alias():
    cmd = server.serve_command(_llama(served_name="m"), cache_type="q8_0", spec_type="draft-mtp",
                               draft_p_min=0.6, context=100000)
    at = cmd.index("-a") + 2
    assert cmd[at:at + 8] == TUNED_FLAGS and cmd[at + 8] == "-c"


def test_the_record_fields_win_over_the_keywords():
    cmd = server.serve_command(_llama(cache_type_k="q4_0", cache_type_v="q4_0", speculative="off"),
                               cache_type="q8_0", spec_type="draft-mtp", draft_p_min=0.6)
    assert cmd[cmd.index("--cache-type-k") + 1] == "q4_0" and cmd[cmd.index("--cache-type-v") + 1] == "q4_0"
    assert "--spec-type" not in cmd and "--spec-draft-p-min" not in cmd


def test_swift_gguf_to_tuned_argv(tmp_path):
    gguf = write_gguf(tmp_path / "m.gguf", SWIFT)
    header = fit_header(read_header(gguf, FIT_KEYS))
    spec = _llama(model=str(gguf))
    tuned = derive(spec, header, AutosenseConfig())
    cmd = server.serve_command(spec, cache_type=tuned.cache_type, spec_type=tuned.spec_type,
                               draft_p_min=tuned.draft_p_min, context=100000)
    for flag, value in zip(TUNED_FLAGS[::2], TUNED_FLAGS[1::2]):
        assert cmd.count(flag) == 1 and cmd[cmd.index(flag) + 1] == value
    positions = [cmd.index(flag) for flag in TUNED_FLAGS[::2]] + [cmd.index("-c")]
    assert positions == sorted(positions)
