"""The Strata config copy: only checked keys, flags and paths pass, and nothing planted redirects its write."""
from __future__ import annotations

import json
import os
import stat

import pytest

from localharness.provider import server, strata
from localharness.provider.tuning import MODEL_CONFIG_CAP
from tests.unit.gguf_support import write_gguf
from tests.unit.launch_support import popen  # noqa: F401
from tests.unit.strata_fixture import CONFIG_NAME, strata_config, strata_record, strata_tree
from tests.unit.test_strata_activate import SENSED, SERVED, make_world, sensing, with_args  # noqa: F401


def refusal(what: str) -> str:
    return strata.CONFIG_REFUSAL.format(name=SERVED, what=what, engine_config=CONFIG_NAME)


@pytest.mark.parametrize("planned", [False, True])
async def test_an_env_key_refuses_the_launch_before_anything_starts(make_world, popen, sensing, planned):  # noqa: F811
    world = make_world(lambda shard: strata_config(shard, env={"X": "1"}))
    with pytest.raises(server.LaunchRefused, match="sets env"):
        await world.activate(autosense=SENSED if planned else None)
    assert popen == [] and not strata.copy_path(world.config_dir, 8083).exists()


async def test_a_clean_config_becomes_the_copy(make_world, popen):  # noqa: F811
    world = make_world(lambda shard: strata_config(shard, host="0.0.0.0", port=1, cwd="/", log="/x.log"))
    await world.activate(autosense=None)
    assert popen and world.copy["host"] == "127.0.0.1" and world.copy["port"] == 8083
    assert world.copy["cwd"] == str(world.engine) and world.copy["log"] == str(strata.log_path(world.config_dir, 8083))
    assert strata.copy_path(world.config_dir, 8083) == world.config_dir / "strata" / "8083.json"


async def test_one_refusal_names_every_key_and_flag(make_world, popen):  # noqa: F811
    world = make_world(lambda shard: with_args("--open", "-x")(shard) | {"mcp": {}, "api_key": "k", "env": {}})
    with pytest.raises(server.LaunchRefused) as refused:
        await world.activate(autosense=None)
    assert str(refused.value) == refusal("sets api_key, env, mcp; passes --open, -x")
    assert popen == []


def build(tmp_path, *, extra=(), header=None, **over):
    models = tmp_path / "models"
    models.mkdir(exist_ok=True)
    model = write_gguf(models / "m.gguf", {
        "general.architecture": "qwen4exp", **{f"general.sampling.{k}": v for k, v in (header or {}).items()}})
    config = strata_config(model, **over)
    config["args"] += list(extra)
    engine = strata_tree(tmp_path, config=config, model=model)
    return strata_record(engine, model, model), engine


def sensed(tmp_path, **kw):
    spec, engine = build(tmp_path, **kw)
    return strata.sense(spec), spec, engine


def copy_for(tmp_path, **kw):
    (found, why), spec, _ = sensed(tmp_path, **kw)
    assert why is None, why
    return strata.copy_of(found, spec, context=None, port=8083, config_dir=tmp_path)


OUTSIDE = "names {} outside the Strata install, its data and the model folder"


def test_a_path_under_the_install_its_data_or_the_model_folder_passes(tmp_path):
    engine, models = tmp_path / "Strata", tmp_path / "models"
    inside = [f"{engine}/mtp/rt", f"{tmp_path}/Strata-data/x", f"{models}/shard2.gguf", "mtp/rt"]
    for token in inside:
        assert sensed(tmp_path, extra=("--pack", token))[0][1] is None, token


@pytest.mark.parametrize("token", ["/etc/passwd", "../../x", "~/elsewhere/f.gguf"])
def test_a_path_outside_the_roots_refuses_naming_it(tmp_path, token):
    (_, why), *_ = sensed(tmp_path, extra=("--pack", token))
    assert why == refusal(OUTSIDE.format(token))


@pytest.mark.parametrize("token", ["ngram=/etc/passwd", "a=x,/etc/passwd", "k=v:/etc/passwd", "$HOME/x", "`id`", "%USERPROFILE%/x"])
def test_a_value_that_expands_or_hides_a_path_after_a_separator_is_outside(tmp_path, token):
    assert sensed(tmp_path, extra=("--pack", token))[0][1] == refusal(OUTSIDE.format(token))


def test_a_separated_value_whose_paths_sit_in_the_roots_passes(tmp_path):
    assert sensed(tmp_path, extra=("--pack", f"ngram=mtp/rt,{tmp_path}/Strata-data/x"))[0][1] is None


def test_a_sibling_whose_name_only_starts_with_the_root_is_outside(tmp_path):
    for token in ("../Strata-evil/x", "../Strata-dataX/x", "../models-evil/x"):
        assert sensed(tmp_path, extra=("--pack", token))[0][1] == refusal(OUTSIDE.format(token))


def test_a_model_folder_that_is_shared_is_refused_naming_it(tmp_path, monkeypatch):
    models = tmp_path / "models"
    sensed(tmp_path)
    SHARED = "has its model in {}, which is the root, your home or open to every user"
    models.chmod(0o777)
    assert sensed(tmp_path)[0][1] == refusal(SHARED.format(models))
    models.chmod(0o755)
    assert sensed(tmp_path)[0][1] is None
    monkeypatch.setenv("HOME", str(models))
    assert sensed(tmp_path)[0][1] == refusal(SHARED.format(models))


def test_a_nul_byte_in_an_argument_refuses_with_the_config_sentence(tmp_path):
    spec, _ = build(tmp_path)
    for extra in (("--pack", "/x\x00y"), ("--ple-gguf=a\x00",)):
        assert sensed(tmp_path, extra=extra)[0][1] == refusal("has a NUL byte in an argument")
    config = strata_config(str(spec.model) + "\x00")
    (tmp_path / "Strata" / CONFIG_NAME).write_text(json.dumps(config), "utf-8")
    assert strata.sense(spec)[1] == refusal("has a NUL byte in an argument")


def test_a_symlink_leading_out_of_the_install_is_outside(tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    _, _, engine = sensed(tmp_path)
    (engine / "mtp").symlink_to(outside)
    assert sensed(tmp_path, extra=("--pack", "mtp/rt"))[0][1] == refusal(OUTSIDE.format("mtp/rt"))


def test_a_native_path_through_a_symlink_in_the_model_folder_passes(tmp_path):
    spec, engine = build(tmp_path)
    link = tmp_path / "models" / "link.gguf"
    link.symlink_to(spec.model)
    config = strata_config(link)
    (engine / CONFIG_NAME).write_text(json.dumps(config), "utf-8")
    assert strata.sense(spec)[1] is None


def test_every_offending_path_is_named_in_args_order(tmp_path):
    (_, why), *_ = sensed(tmp_path, extra=("--pack", "/b/x", "--ple-gguf=/a/y"))
    assert why == refusal(OUTSIDE.format("/b/x, /a/y"))


@pytest.mark.parametrize("over,named", [
    ({"gpu": True}, "gpu"), ({"gpu": 1.0}, "gpu"), ({"gpu": -1}, "gpu"), ({"gpu": "1"}, "gpu"),
    ({"parallel": 0}, "parallel"), ({"parallel": "2"}, "parallel"), ({"parallel": True}, "parallel"),
    ({"reasoning_effort": "max"}, "reasoning_effort"), ({"reasoning_effort": 1}, "reasoning_effort"),
    ({"model_name": "a\nb"}, "model_name"), ({"model_name": 5}, "model_name"),
    ({"sampling": {"temp": [1]}}, "sampling"), ({"sampling": {"temp": True}}, "sampling"), ({"sampling": 1}, "sampling"),
])
def test_a_value_of_the_wrong_type_refuses_naming_its_key(tmp_path, over, named):
    (_, why), *_ = sensed(tmp_path, **over)
    assert why == refusal(f"has a bad value for {named}")


@pytest.mark.parametrize("extra", [("--max-context", "65k"), ("--max-context=1e3",), ("--max-context",),
                                   ("--max-context", "²"), ("--max-context", "1234567890"), ("--max-context", "0"),
                                   ("--max-context=000",),
                                   ("--max-context", "9" * 30)])
def test_a_max_context_that_is_not_decimal_digits_refuses_naming_the_flag(tmp_path, extra):
    spec, _ = build(tmp_path, args=[])
    config = strata_config(spec.model)
    config["args"] = [t for t in config["args"] if t not in ("--max-context", "65536")] + list(extra)
    (tmp_path / "Strata" / CONFIG_NAME).write_text(json.dumps(config), "utf-8")
    assert strata.sense(spec)[1] == refusal("has a bad value for --max-context")


def _with_kv(tmp_path, *kv):
    spec, _ = build(tmp_path, args=[])
    config = strata_config(spec.model)
    config["args"] = [t for t in config["args"] if t not in ("--kv", "q8_0")] + list(kv)
    (tmp_path / "Strata" / CONFIG_NAME).write_text(json.dumps(config), "utf-8")
    return strata.sense(spec)[1]


@pytest.mark.parametrize("kv,named", [(("--kv", "fp8"), "--kv fp8"), (("--kv=q9",), "--kv q9"), (("--kv",), "--kv")])
def test_an_unknown_or_missing_kv_value_refuses_naming_the_flag(tmp_path, kv, named):
    assert _with_kv(tmp_path, *kv) == refusal(f"has a bad value for {named}")


def test_a_known_kv_type_passes(tmp_path):
    assert _with_kv(tmp_path, "--kv", "Q4_0") is None


@pytest.mark.parametrize("flag,extra", [
    ("--max-context", ("--max-context", "4096")), ("--max-context", ("--max-context=4096",)),
    ("--native", ("--native", "other.gguf")), ("--kv", ("--kv", "q8_0", "--kv", "f16"))])
def test_a_repeated_max_context_native_or_kv_is_refused_naming_it(tmp_path, flag, extra):
    (_, why), *_ = sensed(tmp_path, extra=extra)
    assert why == refusal(f"repeats {flag}")


@pytest.mark.parametrize("token", ["https://host/x", "file:///etc/passwd", "s3:/bucket/x"])
def test_a_path_argument_with_a_url_scheme_is_outside_even_when_its_tail_would_sit_in_the_install(tmp_path, token):
    assert sensed(tmp_path, extra=("--pack", token))[0][1] == refusal(OUTSIDE.format(token))


def test_inside_is_false_when_the_paths_cannot_be_compared():
    assert strata.inside("relative/x", ("/abs",)) is False


def test_the_levels_counts_and_numbers_that_are_allowed_pass(tmp_path):
    for level in strata.EFFORT_VALUES:
        assert sensed(tmp_path, reasoning_effort=level, parallel=2, gpu=1)[0][1] is None
    assert sensed(tmp_path, header={"temp": 1, "sequence": "top_k;temp"})[0][1] is None


def test_a_single_dash_flag_is_named_and_a_negative_number_is_a_value(tmp_path):
    assert sensed(tmp_path, extra=("-x", "1"))[0][1] == refusal("passes -x")
    assert sensed(tmp_path, extra=("--prefill", "-1"))[0][1] is None


def test_everything_wrong_is_one_sentence_in_a_fixed_order(tmp_path):
    (_, why), *_ = sensed(
        tmp_path, extra=("-x", "--pack", "/z"), gpu=True, env={}, cors_origins=[])
    assert why == refusal(
        "sets cors_origins, env; passes -x; names /z outside the Strata install, its data and the model folder; "
        "has a bad value for gpu")


def test_the_header_fills_sampling_when_the_config_has_none(tmp_path):
    header = {"temp": 1.0, "top_k": 20, "top_p": 0.95}
    assert copy_for(tmp_path, header=header)["sampling"] == header


def test_an_empty_sampling_object_is_filled_and_a_set_one_is_kept(tmp_path):
    header = {"temp": 1.0, "top_k": 20}
    spec, engine = build(tmp_path, header=header)
    for given, wanted in (({}, header), ({"temp": 0.2}, {"temp": 0.2})):
        config = strata_config(spec.model, sampling=given)
        (engine / CONFIG_NAME).write_text(json.dumps(config), "utf-8")
        found, _ = strata.sense(spec)
        assert strata.copy_of(found, spec, context=None, port=8083, config_dir=tmp_path)["sampling"] == wanted


def test_a_header_without_sampling_keys_adds_none(tmp_path):
    assert "sampling" not in copy_for(tmp_path)


def test_a_config_of_exactly_the_cap_is_read_and_one_byte_more_is_refused(tmp_path):
    spec, engine = build(tmp_path)
    path = engine / CONFIG_NAME
    body = path.read_text("utf-8")
    path.write_text(body.ljust(MODEL_CONFIG_CAP), "utf-8")
    assert strata.sense(spec)[1] is None
    path.write_text(body.ljust(MODEL_CONFIG_CAP + 1), "utf-8")
    assert strata.sense(spec)[1] == refusal(f"cannot be read (over {MODEL_CONFIG_CAP} bytes)")


@pytest.mark.parametrize("config", [{}, {"args": []}])
def test_a_config_naming_no_model_refuses_with_the_native_sentence(tmp_path, config):
    spec, engine = build(tmp_path)
    (engine / CONFIG_NAME).write_text(json.dumps(config), "utf-8")
    assert strata.sense(spec)[1] == strata.NATIVE_MISMATCH.format(name=SERVED)


def test_the_native_sentence_wins_over_every_other_offence(tmp_path):
    spec, engine = build(tmp_path)
    (engine / CONFIG_NAME).write_text(json.dumps(strata_config("/else/x.gguf", env={}, gpu=True)), "utf-8")
    assert strata.sense(spec)[1] == strata.NATIVE_MISMATCH.format(name=SERVED)


def test_two_copies_of_the_same_config_are_byte_identical_and_keep_the_args_order(tmp_path):
    first, second = (strata.copy_text(copy_for(tmp_path, header={"temp": 1.0})) for _ in range(2))
    assert first == second
    args = json.loads(first)["args"]
    assert args[:1] == ["--native"] and args[2:4] == ["--max-context", "65536"]


COPY = {"model_name": SERVED, "args": ["--native", "m"], "port": 8083}


def test_a_symlink_planted_at_the_strata_directory_is_replaced_by_a_private_directory(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    (tmp_path / "strata").symlink_to(target)
    strata.write_copy(tmp_path, 8083, COPY)
    directory = tmp_path / "strata"
    assert not directory.is_symlink() and stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert list(target.iterdir()) == []


def test_a_symlink_planted_at_the_copy_is_replaced_and_its_target_untouched(tmp_path):
    directory = tmp_path / "strata"
    directory.mkdir()
    victim = tmp_path / "victim"
    victim.write_text("keep", "utf-8")
    (directory / "8083.json").symlink_to(victim)
    path = strata.write_copy(tmp_path, 8083, COPY)
    assert not path.is_symlink() and json.loads(path.read_text("utf-8")) == COPY
    assert victim.read_text("utf-8") == "keep"


def test_a_symlink_planted_at_the_log_is_replaced_by_a_private_regular_file(tmp_path):
    directory = tmp_path / "strata"
    directory.mkdir()
    victim = tmp_path / "victim"
    victim.write_text("keep", "utf-8")
    (directory / "8083.log").symlink_to(victim)
    strata.write_copy(tmp_path, 8083, COPY)
    log = strata.log_path(tmp_path, 8083)
    assert log.is_file() and not log.is_symlink() and stat.S_IMODE(log.stat().st_mode) == 0o600
    assert victim.read_text("utf-8") == "keep"


def test_a_hard_link_planted_at_the_log_is_replaced_and_its_other_name_untouched(tmp_path):
    directory = tmp_path / "strata"
    directory.mkdir()
    victim = tmp_path / "victim"
    victim.write_text("keep", "utf-8")
    os.link(victim, directory / "8083.log")
    strata.write_copy(tmp_path, 8083, COPY)
    log = strata.log_path(tmp_path, 8083)
    assert log.stat().st_nlink == 1 and log.read_text("utf-8") == ""
    assert victim.read_text("utf-8") == "keep" and stat.S_IMODE(log.stat().st_mode) == 0o600


def test_a_directory_planted_at_the_log_is_replaced_and_an_existing_log_is_emptied_and_closed(tmp_path):
    log = strata.log_path(tmp_path, 8083)
    log.mkdir(parents=True)
    (log / "x").write_text("x", "utf-8")
    strata.write_copy(tmp_path, 8083, COPY)
    assert log.is_file()
    log.write_text("earlier output", "utf-8")
    log.chmod(0o644)
    strata.write_copy(tmp_path, 8083, COPY)
    assert log.read_text("utf-8") == "" and stat.S_IMODE(log.stat().st_mode) == 0o600


def test_the_copy_is_private_byte_identical_across_writes_and_leaves_no_temporary(tmp_path):
    directory = tmp_path / "strata"
    directory.mkdir()
    (directory / ".8083.json.dead.tmp").write_text("{", "utf-8")
    path = strata.write_copy(tmp_path, 8083, COPY)
    first = path.read_bytes()
    strata.write_copy(tmp_path, 8083, COPY)
    assert path.read_bytes() == first and stat.S_IMODE(path.stat().st_mode) == 0o600
    assert sorted(p.name for p in directory.iterdir()) == ["8083.json", "8083.log"]


async def test_a_launch_never_rewrites_the_persons_config(make_world, popen):  # noqa: F811
    world = make_world()
    source = world.engine / CONFIG_NAME
    before = source.read_bytes()
    await world.activate(autosense=None)
    assert popen and source.read_bytes() == before
