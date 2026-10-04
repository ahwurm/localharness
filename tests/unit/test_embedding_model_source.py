"""Memory's embedding model loads only from a vetted folder, by its absolute path (ruling R24).

Two library behaviours made the load a way to run code nobody chose. sentence-transformers resolves a
model NAME against a same-named folder in the current directory first (a cloned project can ship
`Qwen/Qwen3-Embedding-0.6B/`), and for any folder it imports the module classes the model's
`modules.json`, or a Router's config, names FROM that folder, without `trust_remote_code` (5.6;
deprecated, removed in v6). The harness now never hands the library a name: it finds the snapshot in
the local Hugging Face cache itself (downloading it there on a real miss), reads every class reference
the model's files make, and loads by absolute path only when each is the library's own. A model that
names code of its own is refused in one line, and memory's embedding leg is off for the session. A
relative folder is the config folder's, never the current folder's.

The in-process tests give the harness a recording stand-in for sentence-transformers, so what the
library would have been asked to load is the assertion; the vetting reads real files. The last test
runs the real library in a child process with a home of its own: a module the probe model names (it
only writes a marker file) runs when the library loads that folder itself, and not through the
harness."""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

from localharness.memory.plugin import _embedding_check as _REAL_EMBEDDING_CHECK  # before conftest's stub
from localharness.memory.resonance import ResonanceEngine, ResonanceUnavailable

MODEL = "lh-test/embedder"
DEFAULT = "Qwen/Qwen3-Embedding-0.6B"
COMMIT = "0" * 40
REFUSED = ("memory: the embedding model {} ships its own code; LocalHarness does not run it — choose a "
           "model without custom modules")
RELATIVE = ("memory: the embedding model {} is a relative path, read from the config folder as {} (never "
            "from the current folder)")
MISSING = ("memory: the embedding model {} is not on this machine yet — it downloads once from "
           "huggingface.co the first time memory needs it")
DEFAULT_MISSING = ("memory: the embedding model Qwen/Qwen3-Embedding-0.6B is not on this machine yet — it "
                   "downloads once from huggingface.co (about 1.2 GB) the first time memory needs it")
# The shipped default model's own module list, as its modules.json has it.
PURE = [{"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
        {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
        {"idx": 2, "name": "2", "path": "2_Normalize", "type": "sentence_transformers.models.Normalize"}]
PROBE = "modeling_probe.ProbeModule"  # a module the model ships itself
PROBE_CODE = "open('probe-ran', 'w').write('ran')\n\n\nclass ProbeModule:\n    pass\n"
ROUTER = "sentence_transformers.models.Router"


def _router(types_: dict) -> dict:
    return {"types": types_, "structure": {"query": list(types_)}, "parameters": {}}


def _write(folder: Path, files: dict) -> Path:
    for name, body in files.items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body if isinstance(body, str) else json.dumps(body))
    return folder


def _cached(hub: Path, repo: str, files: dict) -> Path:
    """`files` as the Hugging Face cache holds a download: refs/main naming the snapshot folder."""
    home = hub / f"models--{repo.replace('/', '--')}"
    (home / "refs").mkdir(parents=True)
    (home / "refs" / "main").write_text(COMMIT)
    return _write(home / "snapshots" / COMMIT, files)


@pytest.fixture(autouse=True)
def hub(tmp_path, monkeypatch) -> Path:
    """An empty Hugging Face cache of the test's own."""
    from huggingface_hub import constants

    path = tmp_path / "hub"
    path.mkdir()
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(path))
    return path


@pytest.fixture(autouse=True)
def downloads(tmp_path, monkeypatch) -> list[str]:
    """Each repo the hub was asked for. The (fake) download leaves a pure model in a folder of its own,
    so no test here can reach the network."""
    asked: list[str] = []

    def snapshot_download(repo_id, *args, **kwargs):
        asked.append(repo_id)
        return str(_write(tmp_path / "fetched", {"modules.json": PURE}))

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    return asked


@pytest.fixture
def loads(monkeypatch) -> list[str]:
    """What a stand-in sentence-transformers is asked to load, in order."""
    asked: list[str] = []

    class SentenceTransformer:
        prompts: dict = {}

        def __init__(self, name, **kwargs):
            asked.append(name)

        def encode(self, texts, **kwargs):
            return [[0.6, 0.8] for _ in texts]

    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        types.SimpleNamespace(SentenceTransformer=SentenceTransformer))
    return asked


@pytest.fixture
def said():
    """Every line the engine logs at WARNING, through a handler of its own (an earlier test may have
    routed memory's logs to a file, away from the root logger)."""
    lines: list[str] = []
    handler = logging.Handler(logging.WARNING)
    handler.emit = lambda record: lines.append(record.getMessage())
    logger = logging.getLogger("localharness.memory.resonance")
    logger.addHandler(handler)
    yield lines
    logger.removeHandler(handler)


# --------------------------------------------------------------------------- what a model's files name

SHIPS_CODE = {
    "its module list names its own module": {
        "modules.json": [PURE[0] | {"type": PROBE}, *PURE[1:]], "modeling_probe.py": PROBE_CODE},
    "a module type that is not a name": {"modules.json": [PURE[0] | {"type": None}, *PURE[1:]]},
    "a Router names its own module": {
        "modules.json": [PURE[0] | {"type": ROUTER}],
        "router_config.json": _router({"query_0_Transformer": "sentence_transformers.models.Transformer",
                                       "document_1_Probe": PROBE})},
    "a Router inside a Router names its own module": {
        "modules.json": [PURE[0] | {"type": ROUTER}],
        "router_config.json": _router({"inner": ROUTER}),
        "inner/router_config.json": _router({"probe": PROBE})},
    "a Router's older config.json names its own module": {
        "modules.json": [PURE[0] | {"path": "0_Router", "type": ROUTER}],
        "0_Router/config.json": _router({"probe": PROBE})},
    "an Asym (older libraries) names its own module": {
        "modules.json": [PURE[0] | {"type": "sentence_transformers.models.Asym"}],
        "asym_config.json": _router({"probe": PROBE})},
    "a Dense module's activation is not torch's": {
        "modules.json": [*PURE, {"idx": 3, "name": "3", "path": "3_Dense",
                                 "type": "sentence_transformers.models.Dense"}],
        "3_Dense/config.json": {"in_features": 8, "out_features": 4,
                                "activation_function": "modeling_probe.Activation"}},
    "a WordEmbeddings tokenizer is not the library's": {
        "modules.json": [PURE[0] | {"path": "0_WordEmbeddings",
                                    "type": "sentence_transformers.models.WordEmbeddings"}],
        "0_WordEmbeddings/wordembedding_config.json": {"tokenizer_class": "modeling_probe.Tokenizer"}},
    "its config maps a class to its own code": {
        "modules.json": PURE, "config.json": {"model_type": "bert",
                                              "auto_map": {"AutoModel": "modeling_probe.ProbeModel"}}},
    "its tokenizer config maps a class to its own code": {
        "modules.json": PURE, "tokenizer_config.json": {"auto_map": {"AutoTokenizer": [PROBE, None]}}},
    "a sub-module's config maps a class to its own code": {
        "modules.json": [PURE[0] | {"type": ROUTER}],
        "router_config.json": _router({"query_0_Transformer": "sentence_transformers.models.Transformer"}),
        "query_0_Transformer/config.json": {"auto_map": {"AutoModel": "modeling_probe.ProbeModel"}}},
    "a module config turns trust_remote_code on": {
        "modules.json": PURE, "sentence_bert_config.json": {"model_args": {"trust_remote_code": True}}},
}


@pytest.mark.parametrize("files", SHIPS_CODE.values(), ids=SHIPS_CODE.keys())
def test_a_model_that_names_its_own_code_is_refused_in_one_line_and_never_loaded(files, hub, loads,
                                                                                 downloads, said):
    _cached(hub, MODEL, files)
    engine = ResonanceEngine(MODEL)

    for _ in range(2):  # the second use is refused too, without a second line
        with pytest.raises(ResonanceUnavailable) as refused:
            engine.embed_query("x")
        assert str(refused.value) == REFUSED.format(MODEL)

    assert loads == [] and downloads == []
    assert said == [REFUSED.format(MODEL)]


LIBRARY_ONLY = {
    "the shipped model's files": {
        "modules.json": PURE, "1_Pooling/config.json": {"pooling_mode_lasttoken": True},
        "config.json": {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"},
        "tokenizer_config.json": {"tokenizer_class": "Qwen2Tokenizer"},
        "config_sentence_transformers.json": {"prompts": {"query": "Instruct: ...\nQuery:"}}},
    "no module list (a transformers model, pooled by the library)": {"config.json": {"model_type": "bert"}},
    "a Router of the library's modules, one inside another": {
        "modules.json": [PURE[0] | {"type": ROUTER}],
        "router_config.json": _router({"query_0_Transformer": "sentence_transformers.models.Transformer",
                                       "inner": ROUTER}),
        "inner/router_config.json": _router({"pool": "sentence_transformers.models.Pooling"})},
    "a Dense module with torch's activation": {
        "modules.json": [*PURE, {"idx": 3, "name": "3", "path": "3_Dense",
                                 "type": "sentence_transformers.models.Dense"}],
        "3_Dense/config.json": {"in_features": 8, "out_features": 4,
                                "activation_function": "torch.nn.modules.activation.Tanh"}},
    "a WordEmbeddings module with the library's tokenizer": {
        "modules.json": [PURE[0] | {"path": "0_WordEmbeddings",
                                    "type": "sentence_transformers.models.WordEmbeddings"}],
        "0_WordEmbeddings/wordembedding_config.json": {
            "tokenizer_class": "sentence_transformers.models.tokenizer.WhitespaceTokenizer.WhitespaceTokenizer"}},
    "a model config's own activation setting": {
        "modules.json": PURE, "config.json": {"model_type": "gpt2", "activation_function": "gelu_new"}},
    "trust_remote_code switched off": {
        "modules.json": PURE, "sentence_bert_config.json": {"model_args": {"trust_remote_code": False}}},
}


@pytest.mark.parametrize("files", LIBRARY_ONLY.values(), ids=LIBRARY_ONLY.keys())
def test_a_model_of_the_librarys_own_modules_loads_by_its_absolute_path(files, hub, loads, downloads,
                                                                        said):
    snapshot = _cached(hub, MODEL, files)

    ResonanceEngine(MODEL).embed_query("x")

    assert loads == [str(snapshot)] and os.path.isabs(loads[0])
    assert downloads == [] and said == []


# --------------------------------------------------------------------------- where a model is looked for


@pytest.mark.parametrize("cached", [True, False], ids=["cached", "not-cached"])
def test_a_same_named_folder_in_the_current_directory_is_never_consulted(cached, tmp_path, monkeypatch,
                                                                         hub, loads, downloads, said):
    """A cloned project can ship a folder named like the model: the library would load it before the
    cache, and the harness itself used to look there first (`os.path.exists(model)`)."""
    monkeypatch.chdir(_write(tmp_path / "project", {f"{MODEL}/modules.json": [PURE[0] | {"type": PROBE}],
                                                    f"{MODEL}/modeling_probe.py": PROBE_CODE}))
    snapshot = _cached(hub, MODEL, {"modules.json": PURE}) if cached else tmp_path / "fetched"

    ResonanceEngine(MODEL).embed_query("x")

    assert loads == [str(snapshot)]
    assert (downloads, said) == (([], []) if cached else ([MODEL], [MISSING.format(MODEL)]))


@pytest.mark.parametrize("cached", [True, False], ids=["cached", "not-cached"])
@pytest.mark.parametrize("name, repo", [("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2"),
                                        ("bert-base-uncased", "bert-base-uncased")])
def test_a_name_with_no_owner_is_the_repo_the_library_would_fetch(name, repo, cached, tmp_path,
                                                                   monkeypatch, hub, downloads):
    """Handed a name, the library itself fetched `sentence-transformers/<name>` (but kept the original
    transformers models' names); the harness, which no longer hands it a name, resolves it the same."""
    sentence_transformers = pytest.importorskip("sentence_transformers")
    handed: list[str] = []

    class Recording:
        default_huggingface_organization = (
            sentence_transformers.SentenceTransformer.default_huggingface_organization)
        prompts: dict = {}

        def __init__(self, folder, **kwargs):
            handed.append(folder)

        def encode(self, texts, **kwargs):
            return [[0.6, 0.8] for _ in texts]

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", Recording)
    snapshot = _cached(hub, repo, {"modules.json": PURE}) if cached else tmp_path / "fetched"

    ResonanceEngine(name).embed_query("x")

    assert handed == [str(snapshot)] and downloads == ([] if cached else [repo])


@pytest.mark.parametrize("named", ["explicitly", "by-LOCALHARNESS_DIR"])
def test_a_relative_folder_is_read_from_the_config_folder_never_the_current_one(named, tmp_path,
                                                                               monkeypatch, loads,
                                                                               downloads, said):
    config = _write(tmp_path / "config", {"models/embedder/modules.json": PURE})
    monkeypatch.chdir(_write(tmp_path / "project",
                             {"models/embedder/modules.json": [PURE[0] | {"type": PROBE}]}))
    if named == "by-LOCALHARNESS_DIR":
        monkeypatch.setenv("LOCALHARNESS_DIR", str(config))

    ResonanceEngine("models/embedder", **({"config_dir": config} if named == "explicitly" else {})
                    ).embed_query("x")

    folder = config / "models" / "embedder"
    assert loads == [str(folder)] and downloads == []
    assert said == [RELATIVE.format("models/embedder", folder)]


def test_a_relative_path_the_config_folder_does_not_hold_is_a_model_id(tmp_path, monkeypatch, loads,
                                                                       downloads, said):
    config = _write(tmp_path / "config", {})
    monkeypatch.chdir(_write(tmp_path / "project",
                             {"models/embedder/modules.json": [PURE[0] | {"type": PROBE}]}))

    ResonanceEngine("models/embedder", config_dir=config).embed_query("x")

    assert downloads == ["models/embedder"] and loads == [str(tmp_path / "fetched")]


async def test_the_memory_plugin_reads_a_relative_folder_from_its_own_config_folder(tmp_path, monkeypatch,
                                                                                   loads, downloads):
    from localharness.memory.plugin import MemoryPlugin
    from tests.unit.test_memory_plugin import _ctx

    ctx = _ctx(tmp_path, embedding_model="models/embedder")  # its config folder: tmp_path / "global"
    _write(tmp_path / "global", {"models/embedder/modules.json": PURE})
    monkeypatch.chdir(_write(tmp_path / "project",
                             {"models/embedder/modules.json": [PURE[0] | {"type": PROBE}]}))
    plugin = MemoryPlugin()
    await plugin.tools(ctx)

    plugin._engine.embed_query("x")

    assert loads == [str(tmp_path / "global" / "models" / "embedder")] and downloads == []


def test_a_folder_given_by_absolute_path_is_vetted_the_same_way(tmp_path, loads, downloads, said):
    folder = _write(tmp_path / "probe-model", SHIPS_CODE["its module list names its own module"])

    with pytest.raises(ResonanceUnavailable) as refused:
        ResonanceEngine(str(folder)).embed_query("x")

    assert str(refused.value) == REFUSED.format(folder)
    assert loads == [] and downloads == [] and said == [REFUSED.format(folder)]


def test_a_folder_that_is_not_there_is_an_error_never_a_download(tmp_path, loads, downloads):
    with pytest.raises(ResonanceUnavailable, match="no such folder"):
        ResonanceEngine(str(tmp_path / "nowhere")).embed_query("x")

    assert loads == [] and downloads == []


def test_a_downloaded_model_is_vetted_before_it_loads(tmp_path, monkeypatch, loads, said):
    asked: list[str] = []

    def snapshot_download(repo_id, *args, **kwargs):
        asked.append(repo_id)
        return str(_write(tmp_path / "fetched", SHIPS_CODE["a Router names its own module"]))

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)

    with pytest.raises(ResonanceUnavailable):
        ResonanceEngine(MODEL).embed_query("x")

    assert asked == [MODEL] and loads == []
    assert said == [MISSING.format(MODEL), REFUSED.format(MODEL)]


def test_the_start_notice_and_the_setup_step_never_read_the_current_folder(tmp_path, monkeypatch):
    """A folder named like the model in the current folder used to read as "a local model": no start
    line before the download, and nothing for the setup step to fetch."""
    from localharness.memory import plugin as memory_plugin
    from localharness.provider import server
    from tests.unit.test_memory_setup_step import _ctx
    from tests.unit.test_startup_egress import _spec_found_for

    monkeypatch.setattr(memory_plugin, "_embedding_check", _REAL_EMBEDDING_CHECK)
    monkeypatch.setattr(memory_plugin, "_embedding_package_installed", lambda: True)
    monkeypatch.setattr("importlib.util.find_spec", _spec_found_for("sentence_transformers"))
    fetched: list[str] = []
    monkeypatch.setattr(server, "download_model", lambda repo_id: fetched.append(repo_id) or "/cache")
    monkeypatch.chdir(_write(tmp_path / "project", {f"{DEFAULT}/modules.json": PURE}))
    config = _write(tmp_path / "config", {})

    assert memory_plugin._embedding_download_notice(DEFAULT, config) == DEFAULT_MISSING
    memory_plugin.MemoryPlugin().setup_action(_ctx(config))
    assert fetched == [DEFAULT]


# --------------------------------------------------------------------------- the real library, end to end

CHILD = r"""
import json, sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

from localharness.memory.resonance import ResonanceEngine, ResonanceUnavailable

given, out = json.loads(sys.argv[1]), {}
try:
    ResonanceEngine(given["probe"]).embed_query("hello memory")
    out["probe"] = "loaded"
except ResonanceUnavailable as exc:
    out["probe"] = str(exc)
out["probe ran"] = Path(given["probe marker"]).exists()
out["tiny"] = len(ResonanceEngine(given["tiny"]).embed_query("hello memory"))
out["decoy ran"] = Path(given["decoy marker"]).exists()
for name, marker in ((given["probe snapshot"], "probe marker"), (given["tiny"], "decoy marker")):
    try:  # the library itself, handed the folder or the name
        SentenceTransformer(name, device="cpu")
    except Exception:
        pass
    out[f"{marker} by the library"] = Path(given[marker]).exists()
print(json.dumps(out))
"""


def test_through_the_real_library_a_models_own_code_never_runs_and_the_current_folder_is_not_read(
        tmp_path):
    """The ruling's probe, run for real in a child process whose home, Hugging Face home and config
    folder are the test's own (the library copies a module it imports into the Hugging Face home)."""
    pytest.importorskip("sentence_transformers")
    from tests.unit.test_startup_egress import _tiny_model_in_the_cache

    hub = tmp_path / "hf" / "hub"
    _tiny_model_in_the_cache(hub, "lh-test/tiny", tmp_path)
    tiny = hub / "models--lh-test--tiny" / "snapshots" / COMMIT
    markers = {name: tmp_path / f"{name}-ran" for name in ("probe", "decoy")}

    def with_probe(dest: Path, marker: Path) -> Path:
        shutil.copytree(tiny, dest)
        modules = json.loads((dest / "modules.json").read_text())
        (dest / "modules.json").write_text(json.dumps([modules[0] | {"type": PROBE}, *modules[1:]]))
        (dest / "modeling_probe.py").write_text(PROBE_CODE.replace("'probe-ran'", repr(str(marker))))
        return dest

    (hub / "models--lh-test--probe" / "refs").mkdir(parents=True)
    (hub / "models--lh-test--probe" / "refs" / "main").write_text(COMMIT)
    probe = with_probe(hub / "models--lh-test--probe" / "snapshots" / COMMIT, markers["probe"])
    project = tmp_path / "project"
    with_probe(project / "lh-test" / "tiny", markers["decoy"])
    env = {**os.environ, "HOME": str(tmp_path / "home"), "HF_HOME": str(tmp_path / "hf"),
           "HF_HUB_CACHE": str(hub), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "LOCALHARNESS_DIR": str(tmp_path / "config")}
    given = {"probe": "lh-test/probe", "tiny": "lh-test/tiny", "probe snapshot": str(probe),
             "probe marker": str(markers["probe"]), "decoy marker": str(markers["decoy"])}

    run = subprocess.run([sys.executable, "-c", CHILD, json.dumps(given)], cwd=project, env=env,
                         capture_output=True, text=True, timeout=300)

    assert run.returncode == 0, run.stderr[-3000:]
    assert json.loads(run.stdout.strip().splitlines()[-1]) == {
        "probe": REFUSED.format("lh-test/probe"), "probe ran": False,
        "tiny": 8, "decoy ran": False,
        # The controls: the same folder and the same name, handed to the library directly, run the code.
        "probe marker by the library": True, "decoy marker by the library": True}
