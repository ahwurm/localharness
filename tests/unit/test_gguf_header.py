"""AUTO-03 (reader half): the bounded GGUF header reader (D-12, D-13, D-14, D-15).

The fixtures come from tests/unit/gguf_support.py; hostile files are built with `Raw` values and
hand-written counts. No test writes a 16 MiB file: the byte-budget case is a sparse file whose string
lengths are declared and seeked over, never stored."""
from __future__ import annotations

import os
import struct
import threading

import pytest

from localharness.agent import gguf_header
from localharness.agent.gguf_header import GgufHeaderError, read_header
from localharness.core.private_files import NotRegularFile
from tests.unit.gguf_support import Raw, gguf_bytes, long_string_array, write_gguf
from tests.unit.launch_support import record_opens

FIT_KEYS = {"block_count", "head_count_kv", "key_length", "context_length", "embedding_length", "head_count"}
QWEN = {
    "general.architecture": "qwen3",
    "qwen3.block_count": 48,
    "qwen3.attention.head_count_kv": 8,
    "qwen3.attention.key_length": 128,
    "qwen3.context_length": 262144,
    "qwen3.embedding_length": 5120,
    "qwen3.attention.head_count": 40,
}
QWEN_VALUES = {"block_count": 48, "head_count_kv": 8, "key_length": 128, "context_length": 262144,
               "embedding_length": 5120, "head_count": 40}


def _string_array_header(n: int, elem_type: int = 8) -> Raw:
    return Raw(9, struct.pack("<I", elem_type) + struct.pack("<Q", n))


# ----------------------------------------------------------------------------------- the happy path


def test_the_fits_five_keys_round_trip(tmp_path):
    path = write_gguf(tmp_path / "m.gguf", QWEN)
    assert read_header(path, FIT_KEYS) == QWEN_VALUES


def test_a_per_layer_head_count_kv_array_comes_back_as_a_list(tmp_path):
    path = write_gguf(tmp_path / "m.gguf", {**QWEN, "qwen3.attention.head_count_kv": [8, 8, 4]})
    assert read_header(path, {"head_count_kv"}) == {"head_count_kv": [8, 8, 4]}


def test_a_scalar_head_count_kv_comes_back_as_an_int(tmp_path):
    result = read_header(write_gguf(tmp_path / "m.gguf", QWEN), {"head_count_kv"})
    assert result == {"head_count_kv": 8}
    assert type(result["head_count_kv"]) is int


def test_the_architecture_after_its_keys_still_resolves_them(tmp_path):
    kv = {k: v for k, v in QWEN.items() if k != "general.architecture"}
    kv["general.architecture"] = "qwen3"
    assert read_header(write_gguf(tmp_path / "m.gguf", kv), FIT_KEYS) == QWEN_VALUES


def test_an_empty_want_returns_nothing_after_the_counts(tmp_path):
    path = write_gguf(tmp_path / "m.gguf", QWEN)
    with record_opens():
        assert read_header(path, set()) == {}


def test_a_file_with_no_pairs_returns_nothing(tmp_path):
    assert read_header(write_gguf(tmp_path / "m.gguf", {}), {"block_count"}) == {}


def test_a_non_ascii_key_and_value_round_trip(tmp_path):
    kv = {"general.architecture": "qwen3", "qwen3.name": "Qwén ✓"}
    assert read_header(write_gguf(tmp_path / "m.gguf", kv), {"name"}) == {"name": "Qwén ✓"}


def test_invalid_utf8_in_a_wanted_string_is_replaced_not_raised(tmp_path):
    kv = {"general.architecture": "qwen3", "qwen3.name": b"\xff\xfe"}
    result = read_header(write_gguf(tmp_path / "m.gguf", kv), {"name"})
    assert result == {"name": "��"}


def test_every_scalar_type_parses(tmp_path):
    kv = {"general.architecture": "qwen3",
          "qwen3.a_u8": Raw(0, struct.pack("<B", 200)), "qwen3.a_i8": Raw(1, struct.pack("<b", -3)),
          "qwen3.a_u16": Raw(2, struct.pack("<H", 60000)), "qwen3.a_i16": Raw(3, struct.pack("<h", -300)),
          "qwen3.a_f32": 0.5, "qwen3.a_bool": True, "qwen3.a_u64": 2**40,
          "qwen3.a_i64": Raw(11, struct.pack("<q", -2**40)), "qwen3.a_f64": Raw(12, struct.pack("<d", 1.25)),
          "qwen3.a_i32": -7}
    want = {k.split(".")[1] for k in kv if k != "general.architecture"}
    assert read_header(write_gguf(tmp_path / "m.gguf", kv), want) == {
        "a_u8": 200, "a_i8": -3, "a_u16": 60000, "a_i16": -300, "a_f32": 0.5, "a_bool": True,
        "a_u64": 2**40, "a_i64": -2**40, "a_f64": 1.25, "a_i32": -7}


def test_a_header_with_a_huge_token_array_before_the_keys_still_resolves(tmp_path):
    kv = {"general.architecture": "qwen3", "tokenizer.ggml.tokens": long_string_array(200_000),
          "qwen3.context_length": 262144}
    path = write_gguf(tmp_path / "m.gguf", kv)
    assert path.stat().st_size < gguf_header.HEADER_BYTE_BUDGET
    assert read_header(path, {"context_length"}) == {"context_length": 262144}


def test_a_key_under_another_prefix_is_ignored(tmp_path):
    kv = {"general.architecture": "qwen3", "other.block_count": 99}
    assert read_header(write_gguf(tmp_path / "m.gguf", kv), {"block_count"}) == {}


# --------------------------------------------------------------------------------- hostile headers


def test_a_bad_magic_is_refused(tmp_path):
    with pytest.raises(GgufHeaderError):
        read_header(write_gguf(tmp_path / "m.gguf", QWEN, magic=b"GGUX"), FIT_KEYS)


@pytest.mark.parametrize("version", [1, 4])
def test_an_unsupported_version_is_refused(tmp_path, version):
    with pytest.raises(GgufHeaderError):
        read_header(write_gguf(tmp_path / "m.gguf", QWEN, version=version), FIT_KEYS)


def test_version_two_is_read(tmp_path):
    assert read_header(write_gguf(tmp_path / "m.gguf", QWEN, version=2), FIT_KEYS) == QWEN_VALUES


def test_more_than_the_kv_cap_is_refused(tmp_path):
    with pytest.raises(GgufHeaderError):
        read_header(write_gguf(tmp_path / "m.gguf", {}, n_kv=gguf_header.MAX_KV + 1), {"block_count"})


def test_a_string_past_the_cap_is_refused(tmp_path):
    kv = {"general.architecture": Raw(8, struct.pack("<Q", gguf_header.MAX_STRING + 1))}
    with pytest.raises(GgufHeaderError):
        read_header(write_gguf(tmp_path / "m.gguf", kv), {"block_count"})


def test_an_array_declaring_too_many_elements_is_refused(tmp_path):
    kv = {"qwen3.block_count": _string_array_header(2_000_000)}
    with pytest.raises(GgufHeaderError):
        read_header(write_gguf(tmp_path / "m.gguf", kv), {"nothing"})


def test_a_header_past_the_byte_budget_is_refused_without_a_16_mib_file(tmp_path):
    # No 16 MiB file is written: the file is sparse and the reader seeks over the declared strings.
    path = tmp_path / "m.gguf"
    each, count = gguf_header.MAX_STRING, 18  # 18 MiB declared > the 16 MiB budget
    with open(path, "wb") as fh:
        fh.write(gguf_bytes({"tokenizer.ggml.tokens": _string_array_header(count)}))
        for _ in range(count):
            fh.write(struct.pack("<Q", each))
            fh.seek(each, os.SEEK_CUR)
        fh.truncate(fh.tell())
    assert path.stat().st_size > gguf_header.HEADER_BYTE_BUDGET
    with pytest.raises(GgufHeaderError):
        read_header(path, {"block_count"})


def test_a_truncated_file_is_refused(tmp_path):
    path = tmp_path / "m.gguf"
    path.write_bytes(gguf_bytes(QWEN)[:-7])
    with pytest.raises(GgufHeaderError):
        read_header(path, FIT_KEYS)


# ------------------------------------------------------------------------- what the file may be


def test_a_missing_path_raises_the_plain_oserror(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_header(tmp_path / "absent.gguf", FIT_KEYS)


def test_a_fifo_is_refused_without_blocking(tmp_path):
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    raised: list[BaseException] = []

    def run() -> None:
        try:
            read_header(fifo, FIT_KEYS)
        except BaseException as exc:  # noqa: BLE001 - the thread reports whatever came out
            raised.append(exc)

    with record_opens() as opened:
        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        worker.join(5)
    assert not worker.is_alive(), "read_header blocked on a FIFO"
    assert len(raised) == 1 and isinstance(raised[0], NotRegularFile)
    assert opened.count(str(fifo)) == 1  # the one O_NONBLOCK open the reader made, nothing else


def test_a_symlink_to_a_device_is_refused(tmp_path):
    os.symlink("/dev/null", tmp_path / "dev")
    with pytest.raises(NotRegularFile):
        read_header(tmp_path / "dev", FIT_KEYS)


def test_a_directory_is_refused(tmp_path):
    with pytest.raises(NotRegularFile):
        read_header(tmp_path, FIT_KEYS)


# ------------------------------------------------- budget, stop-early and error texts (Task 2)


class _CountingFile:
    """Wraps the file object read_header opens: records every read and seek and the furthest offset."""

    def __init__(self, inner) -> None:
        self._inner, self.reads, self.seeks, self.furthest = inner, 0, 0, 0

    def __enter__(self):
        self._inner.__enter__()
        return self

    def __exit__(self, *exc):
        return self._inner.__exit__(*exc)

    def read(self, n=-1):
        self.reads += 1
        data = self._inner.read(n)
        self.furthest = max(self.furthest, self._inner.tell())
        return data

    def seek(self, offset, whence=0):
        self.seeks += 1
        pos = self._inner.seek(offset, whence)
        self.furthest = max(self.furthest, pos)
        return pos


@pytest.fixture
def counted(monkeypatch):
    """The _CountingFile of each read_header call, in order."""
    files: list[_CountingFile] = []
    real = os.fdopen

    def fdopen(fd, mode="r", *a, **kw):
        files.append(_CountingFile(real(fd, mode, *a, **kw)))
        return files[-1]

    monkeypatch.setattr(os, "fdopen", fdopen)
    return files


def test_the_reader_stops_at_the_second_value_of_fifty_pairs(tmp_path, counted):
    head = {"general.architecture": "qwen3", "qwen3.context_length": 262144}
    rest = {f"pad.key_{i:02d}": i for i in range(48)}
    full = write_gguf(tmp_path / "full.gguf", {**head, **rest})
    assert read_header(full, {"context_length"}) == {"context_length": 262144}
    assert counted[0].furthest == len(gguf_bytes(head))  # not one byte of the other 48 pairs
    only_head = write_gguf(tmp_path / "head.gguf", head)
    assert read_header(only_head, {"context_length"}) == {"context_length": 262144}
    assert counted[0].reads == counted[1].reads  # the same reads as a file that ends after the 2nd pair


def test_an_unwanted_numeric_array_is_skipped_with_one_seek(tmp_path, counted):
    kv = {"general.architecture": "qwen3", "qwen3.rope.dimension_sections": [1, 2, 3, 4],
          "qwen3.block_count": 48}
    assert read_header(write_gguf(tmp_path / "m.gguf", kv), {"block_count"}) == {"block_count": 48}
    assert counted[0].seeks == 1


def test_an_unwanted_numeric_array_is_charged_to_the_budget(tmp_path, monkeypatch):
    big = Raw(9, struct.pack("<I", 4) + struct.pack("<Q", 1_000_000))
    path = write_gguf(tmp_path / "m.gguf", {"general.architecture": "qwen3", "tokenizer.ggml.token_type": big})
    monkeypatch.setattr(gguf_header, "HEADER_BYTE_BUDGET", 3_000_000)  # the 4,000,000 bytes skipped exceed it
    with pytest.raises(GgufHeaderError) as raised:
        read_header(path, {"block_count"})
    assert str(raised.value) == "header exceeds byte budget"


def test_the_same_array_within_the_budget_is_skipped_unread(tmp_path, counted):
    big = Raw(9, struct.pack("<I", 4) + struct.pack("<Q", 1_000_000))
    path = write_gguf(tmp_path / "m.gguf", {"general.architecture": "qwen3", "x.token_type": big})
    assert read_header(path, {"block_count"}) == {}  # a truncated array body is not read, so not noticed
    assert counted[0].seeks == 1


def test_a_wanted_string_array_comes_back_as_a_list(tmp_path):
    kv = {"general.architecture": "qwen3", "qwen3.layer_names": ["a", "bb", "Qwén"]}
    assert read_header(write_gguf(tmp_path / "m.gguf", kv), {"layer_names"}) == {
        "layer_names": ["a", "bb", "Qwén"]}


def _expect(tmp_path, data: bytes, message: str, want=("block_count",)) -> None:
    path = tmp_path / "bad.gguf"
    path.write_bytes(data)
    with pytest.raises(GgufHeaderError) as raised:
        read_header(path, set(want))
    assert str(raised.value) == message


def test_the_error_texts_are_exact(tmp_path):
    _expect(tmp_path, gguf_bytes(QWEN, magic=b"GGUX"), "bad magic")
    _expect(tmp_path, gguf_bytes(QWEN, version=1), "unsupported GGUF version 1")
    _expect(tmp_path, gguf_bytes(QWEN, version=4), "unsupported GGUF version 4")
    _expect(tmp_path, gguf_bytes({}, n_kv=4097), "too many KV pairs")
    _expect(tmp_path, gguf_bytes({"general.architecture": Raw(8, struct.pack("<Q", (1 << 20) + 1))}),
            "string too long")
    _expect(tmp_path, gguf_bytes({"general.architecture": "qwen3", "k.block_count": _string_array_header(2_000_000)}),
            "array too large")
    _expect(tmp_path, gguf_bytes(QWEN)[:-7], "truncated", want=FIT_KEYS)
    _expect(tmp_path, gguf_bytes({"general.architecture": Raw(99, b"")}), "unknown value type")


def test_an_array_of_an_unknown_element_type_is_refused(tmp_path):
    bad = Raw(9, struct.pack("<I", 9) + struct.pack("<Q", 1))  # an array of arrays
    _expect(tmp_path, gguf_bytes({"general.architecture": "qwen3", "qwen3.block_count": bad}),
            "unknown value type")


def test_a_string_key_past_the_cap_is_refused(tmp_path):
    data = gguf_bytes({}, n_kv=1) + struct.pack("<Q", (1 << 20) + 1)
    _expect(tmp_path, data, "string too long")


def test_a_key_under_neither_general_nor_the_architecture_is_ignored(tmp_path):
    kv = {"other.block_count": 99, "general.architecture": "qwen3", "qwen3.block_count": 48,
          "qwen3.other.block_count": 7}
    assert read_header(write_gguf(tmp_path / "m.gguf", kv), {"block_count"}) == {"block_count": 48}


# ---- D-47: a dotted wanted name is an exact key -------------------------------------------------------

SPLIT_WANT = {"split.no", "split.count", "clip.vision.projection_dim"}


def u16(n: int) -> Raw:
    return Raw(2, struct.pack("<H", n))


def test_dotted_names_are_exact_keys_whatever_the_architecture(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", {"general.architecture": "qwen4exp", "qwen4exp.embedding_length": 2560,
                                          "split.no": u16(0), "split.count": u16(2)})
    assert read_header(p, SPLIT_WANT) == {"split.no": 0, "split.count": 2}


def test_dotted_names_resolve_with_no_architecture_key_at_all(tmp_path):
    p = write_gguf(tmp_path / "shard2.gguf", {"split.no": u16(1), "split.count": u16(2)})
    assert read_header(p, SPLIT_WANT) == {"split.no": 1, "split.count": 2}


def test_a_dotted_name_before_the_architecture_key_is_still_found(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", {"clip.vision.projection_dim": 5120, "general.architecture": "clip",
                                          "general.type": "mmproj"})
    assert read_header(p, {"type", "clip.vision.projection_dim"}) == {"type": "mmproj",
                                                                       "clip.vision.projection_dim": 5120}


def test_a_mixed_want_returns_suffix_names_and_full_dotted_names(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", {"general.architecture": "qwen3", "general.type": "model",
                                          "qwen3.block_count": 48, "split.no": u16(0)})
    assert read_header(p, {"type", "block_count", "split.no"}) == {"type": "model", "block_count": 48,
                                                                    "split.no": 0}


def test_an_exact_name_is_not_a_suffix_match(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", {"general.architecture": "llama", "other.split.no": u16(3)})
    assert read_header(p, {"split.no"}) == {}


def test_an_absent_dotted_name_is_simply_missing(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", {"general.architecture": "llama", "llama.block_count": 8})
    assert read_header(p, {"split.no", "block_count"}) == {"block_count": 8}
