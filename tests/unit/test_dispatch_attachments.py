"""G3 (Phase 49): typed artifacts go OUT as file replies; inbound attachments stay metadata only.

Outbound: an Observation carrying an ArtifactRef, while a message is being answered, is posted to
that conversation as one file — resolved only as `<id><suffix>` under core's
`artifact_root(state_dir, ref.plugin)`, one regular non-symlink file, mime on the core allowlist.
Anything else: one warning, nothing sent, the turn goes on. Driven through the REAL DiscordAdapter
on the recording fake `discord` (tests.dispatch_support), whose channel logs a file send as
`("send", c<id>, "")` followed by `("file", c<id>, <filename>)` and keeps the `File` in `fake.files`.

Inbound: the adapter normalises uploads to `(filename, size, content_type)`; picture uploads are
read (only once the allow gate admits the author) into image parts that `take_images()` hands the
REPL, and every picture that cannot be read is named in the text. Other files stay metadata only.
"""
from __future__ import annotations

import asyncio
import types

import pytest
from structlog.testing import capture_logs

from localharness.core.artifacts import artifact_root, write_artifact
from localharness.core.bus import EventBus
from localharness.core.content import MAX_IMAGE_BYTES
from localharness.core.events import ArtifactRef, Observation
from tests.dispatch_support import build_dispatch_discord, install_fake_discord, isolate_discord_env
from tests.unit.test_image_content import png_bytes

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


@pytest.fixture
def fake(monkeypatch, tmp_path):
    isolate_discord_env(monkeypatch, tmp_path)
    return install_fake_discord(monkeypatch)


async def _answering(fake, tmp_path, *, state_dir="default"):
    """A started channel that is answering one allowlisted message in conversation 7."""
    sd = tmp_path / "state" if state_dir == "default" else state_dir
    ch = build_dispatch_discord(EventBus(), allow={"42"}, channels=set(), ack="", state_dir=sd)
    await ch.start()
    await fake.deliver(fake.message(42, 7, "draw me a cat"))
    assert await ch.read_input() == "draw me a cat"
    return ch, sd


def _obs(ref=None) -> Observation:
    return Observation(agent_id="a", session_id="s", observation_type="tool_result",
                       tool_name="generate_image", output="ok", artifact=ref)


def _sends(fake) -> list:
    return [r for r in fake.log if r[0] in ("send", "file")]


async def test_a_real_artifact_is_posted_as_one_file_to_the_conversation(fake, tmp_path):
    ch, sd = await _answering(fake, tmp_path)
    ref = write_artifact(artifact_root(sd, "image"), "image", PNG, "image/png")
    path = artifact_root(sd, "image") / f"{ref.id}.png"
    await ch.on_observation(_obs(ref))
    assert _sends(fake) == [("send", "c7", ""), ("file", "c7", f"{ref.id}.png")], fake.log
    assert [f.fp for f in fake.files] == [str(path)]
    await ch.stop()


async def test_send_file_gets_the_conversation_path_and_mime(fake, tmp_path):
    ch, sd = await _answering(fake, tmp_path)
    ref = write_artifact(artifact_root(sd, "image"), "image", PNG, "image/png")
    calls = []

    async def send_file(conversation, path, mime):
        calls.append((conversation.id, path, mime))

    ch._adapter.send_file = send_file
    await ch.on_observation(_obs(ref))
    assert calls == [(7, artifact_root(sd, "image") / f"{ref.id}.png", "image/png")]
    await ch.stop()


def _missing(sd):
    return ArtifactRef(plugin="image", kind="image", id="art-20261002-120000-abcdef", mime="image/png")


def _symlink(sd):
    real = write_artifact(sd / "elsewhere", "image", PNG, "image/png")
    root = artifact_root(sd, "image")
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{real.id}.png").symlink_to(sd / "elsewhere" / f"{real.id}.png")
    return real


def _off_allowlist(sd):
    ref = write_artifact(artifact_root(sd, "image"), "image", PNG, "image/png")
    (artifact_root(sd, "image") / f"{ref.id}.gif").write_bytes(PNG)
    return ArtifactRef.model_construct(plugin="image", kind="image", id=ref.id, mime="image/gif")


def _wrong_suffix_dir(sd):
    """The stem names a directory, not a file."""
    ref = ArtifactRef(plugin="image", kind="image", id="art-20261002-120000-abc123", mime="image/png")
    (artifact_root(sd, "image") / f"{ref.id}.png").mkdir(parents=True)
    return ref


@pytest.mark.parametrize("make", [_missing, _symlink, _off_allowlist, _wrong_suffix_dir])
async def test_anything_but_one_regular_allowlisted_file_sends_nothing(fake, tmp_path, make):
    ch, sd = await _answering(fake, tmp_path)
    ref = make(sd)
    with capture_logs() as logs:
        await ch.on_observation(_obs(ref))
    assert _sends(fake) == [] and fake.files == [], fake.log
    assert [e["event"] for e in logs if e["log_level"] == "warning"] == ["discord_artifact_skipped"], logs
    await ch._send("the turn goes on")
    assert _sends(fake) == [("send", "c7", "the turn goes on")]
    await ch.stop()


async def test_no_state_dir_sends_nothing(fake, tmp_path):
    ch, _ = await _answering(fake, tmp_path, state_dir=None)
    sd = tmp_path / "state"
    ref = write_artifact(artifact_root(sd, "image"), "image", PNG, "image/png")
    with capture_logs() as logs:
        await ch.on_observation(_obs(ref))
    assert _sends(fake) == [] and [e["event"] for e in logs] == ["discord_artifact_skipped"]
    await ch.stop()


async def test_no_message_being_answered_sends_nothing(fake, tmp_path):
    sd = tmp_path / "state"
    ch = build_dispatch_discord(EventBus(), allow={"42"}, channels=set(), ack="", state_dir=sd)
    await ch.start()
    ref = write_artifact(artifact_root(sd, "image"), "image", PNG, "image/png")
    await ch.on_observation(_obs(ref))
    assert _sends(fake) == [] and fake.files == []
    await ch.stop()


async def test_an_observation_without_artifact_stays_silent(fake, tmp_path):
    ch, _ = await _answering(fake, tmp_path)
    await ch.on_observation(_obs(None))
    assert _sends(fake) == []
    await ch.stop()


async def test_a_failed_upload_is_logged_and_swallowed(fake, tmp_path):
    ch, sd = await _answering(fake, tmp_path)
    ref = write_artifact(artifact_root(sd, "image"), "image", PNG, "image/png")

    async def boom(*_):
        raise RuntimeError("413 payload too large")

    ch._adapter.send_file = boom
    with capture_logs() as logs:
        await ch.on_observation(_obs(ref))
    assert [e["event"] for e in logs if e["log_level"] == "error"] == ["discord_send_file_failed"]
    await ch.stop()


def _upload(name, data=b"", content_type=None, size=None):
    async def read():
        return data
    return types.SimpleNamespace(filename=name, size=len(data) if size is None else size,
                                 content_type=content_type, url=f"https://cdn.example/{name}", read=read)


async def _started(fake, **kw):
    ch = build_dispatch_discord(EventBus(), allow={"42"}, channels=set(), ack="", **kw)
    await ch.start()
    return ch


async def test_an_image_with_text_reaches_the_turn_as_a_part(fake, tmp_path):
    ch = await _started(fake)
    await fake.deliver(fake.message(42, 7, "what is this?",
                                    attachments=[_upload("shot.png", png_bytes(265, 51), "image/png")]))
    assert await asyncio.wait_for(ch.read_input(), 1) == "what is this?"
    [part] = ch.take_images()
    assert part["type"] == "image_url" and part["image_url"]["url"].startswith("data:image/png;base64,")
    assert (part["_lh"]["width"], part["_lh"]["height"], part["_lh"]["name"]) == (265, 51, "shot.png")
    assert ch.take_images() == [], "take_images is consumed once"
    await ch.stop()


async def test_an_image_only_message_is_a_turn_named_by_its_label(fake, tmp_path):
    ch = await _started(fake)
    await fake.deliver(fake.message(42, 7, "", attachments=[_upload("shot.png", png_bytes(265, 51))]))
    assert await asyncio.wait_for(ch.read_input(), 1) == "[image: 265×51 png, shot.png]"
    assert len(ch.take_images()) == 1
    await ch.stop()


async def test_a_png_name_that_is_not_an_image_is_named_in_the_text(fake, tmp_path):
    ch = await _started(fake)
    await fake.deliver(fake.message(42, 7, "look", attachments=[_upload("fake.png", b"hello", "image/png")]))
    text = await asyncio.wait_for(ch.read_input(), 1)
    assert text.startswith("look\n[attachment fake.png not read: not a PNG, JPEG, GIF or WebP"), text
    assert ch.take_images() == []
    await ch.stop()


async def test_failed_download_oversize_and_over_cap_are_each_named(fake, tmp_path):
    async def boom():
        raise OSError("404")
    broken = _upload("gone.png", content_type="image/png")
    broken.read = boom
    shots = [_upload(f"s{i}.png", png_bytes(40, 40)) for i in range(5)]
    huge = _upload("huge.png", b"x", "image/png", size=MAX_IMAGE_BYTES + 1)
    ch = await _started(fake)
    await fake.deliver(fake.message(42, 7, "", attachments=[broken, huge, *shots]))
    text = await asyncio.wait_for(ch.read_input(), 1)
    assert text.splitlines() == [
        "[attachment gone.png not read: download failed (OSError)]",
        "[attachment huge.png not read: 20.0 MiB is over the 20 MiB limit]",
        "[attachment s4.png not read: only 4 images per message]"], text
    assert [p["_lh"]["name"] for p in ch.take_images()] == ["s0.png", "s1.png", "s2.png", "s3.png"]
    await ch.stop()


async def test_non_image_attachments_stay_metadata_only(fake, tmp_path):
    ch = build_dispatch_discord(EventBus(), allow={"42"}, channels=set(), ack="")
    seen = []
    gate = ch._on_message

    async def record(msg):
        seen.append(msg)
        await gate(msg)

    ch._on_message = record  # the adapter is handed this at connect
    await ch.start()
    upload = _upload("notes.pdf", b"%PDF-1.4", "application/pdf")
    await fake.deliver(fake.message(42, 7, "", attachments=[upload]))          # attachment only
    await fake.deliver(fake.message(42, 7, "what is this?", attachments=[upload]))
    assert [m.attachments for m in seen] == [(("notes.pdf", 8, "application/pdf"),)] * 2
    assert [m.load_images for m in seen] == [None, None]
    assert ch._queue.qsize() == 1, "the attachment-only message was enqueued"
    assert await asyncio.wait_for(ch.read_input(), 1) == "what is this?"
    assert ch.take_images() == []
    await ch.stop()


async def test_a_strangers_image_is_never_downloaded(fake, tmp_path):
    reads = []
    upload = _upload("shot.png", png_bytes(40, 40))
    orig = upload.read

    async def counting():
        reads.append(1)
        return await orig()
    upload.read = counting
    ch = await _started(fake)
    await fake.deliver(fake.message(99, 7, "", attachments=[upload]))
    assert reads == [] and ch._queue.empty()
    await ch.stop()
