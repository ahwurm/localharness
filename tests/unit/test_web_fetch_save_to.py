"""web_fetch(save_to): the whole page lands in the project's downloads folder through no symlink, and a receipt replaces the window."""
from __future__ import annotations

import os
import stat

import httpx
import pytest

from localharness.agent.context import ContentStore
from localharness.tools.builtin import web_tool
from localharness.tools.builtin.web_tool import WebFetchTool, WebPageQueryTool, download_notice
from tests.unit.test_web_fetch_guard import _serve, _text, public_web

BIG = "\n".join(f"line {i} of the saved page" for i in range(2000))


@pytest.fixture
def project(tmp_path, monkeypatch):
    public_web(monkeypatch)
    root = (tmp_path / "proj").resolve()
    root.mkdir()
    return root


def fetch(project, save_to="x.md", url="http://example.com/page", **kw):
    return WebFetchTool(ContentStore(), boundary=project).run(url=url, save_to=save_to, **kw)


def downloads(project):
    return project / ".localharness" / "downloads"


async def test_a_page_past_the_window_is_saved_whole_without_a_cap_notice(project, monkeypatch):
    _serve(monkeypatch, lambda request: _text(BIG))
    res = await fetch(project)
    assert res.success
    assert (downloads(project) / "x.md").read_text() == BIG
    assert "download capped" not in (downloads(project) / "x.md").read_text()
    lines = res.output.splitlines()
    assert lines[1].startswith(f"saved {len(BIG)} chars to .localharness/downloads/x.md (fetch_id=")
    first, last = (next(line for line in lines if line.startswith(p)) for p in ("first:", "last:"))
    assert len(first) <= len("first: ") + 200 and len(last) <= len("last: ") + 200
    assert "line 1999 of the saved page" in last
    assert res.output.startswith(web_tool.UNTRUSTED_BANNER)
    assert "PREFIX ONLY" not in res.output and "replaced" not in res.output


async def test_the_saved_page_stays_queryable_by_its_fetch_id(project, monkeypatch):
    _serve(monkeypatch, lambda request: _text(BIG))
    store = ContentStore()
    res = await WebFetchTool(store, boundary=project).run(url="http://example.com/p", save_to="q.md")
    hit = await WebPageQueryTool(store).run(fetch_id=res.metadata["fetch_id"], pattern="line 1500")
    assert "line 1500 of the saved page" in hit.output


@pytest.mark.parametrize("name", ["con", "NUL.txt", "com1", "LPT9.log", "Aux"])
async def test_a_windows_reserved_name_is_refused_and_nothing_is_written(project, monkeypatch, name):
    _serve(monkeypatch, lambda request: _text(BIG))
    res = await fetch(project, save_to=name)
    assert not res.success and "nothing saved" in res.error
    assert not downloads(project).exists()


async def test_a_name_with_spaces_and_unicode_is_saved_as_a_slug(project, monkeypatch):
    _serve(monkeypatch, lambda request: _text(BIG))
    res = await fetch(project, save_to="My Page ünï.md")
    assert res.success
    (saved,) = downloads(project).iterdir()
    assert saved.name == "my-page-uni.md"
    assert ".localharness/downloads/my-page-uni.md" in res.output


async def test_a_leading_dot_is_stripped_and_a_long_name_cut(project, monkeypatch):
    _serve(monkeypatch, lambda request: _text(BIG))
    await fetch(project, save_to="..hidden")
    await fetch(project, save_to="a" * 300)
    names = sorted(p.name for p in downloads(project).iterdir())
    assert names == ["a" * 128, "hidden"]


async def test_a_name_that_slugifies_to_nothing_takes_the_url_segment_else_the_title(project, monkeypatch):
    html = ("<html><head><title>Quarterly Results</title></head><body>"
            + "<p>" + "revenue grew steadily " * 20 + "</p></body></html>")
    _serve(monkeypatch, lambda request: httpx.Response(200, text=html, headers={"content-type": "text/html"}))
    await fetch(project, save_to="日本語", url="http://example.com/docs/guide.html")
    await fetch(project, save_to="!!!", url="http://example.com/")
    assert sorted(p.name for p in downloads(project).iterdir()) == ["guide.html", "quarterly-results"]


async def test_nothing_to_derive_a_name_from_is_refused(project, monkeypatch):
    _serve(monkeypatch, lambda request: _text(BIG))
    res = await fetch(project, save_to="!!!", url="http://example.com/")
    assert not res.success and "letters or digits" in res.error


async def test_no_boundary_refuses_and_says_to_start_in_a_project_folder(tmp_path, monkeypatch):
    public_web(monkeypatch)
    seen = _serve(monkeypatch, lambda request: _text(BIG))
    res = await WebFetchTool(ContentStore()).run(url="http://example.com/p", save_to="x.md")
    assert not res.success and "start localharness in a project folder" in res.error
    assert not seen and not (tmp_path / ".localharness").exists()


async def test_file_and_folder_modes_are_private(project, monkeypatch):
    _serve(monkeypatch, lambda request: _text(BIG))
    await fetch(project)
    assert stat.S_IMODE(os.stat(downloads(project)).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(downloads(project) / "x.md").st_mode) == 0o600


async def test_a_symlinked_workspace_folder_is_refused(project, tmp_path, monkeypatch):
    _serve(monkeypatch, lambda request: _text(BIG))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (project / ".localharness").symlink_to(elsewhere)
    res = await fetch(project)
    assert not res.success and "nothing saved" in res.error
    assert list(elsewhere.iterdir()) == []


async def test_a_symlinked_downloads_folder_is_refused(project, tmp_path, monkeypatch):
    _serve(monkeypatch, lambda request: _text(BIG))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (project / ".localharness").mkdir()
    downloads(project).symlink_to(elsewhere)
    res = await fetch(project)
    assert not res.success and "nothing saved" in res.error
    assert list(elsewhere.iterdir()) == []


async def test_a_symlinked_target_file_is_refused_and_its_destination_untouched(project, tmp_path, monkeypatch):
    _serve(monkeypatch, lambda request: _text(BIG))
    victim = tmp_path / "victim.txt"
    victim.write_text("keep")
    downloads(project).mkdir(parents=True)
    (downloads(project) / "x.md").symlink_to(victim)
    res = await fetch(project)
    assert not res.success and "nothing saved" in res.error
    assert victim.read_text() == "keep"
    assert not [p for p in downloads(project).iterdir() if p.name.startswith(".tmp-")]


async def test_a_hardlinked_target_is_replaced_and_the_other_name_keeps_its_content(project, tmp_path, monkeypatch):
    _serve(monkeypatch, lambda request: _text(BIG))
    other = tmp_path / "other.txt"
    other.write_text("old")
    downloads(project).mkdir(parents=True)
    os.link(other, downloads(project) / "x.md")
    res = await fetch(project)
    assert res.success and "(replaced an existing file)" in res.output
    assert other.read_text() == "old"
    assert (downloads(project) / "x.md").read_text() == BIG


async def test_over_the_file_cap_the_oldest_go_and_the_receipt_says_so(project, monkeypatch):
    _serve(monkeypatch, lambda request: _text(BIG))
    monkeypatch.setattr(web_tool, "DOWNLOADS_MAX_FILES", 3)
    downloads(project).mkdir(parents=True)
    for age, name in enumerate(["a", "b", "c", "d", "e"]):
        old = downloads(project) / name
        old.write_text(name)
        os.utime(old, (1_000_000 + age, 1_000_000 + age))
    res = await fetch(project)
    assert sorted(p.name for p in downloads(project).iterdir()) == ["d", "e", "x.md"]
    assert "removed 3 older downloads" in res.output
    assert res.metadata["removed"] == 3


async def test_over_the_byte_cap_the_oldest_go(project, monkeypatch):
    _serve(monkeypatch, lambda request: _text(BIG))
    monkeypatch.setattr(web_tool, "DOWNLOADS_MAX_BYTES", len(BIG) + 10)
    downloads(project).mkdir(parents=True)
    old = downloads(project) / "old"
    old.write_text("z" * 100)
    os.utime(old, (1_000_000, 1_000_000))
    res = await fetch(project)
    assert [p.name for p in downloads(project).iterdir()] == ["x.md"]
    assert "removed 1 older downloads" in res.output


async def test_an_overwrite_is_reported(project, monkeypatch):
    _serve(monkeypatch, lambda request: _text(BIG))
    first = await fetch(project)
    second = await fetch(project)
    assert "(replaced an existing file)" not in first.output
    assert "(replaced an existing file)" in second.output


async def test_a_capped_download_is_flagged_as_a_prefix(project, monkeypatch):
    monkeypatch.setattr(web_tool, "FETCH_MAX_BODY_BYTES", 100_000)

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for _ in range(10):
                yield b"y" * 50_000

    _serve(monkeypatch, lambda request: httpx.Response(
        200, headers={"content-type": "text/plain"}, stream=Stream()))
    res = await fetch(project)
    assert "PREFIX ONLY: download capped at 100000 bytes" in res.output
    saved = (downloads(project) / "x.md").read_text()
    assert len(saved) == 100_000
    assert "download capped" not in saved


async def test_a_js_page_refuses_save_to_and_keeps_the_alternatives(project, monkeypatch):
    _serve(monkeypatch, lambda request: httpx.Response(
        200, text="<html><title>App</title><script>x</script></html>", headers={"content-type": "text/html"}))
    res = await fetch(project)
    assert not res.success and "nothing saved" in res.error and "static-HTML source" in res.error
    assert not downloads(project).exists()


def test_download_notice_speaks_only_for_a_replacement_or_a_removal():
    plain = "saved 5 chars to .localharness/downloads/x.md (fetch_id=pg-1)\nfirst: a\nlast: a"
    assert download_notice("web_fetch", plain) is None
    both = plain + "\n(replaced an existing file)\nremoved 2 older downloads"
    line = download_notice("web_fetch", both)
    assert line and "\n" not in line and ".localharness/downloads/x.md" in line
    assert "replaced an existing file" in line and "removed 2 older downloads" in line
    assert download_notice("read", both) is None
