"""The image check asks ComfyUI about each node class by a well-formed URL: the class name from
the workflow template is percent-quoted as one path segment, so a `/` or `?` in it can neither
reach another route nor start a query string."""
from __future__ import annotations

import json

import httpx

from localharness.tools.builtin import generate_image_tool


def test_the_class_type_is_quoted_into_one_path_segment(tmp_path, monkeypatch):
    template = tmp_path / "workflow.json"
    template.write_text(json.dumps({"1": {"class_type": "a/b?c", "inputs": {}}}), encoding="utf-8")
    paths: list[bytes] = []

    def answer(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.raw_path)
        if request.url.path == "/system_stats":
            return httpx.Response(200, json={})
        return httpx.Response(200, json={"a/b?c": {"input": {}}})

    monkeypatch.setattr(generate_image_tool, "_TRANSPORT", httpx.MockTransport(answer))

    generate_image_tool.probe("http://comfy.test:8188", str(template))

    assert paths == [b"/system_stats", b"/object_info/a%2Fb%3Fc"]
