"""Web tools: keyless DuckDuckGo search + URL fetch with output-clipping.

web_search — DuckDuckGo (no API key), returns title/url/snippet per hit.
web_fetch  — fetch a URL, strip HTML to readable text, and CLIP to a char budget so a
             large page can never blow the model's context window (the failure mode that
             hard-killed a turn: raw `curl | head -500` HTML ballooned input past 32k).
"""
from __future__ import annotations

import asyncio
import re

import httpx

from localharness.tools.base import Tool, ToolResult, ToolSchema

_FETCH_DEFAULT_CHARS = 8000   # ~2k tokens — safe default
_FETCH_MAX_CHARS = 20000      # hard ceiling regardless of caller request
_UA = "Mozilla/5.0 (compatible; LocalHarness/0.1; +https://localharness.dev)"


def _html_to_text(html: str) -> str:
    """Cheap HTML -> readable text: drop script/style, strip tags, collapse whitespace."""
    html = re.sub(r"(?is)<(script|style|noscript|svg|head)\b.*?</\1>", " ", html)
    html = re.sub(r"(?is)<br\s*/?>", "\n", html)
    html = re.sub(r"(?is)</(p|div|li|h[1-6]|tr|section|article)>", "\n", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    text = (text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<")
                .replace("&gt;", ">").replace("&#39;", "'").replace("&quot;", '"'))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


class WebSearchTool(Tool):
    timeout_s = 25.0

    def info(self) -> ToolSchema:
        return ToolSchema(
            name="web_search",
            description=(
                "Search the web (DuckDuckGo, no API key). Returns ranked results as "
                "title / URL / snippet. Use this for current information, news, docs, or "
                "anything not in your training data. Follow up with web_fetch to read a result."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 5).",
                        "default": 5, "minimum": 1, "maximum": 15,
                    },
                },
                "required": ["query"],
            },
            destructive=False,
            estimated_tokens=500,
        )

    async def _execute(self, query: str, max_results: int = 5) -> ToolResult:
        try:
            from ddgs import DDGS
        except ImportError:
            return self.err(
                "web search unavailable: 'ddgs' not installed (install the 'dispatch' extra)",
                error_type="execution_error",
            )
        loop = asyncio.get_running_loop()
        try:
            results = await loop.run_in_executor(
                None, lambda: DDGS().text(query, max_results=max_results)
            )
        except Exception as exc:  # noqa: BLE001
            return self.err(f"web search failed: {exc}")
        if not results:
            return self.ok(f"No results for: {query}", result_count=0)
        lines = []
        for i, r in enumerate(results, 1):
            title = (r.get("title") or "").strip()
            url = (r.get("href") or r.get("url") or "").strip()
            snippet = (r.get("body") or r.get("snippet") or "").strip()
            lines.append(f"{i}. {title}\n   {url}\n   {snippet}")
        return self.ok("\n\n".join(lines), result_count=len(results))


class WebFetchTool(Tool):
    timeout_s = 25.0

    def info(self) -> ToolSchema:
        return ToolSchema(
            name="web_fetch",
            description=(
                "Fetch a URL and return its readable text content (HTML stripped). Output is "
                "CLIPPED to a character budget so large pages can't overflow context — increase "
                "max_chars only if you truly need more. Use after web_search to read a page."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch (http/https)."},
                    "max_chars": {
                        "type": "integer",
                        "description": f"Max characters to return (default {_FETCH_DEFAULT_CHARS}, "
                                       f"hard cap {_FETCH_MAX_CHARS}).",
                        "default": _FETCH_DEFAULT_CHARS, "minimum": 500, "maximum": _FETCH_MAX_CHARS,
                    },
                },
                "required": ["url"],
            },
            destructive=False,
            estimated_tokens=_FETCH_DEFAULT_CHARS // 4,
        )

    async def _execute(self, url: str, max_chars: int = _FETCH_DEFAULT_CHARS) -> ToolResult:
        if not re.match(r"^https?://", url):
            return self.err(f"Invalid URL (must be http/https): {url}", error_type="validation_error")
        cap = max(500, min(int(max_chars), _FETCH_MAX_CHARS))
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=20.0,
                                         headers={"User-Agent": _UA}) as client:
                resp = await client.get(url)
                resp.raise_for_status()
        except httpx.HTTPError as exc:
            return self.err(f"fetch failed: {exc}")
        ctype = resp.headers.get("content-type", "")
        body = resp.text
        text = body if ("text/plain" in ctype or "json" in ctype) else _html_to_text(body)
        full_len = len(text)
        if full_len > cap:
            return ToolResult(
                output=text[:cap] + f"\n\n[... clipped {full_len - cap} of {full_len} chars; "
                                    f"call web_fetch with a higher max_chars to read more]",
                success=True, truncated=True, original_length=full_len,
                metadata={"url": str(resp.url), "content_type": ctype},
            )
        return self.ok(text, url=str(resp.url), content_type=ctype, original_length=full_len)
