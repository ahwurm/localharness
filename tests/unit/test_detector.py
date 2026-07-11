"""Tests for provider auto-detection (detector.py)."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from localharness.provider.detector import (
    DEFAULT_PORTS,
    LOCAL_INFERENCE_TIMEOUT_MIN,
    DetectorResult,
    ProviderType,
    _identify_provider,
    _normalize_model_list,
    detect_provider,
    is_local_endpoint,
)


# ---------------------------------------------------------------------------
# is_local_endpoint
# ---------------------------------------------------------------------------


def test_is_local_endpoint_localhost():
    assert is_local_endpoint("http://localhost:8000/v1") is True


def test_is_local_127():
    assert is_local_endpoint("http://127.0.0.1:8000/v1") is True


def test_is_local_endpoint_remote():
    assert is_local_endpoint("http://api.openai.com/v1") is False


def test_is_local_endpoint_0_0_0_0():
    assert is_local_endpoint("http://0.0.0.0:8000/v1") is True


def test_is_local_endpoint_dotlocal():
    assert is_local_endpoint("http://myhost.local:8000/v1") is True


# ---------------------------------------------------------------------------
# _normalize_model_list
# ---------------------------------------------------------------------------


def test_normalize_model_list_ollama():
    response = {"models": [{"name": "qwen3:122b"}, {"name": "llama3:8b"}]}
    result = _normalize_model_list("ollama", response)
    assert result == ["qwen3:122b", "llama3:8b"]


def test_normalize_model_list_openai():
    response = {"data": [{"id": "model-a"}, {"id": "model-b"}]}
    result = _normalize_model_list("vllm", response)
    assert result == ["model-a", "model-b"]


def test_normalize_model_list_unknown():
    response = {"data": [{"id": "model-x"}]}
    result = _normalize_model_list("unknown", response)
    assert result == ["model-x"]


def test_normalize_model_list_null_data():
    """Ollama with zero models pulled returns {"data": null}; must not raise TypeError."""
    assert _normalize_model_list("vllm", {"object": "list", "data": None}) == []


def test_normalize_model_list_null_models():
    """Symmetric guard for the ollama-shaped {"models": null}."""
    assert _normalize_model_list("ollama", {"models": None}) == []


# ---------------------------------------------------------------------------
# _identify_provider
# ---------------------------------------------------------------------------


def test_identify_provider_ollama():
    headers = MagicMock()
    headers.get = MagicMock(return_value=None)
    result = _identify_provider(11434, {}, headers)
    assert result == "ollama"


def test_identify_provider_vllm():
    headers = MagicMock()
    headers.get = MagicMock(return_value=None)
    response = {"data": [{"id": "model-a", "object": "model"}]}
    result = _identify_provider(8000, response, headers)
    assert result == "vllm"


def test_identify_provider_lmstudio():
    headers = MagicMock()
    headers.get = MagicMock(side_effect=lambda k, d=None: "1.0" if k == "x-lm-studio" else d)
    result = _identify_provider(1234, {}, headers)
    assert result == "lmstudio"


def test_identify_provider_llamacpp():
    headers = MagicMock()
    headers.get = MagicMock(return_value=None)
    result = _identify_provider(8080, {}, headers)
    assert result == "llamacpp"


def test_identify_provider_llamacpp_off_port_self_identifies():
    """Issue #12.2: llama.cpp behind a proxy on a non-8080 port self-identifies via
    owned_by, so it is llamacpp — not the old 'data'-catch-all vllm."""
    headers = MagicMock()
    headers.get = MagicMock(return_value=None)
    response = {"data": [{"id": "model.gguf", "owned_by": "llamacpp", "meta": {"n_ctx": 4096}}]}
    assert _identify_provider(9000, response, headers) == "llamacpp"


def test_identify_provider_vllm_off_port_self_identifies():
    """vLLM self-identifies via owned_by on any port."""
    headers = MagicMock()
    headers.get = MagicMock(return_value=None)
    response = {"data": [{"id": "m", "owned_by": "vllm", "max_model_len": 131072}]}
    assert _identify_provider(9001, response, headers) == "vllm"


def test_identify_provider_lmstudio_default_port_no_header():
    """Issue #12.1: LM Studio 0.4.x drops x-lm-studio; on its default port the tie-break
    still classifies it lmstudio (not vllm)."""
    headers = MagicMock()
    headers.get = MagicMock(return_value=None)
    response = {"data": [{"id": "qwen", "owned_by": "organization_owner"}]}
    assert _identify_provider(1234, response, headers) == "lmstudio"


# ---------------------------------------------------------------------------
# detect_provider — parallel probing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_probe_vllm_port():
    """Port 8000 with OpenAI data array -> vllm DetectorResult."""
    vllm_response = {"data": [{"id": "model-a", "object": "model"}]}

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = vllm_response
    mock_response.headers = MagicMock()
    mock_response.headers.get = MagicMock(return_value=None)

    async def fake_get(url, **kwargs):
        if ":8000" in url:
            return mock_response
        raise Exception("connection refused")

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=fake_get)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await detect_provider(timeout_seconds=1.0, ports=[8000])

    assert result.found is True
    assert result.provider_type == "vllm"
    assert "model-a" in result.models


@pytest.mark.asyncio
async def test_probe_ollama_port():
    """Port 11434 with models array -> ollama DetectorResult."""
    ollama_response = {"models": [{"name": "qwen3:122b"}]}

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = ollama_response
    mock_response.headers = MagicMock()
    mock_response.headers.get = MagicMock(return_value=None)

    async def fake_get(url, **kwargs):
        if ":11434" in url:
            return mock_response
        raise Exception("connection refused")

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=fake_get)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await detect_provider(timeout_seconds=1.0, ports=[11434])

    assert result.found is True
    assert result.provider_type == "ollama"
    assert "qwen3:122b" in result.models


@pytest.mark.asyncio
async def test_probe_lmstudio_port():
    """Port 1234 with x-lm-studio header -> lmstudio."""
    lm_response = {"data": [{"id": "local-model"}]}

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = lm_response
    mock_response.headers = MagicMock()
    mock_response.headers.get = MagicMock(side_effect=lambda k, d=None: "1.0" if k == "x-lm-studio" else d)

    async def fake_get(url, **kwargs):
        if ":1234" in url:
            return mock_response
        raise Exception("connection refused")

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=fake_get)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await detect_provider(timeout_seconds=1.0, ports=[1234])

    assert result.found is True
    assert result.provider_type == "lmstudio"


def _mk_response(body: dict, lm_header: str | None = None) -> MagicMock:
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = body
    r.headers = MagicMock()
    r.headers.get = MagicMock(side_effect=lambda k, d=None: lm_header if k == "x-lm-studio" else d)
    return r


async def _run_detect(routes: dict, ports: list[int]):
    """routes: URL-fragment -> mock response (checked in insertion order); else 'refused'."""
    async def fake_get(url, **kwargs):
        for frag, resp in routes.items():
            if frag in url:
                return resp
        raise Exception("connection refused")

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=fake_get)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client
        return await detect_provider(timeout_seconds=1.0, ports=ports)


@pytest.mark.asyncio
async def test_probe_lmstudio_off_port_via_api_v0():
    """Issue #12.1: LM Studio 0.4.x behind a proxy on :8000 with generic headers
    (no x-lm-studio) — /v1/models looks like vLLM's, but /api/v0/models identifies it.
    Must classify lmstudio, NOT vllm."""
    v1 = _mk_response({"data": [{"id": "qwen", "owned_by": "organization_owner"}]})
    v0 = _mk_response(
        {"object": "list", "data": [
            {"id": "qwen", "state": "loaded", "max_context_length": 32768, "loaded_context_length": 8192},
        ]}
    )
    result = await _run_detect(
        {"8000/api/v0/models": v0, "8000/v1/models": v1}, ports=[8000]
    )
    assert result.found is True
    assert result.provider_type == "lmstudio"
    assert "qwen" in result.models


@pytest.mark.asyncio
async def test_probe_llamacpp_off_port():
    """Issue #12.2: llama.cpp on a non-8080 port self-identifies via owned_by -> llamacpp."""
    v1 = _mk_response({"data": [{"id": "model.gguf", "owned_by": "llamacpp", "meta": {"n_ctx": 4096}}]})
    result = await _run_detect({"9000/v1/models": v1}, ports=[9000])
    assert result.found is True
    assert result.provider_type == "llamacpp"
    assert "model.gguf" in result.models


@pytest.mark.asyncio
async def test_probe_ollama_null_data_no_crash():
    """Issue #12.3: zero-model Ollama on a non-11434 port returns {"data": null};
    normalization must not raise TypeError and detection still returns a result."""
    v1 = _mk_response({"object": "list", "data": None})
    result = await _run_detect({"8000/v1/models": v1}, ports=[8000])
    assert result.found is True
    assert result.models == []


@pytest.mark.asyncio
async def test_probe_no_server():
    """All ports timeout -> found=False."""
    async def fake_get(url, **kwargs):
        raise Exception("connection refused")

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=fake_get)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = await detect_provider(timeout_seconds=1.0)

    assert result.found is False
    assert result.models == []
    assert result.base_url == ""


@pytest.mark.asyncio
async def test_parallel_probing():
    """detect_provider uses asyncio.gather — all probes launch concurrently."""
    gather_calls = []

    original_gather = asyncio.gather

    async def mock_gather(*coros, **kwargs):
        gather_calls.append(len(coros))
        return await original_gather(*coros, **kwargs)

    async def fake_get(url, **kwargs):
        raise Exception("connection refused")

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=fake_get)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        with patch("asyncio.gather", side_effect=mock_gather):
            await detect_provider(timeout_seconds=1.0)

    # asyncio.gather was called with all probes at once
    assert len(gather_calls) >= 1
    assert gather_calls[0] == len(DEFAULT_PORTS)


def test_all_base_urls_include_v1():
    """All providers use /v1 suffix for OpenAI-compat API."""
    from localharness.provider.detector import _build_base_url
    assert _build_base_url(11434) == "http://localhost:11434/v1"
    assert _build_base_url(8000) == "http://localhost:8000/v1"


def test_default_ports_constant():
    # 8081 = harness-managed vLLM (init guided setup); 8000 = stock vLLM default.
    assert DEFAULT_PORTS == [8081, 8000, 11434, 1234, 8080]


def test_local_inference_timeout_min():
    assert LOCAL_INFERENCE_TIMEOUT_MIN == 300.0
