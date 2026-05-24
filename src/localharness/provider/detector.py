"""Parallel port probe for local LLM inference backend auto-detection."""
import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlparse

import httpx

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_PORTS: list[int] = [8000, 11434, 1234, 8080]
"""Probe order: vLLM, Ollama, LM Studio, llama.cpp"""

LOCAL_INFERENCE_TIMEOUT_MIN: float = 300.0
"""Minimum allowed timeout (seconds) for local endpoints."""

_LOCAL_PATTERNS = re.compile(
    r"^(localhost|127\.\d+\.\d+\.\d+|0\.0\.0\.0|::1|.*\.local)$",
    re.IGNORECASE,
)

ProviderType = Literal["ollama", "vllm", "llamacpp", "lmstudio", "unknown"]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class DetectorResult:
    found: bool
    provider_type: ProviderType
    base_url: str
    models: list[str]
    suggested_model: str
    probe_duration_ms: float
    raw_response: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


async def detect_provider(
    timeout_seconds: float = 1.0,
    ports: list[int] | None = None,
) -> DetectorResult:
    """Probe known inference server ports in parallel. Returns the first server found.

    Uses asyncio.gather so all probes run concurrently — total time is bounded
    by the slowest individual probe, not their sum.

    Raises:
        Never raises. All probe errors are caught and logged at DEBUG level.
    """
    probe_ports = ports if ports is not None else DEFAULT_PORTS
    start = time.monotonic()

    async with httpx.AsyncClient() as client:
        probe_coros = [_probe_port(client, port, timeout_seconds) for port in probe_ports]
        results = await asyncio.gather(*probe_coros, return_exceptions=True)

    elapsed_ms = (time.monotonic() - start) * 1000

    # Return first non-None, non-exception result (preserves probe_ports priority order)
    for result in results:
        if isinstance(result, DetectorResult):
            result.probe_duration_ms = elapsed_ms
            return result

    return DetectorResult(
        found=False,
        provider_type="unknown",
        base_url="",
        models=[],
        suggested_model="",
        probe_duration_ms=elapsed_ms,
        raw_response={},
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_base_url(port: int) -> str:
    """Return OpenAI-compatible base URL for a port. Always includes /v1."""
    return f"http://localhost:{port}/v1"


async def _probe_port(
    client: httpx.AsyncClient,
    port: int,
    timeout: float,
) -> DetectorResult | None:
    """Attempt a single port probe. Returns None on any failure."""
    if port == 11434:
        url = f"http://localhost:{port}/api/tags"
    else:
        url = f"http://localhost:{port}/v1/models"

    try:
        response = await client.get(url, timeout=timeout)
        response_json = response.json()
        provider_type = _identify_provider(port, response_json, response.headers)
        models = _normalize_model_list(provider_type, response_json)
        base_url = _build_base_url(port)
        return DetectorResult(
            found=True,
            provider_type=provider_type,
            base_url=base_url,
            models=models,
            suggested_model=models[0] if models else "",
            probe_duration_ms=0.0,  # filled in by detect_provider
            raw_response=response_json,
        )
    except Exception as exc:
        log.debug("Port %d probe failed: %s", port, exc)
        return None


def _identify_provider(
    port: int,
    response_json: dict,
    response_headers: httpx.Headers,
) -> ProviderType:
    """Heuristic provider identification from port, response shape, and headers."""
    if port == 11434:
        return "ollama"
    if port == 8080:
        return "llamacpp"
    if port == 1234 and response_headers.get("x-lm-studio") is not None:
        return "lmstudio"
    if "data" in response_json:
        return "vllm"
    return "unknown"


def _normalize_model_list(
    provider_type: ProviderType,
    response_json: dict,
) -> list[str]:
    """Extract model ID list regardless of API response shape."""
    if provider_type == "ollama":
        return [m["name"] for m in response_json.get("models", [])]
    return [m["id"] for m in response_json.get("data", [])]


def is_local_endpoint(base_url: str) -> bool:
    """Return True if base_url resolves to a local address."""
    host = urlparse(base_url).hostname or ""
    return bool(_LOCAL_PATTERNS.match(host))
