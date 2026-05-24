"""Provider layer: LLM client, auto-detection, XML fn_call converter."""
from localharness.provider.client import (
    CapabilityResult,
    LLMClient,
    LLMConfig,
    MalformedResponseError,
    ProviderAPIError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from localharness.provider.detector import DetectorResult, detect_provider
from localharness.provider.fn_call import FnCallConverter

__all__ = [
    "LLMClient",
    "LLMConfig",
    "CapabilityResult",
    "ProviderError",
    "ProviderConnectionError",
    "ProviderTimeoutError",
    "ProviderRateLimitError",
    "ProviderAPIError",
    "MalformedResponseError",
    "DetectorResult",
    "detect_provider",
    "FnCallConverter",
]
