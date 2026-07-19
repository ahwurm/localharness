"""LocalHarness: Model-agnostic hierarchical agent harness for local LLMs."""
__version__ = "0.9.21"


def resolved_version() -> str:
    """Version string for the startup banner and `--version`. Prefers the in-source
    ``__version__`` — the source of truth. Editable/live installs read STALE dist metadata
    (#97: the banner showed v0.9.16 while the source was already v0.9.19), so dist metadata is
    the fallback only, reached in the rare case the in-source constant is somehow empty."""
    if __version__:
        return __version__
    try:
        from importlib.metadata import version
        return version("localharness")
    except Exception:
        return "unknown"
