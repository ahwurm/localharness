"""The copyable LocalHarness plugin.

It contributes one of each thing a plugin can add: a tool (`example_swatch`, which renders a
solid-color PNG into the artifact directory core assigns it), a command (`localharness example`), a
slash command (`/example`), a `doctor` check, and two settings (`example.color`,
`agent.example.size`). See README.md next to pyproject.toml.
"""
import os
from pathlib import Path

# --- import sentinel: TEST INSTRUMENTATION. Delete these lines in your copy. ---
# LocalHarness's own tests set this variable to prove the harness never imports a plugin that is not
# enabled (ENAB-06): the file appears only when this module is actually imported.
if _sentinel := os.environ.get("LOCALHARNESS_EXAMPLE_PLUGIN_SENTINEL"):
    Path(_sentinel).write_text("imported\n", encoding="utf-8")
# --- end of test instrumentation ---

from localharness_plugin_example.plugin import ExamplePlugin, __version__  # noqa: E402

__all__ = ["ExamplePlugin", "__version__"]
