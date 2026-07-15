"""GlobTool: Find files matching a glob pattern."""
from pathlib import Path

from localharness.tools.builtin.paths import resolve_user_path

from localharness.tools.base import Tool, ToolResult, ToolSchema


class GlobTool(Tool):
    def info(self) -> ToolSchema:
        return ToolSchema(
            name="glob",
            description=(
                "Find files matching a glob pattern. Returns newline-separated "
                "absolute paths. Use ** for recursive matching (a trailing bare "
                "'**' matches files too, not only directories). '~' is expanded "
                "automatically; a relative pattern roots at the process's current "
                "working directory — pass base_dir or an absolute/'~' pattern to "
                "search elsewhere."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern, e.g. 'src/**/*.py' or '*.yaml'",
                    },
                    "base_dir": {
                        "type": "string",
                        "description": "Directory to resolve pattern against. Defaults to CWD.",
                        "default": ".",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return.",
                        "default": 500,
                        "minimum": 1,
                        "maximum": 5000,
                    },
                },
                "required": ["pattern"],
            },
            destructive=False,
            estimated_tokens=200,
        )

    async def _execute(self, pattern: str, base_dir: str = ".", limit: int = 500) -> ToolResult:
        # Models routinely pass ~ or absolute patterns (observed live) — normalize both.
        if pattern.startswith("~"):
            pattern = str(Path(pattern).expanduser())
        anchor = Path(pattern).anchor
        if anchor:
            # pathlib's Path.glob() rejects non-relative patterns outright ("Non-relative
            # patterns are unsupported"), so split off the absolute anchor (POSIX '/', or a
            # Windows drive like 'C:\\') and glob the remainder relative to it. Path(...).parts
            # normalizes '\\' vs '/' for us, so this covers POSIX '/…', Windows 'C:\\…', and
            # even mixed separators (e.g. an f-string gluing a WindowsPath to a literal '/**').
            base = Path(anchor)
            pattern = "/".join(Path(pattern).parts[1:])
        else:
            base = resolve_user_path(base_dir)
        if not base.exists():
            return self.err(f"base_dir does not exist: {base}")
        # pathlib's Path.glob() yields DIRECTORIES ONLY for a trailing bare '**' (issue #74:
        # '~/.localharness/agents/**' returned the agents dir but never its .yaml files).
        # Rewrite a trailing '**' to '**/*' so files at any depth match. Permission deny
        # matching runs pre-expansion on the raw string, so a glob deny rule wants an
        # unanchored leading '*' to catch every shape (see config/models.py deny_patterns).
        if pattern == "**" or pattern.endswith("/**"):
            pattern = pattern + "/*"
        matches = sorted(base.glob(pattern))[:limit]
        if not matches:
            return self.ok("(no matches)")
        return self.ok("\n".join(str(p) for p in matches), match_count=len(matches))
