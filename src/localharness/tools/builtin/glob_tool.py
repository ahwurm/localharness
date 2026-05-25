"""GlobTool: Find files matching a glob pattern."""
from pathlib import Path

from localharness.tools.base import Tool, ToolResult, ToolSchema


class GlobTool(Tool):
    def info(self) -> ToolSchema:
        return ToolSchema(
            name="glob",
            description=(
                "Find files matching a glob pattern. Returns newline-separated "
                "absolute paths. Use ** for recursive matching."
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
        base = Path(base_dir).resolve()
        if not base.exists():
            return self.err(f"base_dir does not exist: {base}")
        matches = sorted(base.glob(pattern))[:limit]
        if not matches:
            return self.ok("(no matches)")
        return self.ok("\n".join(str(p) for p in matches), match_count=len(matches))
