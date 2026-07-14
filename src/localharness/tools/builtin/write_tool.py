"""WriteTool: Write or overwrite a file."""
import asyncio
from pathlib import Path

from localharness.tools.base import Tool, ToolResult, ToolSchema


class WriteTool(Tool):
    def info(self) -> ToolSchema:
        return ToolSchema(
            name="write",
            description=(
                "Write or overwrite a file. Creates parent directories if needed. "
                "Returns the absolute path written and byte count. For a large file, write "
                "it in several smaller calls (first call creates it, then add the rest with "
                "mode=append) rather than one huge call — an oversized content argument can "
                "be cut off at the output-token limit and will not be executed."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "File content to write.",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["overwrite", "append"],
                        "description": "Write mode. Default: overwrite.",
                        "default": "overwrite",
                    },
                },
                "required": ["path", "content"],
            },
            destructive=True,
            estimated_tokens=150,
        )

    async def _execute(self, path: str, content: str, mode: str = "overwrite") -> ToolResult:
        target = Path(path).expanduser().resolve()

        forbidden_suffixes = {".env", ".secret", ".token", ".pem", ".key"}
        if target.suffix in forbidden_suffixes or target.name.startswith(".env"):
            return self.err(
                f"Write to credential/secret file blocked: {target}",
                error_type="permission_denied",
            )

        if (denied := self._outside_workspace(target)) is not None:
            return denied

        target.parent.mkdir(parents=True, exist_ok=True)

        new_bytes = content.encode()
        n = len(new_bytes)
        loop = asyncio.get_running_loop()

        # Overwrite reports create/overwrite/no-op honestly: a re-write of byte-identical
        # content returns a STOP signal (unchanged=True) instead of another "success" line
        # a stuck model reacts to by rewriting the same file forever. Append is unchanged.
        if mode == "overwrite":
            old_bytes = None
            if target.exists():
                try:
                    old_bytes = await loop.run_in_executor(None, target.read_bytes)
                except OSError:
                    old_bytes = None
            if old_bytes == new_bytes:  # only True when the file existed AND matched
                return self.ok(
                    f"No change: {target} already contains exactly this content "
                    f"({n} bytes). The file is already written — do not rewrite it; "
                    f"take the next step.",
                    path=str(target), bytes_written=n, unchanged=True,
                )
            message = (
                f"Created {target} ({n} bytes)" if old_bytes is None
                else f"Overwrote {target} (was {len(old_bytes)} bytes, now {n} bytes)"
            )
        else:
            message = f"Written {n} bytes to {target}"

        open_mode = "a" if mode == "append" else "w"
        try:
            await loop.run_in_executor(
                None, lambda: target.open(open_mode, encoding="utf-8").write(content)
            )
        except PermissionError:
            return self.err(f"Permission denied: {target}", error_type="permission_denied")
        except OSError as exc:
            return self.err(str(exc))

        return self.ok(message, path=str(target), bytes_written=n, unchanged=False)
