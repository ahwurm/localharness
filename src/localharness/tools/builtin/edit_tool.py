"""EditTool: surgical in-place string replacement (avoids full-file rewrites)."""
import asyncio

from localharness.tools.builtin.paths import resolve_user_path

from localharness.tools.base import Tool, ToolResult, ToolSchema


class EditTool(Tool):
    """Replace an exact string in a file. Emitting a small diff instead of the whole file
    keeps generation cheap — critical on a bandwidth-bound local model where regenerating a
    large file as output tokens is the slow path (and can outrun request timeouts)."""

    def info(self) -> ToolSchema:
        return ToolSchema(
            name="edit",
            description=(
                "Make a surgical edit to a file by replacing an exact string. Prefer this over "
                "`write` for changing an existing file — you emit only the changed snippet, not the "
                "whole file. `old_string` must match exactly (including whitespace) and be unique "
                "unless replace_all=true. Returns the number of replacements."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the file to edit."},
                    "old_string": {"type": "string", "description": "Exact text to replace."},
                    "new_string": {"type": "string", "description": "Replacement text."},
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace every occurrence (default false = require a unique match).",
                        "default": False,
                    },
                },
                "required": ["path", "old_string", "new_string"],
            },
            destructive=True,
            estimated_tokens=120,
        )

    async def _execute(self, path: str, old_string: str, new_string: str,
                       replace_all: bool = False) -> ToolResult:
        target = resolve_user_path(path)

        forbidden_suffixes = {".env", ".secret", ".token", ".pem", ".key"}
        if target.suffix in forbidden_suffixes or target.name.startswith(".env"):
            return self.err(
                f"Edit of credential/secret file blocked: {target}",
                error_type="permission_denied",
            )
        if (denied := self._outside_workspace(target)) is not None:
            return denied
        if not target.exists():
            return self.err(f"File not found: {target}", error_type="not_found")
        if target.is_dir():
            return self.err(f"Path is a directory, not a file: {target}")
        if old_string == new_string:
            return self.err("old_string and new_string are identical", error_type="validation_error")

        loop = asyncio.get_running_loop()
        try:
            text = await loop.run_in_executor(None, target.read_text, "utf-8")
        except PermissionError:
            return self.err(f"Permission denied: {target}", error_type="permission_denied")
        except (OSError, UnicodeDecodeError) as exc:
            return self.err(str(exc))

        count = text.count(old_string)
        if count == 0:
            return self.err("old_string not found in file", error_type="not_found")
        if count > 1 and not replace_all:
            return self.err(
                f"old_string is not unique ({count} matches) — add surrounding context to "
                "disambiguate, or set replace_all=true.",
                error_type="validation_error",
            )

        updated = text.replace(old_string, new_string)
        try:
            await loop.run_in_executor(None, target.write_text, updated, "utf-8")
        except PermissionError:
            return self.err(f"Permission denied: {target}", error_type="permission_denied")
        except OSError as exc:
            return self.err(str(exc))

        return self.ok(
            f"Replaced {count} occurrence{'s' if count != 1 else ''} in {target}",
            path=str(target),
            replacements=count,
        )
