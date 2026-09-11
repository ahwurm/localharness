"""WriteTool: Write or overwrite a file."""
import asyncio

from localharness.tools.builtin.paths import resolve_user_path

from localharness.tools.base import Tool, ToolResult, ToolSchema


# An overwrite that changes only a small slice of a large file is the case `edit` exists
# for: the model regenerated the whole file as output tokens to alter a few lines. The
# write still lands (refusing would cost a full round-trip — minutes on a local model);
# the result says so, with the numbers, so the next change goes through `edit`.
_EDIT_HINT_MIN_LINES = 20        # a file this long is worth a snippet edit
_EDIT_HINT_MAX_CHANGED_FRACTION = 0.2  # …when at most this share of its lines changed


def overwrite_diff_stat(old: str, new: str) -> tuple[int, int, int]:
    """(lines added, lines removed, lines in the old file) for an old→new overwrite."""
    import difflib
    old_lines, new_lines = old.splitlines(), new.splitlines()
    added = removed = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old_lines, new_lines).get_opcodes():
        if tag == "equal":
            continue
        removed += i2 - i1
        added += j2 - j1
    return added, removed, len(old_lines)


class WriteTool(Tool):
    def info(self) -> ToolSchema:
        return ToolSchema(
            name="write",
            group="fs.write",
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
        if (unpaired := self._editor_hooks_unpaired()) is not None:
            return unpaired
        target = resolve_user_path(path)

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
                    # PRD §4: with an editor attached the "is this already written?" question is
                    # about the BUFFER — a disk read would call an unsaved edit a no-op.
                    if self.file_read_hook is not None:
                        old_bytes = (await self.file_read_hook(target)).encode()
                    else:
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
            if old_bytes is not None:
                try:
                    added, removed, old_n = overwrite_diff_stat(old_bytes.decode("utf-8"), content)
                except UnicodeDecodeError:
                    added = removed = old_n = 0
                changed = max(added, removed)
                if old_n >= _EDIT_HINT_MIN_LINES and changed <= old_n * _EDIT_HINT_MAX_CHANGED_FRACTION:
                    message += (
                        f"\n+{added} −{removed} of {old_n} lines changed. A change this small to a "
                        f"file this long is what `edit` is for: pass only the snippet "
                        f"(old_string → new_string) instead of regenerating the whole file."
                    )
        else:
            message = f"Written {n} bytes to {target}"

        open_mode = "a" if mode == "append" else "w"
        try:
            if self.file_write_hook is not None:
                # ACP's fs/write_text_file replaces the WHOLE file — there is no append mode in
                # the protocol (PRD §4) — so append becomes read-then-concatenate through the
                # same editor-backed pair rather than a silent disk write behind the buffer.
                # The read half is guaranteed present (`_editor_hooks_unpaired` above): making it
                # optional here is exactly how an append used to TRUNCATE the file (R5).
                text = content
                if mode == "append" and target.exists():
                    text = await self.file_read_hook(target) + content
                await self.file_write_hook(target, text)
            else:
                await loop.run_in_executor(
                    None, lambda: target.open(open_mode, encoding="utf-8").write(content)
                )
        except PermissionError:
            return self.err(f"Permission denied: {target}", error_type="permission_denied")
        except OSError as exc:
            return self.err(str(exc))

        return self.ok(message, path=str(target), bytes_written=n, unchanged=False)
