"""Binding, the app token, and the two path-confinement checks (§7).

This listener runs arbitrary shell with the owner's privileges. SECURITY.md already tells users
never to stand up a bare unauthenticated endpoint, so the posture here is deliberately belt and
braces, and each brace is here for a reason that is written down beside it.
"""
from __future__ import annotations

import os
import secrets
import stat
from pathlib import Path
from typing import Optional

TOKEN_BYTES = 32
"""How much entropy the app token carries, in bytes, before URL-safe encoding.

256 bits. Not a chosen-looking round number in a different unit: this is the width
`secrets.token_urlsafe` is documented against, and the token is machine-generated and
machine-copied (§7.2's QR), so there is no length at which a human's patience argues for less.
"""

TOKEN_FILE_MODE = 0o600
"""Owner read/write only. Every local process on this box can reach the loopback port, so the
file that decides which of them may drive it is not world-readable."""

LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "::1", "localhost"})
"""The only hosts this server binds without an explicit override.

Not `0.0.0.0`: that publishes a shell-running endpoint to whatever café wifi the laptop happens
to have joined. A fronting proxy (`tailscale serve`, or any reverse proxy already in use)
terminates TLS and reaches this port over loopback — which is also what keeps Tailscale's own
identity headers trustworthy, since they are only trustworthy if nothing but `tailscaled` can
reach the backend.
"""

UNSAFE_BIND_ERROR = (
    "refusing to bind {host}: this endpoint runs shell commands with your privileges, so it "
    "binds loopback only ({safe}). Put `tailscale serve` or another reverse proxy in front of "
    "it, or pass --allow-unsafe-bind if you genuinely mean to publish it."
)

CONTENT_TYPE_ERROR = (
    "POST bodies must be application/json. The simple content types (text/plain, "
    "application/x-www-form-urlencoded, multipart/form-data) are refused because they are "
    "exactly the ones a cross-origin <form> can send with no preflight."
)

UNAUTHENTICATED_ERROR = "missing or invalid credentials"

AUTH_COOKIE = "lh_web"
"""The cookie the SSE stream authenticates with.

`EventSource` cannot send an `Authorization` header — a real constraint a naive plan discovers
late — and the two obvious workarounds are both bad: a token in the query string lands in logs
and referrers, and a plain cookie alone reintroduces CSRF on the POST verbs. So the stream uses
a `Secure`, `HttpOnly`, `SameSite=Strict` cookie (no other site can cause it to be sent) while
every POST additionally requires the bearer header AND a JSON content type, which together force
a preflight the server refuses for a foreign origin. The POST surface is then structurally
CSRF-safe rather than leaning on the cookie's `SameSite` alone.
"""


def token_path(config_dir: Optional[str | Path] = None) -> Path:
    """Where the app token lives: `<global config dir>/web/token`.

    GLOBAL on purpose, like the GPU daemon's pidfile: there is one listener per machine and the
    phone enrolled against it does not know or care which project directory the process was
    started in.
    """
    from localharness.config.paths import global_config_dir

    return global_config_dir(config_dir) / "web" / "token"


def load_or_create_token(config_dir: Optional[str | Path] = None) -> tuple[str, bool]:
    """The app token, generated on first run. Returns `(token, was_created)`.

    Created rather than configured because a setup step a user can skip is a setup step that
    produces an unauthenticated endpoint. The file is written `0600` and re-chmod'd on every
    read, so a token that predates this rule, or that an editor rewrote with a friendlier mode,
    is tightened rather than trusted.
    """
    path = token_path(config_dir)
    if path.exists():
        existing = path.read_text(encoding="utf-8").strip()
        if existing:
            _tighten(path)
            return existing, False
    return rotate_token(config_dir), True


def rotate_token(config_dir: Optional[str | Path] = None) -> str:
    """Mint a new token, invalidating every enrolled client. Returns the new token.

    The symmetric answer to §5.3's grant-permanence worry, and the honest answer to "my phone was
    stolen while the app was still enrolled." **Named gap:** there is no per-DEVICE revoke —
    rotation is all-or-nothing and every other device re-enrols. Tracked, not closed.
    """
    path = token_path(config_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(TOKEN_BYTES)
    # Write through a mode-restricted descriptor rather than write_text-then-chmod: the latter
    # leaves a world-readable window between the two calls, which is a window on a box that runs
    # agents of its own.
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, TOKEN_FILE_MODE)
    try:
        os.write(fd, (token + "\n").encode("utf-8"))
    finally:
        os.close(fd)
    _tighten(path)
    return token


def _tighten(path: Path) -> None:
    """Best effort `0600`. A filesystem that cannot express it (Windows, some mounts) is not a
    reason to refuse to start — it is a reason not to pretend the mode was applied."""
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass


def check_bind(host: str, *, allow_unsafe: bool = False) -> str:
    """Return the host to bind, or raise `ValueError` naming the refusal.

    Startup refuses a non-loopback bind unless the override is passed explicitly, so publishing
    this port is something a person did on purpose and can be seen to have done on purpose.
    """
    if host in LOOPBACK_HOSTS or allow_unsafe:
        return host
    raise ValueError(UNSAFE_BIND_ERROR.format(host=host, safe=", ".join(sorted(LOOPBACK_HOSTS))))


def confine(root: Path, candidate: str) -> Optional[Path]:
    """Resolve `candidate` beneath `root`, or None if it escapes (WEBCH-40).

    `--ui-dir` and `--replay` both take a user path, so both get the same treatment the harness
    already gives its trust and grant stores: compare REALPATHS. Resolving both sides is what
    closes the symlink case — a link inside the UI directory pointing at `/etc/passwd` resolves
    outside the root and is refused, where a string-prefix check would have served it.
    """
    root_real = root.resolve()
    try:
        target = (root_real / candidate.lstrip("/")).resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    if target == root_real:
        return target
    return target if root_real in target.parents else None


def constant_time_match(presented: Optional[str], expected: str) -> bool:
    """Compare a presented credential against the real one without leaking its length in time."""
    if not presented:
        return False
    return secrets.compare_digest(presented, expected)
