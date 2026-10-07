# Spec 15 — `/mcp` Slash Command: In-Session MCP Server Management

**Status:** Draft (revised — grounded in actual config shape)
**Version target:** 0.17.0
**API version:** `PLUGIN_API_VERSION` unchanged ("1")

---

## 1. Problem

Today, MCP servers are configured by hand-editing YAML in an agent file (`agents/<name>.yaml` → `tools.mcp_servers`). There is no in-session way to:

- See which MCP servers are configured for the session's agent and whether they are connected.
- Add a new server without leaving the terminal.
- Edit an existing server's command, args, env, or URL.
- Test connectivity before committing the config.
- Remove a server.

This is the same gap `/plugins` fills for plugins (spec 09): a config surface that is currently file-only, made interactive.

## 2. Config shape (ground truth)

MCP servers are **per-agent**, not a top-level config key. The relevant model:

```python
# src/localharness/config/models.py
class ToolConfig(BaseModel):
    inherit: list[Literal["global", "division", "org"]] = ...
    add: list[str] = ...
    deny: list[str] = ...
    mcp_servers: list["MCPServerConfig"] = ...  # ← here

class MCPServerConfig(BaseModel):
    name: str
    transport: Literal["stdio", "streamable_http"]
    command: Optional[str] = None      # required for stdio
    args: list[str] = []
    env: dict[str, SecretStr] = {}
    url: Optional[str] = None          # required for streamable_http
    headers: dict[str, SecretStr] = {}
    timeout_seconds: float = 30.0
```

MCP servers live in `agents/<name>.yaml`:

```yaml
# .localharness/agents/orchestrator.yaml
tools:
  mcp_servers:
    - name: gsd
      transport: stdio
      command: node
      args: ["/path/to/gsd-mcp.cjs"]
      env:
        GSD_PROJECT_PATH: "/home/awurm/myproject"
```

The session's agent is resolved by `default_root_agent(agents: list[dict]) -> dict` in `start_cmd.py:166`: prefers `"default"`, then `"orchestrator"`, then the first discovered agent. It returns the agent's **data dict** (read `.get("name")` for the name), not a bare string.

**There is no global `mcp_servers` list.** Every MCP server is scoped to an agent. The "layer" for an MCP server is the agent file layer (workspace agent file vs global agent file), not the config override layer.

## 3. Existing trust gate (G10 — already implemented)

The trust gate for workspace MCP servers **already exists** in `config/trust.py` + `start_cmd.py`:

- `executables_snapshot(workspace_dir)` snapshots every `tools.mcp_servers` entry in `<workspace>/agents/*.yaml`, as `{file, name, transport, command, args, env, url, headers}` (env/header **names** only, never values).
- `fingerprint(snapshot)` computes a SHA-256 digest of the canonical JSON.
- `recorded_executables(workspace_root)` reads the previously approved fingerprint from `~/.localharness/trusted_workspaces.yaml`.
- `cli/workspace.py` exposes `decide_project_trust()` (documented in `trust.py:19`); `start_cmd.py` calls it **once at start** (and again at `/plugins` restart). It compares the current snapshot's fingerprint to the recorded one. On mismatch (new/changed MCP servers) it prompts the user; on `y` it calls `record_executables()` to store the new fingerprint.
- The loader (`loader.py:1545-1556`, `_without_project_mcp`) **strips** `tools.mcp_servers` from an untrusted project's **workspace** agent file before the merge (call sites `loader.py:1239-1240` and `1993-1994`), so no untrusted workspace MCP server can connect at load time.

**Known gap in the existing gate (must be addressed by this spec, §7.5/§9):** `executables_snapshot` fingerprints `command`, `args`, and `url` as full values, but env/`headers` as **names only** (`trust.py:258`, `_names` at `trust.py:230-232`). A value-only change (e.g. swapping a `TOKEN` env value) does **not** trip the fingerprint. This spec adds value-hashing to the snapshot (§7.5) so a secret swap re-prompts.

**Fail-open default (load-bearing):** `ConfigLoader.__init__` defaults to `project_trusted: bool = True` (`loader.py:716`) — "only `start` decides trust, and nothing else starts a server." Any in-session code path that reads/writes the agent file **does not** pass through the startup gate. This is why §7.6 must re-run the trust check before hot-connecting, and why §7.5 must not rely on the loader strip to protect an in-session add.

**The spec does NOT propose a new trust gate for the startup/load path.** It reuses the existing mechanism, adds value-hashing to the snapshot, and makes `/mcp` aware of trust state: when the session's agent file is in an untrusted workspace, `/mcp` shows a notice that the servers are withheld until trust is granted.

## 4. Goals

| # | Goal | Priority |
|---|------|----------|
| G1 | List all MCP servers configured for the session's agent, with connection status | Must |
| G2 | Add a new MCP server (stdio or streamable_http) interactively | Must |
| G3 | Edit an existing server's fields | Must |
| G4 | Test connectivity (connect, list tools, show count) without persisting | Must |
| G5 | Remove a server from config | Must |
| G6 | Frictionless: ≤ 3 keystrokes from `/mcp` to the most common action (list) | Should |
| G7 | Narrow changes: no new module in `src/localharness/tools/`; reuse existing `MCPServerClient` for test | Should |
| G8 | Config writes go to the session's agent file (`agents/<name>.yaml` → `tools.mcp_servers`), at the layer matching the session context (workspace agent file when in a project, global agent file otherwise) | Must |
| G9 | Provenance: the list view shows which agent file defines each server (workspace agent vs global agent) | Should |
| G10 | Trust gate: reuse the existing `config/trust.py` mechanism — no new trust logic | Must (already done) |

## 5. Non-Goals

- No MCP server discovery/install (that's a `plugins install`-class feature, out of scope).
- No per-agent MCP scoping from `/mcp` (the command operates on the session's agent; other agents' MCP servers are edited by hand).
- No MCP tool execution from `/mcp` (tools are already callable by the model once connected).
- No change to `MCPServerClient`, `MCPClientManager`, or `MCPToolWrapper` internals.
- No new trust gate — the existing `config/trust.py` mechanism is reused (with one fix: value-hashing of env/header values in `executables_snapshot`, §7.5).
- No new config merge logic — the existing agent-file layering (workspace agent shadows same-name global agent) already handles provenance.
- No top-level `mcp_servers` key — MCP servers remain per-agent.

## 6. UX

### 6.1 Command surface

```
/mcp                    # list all servers for the session's agent (default action)
/mcp add                # add a new server to the session's agent
/mcp edit <name>        # edit an existing server
/mcp test <name>        # test connectivity (one-shot, no persist)
/mcp remove <name>      # remove a server
/mcp help               # show this help
```

**Agent scope (G8):** all commands operate on the **session's agent** (resolved by `default_root_agent()`). The write target is that agent's YAML file at the layer matching the session context: the workspace agent file (`.localharness/agents/<name>.yaml`) when running inside a project, the global agent file (`~/.localharness/agents/<name>.yaml`) otherwise.

If the session's agent file exists in both layers (workspace shadows global), `/mcp` operates on the **workspace** file. There is no `--global` flag for v0.17 — the session's agent is the scope, and the layer is determined by the session context.

### 6.2 List view (`/mcp`)

Renders a table (rich `Table`, same style as `/plugins`):

```
┌────────────────────────────────────────────────────────────────────────┐
│ MCP Servers — agent: orchestrator                                      │
├──────────┬────────────────┬──────────┬──────────────┬─────────────────┤
│ Name     │ Transport      │ Status   │ Source       │ Tools           │
├──────────┼────────────────┼──────────┼──────────────┼─────────────────┤
│ gsd      │ stdio          │ ● connected │ [workspace] │ 12 tools       │
│ github   │ streamable_http│ ○ not connected │ [global] │ —             │
└──────────┴────────────────┴──────────┴──────────────┴─────────────────┘

  /mcp add        Add a new server
  /mcp edit <n>   Edit a server
  /mcp test <n>   Test connectivity
  /mcp remove <n> Remove a server
```

- **Agent** is shown in the table title (the session's agent name).
- **Status** is read from `MCPClientManager.connected_servers` if a session is running; `○ not connected` if the server is configured but not started.
- **Tools** count comes from `MCPServerClient.tools` length when connected; `—` otherwise.
- **Source** shows provenance (G9): `[workspace]` when the entry comes from the workspace agent file, `[global]` when from the global agent file. A same-name server defined in both shows `[workspace]` (it shadows the global one) and the global one is hidden.
- If no servers are configured: one line — `No MCP servers configured for 'orchestrator'. /mcp add to create one.`
- If the workspace is untrusted and servers are withheld: a notice line — `⚠ 2 servers withheld (workspace not trusted). Run 'localharness start' to grant trust.`

### 6.3 Add flow (`/mcp add`)

Step-by-step prompts (each a single-line input, pre-filled where sensible):

```
Name:            gsd
Transport:       [1] stdio  [2] streamable_http  → 1
Command:         node
Args:            /home/awurm/.npm/_npx/gsd-core/bin/gsd-mcp.cjs
Env (key=value, blank to skip):
  GSD_PROJECT_PATH=/home/awurm/myproject
  (blank to finish)
```

For `streamable_http`:
```
Name:            github
Transport:       [1] stdio  [2] streamable_http  → 2
URL:             http://127.0.0.1:3000/mcp
Headers (key=value, blank to skip):
  Authorization=Bearer <token>
  (blank to finish)
```

After collection:
1. Validate with `MCPServerConfig` (pydantic).
2. Run a **one-shot connect test** (see §7.2). Show tool count or the error.
3. Ask: `Save? [y/N]` — on `y`, write to the agent file.
4. If the session is running and the server connects: register its tools in the live `ToolRegistry` (hot-add, no restart).

**Frictionless path:** the most common case (stdio, one command) is 4 prompts: name, transport (default 1, just Enter), command, args. Env is optional.

### 6.4 Edit flow (`/mcp edit <name>`)

Pre-fills all fields from the existing config. User can change any field; blank = keep current. Same validation + test + save as add.

### 6.5 Test flow (`/mcp test <name>`)

One-shot: build an `MCPServerConfig` from the saved config, call `MCPServerClient.connect()`, print tool count and names, then `disconnect()`. No config write. If the server is already connected in the live session, just re-list its tools (no re-connect).

```
Testing 'gsd'…
  ✓ Connected — 12 tools:
    gsd__state_load, gsd__state_update, gsd__phase_add, …
```

On failure:
```
Testing 'gsd'…
  ✗ Connection failed: spawn ENOENT — command 'node' not found
```

### 6.6 Remove flow (`/mcp remove <name>`)

```
Remove MCP server 'gsd'? [y/N]
```

On `y`: remove the entry from the agent file, and if the session is running, `disconnect()` + unregister its tools from the live registry.

## 7. Implementation

### 7.1 New file

**`src/localharness/cli/mcp_cmd.py`** — the slash command handler.

This is the ONLY new source file. It contains:

- `run_mcp_slash(repl, args: str) -> None` — the entry point, called by the `OrchestratorREPL._slash_mcp` method (§7.7). `repl` is the `OrchestratorREPL` instance (session state is reached via `repl`, not a passed `ctx`); `args` is the text after `/mcp` as a `str`.
- Internal helpers: `_list(repl)`, `_add(repl)`, `_edit(repl, name)`, `_test(repl, name)`, `_remove(repl, name)`.
- Agent file read/write via the existing agent-file layering in `config/loader.py` (workspace agent shadows same-name global agent). The concrete enumeration helper is `layer_files(layer_dir, "agents", contained=True)` (`loader.py:562`), which lists one layer's `agents/*.yaml`.
- Trust status via the existing `config/trust.py` (`recorded_executables`, `executables_snapshot`).

The slash handler is a **core slash command** (registered in `src/localharness/cli/slash_commands.py`'s core table), not a plugin. This keeps it always available, no plugin enable required.

### 7.2 Test connectivity (one-shot)

Reuses `MCPServerClient` directly:

```python
from localharness.tools.mcp import MCPServerClient

async def _test_connect(config: MCPServerConfig) -> tuple[bool, str, list[str]]:
    client = MCPServerClient(config)
    try:
        await client.connect()
        names = [t.info().name for t in client.tools]
        return True, f"{len(names)} tools", names
    except Exception as exc:
        return False, str(exc), []
    finally:
        await client.disconnect()
```

No changes to `mcp.py`.

### 7.3 Config write (agent-file, G8)

MCP servers live in `agents/<name>.yaml` → `tools.mcp_servers`. The write path:

- **Agent resolution:** the session's agent name (from `default_root_agent(agents)` — read `.get("name")` — or the `--agent` flag).
- **Layer resolution:** the workspace agent file is `.localharness/agents/<name>.yaml` (under the session's `local_config_dir`); the global one is `~/.localharness/agents/<name>.yaml`. `/mcp` picks the workspace file when it exists (it shadows the global one), else the global file. `layer_files` (`loader.py:562`) enumerates each layer's `agents/*.yaml` to confirm existence.
- **Read:** load the agent file at the active layer → `tools.mcp_servers` list.
- **Write:** update the `tools.mcp_servers` list in the agent file at the active layer. The write is a targeted YAML edit (read the file, modify the `tools.mcp_servers` list, write back), not a full-file rewrite.

The agent file is the source of truth for MCP servers. There is no `overrides.yaml` involvement — MCP servers are not a config-override key, they are an agent-file key.

**Edit/remove layer rule:** when the agent file exists in both layers, `/mcp` operates on the **workspace** file (it shadows the global one). To edit the global agent file, the user edits it by hand (out of scope for v0.17).

### 7.4 Provenance tracking (G9)

The list view needs to show which agent file each server entry comes from. The existing agent-file layering in `config/loader.py` already resolves which file wins (workspace shadows same-name global). The `/mcp` list view:

- Reads the workspace agent file's `tools.mcp_servers` (if it exists).
- Reads the global agent file's `tools.mcp_servers` (if it exists).
- For each server name, determines the source: `[workspace]` if in the workspace file, `[global]` if only in the global file.
- A same-name server in both files shows `[workspace]` (it shadows the global one); the global entry is hidden.

No new merge logic — this is a read-only inspection of the two agent files.

### 7.5 Trust gate awareness + value-hashing fix (G10)

The existing trust gate in `config/trust.py` + `cli/workspace.py` (`decide_project_trust`) already handles workspace MCP server trust at **start**. `/mcp` is **aware** of it and **closes one gap** in it:

**Awareness (no new gate for the startup/load path):**
- At list time, `/mcp` checks `recorded_executables(workspace_root)` vs `fingerprint(executables_snapshot(workspace_dir))` to determine if the workspace is trusted.
- If the workspace is untrusted and the agent file has `tools.mcp_servers` entries, the list view shows a notice: `⚠ N servers withheld (workspace not trusted).`
- `/mcp add` in an untrusted workspace: the write succeeds (the agent file is updated), but the server will **not** hot-connect in-session (see §7.6). It will not connect at all until trust is granted at the next `start`. The add flow shows: `Note: this workspace is not trusted. The server will not connect until you grant trust at 'localharness start'.`

**Value-hashing fix (the change to `trust.py`):** `executables_snapshot` (workspace, `trust.py:258`) **and** `machine_snapshot` (global, `trust.py:484-485`) both fingerprint env/`headers` as **names only** via `_names` (`trust.py:230-232`), so a value-only change (swapping a `TOKEN` env value) does not trip **either** fingerprint. This spec adds a small shared helper, e.g. `_value_digests(mapping: Mapping[str, SecretStr]) -> dict[str, str]`, that maps each env/header key to the **SHA-256 hex of its value**. There is **no salt** — the value is already non-public and the digest is one-way, so a salt only adds management for no security gain, and an unsalted digest is stable across runs (a salted one would re-prompt on every run if the salt weren't persisted). Both snapshots use the helper in place of `_names`, so a value change now re-prompts on **both** the workspace and the global/machine path. The raw value is never stored — the no-plaintext property is preserved.

**One-time migration (documented, safe):** because the snapshot's canonical JSON changes, **every existing trusted workspace re-prompts exactly once** on the next `start` after this ships (its stored name-only fingerprint no longer matches). This is a single, bounded re-confirmation in the fail-closed direction — acceptable, no code migration required. The stored `servers` name-list (`record_executables`, `trust.py:284-294`) remains valid for the name-comparison in §7.6.

**Trusted-when-unrecorded:** `recorded_executables` returns `None` for a never-recorded workspace (`trust.py:276`). `/mcp` treats `None` as **untrusted/withheld** (fail-closed), consistent with the loader strip.

- No new trust gate. The existing `decide_project_trust()` / `record_executables()` / loader-stripping mechanism is unchanged apart from the value-hashing above.

### 7.6 Hot-add / hot-remove in a live session

When the session is running (the slash command is called from the terminal REPL), the `MCPClientManager` and `ToolRegistry` instances are reached **via the `OrchestratorREPL` (`repl`)** — the handler has no `ctx` parameter; it accesses session state through `repl` (confirm the exact attribute names on the REPL/orchestrator during implementation).

**Trust re-check before hot-connect (closes the in-session bypass):** the startup trust gate runs only at `start` (`cli/workspace.py`), and `ConfigLoader` defaults to `project_trusted=True` (`loader.py:716`), so an in-session add would otherwise connect a new server with **no** trust prompt. To prevent that, before hot-connecting on add/edit, `/mcp` must:
1. Recompute `executables_snapshot(workspace_dir)` for the updated agent file.
2. If the workspace is **untrusted** (fingerprint not in `recorded_executables`), **do not hot-connect** — show the §7.5 notice and stop. The server only connects at the next trusted `start`.
3. If the workspace is **trusted** but the fingerprint **changed** (a new/changed server), isolate the changed/new servers by a **whole-entry comparison**: `record_executables` (`trust.py:284-294`) persists the full `servers` snapshot (name + command + args + env + url + headers), so compare each current `executables_snapshot` entry against the recorded entry of the same name — flagging a server as changed if its name is new **or** any field (including a value-only `command`/`args`/`env`/`url` change) differs. This is the same whole-entry comparison `decide_project_trust` already uses (`cli/workspace.py:491`); no new function is required. Prompt a TTY `y/N` for each new/changed server before `connect()`. On `n`, the agent file is updated but that server is not connected this session.

Only after that gate passes:
- **Add:** create an `MCPServerClient`, `await connect()`, `await registry.register(tool, scope="mcp")` for each tool (both are `async`). The model sees the new tools on the next turn. No restart.
- **Remove:** `await registry.unregister(name, scope="mcp")` for each tool, then `await client.disconnect()`.
- **Edit:** disconnect old, update agent file, run the trust re-check, connect new, register new tools.

**Defense-in-depth:** the hot-connect helper in `mcp_cmd.py` re-verifies the trust state **itself** (not just the calling flow), so a future code path that reaches `connect()` cannot skip the re-check.

This resolves the §7.5/§7.6 tension: hot-connect is **withheld** for an untrusted workspace and **gated by a per-server confirm** for a trusted-but-changed one.

### 7.7 Slash table registration

The slash table row is a `SlashCommand` (`cli/slash_commands.py:18`), **not** `SlashRow`. Its fields are `name`, `description` (not `help`), and `handler`. For a **core** row, `handler` is the **name of an `OrchestratorREPL` method** (a `str`), not a `module:func` path — a `module:func` string is only legal for a *plugin* row (via `set_plugin_rows`), and this command is a core row, not a plugin. So the registration has two parts:

1. **`cli/slash_commands.py`** — add one row to the core table:
   ```python
   SlashCommand(
       name="/mcp",
       description="Manage MCP servers for the session's agent (list, add, edit, test, remove)",
       handler="_slash_mcp",
       takes_args=True,
   )
   ```
2. **`cli/repl.py`** — add the thin `OrchestratorREPL` method that the dispatcher calls. Core handlers have the contract `async def _slash_X(self, args: str, args_lower: str) -> None` (`repl.py:1263+`): `args` is the **text after the command** as a `str` (not a `list`), there is **no `ctx` parameter**, and the method renders directly (returns `None`). It delegates to `mcp_cmd.py`:
   ```python
   async def _slash_mcp(self, args: str, args_lower: str) -> None:
       from localharness.cli.mcp_cmd import run_mcp_slash
       await run_mcp_slash(self, args)   # `self` is the OrchestratorREPL — session state is reached via `self`, not a passed ctx
   ```

The `SlashCommandCompleter` picks the row up automatically (it reads `all_rows()` at completion time).

### 7.8 Files touched (narrow-change audit)

| File | Change | Lines (est.) |
|------|--------|-------------|
| `src/localharness/cli/mcp_cmd.py` | **NEW** — all slash logic (list, add, edit, test, remove, agent-file writes, provenance, trust awareness, secret masking) | ~300 |
| `src/localharness/cli/slash_commands.py` | Add one `SlashCommand` core row (`handler="_slash_mcp"`) | +6 |
| `src/localharness/cli/repl.py` | Add the thin `OrchestratorREPL._slash_mcp` method that delegates to `mcp_cmd` | +5 |
| `src/localharness/config/models.py` | No change (MCPServerConfig already exists) | 0 |
| `src/localharness/config/loader.py` | No change (agent-file layering + `layer_files` already exist) | 0 |
| `src/localharness/config/trust.py` | **Value-hashing:** shared `_value_digests` helper; `executables_snapshot` **and** `machine_snapshot` hash env/header **values** (SHA-256), not just names (§7.5) | ~15 |
| `src/localharness/tools/mcp.py` | No change | 0 |
| `src/localharness/cli/start_cmd.py` | No change (trust gate already wired) | 0 |
| `src/localharness/channels/terminal.py` | No change (slash dispatch is generic) | 0 |
| `tests/test_mcp_cmd.py` | **NEW** — unit tests | ~200 |
| `tests/test_trust_value_hash.py` | **NEW** — value-hashing regression tests | ~60 |

Total: **1 new source file (`mcp_cmd.py`) + 2 new test files, plus small edits to `slash_commands.py`, `repl.py`, `trust.py`.** The "narrow change" claim (G7) still holds — no new module in `tools/`, no new merge logic, `MCPServerClient` reused as-is — but it is **not** "1 new file + 1 line": `repl.py` and `trust.py` both change, and the `trust.py` change is behavioral (it changes a persisted fingerprint format, §7.5).

## 8. Config schema (unchanged)

The existing `MCPServerConfig` in `config/models.py` is used as-is:

```python
class MCPServerConfig(BaseModel):
    name: str
    transport: Literal["stdio", "streamable_http"]
    command: Optional[str] = None      # required for stdio
    args: list[str] = []
    env: dict[str, SecretStr] = {}
    url: Optional[str] = None          # required for streamable_http
    headers: dict[str, SecretStr] = {}
    timeout_seconds: float = 30.0
```

No new fields. If GSD's MCP server needs a specific env var, it goes in `env:`.

## 9. Security

- **Trust model:** an MCP server is external code. The existing trust model applies: every MCP tool asks once per (server, tool) on first call; `mcp_trusted_servers` (global-only) skips the ask. `/mcp add` does NOT auto-add to `mcp_trusted_servers` — the user must do that separately if they want to skip the per-tool ask.
- **Workspace trust gate (existing):** `config/trust.py` snapshots a project's agent-file `tools.mcp_servers`, fingerprints them, and the `start` path (via `cli/workspace.decide_project_trust`, called by `start_cmd.py`) gates untrusted projects. The loader strips `tools.mcp_servers` from untrusted projects before merge. `/mcp` is aware of this state (shows a notice) and adds value-hashing to the snapshot (§7.5); it does not change the gate mechanism.
- **Agent-file write:** `/mcp` writes to the session's agent file at the active layer (workspace agent file when in a project, global agent file otherwise). The write is a targeted YAML edit of the `tools.mcp_servers` list, not a full-file rewrite.
- **No secret exposure:** `env` and `headers` values are `SecretStr` in the config model. The list view shows `●` for set values, never the value itself. The edit flow pre-fills with `●●●●` and only writes back if the user types a new value.
- **Secret masking in the test path:** `_test_connect` (§7.2) returns `str(exc)` on failure. Before display, that string must be **sanitized**: any configured `url` (which may carry a query-string token) and any env/header **value** is replaced with `●●●●`. A raw `str(exc)` from `MCPServerClient.connect()` can embed the URL or a header, so it is never printed unsanitized.
- **Secret-safe write path:** the targeted YAML edit must preserve existing secret values **opaquely** — it reads the raw YAML node and rewrites only the `tools.mcp_servers` list, and must **never** `str()`/`json.dumps` a `SecretStr` (which would persist the plaintext). The write path and any log line must not serialize a config object. `test_secret_not_exposed` (§10.1) covers the list, test-error, and write paths.
- **In-session trust re-check:** `/mcp add`/`edit` re-run the trust check before hot-connecting (§7.6), so a mid-session add cannot bypass the startup gate — the highest-risk action is gated, not just the load path.

## 10. Test plan

### 10.1 Unit tests (`tests/test_mcp_cmd.py`)

- `test_list_empty` — no servers configured → "No MCP servers" message.
- `test_list_configured` — two servers, one connected → correct table with Source column.
- `test_list_provenance` — one server in workspace agent file, one in global → correct `[workspace]` / `[global]` labels.
- `test_list_shadowing` — same-name server in both agent files → only `[workspace]` shown, global hidden.
- `test_list_untrusted` — workspace untrusted, servers withheld → notice line shown.
- `test_add_stdio_workspace` — inside a project, mock prompts, verify write to workspace agent file.
- `test_add_stdio_global` — outside a project, mock prompts, verify write to global agent file.
- `test_add_http` — mock prompts, verify config write.
- `test_add_untrusted_notice` — inside an untrusted project, verify notice shown after save.
- `test_edit` — pre-fill, change one field, verify write to correct agent file.
- `test_edit_shadowed` — same-name in both agent files, edit → workspace file changed.
- `test_test_connect_success` — mock `MCPServerClient.connect`, verify tool count.
- `test_test_connect_failure` — mock `connect` raising, verify error message.
- `test_remove` — verify agent-file write + tool unregistration.
- `test_remove_not_found` — "unknown server" message.
- `test_secret_not_exposed` — env values never appear in list output.

### 10.2 Integration test (next session, with GSD)

- Install GSD (`npx @opengsd/gsd-core@latest`).
- `/mcp add` → name `gsd`, transport stdio, command `node`, args `[<gsd-mcp-path>]`.
- Verify: connect succeeds, tool count > 0, tools appear in the session's tool list.
- `/mcp test gsd` → re-verify.
- `/mcp edit gsd` → change an env var, verify re-connect.
- `/mcp remove gsd` → verify tools gone.

## 11. Open questions

| # | Question | Default (if unanswered) |
|---|----------|------------------------|
| Q1 | Should `/mcp add` auto-add the server to `mcp_trusted_servers`? | No — explicit opt-in. |
| Q2 | Should `/mcp` support a `--agent <name>` flag to operate on a non-session agent? | No for v0.17 — the session's agent is the scope. |
| Q3 | Should the test flow show tool schemas, not just names? | Names only for v0.17; schemas in a `/mcp tools <name>` subcommand later. |
| Q4 | Should `/mcp` show MCP servers from ALL agents, not just the session's? | No for v0.17 — the session's agent is the scope. A `/mcp --all` flag is a future extension. |

## 12. Out of scope (future)

- `/mcp tools <name>` — show full tool schemas.
- `/mcp call <name> <tool> [args]` — invoke a tool from the slash (debugging).
- `/mcp --agent <name>` — operate on a non-session agent.
- `/mcp --all` — list MCP servers across all agents.
- MCP server discovery from a registry (analogous to `plugins install`).
- MCP server health monitoring / auto-reconnect (the existing `reconnect_server` is there; `/mcp` can expose it later).
