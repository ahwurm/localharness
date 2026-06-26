# Security Policy

## Supported versions

LocalHarness is early-stage (v0.x). Security fixes land on the latest `main`;
there are no long-term-support branches yet.

| Version | Supported |
|---------|-----------|
| `main` (latest) | ✓ |
| older tags | ✗ |

## Reporting a vulnerability

Please report security issues **privately** — do not open a public issue.

Use GitHub's private vulnerability reporting:
[**Report a vulnerability**](https://github.com/ahwurm/localharness/security/advisories/new).
Include reproduction steps and impact. You will get an acknowledgment, and a fix
or mitigation will be coordinated before any public disclosure.

## Trust boundaries

LocalHarness runs tools — including `bash` and file writes — on the machine where
the harness runs, driven by a local model. **Treat agent definitions and any
connected MCP servers as trusted code**: review them the way you would review code,
because they decide what the agents are allowed to do.

## Threat model: prompt injection

Agents fetch web pages and call tools, then act on what they read. The central risk
is **prompt injection**: attacker-controlled text in a fetched page or tool result
trying to make an agent take a host action it should not. This is not hypothetical —
the companion morning-report job runs agents with `bash` and web tools, on a
schedule, over live pages no human vetted first.

**Primary defense: separation, enforced structurally.** An agent that ingests
untrusted content is never the same agent that can mutate the host. Host-mutating
tools (`bash`, file `write`/`edit`) are kept out of any agent that fetches or ingests
untrusted text. This is enforced where an agent's tools are resolved: a host-mutating
toolset combined with untrusted ingestion is rejected, and the check **fails closed**
(deny on doubt). Untrusted content moves between agents only as opaque handles
carrying a sticky "untrusted" tag; its raw bytes resolve only inside an agent that
holds no host-mutating tools. This covers built-in web and tool-result ingestion and
MCP tools today; one known gap remains — a plugin pulled in through inherited global
scope still needs a per-tool ingestion tag to be caught.

**Not yet built: sandboxing.** Host-mutating tools currently run with the machine's
full trust; there is no OS-level sandbox (e.g. bubblewrap) around them yet. That is on
the roadmap. Until it ships, the separation above is the containment — so **run the
harness as a non-privileged user**, and isolate it in a container or VM if it will
process untrusted content on a machine you care about.

**Known residual (named, not closed).** The separation blocks *verbatim* untrusted
bytes from reaching a host-mutating agent. It does not fully block *laundered*
influence: an agent with no host tools can read untrusted content and hand a summary
to an agent that has them. Summarizing degrades an attacker's control but does not
eliminate it. Closing this fully is a larger change, deferred until a live test shows
it is exploitable on the target model.

**Not a current vector: memory.** Tool output is written to per-agent history, not to
the queryable facts memory, and no code path promotes tool output into the facts an
agent recalls. If that changes, this section changes with it.

## Securing the endpoint

Inference servers ship with no authentication. On a network with untrusted devices,
start the server with an API key and set `provider.api_key` to match; for access
beyond your LAN use a private overlay network (Tailscale/WireGuard). Never port-forward
a bare, unauthenticated endpoint to the internet. See "Running the harness on a
different machine than the model" in the README.
