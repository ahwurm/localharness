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

## Scope notes

LocalHarness runs tools — including `bash` and file writes — on the machine
where the harness runs, driven by a local model. Treat agent definitions and any
connected MCP servers as trusted code. Secure your inference endpoint (see
"Running the harness on a different machine than the model" in the README):
never expose a bare, unauthenticated endpoint to an untrusted network.
