# Changelog

All notable changes to LocalHarness are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/) (pre-1.0: interfaces may change).

## [0.5.1] — 2026-06-26

### Added
- **Number provenance for over-window document reading.** LocalHarness reads documents
  larger than the model's context window by fully reading every section and combining the
  notes — losslessly, on local hardware. As of 0.5.1, when that combine runs in multiple
  stages, every figure in the final answer is checked back against the verbatim per-section
  notes the model actually read, and any figure that doesn't trace back is surfaced. This
  keeps the numbers in an over-window answer anchored to their source — built for financial
  filings, contracts, and long reports, where a drifted figure is unacceptable.

  Verified by deterministic tests in both directions (through the real reduction path), an
  independent adversarial review, and a live run on a real 27B local model over a
  ~600k-character filing (0 of 24 figures flagged on a faithful answer; all 24 confirmed
  present in the notes). Surfaced as a warning, never a silent rewrite; engages on large
  multi-stage reductions; figure-matching is heuristic.

## [0.5.0] — 2026-06-25

### Added
- **Lossless, secured over-window context.** Cross-agent content grants (hand a child a
  handle, not re-inlined bytes) plus a "cruncher" subagent that reads an over-window
  document by handle and faithfully map-reduces it — every chunk fully read, nothing
  truncated — with structure-aware splitting and per-section context headers.
- A **capability floor**: untrusted-ingested web/tool content cannot co-reside with
  host-dangerous tools in the same agent (defense-in-depth above model refusal).
- Bench instrumentation for over-window eviction, plus a scored faithfulness scenario.
