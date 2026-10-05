# LocalHarness 0.16.4 validation

Release contributors: [ahwurm](https://github.com/ahwurm) and Codex (AI coding assistant).

Baseline: `e33a5f247717bf5c6a0cdba2dc91bba625d85617` (0.16.3 plus its CI fix).
The release tag identifies the candidate. No persistent-memory retrieval or writes were used
in the behavioral comparison. Private source evidence and raw evaluation traces are excluded
from the repository and distributions.

## Deterministic checks

Feature regressions live in `test_message_provenance.py`, `test_active_references.py`, and
`test_task_context.py` under `tests/unit/`. They exercise provider-native/XML payloads,
quoted human markers, summary lineage, real eviction and compaction, repaired tool pairs,
bounded/missing snapshots, complete prompt and XML overflow, revision-sensitive receipts,
failed gates, missing children, checkpoints, and the direct ordinary-query path.

Existing context, agent-loop, self-check, steering, restart/resume, and prompt snapshot tests
were also run. Snapshot updates contain the intended compact role guidance and canonical
origin metadata. Full-suite CI on the exact release commit is a publication gate.

Distribution checks include wheel and sdist builds, a clean wheel install with freshly resolved
dependencies, upgrade from PyPI 0.16.3, version/CLI alignment, validation of an unchanged 0.16.3
agent configuration, and a live ordinary query in each installed environment. Both installed
environments answered `17 + 25` as `42` in one call. Fresh-wheel plugin mount/example checks
passed (39 tests). Distribution contents were inspected for unrelated/private files.

## Preregistered behavioral comparison

Eight cases and criteria were recorded before either run. Both variants used the configured
`swift-qwen3.8-27b` model, temperature 0, a 2,048-token output cap, a 32,768-token context window,
three permitted tool actions and a three-minute turn budget. The real AgentLoop and provider
client ran against the same endpoint. One deterministic fixture tool read task-local disk data;
no memory plugin or memory tools were loaded, and prior-session compact files were disabled.

Release criteria: both quick cases must use one model call in both variants; deterministic
regressions must pass; reviewed candidate outcomes must have no unresolved material regression.

| Case | Reviewed outcome in both variants |
| --- | --- |
| Quick arithmetic | `42`, one call |
| Answer-only explanation | One sentence, one call, no execution |
| Missing source | Reports missing interview material; invents no report |
| Outline checkpoint | Stops before drafting; retains seven-versus-six mismatch |
| Reference and gate | Revises one sentence, reports failed old gate and absent rerun |
| Failed critic | Keeps critique pending; invents no findings |
| Disk continuation/conflict | Preserves prior progress; asks which conflicting rule takes priority |
| Missing child | Preserves known work; leaves reviewer outcome unresolved |

| Measurement (8 cases per variant) | Baseline | Candidate |
| --- | ---: | ---: |
| Model calls | 14 | 14 |
| Fixture tool dispatches | 6 | 6 |
| Reported input tokens | 6,754 | 8,115 |
| Reported output tokens | 1,765 | 1,982 |
| Aggregate elapsed seconds | 63.90 | 73.78 |

All eight cases met the stated outcome criteria in both variants. No additional model call
was introduced. Candidate prompt overhead and sampled latency increased; there is no claim
of faster responses. Review was performed by the implementing agent, unblinded, with one sample
per case. This is not an independent quality assessment or a statistically powered experiment.
The revised example sentences still make an unsupported frequency judgment ("usually"); the
comparison does not establish good prose or general factual accuracy.

The live cases do not force compaction or exercise a stalled child. Reference packing,
provenance markers, and structured completion enforcement are covered deterministically;
universal model attribution, reference selection, quality, and child liveness remain unproven.
