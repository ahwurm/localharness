# Running agents locally: where LocalHarness fits

**The harness for local.** A polished agent harness built for the models you run yourself — smart endpoint and model detection, context management for small windows, defaults tuned for local instead of retrofitted from a cloud tool. The payoff: a working agent on hardware you own, at zero marginal cost.

*An honest map of the field, reviewed 2026-06-16 against official docs (sources at the bottom). If something here is out of date or wrong, open an issue — we keep it accurate, including where other tools are the better choice.*

## Who this is for

You're here for one of two reasons: you want agents running **on your own hardware**, or you want the inference bill to go **to zero**. Maybe both.

This page is not for reducing costs by routing to a cheaper hosted model or running a smaller model behind a metered API. Those still bill per token and the work still isn't yours. This is about local and zero — and the shortest path to it.

## The two things that make local actually work

A cloud tool pointed at `localhost` is not the same as a tool built for local. Local models bring real constraints a frontier API hides:

- **Endpoint and model detection.** `localharness init` probes vLLM, Ollama, LM Studio, and llama.cpp, finds your running model, and checks whether it supports native tool-calling — so you don't hand-wire any of it.
- **Context management for small windows.** Local context budgets are tighter than frontier ones. The harness manages context, caps tool output, and compacts before it overruns, which is the difference between an agent that finishes and one that derails.

That's the substance behind "zero cost." A local model holds up here because the harness does the work the frontier used to do for you.

## The field, by what matters for local

| | Runs on a model you own | Zero marginal cost | Built *for* local's pains | Time to a working agent | Agents as config, not code |
|---|---|---|---|---|---|
| **LocalHarness** | Yes — the design center | Yes, on your hardware | Yes — endpoint/model detection + context management | Fast: `init` then `start`, the orchestrator is ready, no wiring | Yes — YAML |
| **Claude Code** | No (Claude models only) | No (subscription / API) | n/a | Fast, but cloud | Settings + markdown |
| **OpenCode** | Can point at a local endpoint | Yes, if you run local | Partial — local is a config option, not the design center | Fast | JSON |
| **Goose** | Can point at a local endpoint | Yes, if you run local | Partial — general agent that supports local | Fast | YAML recipes |
| **LangChain / LangGraph** | Via the model layer | Yes, if you run local | No — you build the local handling yourself | Slower: you write the orchestration | Code |
| **Ollama** | It *is* your local model server | Yes | It's the inference layer, not the agent | n/a — you still need a harness on top | n/a |
| **OpenRouter / hosted routing** | No (hosted models) | No (metered per token) | n/a | Fast | n/a |

A few honest notes on the table:
- **OpenCode and Goose can absolutely run on local models** — they're good tools. The difference is that local is one option among many for them, where it's the entire design center here. If local is your main case, defaults built around it matter.
- **LangChain/LangGraph is a different category** — a powerful framework you build agents *with*, the most mature in the space, with the best evaluation tooling (LangSmith). If you want to build a custom agent system against any model, it's a strong choice. It just isn't trying to be the short path to a local agent.
- **Ollama is complementary, not a competitor.** It runs the model; LocalHarness runs the agent on top of it (or vLLM, LM Studio, llama.cpp).

## Where LocalHarness is honestly behind

- **Maturity.** It's early (v0.1). The tools above have far more production usage and larger communities.
- **Ecosystem and integrations.** Fewer prebuilt integrations than LangChain or the larger coding agents.
- **Sandboxing.** Deny-first permission patterns today, no OS-level sandbox yet (on the roadmap). For hard isolation now, run it in a container or VM.
- **Eval depth.** The built-in benchmark measures the harness against your own model; it's not a general-purpose eval platform like LangSmith.

## The short version

If you want to build a custom agent system against cloud models with the deepest ecosystem, reach for a general framework. If you want a working agent **on a model you own, at zero marginal cost, fast** — that's the one job LocalHarness is built for, and it handles the local-specific work that general tools leave to you.

```bash
git clone https://github.com/ahwurm/localharness.git
cd localharness && uv sync
uv run localharness init    # detects your local endpoint + model
uv run localharness start   # working agent, zero config
```

---

### Sources
Claude Code: https://code.claude.com/docs/en/overview · https://code.claude.com/docs/en/model-config ·
OpenCode: https://opencode.ai/docs/providers/ · https://opencode.ai/docs/config/ ·
Goose: https://github.com/block/goose · https://goose-docs.ai/docs/getting-started/providers/ ·
LangGraph: https://docs.langchain.com/oss/python/langgraph/overview · LangSmith: https://docs.langchain.com/langsmith/evaluation-concepts ·
Ollama: https://github.com/ollama/ollama · OpenRouter: https://openrouter.ai/docs ·
LocalHarness: https://github.com/ahwurm/localharness
