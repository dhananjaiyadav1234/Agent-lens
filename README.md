# AgentLens

**Framework-agnostic observability, debugging, and AI-powered failure analysis for AI agents.**

AgentLens captures what your agents actually do, normalizes it into a universal
event model, and helps you find out *why* a run failed — first with deterministic
detectors, then with optional LLM-based analysis.

## Why

Agent frameworks (LangGraph, CrewAI, and others) each emit their own traces in
their own shapes. AgentLens sits underneath them:

```
AI Agent Framework
        │
Framework Adapter
        │
Universal AgentLens Event Model
        │
Tracing and Storage
        │
Deterministic Detectors
        │
AI Analysis
        │
REST API and Dashboard
```

## Principles

- Works locally with minimal infrastructure — SQLite, no cloud dependencies.
- The core is framework-agnostic; frameworks integrate through adapters.
- All framework-specific data is converted into one universal internal event model.
- Deterministic detection comes before LLM-based analysis; LLM analysis
  complements the detectors, it does not replace them.
- Modular, testable, production-quality code with automated tests for every
  significant feature.

## Status

Early development. This repository currently contains the project foundation and
package skeleton only — tracing, storage, and detectors are not yet implemented.

## Project layout

| Path                    | Purpose                                              |
| ----------------------- | ---------------------------------------------------- |
| `agentlens/core/`       | Core runtime and orchestration primitives           |
| `agentlens/models/`     | Pydantic models for the universal event model       |
| `agentlens/detectors/`  | Deterministic failure detectors                     |
| `agentlens/storage/`    | Tracing persistence (SQLite for local development)  |
| `agentlens/integrations/` | Framework adapters (LangGraph, CrewAI, …)         |
| `tests/unit/`           | Fast, isolated unit tests                            |
| `tests/integration/`    | Cross-component integration tests                    |
| `tests/e2e/`            | End-to-end scenario tests                            |
| `examples/`             | Runnable usage examples                              |
| `benchmarks/`           | Performance and detection-quality benchmarks         |
| `docs/`                 | Documentation                                        |

## Development

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the tests:

```bash
pytest
```

Run the linter:

```bash
ruff check .
```

## Technology

Python 3.11+ · Pydantic · FastAPI · SQLAlchemy · SQLite · pytest · Ruff ·
Next.js + TypeScript (dashboard) · Docker / Docker Compose

## License

Apache-2.0.
