# AgentLens

**Framework-agnostic observability, debugging, and deterministic issue detection for AI agents.**

AgentLens captures what your agents actually do, normalizes it into a universal
event model, and helps you find out *why* a run went wrong — with deterministic,
offline detectors you run on demand.

## What is AgentLens?

AgentLens is a local, framework-agnostic library for recording and inspecting AI
agent activity. Concretely, it:

- **Records** agent activity (tool calls, LLM calls, decisions, errors) as an
  ordered stream of events on a run.
- **Stores** runs, events, and issues — in memory by default, or in a local
  SQLite database.
- **Detects issues explicitly.** Nothing runs automatically in the background;
  you call `lens.detect(run_id)` when you want deterministic detectors (loops,
  excessive retries, duplicate tool calls, inefficiencies) analysed.
- **Produces reports** that aggregate a run with its events and issues, and
  **exports** them as deterministic JSON or Markdown strings.
- **Provides a read-only CLI** for inspecting a persisted database.
- **Provides optional framework adapters** (LangChain, the OpenAI Agents SDK,
  CrewAI) that record activity from those frameworks into an AgentLens run using
  the same public API application code uses.

AgentLens does not automatically fix agents, does not call any model or API on
your behalf, and does not run anything in the background. Detection is
deterministic and explicit; there is no LLM-based analysis in the current
version (see [Why AgentLens?](#why-agentlens) for that as a design principle,
not a shipped feature).

## Project layout

| Path                       | Purpose                                                |
| --------------------------- | ------------------------------------------------------ |
| `agentlens/core/`           | Core runtime: `AgentLens`, trace lifecycle, run manager |
| `agentlens/models/`         | Pydantic models for the universal event model           |
| `agentlens/detectors/`      | Deterministic failure detectors                         |
| `agentlens/storage/`        | Persistent storage (`SQLiteTraceStore`)                 |
| `agentlens/export/`         | JSON / Markdown report export                            |
| `agentlens/cli/`            | The read-only `agentlens` command-line tool              |
| `agentlens/integrations/`   | Optional framework adapters (LangChain, OpenAI Agents SDK, CrewAI) |
| `tests/unit/`               | Fast, isolated unit tests                                |
| `tests/integration/`        | Cross-component integration tests                        |
| `examples/`                 | Runnable, offline usage examples                          |
| `scripts/`                  | Release-readiness / clean-install verification            |

## Why AgentLens?

Agent frameworks (LangChain, CrewAI, the OpenAI Agents SDK, and others) each
emit their own traces in their own shapes. AgentLens sits underneath them as a
common, framework-agnostic layer: adapters translate each framework's native
events into one universal event model, so the same storage, detectors, reports,
CLI, and export logic work no matter which framework — or no framework at all —
produced the run. See [Architecture](#architecture) for the exact dependency
flow.

### Principles

- Works locally with minimal infrastructure — SQLite, no cloud dependencies, no
  network calls made by AgentLens itself.
- The core is framework-agnostic; frameworks integrate through optional
  adapters that depend on AgentLens, never the other way around.
- All framework-specific data is converted into one universal internal event
  model before anything else touches it.
- Detection is deterministic and explicit today. The design leaves room for
  optional LLM-based analysis to complement (not replace) the deterministic
  detectors later, but that analysis is **not implemented** in the current
  version — nothing in this repository calls a model.
- Modular, testable code with automated tests for every significant feature,
  enforced in CI on every push and pull request (see [Supported Python
  Versions](#supported-python-versions)).

## Installation

The PyPI distribution name is **`agentlens-evaluator`** (the `agentlens` name
was already taken by an unrelated package); the Python import name is
unaffected and remains `agentlens`. As of this writing the package has **not
yet been published**, so `pip install agentlens-evaluator` will not work until
a release is made — install from a local clone in the meantime (see below).
Once published, installation will be:

```bash
pip install agentlens-evaluator
```

```python
from agentlens import AgentLens  # import name is unchanged
```

### Core (from source, until the first PyPI release)

```bash
git clone https://github.com/dhananjaiyadav1234/Agent-lens.git
cd Agent-lens
python -m venv .venv
source .venv/bin/activate  # .venv\Scripts\activate on Windows
pip install --upgrade pip
pip install .
```

Verify the install:

```bash
python -c "from agentlens import AgentLens; print('AgentLens OK')"
agentlens --help
```

Core installation depends on **only** `pydantic` — no framework, no network
client, no optional dependency is required or imported.

### Optional framework integrations

Each framework adapter is an install extra; none is required for core usage.
From PyPI (once published):

```bash
pip install "agentlens-evaluator[langchain]"
pip install "agentlens-evaluator[openai-agents]"
pip install "agentlens-evaluator[crewai]"
```

From a local clone today, the same extras apply to the dot-path form:

```bash
pip install ".[langchain]"
pip install ".[openai-agents]"
pip install ".[crewai]"
```

Extras can be combined:

```bash
pip install ".[langchain,openai-agents,crewai]"
```

### Development install

```bash
pip install -e ".[dev]"
```

See [Development / Testing](#development--testing) for running the test suite
and linter, and [Framework Integrations](#framework-integrations) for adapter
usage.

## Quick Start

This is the complete basic workflow, using only core AgentLens APIs — no
framework, no network, no API key:

```python
from agentlens import AgentLens
from agentlens.models import EventType
from agentlens.export import export_json, export_markdown

lens = AgentLens()

with lens.trace("Look up a customer's order status") as trace:
    trace.record_event(
        event_type=EventType.TOOL_CALL_STARTED,
        name="lookup_customer",
        input={"customer_id": "123"},
    )
    trace.record_event(
        event_type=EventType.TOOL_CALL_COMPLETED,
        name="lookup_customer",
        input={"customer_id": "123"},
        output={"open_orders": 2},
        status="ok",
    )

run_id = trace.run_id

issues = lens.detect(run_id)  # explicit: detectors never run on their own
report = lens.get_report(run_id)  # explicit: aggregates the run + events + issues

print(report.summary.total_events)  # 4 (RUN_STARTED, 2 tool events, RUN_COMPLETED)
print(export_json(report))  # deterministic JSON string
print(export_markdown(report))  # deterministic Markdown string
```

This exact workflow is exercised by `tests/unit/test_readme_quickstart.py`, so
it will not silently drift from what's documented here.

## Detecting Issues

**Recording activity never automatically runs detectors.** `record_event` only
appends an event to the run; nothing analyses it until you explicitly call
`lens.detect(run_id)`. This is true with and without a framework integration —
adapters record events the same way application code does, and detection stays
a separate, explicit step you control.

Detectors are pure analysis components: given an `AgentRun` and its ordered
`AgentEvent` list they return validated `AgentIssue` objects. They depend only
on the universal models — never on `AgentLens`, storage, or a framework — so
they run on AgentLens traces, imported JSON traces, and framework-adapter
traces alike.

### `LoopDetector`

Identifies pathological repeated execution cycles. A **loop** is a contiguous run
of one or more logically-equivalent events repeated consecutively at least
`minimum_repetitions` times — `A B A B A B` is `cycle_length=2`, `repetitions=3`.

- Events are compared by a fingerprint of `event_type` + `name` + canonical JSON
  of `input` only. `output`, `status`, `duration_ms`, `metadata`, ids, and
  timestamps are ignored, so cycles that vary only in, say, `metadata.iteration`
  still match.
- Events are analysed in `sequence_number` order; the caller's list is not
  modified.
- `RUN_STARTED` / `RUN_COMPLETED` are excluded. `ERROR` events break a cycle —
  repeated failure/retry sequences are a retry pattern and get their own detector,
  not a loop.
- Default `minimum_repetitions` is **3** (must be an `int` ≥ 2).
- Severity by repetition count: **3 → MEDIUM**, **4–5 → HIGH**, **6+ → CRITICAL**.

```python
from agentlens import AgentLens
from agentlens.detectors import LoopDetector
from examples.demo_agents import run_looping_agent

lens = AgentLens()
run_id = run_looping_agent(lens)

run = lens.get_run(run_id)
events = lens.get_events(run_id)

issues = LoopDetector().detect(run, events)
# -> one AgentIssue: issue_type=agent_loop, severity=medium,
#    metadata={"detector": "loop_detector", "cycle_length": 2,
#              "repetitions": 3, "pattern": [...]}
```

`python -m examples.detect_loops` runs the detector against every demo scenario.

### `RetryDetector`

Identifies excessive retries of the same tool operation. A **failed attempt** is
a `TOOL_CALL_STARTED` immediately followed (in `sequence_number` order) by an
associated `ERROR`; a **retry streak** is two or more *consecutive* failed
attempts of the same operation. A streak that reaches `minimum_failed_attempts`
produces one issue describing those failures.

- **Operation identity** = tool `name` + canonical JSON of `input` (key order
  irrelevant; list order and primitive types significant). `output`, `status`,
  `metadata`, ids, and timestamps are ignored.
- **Error association**: if `ERROR.output` is a mapping with a string
  `"operation"`, it must equal the active tool name; if `"operation"` is absent,
  the adjacent `ERROR` is treated as associated; a mismatch breaks the streak.
- Any unrelated event — a different operation, a `DECISION`, a success, a
  lifecycle boundary — breaks the streak. Streaks never span `RUN_STARTED` /
  `RUN_COMPLETED`.
- A successful final attempt (`TOOL_CALL_STARTED` → `TOOL_CALL_COMPLETED`) ends
  the streak but **does not erase the preceding failures**: they are still
  reported, and the successful attempt is not among `related_event_ids`.
- Default `minimum_failed_attempts` is **3** (must be an `int` ≥ 2).
- Severity by failed-attempt count: **2–3 → MEDIUM**, **4–5 → HIGH**,
  **6+ → CRITICAL**.

```python
from agentlens import AgentLens
from agentlens.detectors import RetryDetector
from examples.demo_agents import run_retrying_agent

lens = AgentLens()
run_id = run_retrying_agent(lens)

issues = RetryDetector().detect(lens.get_run(run_id), lens.get_events(run_id))
# -> one AgentIssue: issue_type=excessive_retry, severity=medium,
#    metadata={"detector": "retry_detector", "operation": "fetch_customer",
#              "failed_attempts": 3, "input": {"customer_id": "123"}}
```

Detection is deterministic and fully offline.
`python -m examples.detect_retries` runs it against every demo scenario.

### `DuplicateToolDetector`

Identifies redundant repeated tool calls: a **second (or later) successful**
execution of the same operation with no new information justifying the redo.

- **Same operation** = identical tool `name` + canonical JSON of `input` (key
  order irrelevant, recursively; list order and primitive types significant).
- **Successful call** = a `TOOL_CALL_STARTED` immediately followed (in
  `sequence_number` order) by a matching `TOOL_CALL_COMPLETED`. A `DECISION`,
  `ERROR`, or second `TOOL_CALL_STARTED` in between invalidates the pair. Failed
  attempts are `RetryDetector`'s concern, not this one.
- **`related_event_ids`** = exactly four, in trace order: the original
  `TOOL_CALL_STARTED` / `TOOL_CALL_COMPLETED` and the duplicate
  `TOOL_CALL_STARTED` / `TOOL_CALL_COMPLETED`. No lifecycle or decision events.
- **Information boundary** — resets what counts as "already done": a successful
  `TOOL_CALL_COMPLETED` for a *different* operation, an `LLM_CALL_COMPLETED`, an
  event whose `metadata`/`output` sets `uses_new_information` /
  `depends_on_new_information` / `new_information` to `true`, or a malformed
  completion. A plain `DECISION`, an explicit `uses_new_information: false`,
  lifecycle events, and `ERROR` events do **not** reset it.
- **Severity** by the duplicate's ordinal in its region: **1st → MEDIUM**,
  **2nd–3rd → HIGH**, **4th+ → CRITICAL**. One issue per duplicate call; each
  names the original and that specific duplicate.

```python
from agentlens import AgentLens
from agentlens.detectors import DuplicateToolDetector
from examples.demo_agents import run_duplicate_tool_agent

lens = AgentLens()
run_id = run_duplicate_tool_agent(lens)

issues = DuplicateToolDetector().detect(lens.get_run(run_id), lens.get_events(run_id))
# -> one AgentIssue: issue_type=duplicate_tool_call, severity=medium,
#    metadata={"detector": "duplicate_tool_detector", "operation": "lookup_customer",
#              "input": {"customer_id": "123"}, "original_sequence_number": ...,
#              "duplicate_sequence_number": ..., "duplicate_count": 1}
```

Detection is deterministic and fully offline.
`python -m examples.detect_duplicate_tools` runs it against every demo scenario.

### `InefficiencyDetector`

Flags a **successful tool operation** only when the trace carries explicit
structured evidence that it was unnecessary or wasteful. Nothing is inferred
from timing, durations, record counts, decision names, or semantics.

- **Successful operation** = a `TOOL_CALL_STARTED` immediately followed by a
  matching `TOOL_CALL_COMPLETED` (same name; same canonical input when the
  completed event carries one). A `DECISION`/`ERROR`/second start in between
  invalidates the pair; failed attempts are never "successful work".
- **Evidence signals** (each read from event `metadata` or a mapping `output`,
  booleans only — `"true"` / `1` do not count):
  - `scope_mismatch` — the operation's own event has `required_scope` and
    `actual_scope` both present, same JSON type, unequal (no ordering assumed);
  - `explicit_wasted_work` — `wasted_work is True` on the operation's own event;
  - `explicit_unnecessary` — `necessary is False` on the operation's own event;
  - `explicit_wasted_operation` — a later event names it via
    `wasted_operation == "<op name>"`, associated with the most recent preceding
    successful operation of that name;
  - `sufficient_information_already_available` — an earlier event set
    `sufficient_to_answer is True`. This never flags an operation on its own; it
    only raises severity of one already flagged by the four signals above.
- **`related_event_ids`** (UUIDs, unique, trace order, no lifecycle events): the
  operation's `TOOL_CALL_STARTED` and `TOOL_CALL_COMPLETED`, plus the recognition
  event and the sufficiency event when those signals are used.
- **Severity** by number of distinct signals: **1 → MEDIUM**, **2 → HIGH**,
  **3+ → CRITICAL**.
- One issue per inefficient operation; multiple issues are returned in trace
  order. The detector is parameterless, deterministic, holds no state between
  calls, and is fully offline.

```python
from agentlens import AgentLens
from agentlens.detectors import InefficiencyDetector
from examples.demo_agents import run_inefficient_agent

lens = AgentLens()
run_id = run_inefficient_agent(lens)

issues = InefficiencyDetector().detect(lens.get_run(run_id), lens.get_events(run_id))
# -> one AgentIssue: issue_type=inefficiency, severity=critical,
#    metadata={"detector": "inefficiency_detector", "operation": "fetch_all_customers",
#              "input": {}, "evidence": ["scope_mismatch", "explicit_wasted_operation",
#              "sufficient_information_already_available"]}
```

`python -m examples.detect_inefficiencies` runs it against every demo scenario.

### Running every detector

`AgentLens.detect(run_id)` runs all four deterministic detectors over one stored
run and returns the combined `list[AgentIssue]` — no need to import and invoke
each detector by hand. It looks the run and its events up in the instance's
store (so only that run's events are analysed), then runs the detectors in a
**fixed order** — loop, retry, duplicate-tool, inefficiency — with their default
configuration. Issues come back in that detector order; within a detector its own
trace ordering is preserved. It is a pure analysis call: issues are not
persisted unless you're using a store that persists them (see
[Persistence](#persistence)), and neither the run nor its events are modified.
An unknown run id raises `AgentLensError`.

```python
from agentlens import AgentLens
from examples.demo_agents import run_inefficient_agent

lens = AgentLens()
run_id = run_inefficient_agent(lens)

issues = lens.detect(run_id)  # or lens.detect(run)
```

The same composition is available framework-agnostically for traces that did not
come from a live `AgentLens`:

```python
from agentlens.detectors import run_detectors

issues = run_detectors(run, events)
```

`python -m examples.detect_all_issues` runs `lens.detect` over every demo scenario.

## Reports and Export

`lens.get_report(run_id)` returns an `AgentReport` that aggregates one stored run
with its events and its already-persisted issues, plus a deterministic
`ReportSummary`:

```python
report = lens.get_report(run_id)

report.run  # the stored AgentRun
report.events  # its events, in sequence_number order
report.issues  # its persisted issues, in save order (no de-duplication)

report.summary.total_events  # len(report.events), lifecycle events included
report.summary.total_issues  # len(report.issues)
report.summary.events_by_type  # {EventType: count}, first-appearance order
report.summary.issues_by_type  # {IssueType: count}, first-appearance order
report.summary.issues_by_severity  # {Severity: count}, first-appearance order
```

`detect(run_id)` is **analysis (+ issue persistence when the store supports it)**;
`get_report(run_id)` is **read-only retrieval + deterministic aggregation** — it
never runs detectors, creates issues, or writes storage, and repeated calls on
unchanged data return byte-identical content. An unknown run id raises
`AgentLensError`. It works identically with the in-memory store and
`SQLiteTraceStore`, including after the database is closed and reopened.

`python -m examples.report_generation` runs a scenario, detects, and prints its
report.

### Export

`agentlens.export` turns a built `AgentReport` into a string — JSON or Markdown.
Both functions operate purely on an `AgentReport` and return a `str`; neither
writes a file:

```python
from agentlens.export import export_json, export_markdown

report = lens.get_report(run_id)

json_report = export_json(report)  # str
markdown_report = export_markdown(report)  # str
```

- **Read-only.** The exporters take an `AgentReport`, not a store or `AgentLens`.
  They never run detectors, never persist anything, and never mutate the report.
- **Deterministic.** `export_json` is `json.dumps(report.model_dump(mode="json"),
  sort_keys=True, indent=2, ensure_ascii=False)` — repeated calls are
  byte-identical and `json.loads` of the result equals `report.model_dump(mode="json")`.
- **Ordering preserved.** `export_markdown` renders events in `report.events`
  order, issues in `report.issues` order, and each summary table in the report's
  existing first-appearance key order — never sorted. Its sections are always
  Run → Summary → Events → Issues; empty issues render `_No issues detected._`.
- No file-writing API exists in either function — callers write the returned
  string to a file themselves if they want one.

`python -m examples.export_report` traces a scenario, detects, builds a report,
and prints both exports.

## Command Line

Installing the package provides a read-only `agentlens` command for inspecting
a persisted SQLite database. It is a thin adapter over the Python API — every
command opens the store, reads, prints, and closes; **no command runs
detectors or writes to the database** (there is deliberately no `detect`
subcommand — run detection through the Python API and persist it to SQLite,
then inspect it with the CLI).

```bash
agentlens runs --db agentlens.db
agentlens issues <run-id> --db agentlens.db
agentlens report <run-id> --db agentlens.db
agentlens export <run-id> --format json --db agentlens.db
agentlens export <run-id> --format markdown --db agentlens.db
```

(Without installing the console script, `python -m agentlens.cli.main <args>`
is equivalent.)

- `--db PATH` is always explicit — there is no environment variable, config
  file, or default database location.
- All commands are **read-only**: they use the *persisted* runs, events, and
  issues and never write to the database.
- `issues` shows already-persisted issues — it does **not** run detection.
  `report` and `export` build the report via `lens.get_report(...)` and likewise
  run no detectors.
- To (re)compute and persist issues, run detection separately through the Python
  API: `lens.detect(run_id)` against a `SQLiteTraceStore`-backed `AgentLens`.
- `report` prints the same Markdown as `export --format markdown`; `export
  --format json` prints exactly `export_json(report)`. Nothing is added around
  the export output.
- Exit codes: `0` on success (including "no runs"/"no issues"), `1` on an
  expected error (invalid run id, unknown run), `2` for argument errors.

`python -m examples.cli_usage` drives every command against a throwaway database.

## Framework Integrations

Three optional adapters record activity from a specific framework into an
existing AgentLens run as ordinary `AgentEvent`s, using the same public
`AgentLens.record_event` API application code uses. Each is:

- **optional** — none is a core dependency; core usage never imports any of
  them (verified by `tests/unit/test_packaging.py`);
- **record-only** — none of them ever calls `lens.detect(...)`, `run_detectors`,
  `save_issues`, or `lens.get_report(...)`. Detection and reporting stay
  explicit steps you call yourself, exactly as in the Quick Start;
- **offline** — each only observes data the framework hands it; none makes a
  network or provider call or reads an environment variable.

### LangChain

```bash
pip install ".[langchain]"
```

```python
from agentlens import AgentLens
from agentlens.integrations.langchain import AgentLensCallbackHandler

lens = AgentLens()
with lens.trace("Answer a customer support question") as trace:
    handler = AgentLensCallbackHandler(lens=lens, run_id=trace.run_id)
    chain.invoke(
        {"question": "Where is my order?"},
        config={"callbacks": [handler]},
    )

issues = lens.detect(trace.run_id)  # detection is explicit
report = lens.get_report(trace.run_id)  # reporting is explicit
```

Attach the handler *inside* the run's `with lens.trace(...)` block; it writes
every event to that one run and never creates another. Tool
starts/completions/errors map to `TOOL_CALL_STARTED` / `TOOL_CALL_COMPLETED` /
`ERROR`; LLM activity to `LLM_CALL_STARTED` / `LLM_CALL_COMPLETED` / `ERROR`;
agent actions/finishes to `DECISION`. Each event carries
`metadata={"framework": "langchain", "langchain_run_id": "..."}`.

Importing `agentlens` (or the CLI, or the export layer) does **not** import
LangChain. Importing `agentlens.integrations.langchain` without the extra
raises a clear `ModuleNotFoundError` telling you to install
`agentlens[langchain]`.

`python -m examples.langchain_integration` runs a fully offline end-to-end demo.

### OpenAI Agents SDK

```bash
pip install ".[openai-agents]"
```

```python
from agents import Runner
from agents.tracing import add_trace_processor

from agentlens import AgentLens
from agentlens.integrations.openai_agents import AgentLensOpenAITracer

lens = AgentLens()
with lens.trace("Answer a customer support question") as trace:
    tracer = AgentLensOpenAITracer(lens=lens, run_id=trace.run_id)
    add_trace_processor(tracer)
    Runner.run_sync(agent, "Where is my order?")

issues = lens.detect(trace.run_id)  # detection is explicit
report = lens.get_report(trace.run_id)  # reporting is explicit
```

`AgentLensOpenAITracer` is a `TracingProcessor` registered through the SDK's own
official tracing extension mechanism (`agents.tracing.add_trace_processor`) —
register it *inside* the run's `with lens.trace(...)` block. It adopts the
OpenAI trace(s) started while that run is the lens's active trace, so several
globally-registered tracers never leak spans between AgentLens runs. `function`
spans → `TOOL_CALL_STARTED` / `TOOL_CALL_COMPLETED` (or `ERROR` on a span
error); `generation` / `response` spans → `LLM_CALL_STARTED` /
`LLM_CALL_COMPLETED` / `ERROR`; `handoff` spans → `DECISION`. `agent`,
`guardrail`, `custom`, and other span types are deliberately not mapped, to
keep the trace analysis-friendly. Each event carries
`metadata={"framework": "openai_agents", "framework_run_id": "...", "framework_span_id": "..."}`.

Importing `agentlens` does not import the SDK; importing
`agentlens.integrations.openai_agents` without the extra raises a clear
`ModuleNotFoundError`.

`python -m examples.openai_agents_integration` runs a fully offline end-to-end demo.

### CrewAI

```bash
pip install ".[crewai]"
```

```python
from crewai import Crew

from agentlens import AgentLens
from agentlens.integrations.crewai import AgentLensCrewAIListener

lens = AgentLens()
with lens.trace("Research a topic") as trace:
    with AgentLensCrewAIListener(lens=lens, run_id=trace.run_id):
        crew.kickoff()

issues = lens.detect(trace.run_id)  # detection is explicit
report = lens.get_report(trace.run_id)  # reporting is explicit
```

`AgentLensCrewAIListener` is a `BaseEventListener` on CrewAI's public
`crewai_event_bus` — CrewAI's official extension point. **It must be used as a
context manager while the AgentLens trace is active**, exactly as shown above:
CrewAI dispatches its event handlers from background worker threads, so the
listener buffers a JSON-safe copy of each event as it arrives and replays those
events into AgentLens — in CrewAI's own emission order — on your thread, when
the `with AgentLensCrewAIListener(...)` block exits. It records only activity
emitted while that run is the lens's active trace, so several listeners
registered on the global bus never leak events between AgentLens runs.

Tool events map to `TOOL_CALL_STARTED` / `TOOL_CALL_COMPLETED` (or `ERROR` on a
tool failure); LLM call events to `LLM_CALL_STARTED` / `LLM_CALL_COMPLETED` /
`ERROR`; agent-execution and task failures to `ERROR`. Crew / task / agent
*lifecycle* events (kickoff, task/agent started/completed, and similar) are
deliberately not mapped, to keep the trace analysis-friendly. **CrewAI exposes
no public "agent decided X" event in the tested version, so this integration
does not currently produce any `DECISION` events** — this is a documented gap,
not a promise of full lifecycle coverage. Each event carries
`metadata={"framework": "crewai", "framework_event_type": "...", "framework_event_id": "..."}`.

Importing `agentlens` does not import CrewAI; importing
`agentlens.integrations.crewai` without the extra raises a clear
`ModuleNotFoundError`.

`python -m examples.crewai_integration` runs a fully offline end-to-end demo.

## Persistence

By default an `AgentLens` keeps runs and events in memory only — `AgentLens()`
behaves exactly as before. To keep them across process restarts, pass a
`SQLiteTraceStore` with an explicit database path:

```python
from agentlens import AgentLens
from agentlens.storage import SQLiteTraceStore

store = SQLiteTraceStore("agentlens.db")
lens = AgentLens(store=store)

with lens.trace("example task") as trace:
    trace.record_event(...)
run_id = trace.run_id

store.close()
```

The database (and its schema) is created automatically.

### Issue persistence

`lens.detect(run_id)` does two things when the underlying store supports it: it
runs the detectors **and** persists the issues it generates. `lens.get_issues(run_id)`
reads back what was persisted — it never re-runs detection.

```python
lens = AgentLens(store=SQLiteTraceStore("agentlens.db"))

issues = lens.detect(run_id)  # runs detectors, stores + returns issues
persisted = lens.get_issues(run_id)  # reads stored issues; no detection
# issues == persisted
```

- Issue storage is **append-only and not de-duplicated**. `AgentIssue.id` is
  generated fresh each detection, so calling `detect` again appends a second
  batch of issue records — `len(lens.get_issues(run_id))` grows. There is no
  "replace previous results" behaviour.
- Detector *content* (type, severity, description, metadata, related event ids,
  detector order) is deterministic across repeated `detect` calls; only the
  `AgentIssue.id`s differ.
- `get_issues` returns `[]` for an unknown run or a run that has never been
  detected.
- All of `AgentRun`, `AgentEvent`, and `AgentIssue` are retrieved back through
  Pydantic validation, so they are semantically identical to what was stored;
  event order is by `sequence_number`, matching the in-memory store.
- `SQLiteTraceStore` uses the standard library's `sqlite3` — no new dependency.
  Call `store.close()` when done, or use it as a context manager. It is not
  thread-safe, and there is no environment-variable or implicit-path config.

`python -m examples.sqlite_persistence` traces a run, closes the database,
reopens it, and detects issues. `python -m examples.persisted_issues` detects
into SQLite, reopens the database, and reads the issues back without re-running
detection.

## Architecture

Dependency direction is one-way, top to bottom. Nothing below a layer imports
anything above it:

```
Framework integrations (LangChain, OpenAI Agents SDK, CrewAI)
        │  record events via the public AgentLens API
        ▼
      AgentLens  (agentlens.core)
        │  opens/closes traces, assigns sequence numbers
        ▼
 Tracing / Events  (AgentRun, AgentEvent — agentlens.models)
        │
        ▼
      Storage  (in-memory, or SQLiteTraceStore — agentlens.storage)
        │
        ▼
 Explicit Detection  (agentlens.detectors — you call lens.detect(...))
        │  produces
        ▼
      Issues  (AgentIssue, persisted alongside the run)
        │  aggregated into
        ▼
      Reports  (AgentReport — agentlens.core / agentlens.models)
        │
        ▼
  Export / CLI  (agentlens.export, agentlens.cli — read-only)
```

- **Framework adapters record events.** They call the same
  `AgentLens.record_event` (or `Trace.record_event`) application code calls;
  they never touch storage, detectors, or reports directly, and none of them
  imports another adapter.
- **Detectors analyse explicitly.** They only run when you call `lens.detect(...)`
  or `run_detectors(...)`; nothing in `AgentLens.record_event` or any adapter
  triggers them.
- **Reports aggregate existing data.** `get_report` reads back the run, its
  events, and its already-persisted issues — it computes a summary but performs
  no analysis of its own.
- **Exports serialize reports.** `export_json` / `export_markdown` take an
  `AgentReport` and return a string; they have no dependency on `AgentLens`,
  storage, or detectors.
- **The CLI reads persisted data.** Every subcommand opens a `SQLiteTraceStore`,
  calls the same `AgentLens` read methods application code would, and prints —
  it never writes.

## Examples

Every example under `examples/` is offline, deterministic, and free of API
keys, network calls, and environment-variable configuration (verified in this
milestone by running each one and grepping for network/secret usage).

| Example | Demonstrates |
| --- | --- |
| `run_demo_scenarios.py` | Core tracing: five controlled scenarios via the real `AgentLens` API |
| `detect_loops.py` | `LoopDetector` |
| `detect_retries.py` | `RetryDetector` |
| `detect_duplicate_tools.py` | `DuplicateToolDetector` |
| `detect_inefficiencies.py` | `InefficiencyDetector` |
| `detect_all_issues.py` | `lens.detect(run_id)` — all four detectors together |
| `report_generation.py` | Building an `AgentReport` |
| `export_report.py` | `export_json` and `export_markdown` |
| `sqlite_persistence.py` | `SQLiteTraceStore` close/reopen |
| `persisted_issues.py` | Issue persistence and read-back without re-detecting |
| `cli_usage.py` | Every `agentlens` CLI command against a throwaway database |
| `langchain_integration.py` | `AgentLensCallbackHandler` |
| `openai_agents_integration.py` | `AgentLensOpenAITracer` |
| `crewai_integration.py` | `AgentLensCrewAIListener` |

Run any of them with `python -m examples.<name>` (see each section above for
the exact command).

## Supported Python Versions

AgentLens targets **Python 3.11 and 3.12**. Both are verified by the CI matrix
(`.github/workflows/ci.yml`) on every push and pull request: the full test
suite and `ruff check` / `ruff format --check` must pass on both interpreters.

## Development / Testing

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the tests:

```bash
pytest
```

Run the linter and formatter check:

```bash
ruff check .
ruff format --check .
```

To verify the package the way an external user would — a clean virtual
environment, a real `pip install .`, the console script, and each optional
extra in isolation — run:

```bash
scripts/verify_release.sh
```

It builds a throwaway venv (removed on exit), never touches your development
environment, and exits non-zero on the first failed check.

## License

Apache-2.0.
