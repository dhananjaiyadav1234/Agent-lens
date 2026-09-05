#!/usr/bin/env bash
# Release-readiness / clean-install verification.
#
# Exercises AgentLens the way an external user would consume it: a fresh
# virtual environment and a real `pip install`, never the developer's existing
# .venv. Builds one throwaway venv per check (removed on exit) and exits
# non-zero on the first failure.
#
# Usage: scripts/verify_release.sh [python-executable]
#   scripts/verify_release.sh            # uses `python3`
#   scripts/verify_release.sh python3.12 # verify against a specific interpreter

set -euo pipefail

PYTHON_BIN="${1:-python3}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

step() { printf '\n==> %s\n' "$1"; }
ok() { printf '    ok: %s\n' "$1"; }

cd "$REPO_ROOT"

step "core install (clean venv, $PYTHON_BIN)"
"$PYTHON_BIN" -m venv "$WORKDIR/core"
"$WORKDIR/core/bin/python" -m pip install --upgrade pip -q
"$WORKDIR/core/bin/python" -m pip install . -q
"$WORKDIR/core/bin/python" -c "import agentlens; print(agentlens.__version__)" >/dev/null
"$WORKDIR/core/bin/python" -c "from agentlens import AgentLens" >/dev/null
ok "import agentlens / from agentlens import AgentLens"
"$WORKDIR/core/bin/agentlens" --help >/dev/null
ok "agentlens --help (installed console script)"

step "optional dependency isolation (core-only venv)"
"$WORKDIR/core/bin/python" -c "
import sys
import agentlens
from agentlens import AgentLens
from agentlens.export import export_json, export_markdown
from agentlens.storage import SQLiteTraceStore
from agentlens.detectors import run_detectors
from agentlens.cli import main
bad = sorted(m for m in sys.modules if m.split('.')[0] in ('langchain', 'langchain_core', 'agents', 'openai', 'crewai'))
assert not bad, f'core install imported optional frameworks: {bad}'
"
ok "core install does not import langchain / agents / openai / crewai"

step "quick-start workflow (core-only venv)"
"$WORKDIR/core/bin/python" -c "
from agentlens import AgentLens
from agentlens.models import EventType
from agentlens.export import export_json, export_markdown

lens = AgentLens()
with lens.trace('smoke test') as trace:
    trace.record_event(event_type=EventType.TOOL_CALL_STARTED, name='t', input={})
    trace.record_event(event_type=EventType.TOOL_CALL_COMPLETED, name='t', input={}, output={}, status='ok')
run_id = trace.run_id
issues = lens.detect(run_id)
report = lens.get_report(run_id)
assert report.summary.total_events == 4
export_json(report)
export_markdown(report)
"
ok "trace -> detect -> report -> export"

for extra in langchain openai-agents crewai; do
    step "optional extra: $extra (clean venv)"
    "$PYTHON_BIN" -m venv "$WORKDIR/$extra"
    "$WORKDIR/$extra/bin/python" -m pip install --upgrade pip -q
    "$WORKDIR/$extra/bin/python" -m pip install ".[$extra]" -q
    case "$extra" in
        langchain)
            "$WORKDIR/$extra/bin/python" -c "from agentlens.integrations.langchain import AgentLensCallbackHandler"
            ;;
        openai-agents)
            "$WORKDIR/$extra/bin/python" -c "from agentlens.integrations.openai_agents import AgentLensOpenAITracer"
            ;;
        crewai)
            "$WORKDIR/$extra/bin/python" -c "from agentlens.integrations.crewai import AgentLensCrewAIListener"
            ;;
    esac
    ok "public API import"
done

step "all extras combined (clean venv)"
"$PYTHON_BIN" -m venv "$WORKDIR/all"
"$WORKDIR/all/bin/python" -m pip install --upgrade pip -q
"$WORKDIR/all/bin/python" -m pip install ".[langchain,openai-agents,crewai]" -q
"$WORKDIR/all/bin/python" -c "
from agentlens.integrations.langchain import AgentLensCallbackHandler
from agentlens.integrations.openai_agents import AgentLensOpenAITracer
from agentlens.integrations.crewai import AgentLensCrewAIListener
"
ok "all three extras import together"

printf '\nRELEASE VERIFICATION PASSED\n'
