"""Packaging guarantees: core import isolation and the installed console script.

These are the "external developer" checks that the rest of the suite does not
otherwise cover: a clean process importing only the public core API, and the
actual installed ``agentlens`` entry point (not ``python -m agentlens.cli.main``).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def _console_script() -> str | None:
    """Find the installed ``agentlens`` entry point.

    Prefers the script installed alongside the running interpreter (so this
    works whether or not that interpreter's ``bin``/``Scripts`` directory is on
    ``PATH``), falling back to a plain ``PATH`` lookup.
    """

    for name in ("agentlens", "agentlens.exe"):
        candidate = Path(sys.executable).parent / name
        if candidate.is_file():
            return str(candidate)
    return shutil.which("agentlens")


def test_core_public_api_imports_in_a_clean_process():
    code = (
        "import agentlens\n"
        "from agentlens import AgentLens\n"
        "from agentlens.export import export_json, export_markdown\n"
        "from agentlens.storage import SQLiteTraceStore\n"
        "from agentlens.detectors import run_detectors\n"
        "from agentlens.cli import main\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_core_process_does_not_import_optional_frameworks():
    code = (
        "import sys\n"
        "import agentlens\n"
        "from agentlens import AgentLens\n"
        "from agentlens.export import export_json, export_markdown\n"
        "from agentlens.storage import SQLiteTraceStore\n"
        "from agentlens.detectors import run_detectors\n"
        "from agentlens.cli import main\n"
        "bad = sorted(\n"
        "    m for m in sys.modules\n"
        "    if m.split('.')[0] in ('langchain', 'langchain_core', 'agents', 'openai', 'crewai')\n"
        ")\n"
        "assert not bad, bad\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_installed_console_script_help():
    """Exercise the actual `agentlens` entry point, not `python -m ...`."""

    console_script = _console_script()
    assert console_script is not None, (
        "the 'agentlens' console script is not on PATH -- install the package "
        "(e.g. `pip install -e .`) before running this test"
    )
    result = subprocess.run([console_script, "--help"], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "agentlens" in result.stdout
    assert "runs" in result.stdout
    assert "issues" in result.stdout
    assert "report" in result.stdout
    assert "export" in result.stdout


def test_installed_console_script_matches_cli_module():
    """The console script and `python -m agentlens.cli.main` must agree."""

    console_script = _console_script()
    assert console_script is not None

    via_script = subprocess.run(
        [console_script, "--help"], capture_output=True, text=True, check=False
    )
    via_module = subprocess.run(
        [sys.executable, "-m", "agentlens.cli.main", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert via_script.returncode == via_module.returncode == 0
    assert via_script.stdout == via_module.stdout
