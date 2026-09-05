"""Tests for the public detector interface / package API."""

from collections.abc import Sequence

from agentlens.detectors import IssueDetector, LoopDetector
from agentlens.models import AgentEvent, AgentIssue, AgentRun


def test_loop_detector_is_importable_from_package_root():
    from agentlens.detectors import LoopDetector as Imported

    assert Imported is LoopDetector


def test_loop_detector_satisfies_the_issue_detector_protocol():
    assert isinstance(LoopDetector(), IssueDetector)


def test_issue_detector_is_structural_only():
    class MyDetector:
        def detect(self, run: AgentRun, events: Sequence[AgentEvent]) -> list[AgentIssue]:
            return []

    assert isinstance(MyDetector(), IssueDetector)


def test_detect_signature_returns_a_list():
    run = AgentRun(task="t")
    result = LoopDetector().detect(run, [])
    assert isinstance(result, list)
