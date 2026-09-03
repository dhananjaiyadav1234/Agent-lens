"""Smoke tests: the package and its subpackages import cleanly."""

import importlib

import pytest

SUBPACKAGES = [
    "agentlens",
    "agentlens.core",
    "agentlens.models",
    "agentlens.detectors",
    "agentlens.storage",
    "agentlens.integrations",
]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name: str) -> None:
    assert importlib.import_module(name) is not None


def test_version_is_exposed() -> None:
    import agentlens

    assert isinstance(agentlens.__version__, str)
    assert agentlens.__version__.count(".") >= 2
