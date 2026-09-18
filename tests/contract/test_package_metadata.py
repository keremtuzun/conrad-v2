"""Every substantial package declares IMPLEMENTATION_METADATA (prompt s8, ch30) and never over-claims."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
PACKAGES = [
    "conrad.core",
    "conrad.core.ecmer",
    "conrad.domains.spatial",
    "conrad.domains.technical",
    "conrad.domains.ecological",
    "conrad.twins.twin2s",
    "conrad.twins.twin2t",
    "conrad.twins.twin2e",
    "conrad.decision",
    "conrad.active",
    "conrad.communication",
    "conrad.sim.kernel",
    "conrad.robotics.estimation",
    "conrad.robotics.navigation",
    "conrad.robotics.control",
    "conrad.robotics.allocation",
    "conrad.robotics.safety",
    "conrad.console",
]
STATUSES = {"FROZEN_CONTRACT", "EXPERIMENTAL_CANDIDATE", "OPEN_BLOCKED", "INTERFACE_FROZEN"}
CLAIMS = {"NONE", "IMPLEMENTED", "EVALUATED"}  # VALIDATED needs evidence nothing in this repo has yet


def _meta(pkg: str) -> dict:
    mod = importlib.import_module(pkg)
    meta = getattr(mod, "IMPLEMENTATION_METADATA", None)
    assert isinstance(meta, dict), f"{pkg} has no IMPLEMENTATION_METADATA"
    return meta


@pytest.mark.parametrize("pkg", PACKAGES)
def test_metadata_is_complete_and_honest(pkg: str) -> None:
    meta = _meta(pkg)
    status = meta["implementation_status"]
    statuses = set(status.values()) if isinstance(status, dict) else {status}
    assert statuses <= STATUSES, (pkg, status)
    assert meta["claim_status"] in CLAIMS, (pkg, meta["claim_status"])
    for key in ("source_sections", "configuration_keys", "assumptions", "baselines", "acceptance_tests"):
        assert key in meta, (pkg, key)


@pytest.mark.parametrize("pkg", PACKAGES)
def test_referenced_acceptance_tests_exist(pkg: str) -> None:
    for ref in _meta(pkg)["acceptance_tests"]:
        path = str(ref).split("::")[0]
        if path.startswith("tests/"):
            assert (ROOT / path).exists(), f"{pkg} references missing test {path}"
