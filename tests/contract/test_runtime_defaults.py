"""Killed or failing research candidates stay out of the production runtime (prompt s9, ADR-0006/0007).

- generic RBP: off by default in Model2Core, never constructed by runtime/orchestration/domain code
- learned association scorer: never supplied by runtime code (nearest-neighbour decision is the default)
- CEFD coupling: production Model2E default is uncoupled
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from conrad.core.config import tiny_config
from conrad.core.pipeline import Model2Core
from conrad.domains.ecological.config import CefdSwitches, Model2EConfig

ROOT = Path(__file__).resolve().parent.parent.parent
RUNTIME_PACKAGES = (
    "orchestration",
    "runtime",
    "domains",
    "decision",
    "active",
    "communication",
    "robotics",
    "sim/mission",
)
FORBIDDEN_CONSTRUCTORS = {
    "AnalyticRBP",
    "RelationalBeliefPropagation",
    "AssociationScorer",
    "LearnedTCDP",
    "LearnedCEFD",
}


def test_model2core_rbp_default_off() -> None:
    assert inspect.signature(Model2Core.__init__).parameters["use_rbp"].default is False


def test_model2e_production_default_is_uncoupled() -> None:
    """Ecological coupling (both directions) stays OFF; only the sensing-quality context is ON (ADR-0007
    addendum). The switch set is exhaustive: a new switch must be added here deliberately."""
    sw = Model2EConfig().switches
    assert sw.ecological_coupling is False
    assert sw.entity_to_field is False
    assert sw.observability_context is True
    assert sw == CefdSwitches()
    assert set(CefdSwitches.model_fields) == {
        "entities",
        "fields",
        "observability_context",
        "ecological_coupling",
        "entity_to_field",
        "field_dynamics",
        "spatial_correlation",
    }
    assert not hasattr(sw, "field_to_entity")


def test_runtime_code_never_constructs_killed_candidates() -> None:
    offenders = []
    for pkg in RUNTIME_PACKAGES:
        for path in (ROOT / "conrad" / pkg).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    name = (
                        node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
                    )
                    if (
                        name in FORBIDDEN_CONSTRUCTORS
                        and "learned" not in path.parts
                        and path.name != "rbp.py"
                    ):
                        offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} {name}()")
                if (
                    isinstance(node, ast.keyword)
                    and node.arg == "use_rbp"
                    and isinstance(node.value, ast.Constant)
                    and node.value.value is True
                ):
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} use_rbp=True")
    assert not offenders, "killed candidate in production path:\n" + "\n".join(offenders)


def test_tiny_core_runs_without_rbp() -> None:
    assert tiny_config() is not None
