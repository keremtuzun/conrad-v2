"""Killed or failing research candidates stay out of the production runtime (prompt s9, ADR-0006/0007).

- generic RBP: off by default in Model2Core, never constructed by runtime/orchestration/domain code
- learned association scorer: never supplied by runtime code (nearest-neighbour decision is the default)
- CEFD coupling: production Model2E default is uncoupled
- TCDP / relational propagation: production Model2T default is NONE (ADR-0009, gate 2T-TCDP FAILED)
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import replace
from pathlib import Path

from conrad.core.config import tiny_config
from conrad.core.pipeline import Model2Core
from conrad.domains.ecological.config import CefdSwitches, Model2EConfig
from conrad.domains.technical import (
    Model2T,
    Model2TConfig,
    PropagationMode,
    model2t_config_from_dict,
    production_propagation_mode,
)
from conrad.orchestration.mission_config import MissionRuntimeConfig
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import EntityCandidate, Evidence, Modality, QualityContext
from conrad.schemas.timebase import stamp

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


def test_model2t_production_default_does_not_propagate() -> None:
    """ADR-0009: gate 2T-TCDP FAILED (TCDP worse than no propagation for corrosion), so the production Model2T
    propagates nothing to neighbours. TCDP / GENERIC are EXPERIMENTAL arms reachable only explicitly."""
    cfg = Model2TConfig()
    assert cfg.tcdp.mode is PropagationMode.NONE
    assert cfg.tcdp.experimental_enabled is False
    for requested in PropagationMode:  # whatever a mission config asks for, no opt-in -> NONE
        assert production_propagation_mode(requested, cfg) is PropagationMode.NONE
    opt_in = replace(cfg, tcdp=replace(cfg.tcdp, experimental_enabled=True))
    assert production_propagation_mode("TCDP", opt_in) is PropagationMode.TCDP
    mission = MissionRuntimeConfig()
    m2t_cfg = model2t_config_from_dict(dict(mission.model2t))
    assert production_propagation_mode(mission.model2t_mode, m2t_cfg) is PropagationMode.NONE


def test_mission_runtime_builds_model2t_through_the_production_resolver() -> None:
    tree = ast.parse((ROOT / "conrad" / "orchestration" / "children.py").read_text(encoding="utf-8"))
    calls = {
        node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    assert "production_propagation_mode" in calls
    assert "PropagationMode" not in calls  # the mode is never built straight from mission_config


def test_default_model2t_never_infers_an_unobserved_neighbour() -> None:
    """A never-observed component next to a badly corroded, observed one stays UNKNOWN (prior only, never
    INFERRED) in the default Model2T."""
    f = IdFactory(seed=5, namespace="registry")
    a, b = f.new(), f.new()
    seg = {"component_type": "SEGMENT", "material": "carbon_steel", "wall_thickness_m": 0.015}
    ctx = {
        "asset_registry": {
            "components": [{"registry_id": a, **seg}, {"registry_id": b, **seg}],
            "relationships": [{"source": a, "target": b, "type": "CONNECTED_TO"}],
        }
    }
    m = Model2T(IdFactory(seed=6, namespace="model2t"))
    m.initialize(ctx)
    ids = IdFactory(seed=7, namespace="evidence")
    ts = stamp(10.0, "sim")
    ev = Evidence(
        evidence_id=ids.new(),
        source_observation_id=ids.new(),
        mission_id=ids.new(),
        run_id=ids.new(),
        trace_id=ids.new(),
        modality=Modality.STRUCTURED,
        timestamp=ts,
        created_time_ns=ts.time_ns,
        embedding=(1.0,),
        entity_candidates=(EntityCandidate(registry_entity_id=a, score=1.0),),
        reliability=0.9,
        aleatoric_uncertainty=0.1,
        sensor_context=QualityContext(),
        measurements={"apparent_wall_loss": 8e-3},
        measurement_units={"apparent_wall_loss": "m"},
        provenance_id=ids.new(),
        encoder_version="contract-test",
    )
    m.ingest([ev])
    m.update_beliefs(ts)
    m.predict(30 * 86400.0, stamp(10.0 + 30 * 86400.0, "sim"))
    hidden = next(x for x in m.export_beliefs() if x.world_entity_id == b)
    statuses = {c.name: c.status.value for c in hidden.state_summary}
    assert statuses["condition"] == "UNKNOWN"
    assert "INFERRED" not in statuses.values() and "OBSERVED" not in statuses.values()
    assert hidden.uncertainty.observational >= 0.9
