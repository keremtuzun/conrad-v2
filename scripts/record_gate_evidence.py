"""Record formal gate evidence for the gates whose criteria are decided by executed tests/experiments.

Each criterion maps to pytest node IDs (all must pass) and/or an experiment check reading a retained artifact.
Gates that need Unity or integrated missions (U0, I1-I7) are recorded by their own harnesses, not here.

Usage: python -m uv run python scripts/record_gate_evidence.py [P0 I0 C1 2S-FIRST 2T 2E]
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from conrad.evaluation.gates import (  # noqa: E402
    CriterionResult,
    CriterionStatus,
    EvidenceClass,
    GateEvidence,
    write_evidence,
)

P0 = "tests/contract/test_p0_contracts.py::"
I0 = "tests/integration/test_i0_fake_full_system.py::"
CC = "tests/contract/test_cc_persistence.py::"
UC = "tests/unit/core/"
SP = "tests/unit/domains/spatial/"


def _exp(path: str, check: Callable[[dict], tuple[bool, str]]) -> Callable[[], tuple[CriterionStatus, str]]:
    def run() -> tuple[CriterionStatus, str]:
        p = ROOT / path
        if not p.exists():
            return CriterionStatus.NOT_RUN, f"artifact missing: {path}"
        ok, measured = check(json.loads(p.read_text(encoding="utf-8")))
        return (CriterionStatus.PASS if ok else CriterionStatus.FAIL), measured

    return run


def _persist_check(d: dict) -> tuple[bool, str]:
    """Main PMBL arm over long sequences: identity, re-identification, provenance, dedupe, CC-09 all at 1.0."""
    s = d.get("summary", d)
    keys = [
        "pmbl_full.identity_purity",
        "pmbl_full.reidentification_after_gap",
        "pmbl_full.provenance_integrity",
        "pmbl_full.duplicate_rejection_rate",
        "cc09_reset_equivalence",
    ]
    vals = {k: float(s[k]["mean"]) for k in keys if k in s}
    ok = len(vals) == len(keys) and all(v >= 0.999 for v in vals.values())
    return ok, json.dumps(vals) if vals else "keys missing"


def _walk(o: object, prefix: str = "") -> list[tuple[str, object]]:
    out: list[tuple[str, object]] = []
    if isinstance(o, dict):
        for k, v in o.items():
            out.append((f"{prefix}{k}", v))
            out += _walk(v, f"{prefix}{k}.")
    return out


PLAN: dict[str, list[tuple[str, list[str], Callable[[], tuple[CriterionStatus, str]] | None]]] = {
    "P0": [
        (
            "serialization/deserialization",
            [P0 + "test_round_trip_serialization", P0 + "test_unknown_keys_rejected"],
            None,
        ),
        ("schema version", [P0 + "test_schema_version_gate"], None),
        ("timestamps", [P0 + "test_time_semantics", P0 + "test_evidence_cannot_predate_observation"], None),
        (
            "coordinate frames",
            [
                P0 + "test_frame_contract",
                P0 + "test_frame_graph_round_trip",
                P0 + "test_every_spatial_type_requires_a_frame",
            ],
            None,
        ),
        (
            "UUID uniqueness",
            [P0 + "test_uuid_uniqueness_and_determinism", P0 + "test_namespaced_id_factories_never_collide"],
            None,
        ),
        ("provenance graph validity", [P0 + "test_provenance_dag"], None),
        (
            "uncertainty contract",
            [P0 + "test_uncertainty_contract", P0 + "test_unknown_is_a_status_not_a_value"],
            None,
        ),
        ("RobotConfig validation", [P0 + "test_robot_config_validation"], None),
    ],
    "I0": [
        ("complete architectural loop executes", [I0 + "test_i0_loop_closes_and_belief_resolves"], None),
        ("timestamps", [I0 + "test_i0_boundaries_preserve_identity"], None),
        ("IDs", [I0 + "test_i0_boundaries_preserve_identity"], None),
        ("frames", [I0 + "test_i0_boundaries_preserve_identity"], None),
        ("provenance", [I0 + "test_i0_full_causal_trace_from_command_to_raw_observation"], None),
        ("uncertainty", [I0 + "test_i0_loop_closes_and_belief_resolves"], None),
        ("logging", [I0 + "test_i0_deterministic_replay"], None),
        ("configs", [I0 + "test_i0_boundaries_preserve_identity"], None),
    ],
    "C1": [
        (
            "persistent identity",
            [
                UC + "test_core_pmbl.py::test_lifecycle_candidate_confirm_active_dormant_reactivated",
                UC + "test_core_pmbl.py::test_cc09_reset_working_memory_preserves_persistence",
            ],
            None,
        ),
        (
            "evidence association",
            [
                UC + "test_core_association.py::test_retrieval_gates_frame_lifecycle_domain_distance_and_k",
                UC + "test_core_association.py::test_nearest_neighbour_baseline_and_engine",
            ],
            None,
        ),
        (
            "no-match/new entity",
            [
                UC + "test_core_association.py::test_no_match_with_empty_candidate_set",
                UC + "test_core_pipeline.py::test_no_match_creates_candidate_then_matches",
            ],
            None,
        ),
        (
            "credible contradiction handling",
            [
                UC
                + "test_core_buo_analytic.py::test_credible_contradiction_raises_uc_and_is_not_averaged_away",
                UC + "test_core_buo_analytic.py::test_unreliable_disagreement_is_ambiguous_not_contradiction",
            ],
            None,
        ),
        (
            "decomposed uncertainty",
            [
                UC + "test_core_buo_analytic.py::test_corrupted_evidence_raises_ua",
                UC + "test_core_buo_analytic.py::test_ood_score_raises_ue_and_unmeasured_ood_keeps_it",
                UC + "test_core_buo_analytic.py::test_low_coverage_raises_uo",
            ],
            None,
        ),
        (
            "relational inference",
            [
                UC
                + "test_core_analytic_tbd_rbp.py::test_analytic_rbp_labels_inferred_and_keeps_observational_uncertainty",
                UC + "test_core_analytic_tbd_rbp.py::test_analytic_rbp_never_overwrites_direct_observation",
                UC + "test_core_pipeline.py::test_relational_step_is_inferred_with_provenance",
            ],
            None,
        ),
        (
            "temporal prediction/correction",
            [
                UC + "test_core_analytic_tbd_rbp.py::test_analytic_tbd_uses_delta_t_and_marks_predicted",
                UC + "test_core_pipeline.py::test_prediction_is_persisted_as_predicted_and_kept_apart",
                UC + "test_core_pipeline.py::test_sequence_index_is_irrelevant_only_physical_time_matters",
            ],
            None,
        ),
        (
            "provenance",
            [
                CC + "test_cc04_merge_preserves_lineage_to_raw_observations",
                CC + "test_cc03_dangling_provenance_rejected",
                UC + "test_core_pmbl.py::test_cc04_merge_preserves_lineage_evidence_and_provenance",
            ],
            None,
        ),
        (
            "long sequence persistence",
            [
                CC + "test_cc01_duplicate_delivery_is_one_contribution",
                CC + "test_cc02_late_observation_never_masquerades_as_current",
            ],
            _exp("artifacts/experiments/core/CORE-PERSIST-E001.json", _persist_check),
        ),
    ],
    "2S-FIRST": [
        (
            "observed != inferred != unknown",
            [
                SP + "test_m2s_inference_temporal.py::test_small_gap_is_inferred_with_relational_provenance",
                SP + "test_m2s_inference_temporal.py::test_large_gap_and_hidden_side_stay_unknown",
                SP + "test_m2s_map_updates.py::test_unknown_is_never_occupied_and_is_free_follows_policy",
            ],
            None,
        ),
        (
            "counterfactual worlds produce uncertainty, not confident hallucination",
            [
                "tests/simulation/spatial/test_m2s_twin2s_loop.py::test_counterfactual_region_is_not_confidently_resolved"
            ],
            None,
        ),
    ],
}


I1S = "tests/integration/test_i1_spatial_loop.py::"
I3S = "tests/integration/test_i3_structural.py::"
# SURROGATE evidence: the same criteria exercised through the Python L1 kernel instead of Unity (ADR-0008).
# It is reported next to the gate and can never promote it.
SURROGATE_PLAN: dict[str, list[tuple[str, list[str], Callable[[], tuple[CriterionStatus, str]] | None]]] = {
    "I1": [
        (
            "robot moves through synthetic world",
            [I1S + "test_spatial_beliefs_persist_with_direct_provenance"],
            None,
        ),
        ("persistent map", [I1S + "test_spatial_beliefs_persist_with_direct_provenance"], None),
        ("geometry", [I1S + "test_free_space_claims_agree_with_truth_where_claimed"], None),
        ("occupancy", [I1S + "test_free_space_claims_agree_with_truth_where_claimed"], None),
        ("coverage", [I1S + "test_map_keeps_unknown_space_next_to_observed_space"], None),
        ("observed/inferred/unknown", [I1S + "test_map_keeps_unknown_space_next_to_observed_space"], None),
        ("uncertainty", [I1S + "test_map_keeps_unknown_space_next_to_observed_space"], None),
        ("provenance", [I1S + "test_spatial_beliefs_persist_with_direct_provenance"], None),
        (
            "no Twin truth leakage",
            [
                I1S + "test_spatial_beliefs_carry_only_registry_identities",
                "tests/leakage/test_dynamic_leakage.py::test_no_world_entity_id_or_twin_key_on_the_runtime_side",
                "tests/leakage/test_dynamic_leakage.py::test_registry_ids_are_not_world_ids",
            ],
            None,
        ),
    ],
    "I3": [
        (
            "Twin2T + Twin2S -> Unity -> ECMER -> 2S + 2T",
            [I3S + "test_lane_views_associate_structural_observations_correctly"],
            None,
        ),
        (
            "robot sees only partial infrastructure",
            [I3S + "test_hidden_target_stays_unknown_from_the_lane"],
            None,
        ),
        (
            "persistent technical belief",
            [I3S + "test_observed_components_get_persistent_direct_beliefs"],
            None,
        ),
    ],
}


def run_nodes(nodes: list[str]) -> dict[str, str]:
    with tempfile.TemporaryDirectory() as tmp:
        xml_path = Path(tmp) / "r.xml"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                f"--junitxml={xml_path}",
                *nodes,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        results: dict[str, str] = {}
        if not xml_path.exists():
            return results
        for case in ET.parse(xml_path).getroot().iter("testcase"):
            file = case.get("classname", "").replace(".", "/") + ".py"
            name = case.get("name", "")
            bad = case.find("failure") is not None or case.find("error") is not None
            skipped = case.find("skipped") is not None
            results[f"{file}::{name}"] = "FAIL" if bad else ("SKIP" if skipped else "PASS")
        return results


def record(gate: str, surrogate: bool = False) -> GateEvidence:
    plan = (SURROGATE_PLAN if surrogate else PLAN)[gate]
    nodes = sorted({n for _, ns, _ in plan for n in ns})
    outcome = run_nodes(nodes)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    criteria = []
    for name, ns, check in plan:
        statuses = [outcome.get(n, "NOT_RUN") for n in ns]
        measured = "; ".join(f"{n.split('::')[-1]}={s}" for n, s in zip(ns, statuses, strict=True))
        if any(s == "FAIL" for s in statuses):
            status = CriterionStatus.FAIL
        elif any(s in ("NOT_RUN", "SKIP") for s in statuses):
            status = CriterionStatus.NOT_RUN
        else:
            status = CriterionStatus.PASS
        if check is not None and status is CriterionStatus.PASS:
            status, extra = check()
            measured += f"; experiment: {extra}"
        criteria.append(CriterionResult(criterion=name, status=status, measured=measured))
    ev = GateEvidence(
        gate_id=gate,
        evidence_class=EvidenceClass.SURROGATE if surrogate else EvidenceClass.FORMAL,
        execution_path=("python L1 kernel instead of Unity; " if surrogate else "")
        + f"pytest nodes + retained experiment artifacts ({gate})",
        git_commit=commit,
        criteria=tuple(criteria),
        artifacts=tuple(nodes),
    )
    write_evidence(ev)
    return ev


if __name__ == "__main__":
    args = sys.argv[1:]
    surrogate = "--surrogate" in args
    args = [a for a in args if a != "--surrogate"]
    for gate in args or list(SURROGATE_PLAN if surrogate else PLAN):
        ev = record(gate, surrogate)
        print(gate, [f"{c.criterion}:{c.status.value}" for c in ev.criteria])
