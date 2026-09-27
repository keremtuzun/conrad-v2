"""Requirement-by-requirement audit for P5-P10 downstream implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from conrad.oceansense.status import (
    QUALIFIED_CHECKPOINT_LABEL,
    _repo_commit,
    _resolve,
    verify_qualified_checkpoint_metadata,
)


REQUIREMENTS: tuple[dict[str, Any], ...] = (
    {
        "phase": "P5",
        "requirement": "task heads for detection, segmentation, anomaly, and condition/state outputs",
        "implementation": "conrad/oceansense/task_heads.py",
        "tests": ("tests/unit/oceansense/test_p5_p10_contracts.py::test_p5_task_heads_emit_schema_outputs_and_preserve_status",),
    },
    {
        "phase": "P6",
        "requirement": "384->512->256 RepresentationAdapter, learned evidence, semantic evidence, and Model2 256D compatibility",
        "implementation": "conrad/oceansense/model2_adapter.py",
        "tests": (
            "tests/unit/oceansense/test_p5_p10_contracts.py::test_p6_adapter_is_384_512_256_and_preserves_direct_measurements",
            "tests/unit/oceansense/test_p5_p10_contracts.py::test_p6_adapter_rejects_wrong_osfm_dimension_and_is_deterministic",
        ),
    },
    {
        "phase": "P6",
        "requirement": "dual-path rule: direct physical measurements bypass latent reconstruction",
        "implementation": "conrad/oceansense/model2_adapter.py",
        "tests": ("tests/unit/oceansense/test_p5_p10_contracts.py::test_p6_adapter_is_384_512_256_and_preserves_direct_measurements",),
    },
    {
        "phase": "P7",
        "requirement": "four-channel uncertainty, provenance-aware update, contradiction handling, competing hypotheses",
        "implementation": "conrad/oceansense/reasoning.py",
        "tests": (
            "tests/unit/oceansense/test_p5_p10_contracts.py::test_p7_uncertainty_channels_and_competing_hypotheses",
            "tests/unit/oceansense/test_p5_p10_contracts.py::test_p7_competing_hypotheses_mark_close_scores_unresolved",
        ),
    },
    {
        "phase": "P8",
        "requirement": "InformationNeed/adaptive inspection hooks over MCBR-style ObservationPlan",
        "implementation": "conrad/oceansense/inspection.py",
        "tests": (
            "tests/unit/oceansense/test_p5_p10_contracts.py::test_p8_information_need_value_of_information_contract",
            "tests/unit/oceansense/test_p5_p10_contracts.py::test_p8_adaptive_inspection_updates_plan_contract_fields",
        ),
    },
    {
        "phase": "P9",
        "requirement": "host capability negotiation, InspectionIntent boundary, degraded/refusal paths",
        "implementation": "conrad/oceansense/host.py",
        "tests": (
            "tests/unit/oceansense/test_p5_p10_contracts.py::test_p9_capability_negotiation_filters_without_inventing_host_authority",
            "tests/unit/oceansense/test_p5_p10_contracts.py::test_p9_gateway_preserves_host_boundary_and_degraded_paths",
            "tests/unit/oceansense/test_p5_p10_contracts.py::test_p9_gateway_refuses_when_host_has_no_supported_fallback",
        ),
    },
    {
        "phase": "P10",
        "requirement": "Alpha integration with persistence, provenance, replay, failure handling, coverage, Asset Memory, degraded operation",
        "implementation": "conrad/oceansense/alpha.py",
        "tests": (
            "tests/unit/oceansense/test_p5_p10_contracts.py::test_p10_alpha_records_asset_memory_replay_and_degraded_operation",
            "tests/unit/oceansense/test_p5_p10_contracts.py::test_p10_alpha_replay_key_and_outputs_are_deterministic",
            "tests/unit/oceansense/test_p5_p10_contracts.py::test_p10_alpha_records_adapter_failure_as_degraded_operation",
        ),
    },
    {
        "phase": "P5-P10",
        "requirement": "strict IMPLEMENTED vs final integrated VALIDATED-RUN separation",
        "implementation": "conrad/oceansense/status.py",
        "tests": (
            "tests/unit/oceansense/test_p5_p10_status.py::test_status_gate_is_implemented_but_not_validated_without_qualified_osfm",
            "tests/unit/oceansense/test_p5_p10_status.py::test_status_gate_allows_validated_run_only_with_qualified_checkpoint_metadata",
        ),
    },
    {
        "phase": "P5-P10",
        "requirement": "single fixture path exercises task heads, Model2 adapter, reasoning, adaptive inspection, host boundary, persistence, replay, and direct evidence preservation",
        "implementation": "conrad/oceansense/",
        "tests": ("tests/unit/oceansense/test_p5_p10_contracts.py::test_p5_p10_end_to_end_fixture_chain_preserves_boundaries",),
    },
)


def evaluate_p5_p10_requirement_audit(
    *,
    tests_passed: bool,
    qualified_checkpoint_metadata: str | Path | None = None,
) -> dict[str, Any]:
    qualified_checkpoint_verified, checkpoint_detail, metadata_supplied = verify_qualified_checkpoint_metadata(
        qualified_checkpoint_metadata
    )
    rows = []
    for req in REQUIREMENTS:
        status = "IMPLEMENTED" if tests_passed else "DESIGNED"
        if qualified_checkpoint_verified:
            status = "VALIDATED-RUN"
        rows.append(
            {
                **req,
                "status": status,
                "final_validation_dependency": QUALIFIED_CHECKPOINT_LABEL,
                "validated_run_allowed": qualified_checkpoint_verified,
            }
        )
    return {
        "schema_version": "1.0.0",
        "audit_id": "P5-P10-REQUIREMENT-AUDIT",
        "source_commit": _repo_commit(),
        "status": "VALIDATED-RUN" if qualified_checkpoint_verified else ("IMPLEMENTED" if tests_passed else "DESIGNED"),
        "qualified_checkpoint_verified": qualified_checkpoint_verified,
        "qualified_checkpoint": {
            "required_label": QUALIFIED_CHECKPOINT_LABEL,
            "metadata_supplied": metadata_supplied,
            "verified": qualified_checkpoint_verified,
            "detail": checkpoint_detail,
        },
        "remaining_blockers": []
        if qualified_checkpoint_verified
        else [
            {
                "blocker_id": "P5-P10-OSFM-01",
                "scope": "EXTERNAL",
                "detail": checkpoint_detail,
            }
        ],
        "requirements": rows,
    }


def write_p5_p10_requirement_audit(report: dict[str, Any], output: str | Path) -> Path:
    path = _resolve(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
