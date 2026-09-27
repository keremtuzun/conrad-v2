"""P4.8 full-launch preflight gate.

This gate is intentionally separate from the trainer. It can be run after a
real-data rehearsal to decide whether a formal P4.8 research launch is allowed,
without spending GPU time or silently relaxing evidence requirements.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from conrad.settings import REPO_ROOT


DEFAULT_REHEARSAL_REPORT = Path("artifacts/gates/P4.8A/rehearsal_l4/reports/p48a_u1_sonar_rehearsal_report.json")
DEFAULT_P47_REPORT = Path("artifacts/gates/P4.7D_L4/osfm_readiness.json")
DEFAULT_IMPLEMENTATION_REVIEW = Path("artifacts/gates/P4.8B/full_run_implementation_review.json")


@dataclass(frozen=True)
class P48Blocker:
    blocker_id: str
    scope: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {"blocker_id": self.blocker_id, "scope": self.scope, "detail": self.detail}


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _load_json(path: str | Path) -> tuple[dict[str, Any] | None, str | None, str | None]:
    p = _resolve(path)
    if not p.exists():
        return None, None, f"missing JSON artifact: {p}"
    try:
        raw = p.read_bytes()
        data = json.loads(raw.decode("utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        return None, None, f"invalid JSON artifact {p}: {exc}"
    if not isinstance(data, dict):
        return None, None, f"JSON artifact must be an object: {p}"
    return data, hashlib.sha256(raw).hexdigest(), None


def _check_partition_evidence(report: dict[str, Any], blockers: list[P48Blocker]) -> None:
    data = report.get("data") if isinstance(report.get("data"), dict) else {}
    train = data.get("train") if isinstance(data.get("train"), dict) else {}
    validation = data.get("validation") if isinstance(data.get("validation"), dict) else {}
    if train.get("partition") != "PRETRAIN_REAL":
        blockers.append(P48Blocker("P48-DATA-TRAIN-01", "INTERNAL", "training evidence is not PRETRAIN_REAL"))
    if validation.get("partition") != "VALIDATION":
        blockers.append(P48Blocker("P48-DATA-VAL-01", "INTERNAL", "validation evidence is not VALIDATION"))
    if data.get("final_test_used") is not False or data.get("ood_test_used") is not False:
        blockers.append(P48Blocker("P48-DATA-LEAK-01", "INTERNAL", "FINAL_TEST/OOD_TEST was used by rehearsal"))

    train_ids = set(train.get("sample_ids") or [])
    val_ids = set(validation.get("sample_ids") or [])
    overlap = sorted(train_ids & val_ids)
    if overlap:
        blockers.append(P48Blocker("P48-DATA-OVERLAP-01", "INTERNAL", f"train/validation overlap: {overlap[:3]}"))

    excluded = set(train.get("normalization_excluded_partitions") or [])
    val_excluded = set(validation.get("normalization_excluded_partitions") or [])
    required_excluded = {"VALIDATION", "FINAL_TEST", "OOD_TEST"}
    if train.get("normalization_fit_partitions") != ["PRETRAIN_REAL"] or not required_excluded.issubset(excluded):
        blockers.append(
            P48Blocker("P48-STATS-TRAIN-ONLY-01", "INTERNAL", "training statistics are not fit on PRETRAIN_REAL only")
        )
    if validation.get("normalization_fit_partitions") != ["PRETRAIN_REAL"] or not required_excluded.issubset(val_excluded):
        blockers.append(
            P48Blocker("P48-STATS-VAL-LEAK-01", "INTERNAL", "validation statistics do not reuse PRETRAIN_REAL-only stats")
        )
    limitation = " ".join(
        str(x)
        for x in (
            data.get("limitation"),
            train.get("rendered_sonar_limitation"),
            validation.get("rendered_sonar_limitation"),
        )
    ).lower()
    if "rendered side-scan imagery" not in limitation or "not raw acoustic backscatter" not in limitation:
        blockers.append(P48Blocker("P48-SONAR-CLAIM-01", "INTERNAL", "rendered-SSS limitation is not preserved"))


def evaluate_p48_full_launch(
    rehearsal_report: str | Path = DEFAULT_REHEARSAL_REPORT,
    p47_report: str | Path = DEFAULT_P47_REPORT,
    implementation_review: str | Path = DEFAULT_IMPLEMENTATION_REVIEW,
) -> dict[str, Any]:
    blockers: list[P48Blocker] = []
    rehearsal, rehearsal_digest, problem = _load_json(rehearsal_report)
    if problem:
        blockers.append(P48Blocker("P48A-EVIDENCE-01", "INTERNAL", problem))
        rehearsal = {}
    p47, p47_digest, p47_problem = _load_json(p47_report)
    if p47_problem:
        blockers.append(P48Blocker("P47-GO-ARTIFACT-01", "INTERNAL", p47_problem))
        p47 = {}

    if p47.get("status") != "VALIDATED-RUN" or p47.get("decision") != "GO" or p47.get("blockers"):
        blockers.append(P48Blocker("P47-GO-ARTIFACT-02", "INTERNAL", "P4.7 artifact is not blocker-free GO"))

    if rehearsal.get("experiment_id") != "P4.8A-U1-SONAR-REAL-REHEARSAL":
        blockers.append(P48Blocker("P48A-EVIDENCE-02", "INTERNAL", "artifact is not the P4.8A U1 sonar rehearsal"))
    if rehearsal.get("formal_p4_8") is not False or rehearsal.get("rehearsal_only") is not True:
        blockers.append(P48Blocker("P48A-EVIDENCE-03", "INTERNAL", "artifact does not preserve rehearsal-only status"))
    gate = rehearsal.get("gate") if isinstance(rehearsal.get("gate"), dict) else {}
    compute = gate.get("compute") if isinstance(gate.get("compute"), dict) else {}
    if compute.get("cuda_available") is not True:
        blockers.append(P48Blocker("P48-COMPUTE-01", "EXTERNAL", "P4.8A rehearsal was not run on CUDA"))
    if max(gate.get("measured_cuda_vram_gb") or [0]) < 16:
        blockers.append(P48Blocker("P48-COMPUTE-02", "EXTERNAL", "P4.8A rehearsal did not prove >=16 GB CUDA VRAM"))
    if gate.get("clean_tracked_git") is not True:
        blockers.append(P48Blocker("P48-GIT-01", "INTERNAL", "P4.8A rehearsal did not run from clean tracked git"))
    if gate.get("p47_report_digest") and p47_digest and gate.get("p47_report_digest") != p47.get("report_digest"):
        blockers.append(P48Blocker("P48-P47-DIGEST-01", "INTERNAL", "P4.8A references a different P4.7 report digest"))

    _check_partition_evidence(rehearsal, blockers)

    health = rehearsal.get("representation_health") if isinstance(rehearsal.get("representation_health"), dict) else {}
    if health.get("finite") is not True:
        blockers.append(P48Blocker("P48-REPRESENTATION-01", "INTERNAL", "rehearsal representation health is not finite"))

    implementation, implementation_digest, implementation_problem = _load_json(implementation_review)
    if implementation_problem:
        blockers.append(P48Blocker("P48-FULL-RUN-IMPLEMENTATION-01", "INTERNAL", implementation_problem))
        implementation = {}
    else:
        required_review = {
            "status": "IMPLEMENTED",
            "supports_full_run": True,
            "requires_full_run_allowed": True,
            "varies_training_batches": True,
            "reload_verification_required": True,
            "final_or_ood_training_access": "forbidden",
            "normalization_scope": "PRETRAIN_REAL_ONLY",
        }
        missing = {key: value for key, value in required_review.items() if implementation.get(key) != value}
        if missing:
            blockers.append(
                P48Blocker(
                    "P48-FULL-RUN-IMPLEMENTATION-02",
                    "INTERNAL",
                    f"implementation review missing required fields: {sorted(missing)}",
                )
            )
    decision = "GO" if not blockers else "NO-GO"
    status = "VALIDATED-RUN" if rehearsal and p47 else "DESIGNED"
    return {
        "schema_version": "1.0.0",
        "gate_id": "P4.8-FULL-LAUNCH",
        "status": status,
        "decision": decision,
        "rehearsal_report": str(_resolve(rehearsal_report)),
        "rehearsal_report_digest": rehearsal_digest,
        "p47_report": str(_resolve(p47_report)),
        "p47_report_digest": p47_digest,
        "implementation_review": str(_resolve(implementation_review)),
        "implementation_review_digest": implementation_digest,
        "blockers": [b.as_dict() for b in blockers],
    }


def write_p48_full_launch_report(report: dict[str, Any], output: str | Path) -> Path:
    path = _resolve(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
