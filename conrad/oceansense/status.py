"""Machine-readable P5-P10 implementation status gate."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from conrad.settings import REPO_ROOT

PHASES = ("P5", "P6", "P7", "P8", "P9", "P10")
CONTRACT_TEST = "tests/unit/oceansense/test_p5_p10_contracts.py"
QUALIFIED_CHECKPOINT_LABEL = "OSFM-S-PRETRAIN-V1"


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _repo_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint_metadata(path: str | Path | None) -> tuple[dict[str, Any] | None, str | None]:
    if path is None:
        return None, "qualified OSFM-S-PRETRAIN-V1 metadata path not supplied"
    resolved = _resolve(path)
    if not resolved.is_file():
        return None, f"qualified OSFM-S-PRETRAIN-V1 metadata missing: {resolved}"
    try:
        data = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"qualified OSFM-S-PRETRAIN-V1 metadata invalid: {exc}"
    if not isinstance(data, dict):
        return None, "qualified OSFM-S-PRETRAIN-V1 metadata must be a JSON object"
    return data, None


def _checkpoint_verified(metadata: dict[str, Any] | None) -> tuple[bool, str]:
    if not metadata:
        return False, "no qualified checkpoint metadata"
    label = metadata.get("checkpoint_label") or metadata.get("produces") or metadata.get("checkpoint_id")
    if label != QUALIFIED_CHECKPOINT_LABEL:
        return False, f"checkpoint label is {label!r}, expected {QUALIFIED_CHECKPOINT_LABEL!r}"
    if metadata.get("status") != "VALIDATED-RUN":
        return False, f"checkpoint status is {metadata.get('status')!r}, expected 'VALIDATED-RUN'"
    if metadata.get("decision") not in {"PROMOTE", "GO"}:
        return False, f"checkpoint decision is {metadata.get('decision')!r}, expected PROMOTE or GO"
    return True, "qualified OSFM-S-PRETRAIN-V1 metadata is present"


def verify_qualified_checkpoint_metadata(path: str | Path | None) -> tuple[bool, str, bool]:
    """Verify qualified checkpoint metadata using the same fail-closed rule as the status gate."""

    metadata, metadata_problem = _load_checkpoint_metadata(path)
    checkpoint_ok, checkpoint_detail = _checkpoint_verified(metadata)
    return checkpoint_ok, metadata_problem or checkpoint_detail, path is not None


def evaluate_p5_p10_status(
    *,
    test_result: str = "NOT_RUN",
    qualified_checkpoint_metadata: str | Path | None = None,
) -> dict[str, Any]:
    """Return fail-closed P5-P10 status without running tests."""

    checkpoint_ok, checkpoint_detail, metadata_supplied = verify_qualified_checkpoint_metadata(qualified_checkpoint_metadata)
    tests_passed = test_result == "PASS"
    implementation_status = "IMPLEMENTED" if tests_passed else "DESIGNED"
    integrated_status = "VALIDATED-RUN" if tests_passed and checkpoint_ok else implementation_status
    blockers = []
    if not tests_passed:
        blockers.append(
            {
                "blocker_id": "P5-P10-TESTS-01",
                "scope": "INTERNAL",
                "detail": f"contract tests are {test_result}; required PASS for IMPLEMENTED",
            }
        )
    if not checkpoint_ok:
        blockers.append(
            {
                "blocker_id": "P5-P10-OSFM-01",
                "scope": "EXTERNAL",
                "detail": checkpoint_detail,
            }
        )
    phases = {
        phase: {
            "status": integrated_status if checkpoint_ok else implementation_status,
            "validated_run_allowed": checkpoint_ok and tests_passed,
            "final_validation_dependency": QUALIFIED_CHECKPOINT_LABEL,
        }
        for phase in PHASES
    }
    return {
        "schema_version": "1.0.0",
        "gate_id": "P5-P10-DOWNSTREAM-CONTRACTS",
        "source_commit": _repo_commit(),
        "contract_test": CONTRACT_TEST,
        "contract_test_result": test_result,
        "qualified_checkpoint": {
            "required_label": QUALIFIED_CHECKPOINT_LABEL,
            "metadata_supplied": metadata_supplied,
            "verified": checkpoint_ok,
            "detail": checkpoint_detail,
        },
        "status": integrated_status,
        "decision": "VALIDATED-RUN" if integrated_status == "VALIDATED-RUN" else implementation_status,
        "phases": phases,
        "blockers": blockers,
        "status_note": (
            "P5-P10 implementation contracts may be IMPLEMENTED with fixtures/dev checkpoints; final integrated "
            "VALIDATED-RUN requires verified OSFM-S-PRETRAIN-V1."
        ),
    }


def write_p5_p10_status(report: dict[str, Any], output: str | Path) -> Path:
    path = _resolve(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def status_digest(path: str | Path) -> str:
    return _sha256(_resolve(path))
