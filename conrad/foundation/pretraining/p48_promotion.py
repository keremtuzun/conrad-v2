"""Fail-closed promotion review for a completed P4.8 U1-sonar research run."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from conrad.foundation.pretraining.p48_gate import _check_partition_evidence
from conrad.settings import REPO_ROOT


@dataclass(frozen=True)
class P48PromotionBlocker:
    blocker_id: str
    scope: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {"blocker_id": self.blocker_id, "scope": self.scope, "detail": self.detail}


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _load_mapping(path: str | Path, *, yaml_allowed: bool = False) -> tuple[dict[str, Any], str | None]:
    resolved = _resolve(path)
    if not resolved.is_file():
        return {}, f"missing artifact: {resolved}"
    try:
        data = yaml.safe_load(resolved.read_text(encoding="utf-8")) if yaml_allowed else json.loads(
            resolved.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        return {}, f"invalid artifact {resolved}: {exc}"
    if not isinstance(data, dict):
        return {}, f"artifact must contain a mapping: {resolved}"
    return data, None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_p48_promotion(
    *,
    run_report: str | Path,
    resolved_config: str | Path,
    run_state: str | Path,
    p47_report: str | Path = "artifacts/gates/P4.7D_L4/osfm_readiness.json",
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    blockers: list[P48PromotionBlocker] = []

    report, problem = _load_mapping(run_report)
    if problem:
        blockers.append(P48PromotionBlocker("P48-PROMOTION-REPORT-01", "INTERNAL", problem))
    config, problem = _load_mapping(resolved_config, yaml_allowed=True)
    if problem:
        blockers.append(P48PromotionBlocker("P48-PROMOTION-CONFIG-01", "INTERNAL", problem))
    state, problem = _load_mapping(run_state)
    if problem:
        blockers.append(P48PromotionBlocker("P48-PROMOTION-STATE-01", "INTERNAL", problem))
    p47, problem = _load_mapping(p47_report)
    if problem:
        blockers.append(P48PromotionBlocker("P48-PROMOTION-P47-01", "INTERNAL", problem))

    if report.get("experiment_id") != "P4.8-U1-SONAR-RESEARCH":
        blockers.append(P48PromotionBlocker("P48-PROMOTION-RUN-01", "INTERNAL", "wrong experiment identity"))
    if report.get("formal_p4_8") is not True or report.get("rehearsal_only") is not False:
        blockers.append(P48PromotionBlocker("P48-PROMOTION-RUN-02", "INTERNAL", "run is not formal P4.8"))
    if config.get("stage_id") != "U1-SONAR-RESEARCH" or config.get("promotable") is not True:
        blockers.append(
            P48PromotionBlocker(
                "P48-PROMOTION-CONFIG-02",
                "INTERNAL",
                "resolved config is not the promotable U1-SONAR-RESEARCH stage",
            )
        )
    configured_steps = int(config.get("optimizer_steps", 0) or 0)
    completed_steps = int((report.get("metrics") or {}).get("optimizer_steps", 0) or 0)
    if configured_steps < 100_000 or completed_steps != configured_steps:
        blockers.append(
            P48PromotionBlocker(
                "P48-PROMOTION-STEPS-01",
                "INTERNAL",
                f"requires exactly the configured >=100000 steps; configured={configured_steps}, completed={completed_steps}",
            )
        )
    if state.get("status") != "COMPLETED" or state.get("comparability") != "COMPARABLE":
        blockers.append(
            P48PromotionBlocker("P48-PROMOTION-STATE-02", "INTERNAL", "run is not sealed COMPLETED/COMPARABLE")
        )

    if p47.get("status") != "VALIDATED-RUN" or p47.get("decision") != "GO" or p47.get("blockers"):
        blockers.append(P48PromotionBlocker("P48-PROMOTION-P47-02", "INTERNAL", "P4.7D is not blocker-free GO"))
    gate = report.get("gate") if isinstance(report.get("gate"), dict) else {}
    if gate.get("p47_report_digest") != p47.get("report_digest"):
        blockers.append(P48PromotionBlocker("P48-PROMOTION-P47-03", "INTERNAL", "P4.7D digest mismatch"))
    if gate.get("clean_tracked_git") is not True:
        blockers.append(P48PromotionBlocker("P48-PROMOTION-GIT-01", "INTERNAL", "training git state was dirty"))
    compute = gate.get("compute") if isinstance(gate.get("compute"), dict) else {}
    if compute.get("cuda_available") is not True or max(gate.get("measured_cuda_vram_gb") or [0]) < 16:
        blockers.append(
            P48PromotionBlocker("P48-PROMOTION-COMPUTE-01", "EXTERNAL", "run lacks qualifying CUDA evidence")
        )

    partition_blockers: list[Any] = []
    _check_partition_evidence(report, partition_blockers)
    blockers.extend(
        P48PromotionBlocker(item.blocker_id, item.scope, item.detail) for item in partition_blockers
    )

    health = report.get("representation_health") if isinstance(report.get("representation_health"), dict) else {}
    if (
        health.get("finite") is not True
        or float(health.get("collapse_score", 0.0) or 0.0) <= 0.0
        or health.get("formal_rank_guard") != "PASS"
        or float(health.get("effective_rank", 0.0) or 0.0) < 64.0
    ):
        blockers.append(
            P48PromotionBlocker(
                "P48-PROMOTION-HEALTH-01",
                "RESEARCH_RESULT",
                "representation health did not pass finite, non-collapse, and effective-rank >=64 gates",
            )
        )
    reload_evidence = report.get("reload") if isinstance(report.get("reload"), dict) else {}
    if reload_evidence.get("matches") is not True or float(reload_evidence.get("max_abs_diff", 1.0)) > 1e-6:
        blockers.append(P48PromotionBlocker("P48-PROMOTION-RELOAD-01", "INTERNAL", "checkpoint reload mismatch"))

    checkpoint_path = _resolve(checkpoint or report.get("checkpoint", ""))
    checkpoint_id = str(report.get("checkpoint_id") or "")
    if not checkpoint_path.is_file():
        blockers.append(
            P48PromotionBlocker("P48-PROMOTION-CHECKPOINT-01", "INTERNAL", f"missing checkpoint: {checkpoint_path}")
        )
    elif _sha256(checkpoint_path) != checkpoint_id:
        blockers.append(
            P48PromotionBlocker("P48-PROMOTION-CHECKPOINT-02", "INTERNAL", "checkpoint digest mismatch")
        )

    promoted = not blockers
    return {
        "schema_version": "1.0.0",
        "gate_id": "P4.8-U1-SONAR-PROMOTION",
        "status": "VALIDATED-RUN" if report and config and state and p47 else "DESIGNED",
        "decision": "PROMOTE" if promoted else "DO-NOT-PROMOTE",
        "stage_id": "U1-SONAR-RESEARCH",
        "checkpoint_id": checkpoint_id or None,
        "promotion_record": {
            "promoted": promoted,
            "produces": "U1-SONAR-RESEARCH-CHECKPOINT" if promoted else None,
            "next_stage": "P4.9" if promoted else None,
        },
        "blockers": [blocker.as_dict() for blocker in blockers],
    }


def write_p48_promotion_report(report: dict[str, Any], output: str | Path) -> Path:
    path = _resolve(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
