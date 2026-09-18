"""Claim ladder and claims registry (ch27 Claims registry, ch27 H-XFER-01 twin usefulness).

Implemented -> Synthetic -> Public Real -> Controlled Hardware -> Representative Environment.
A claim is refused when its evidence sits lower on the ladder than the claim.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum

import numpy as np
from pydantic import Field, model_validator

from conrad.evaluation.metrics.bootstrap import PairedComparison, paired_seed_comparison
from conrad.schemas.base import ConradModel


class ClaimLevel(str, Enum):
    NONE = "NONE"
    IMPLEMENTED = "IMPLEMENTED"
    VALIDATED_IN_SIMULATION = "VALIDATED_IN_SIMULATION"
    VALIDATED_ON_PUBLIC_REAL_DATA = "VALIDATED_ON_PUBLIC_REAL_DATA"
    VALIDATED_ON_CONTROLLED_HARDWARE = "VALIDATED_ON_CONTROLLED_HARDWARE"
    VALIDATED_IN_REPRESENTATIVE_ENVIRONMENT = "VALIDATED_IN_REPRESENTATIVE_ENVIRONMENT"

    @property
    def rank(self) -> int:
        return list(ClaimLevel).index(self)


class ClaimRefused(ValueError):
    pass


class Claim(ConradModel):
    claim_id: str = Field(pattern=r"^CLAIM-[A-Z0-9-]+$")
    statement: str = Field(min_length=1)
    mechanism: str = Field(min_length=1)
    evidence_required: str = Field(min_length=1)
    experiments_supporting: tuple[str, ...] = ()
    current_validation_level: ClaimLevel = ClaimLevel.NONE
    allowed_wording: tuple[str, ...] = ()
    prohibited_wording: tuple[str, ...] = ("world's first", "revolutionary", "proven", "guaranteed")

    @model_validator(mode="after")
    def _supported(self) -> Claim:
        if (
            self.current_validation_level.rank > ClaimLevel.IMPLEMENTED.rank
            and not self.experiments_supporting
        ):
            raise ValueError(f"{self.claim_id}: a validated claim needs supporting experiments")
        return self


def check_claim(claim: Claim, evidence_level: ClaimLevel) -> Claim:
    """Refuse a claim above its evidence, or one that uses prohibited wording."""
    if claim.current_validation_level.rank > evidence_level.rank:
        raise ClaimRefused(
            f"{claim.claim_id} claims {claim.current_validation_level.value} "
            f"but evidence reaches only {evidence_level.value}"
        )
    lowered = claim.statement.lower()
    used = [w for w in claim.prohibited_wording if w.lower() in lowered]
    if used:
        raise ClaimRefused(f"{claim.claim_id} uses prohibited wording {used}")
    return claim


def highest_supported_level(levels: list[ClaimLevel]) -> ClaimLevel:
    return max(levels, key=lambda lv: lv.rank) if levels else ClaimLevel.NONE


# ---- twin usefulness TU_d = Perf_real(Real + Twin_d) - Perf_real(RealOnly) ----------------------


class TransferArm(ConradModel):
    """One experimental group evaluated on the untouched final real test."""

    group: str = Field(min_length=1)
    architecture: str = Field(min_length=1)
    real_train_manifest: str | None = Field(
        description="digest of the real adaptation data; None = no real data"
    )
    real_test_split_hash: str | None
    adaptation_budget: dict[str, float] = Field(description="e.g. gpu_hours, optimizer_steps, real_samples")
    metric: str
    direction: str = Field(pattern="^(lower|higher)$")
    seed_results: dict[int, float] = Field(default_factory=dict)


class TwinUsefulnessStatus(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    INCONCLUSIVE = "INCONCLUSIVE"
    NOT_EVALUABLE = "NOT_EVALUABLE"


class TwinUsefulness(ConradModel):
    status: TwinUsefulnessStatus
    reasons: tuple[str, ...] = ()
    comparison: PairedComparison | None = None


def twin_usefulness(
    real_only: TransferArm,
    twin_plus_real: TransferArm,
    rng: np.random.Generator,
    *,
    budget_tolerance: float = 0.0,
) -> TwinUsefulness:
    """NOT_EVALUABLE without real data, unmatched budgets / architecture / test split or seeds."""
    reasons: list[str] = []
    for arm in (real_only, twin_plus_real):
        if arm.real_train_manifest is None or arm.real_test_split_hash is None:
            reasons.append(f"{arm.group}: no real adaptation data or no final real test split")
        if not arm.seed_results:
            reasons.append(f"{arm.group}: no measured real-test results")
    if real_only.architecture != twin_plus_real.architecture:
        reasons.append("architectures differ")
    if real_only.real_train_manifest != twin_plus_real.real_train_manifest:
        reasons.append("real adaptation data differ")
    if real_only.real_test_split_hash != twin_plus_real.real_test_split_hash:
        reasons.append("final real test splits differ")
    if (real_only.metric, real_only.direction) != (twin_plus_real.metric, twin_plus_real.direction):
        reasons.append("metrics differ")
    reasons.extend(
        _budget_mismatches(real_only.adaptation_budget, twin_plus_real.adaptation_budget, budget_tolerance)
    )
    if set(real_only.seed_results) != set(twin_plus_real.seed_results):
        reasons.append("seed sets differ")
    if reasons:
        return TwinUsefulness(status=TwinUsefulnessStatus.NOT_EVALUABLE, reasons=tuple(reasons))
    comparison = paired_seed_comparison(
        real_only.seed_results,
        twin_plus_real.seed_results,
        rng,
        metric=real_only.metric,
        direction=real_only.direction,
    )
    if comparison.repeatable_benefit:
        status = TwinUsefulnessStatus.POSITIVE
    elif comparison.repeatable_regression:
        status = TwinUsefulnessStatus.NEGATIVE
    else:
        status = TwinUsefulnessStatus.INCONCLUSIVE
    return TwinUsefulness(status=status, comparison=comparison)


def _budget_mismatches(a: Mapping[str, float], b: Mapping[str, float], tolerance: float) -> list[str]:
    if not a or not b:
        return ["adaptation budget not declared"]
    if set(a) != set(b):
        return [f"budget keys differ: {sorted(set(a) ^ set(b))}"]
    out = []
    for key in sorted(a):
        scale = max(abs(a[key]), abs(b[key]), 1e-12)
        if abs(a[key] - b[key]) / scale > tolerance:
            out.append(f"budget {key} differs: {a[key]} vs {b[key]}")
    return out
