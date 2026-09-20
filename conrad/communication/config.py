"""BAAC configuration. All defaults are ENGINEERING_ESTIMATE / SYNTHETIC_ONLY simulation values."""

from __future__ import annotations

from pydantic import Field

from conrad.schemas.base import ConradModel


class BAACConfig(ConradModel):
    model_version: str = "baac-greedy-0.3"
    scheduler_policy: str = Field(
        default="C-B10_baac",
        description="SchedulingPolicy name: C-B10_baac or a baseline C-B0..C-B4 (scheduler.ALL_POLICIES)",
    )
    critical_value: float = Field(
        default=0.8, ge=0, description="mission_value at/above which a unit is critical"
    )
    information_retained: tuple[float, float, float, float, float] = Field(
        default=(0.3, 0.7, 0.85, 0.95, 1.0), description="F0..F4, fraction of mission information retained"
    )
    embedding_summary_dims: int = Field(default=16, gt=0, description="F2 carries this many quantised dims")
    compressed_evidence_fraction: float = Field(default=0.05, gt=0, le=1, description="F3 bytes / raw bytes")
    default_raw_evidence_bytes: int = Field(default=200_000, gt=0)
    preempt_until_delivered: bool = Field(
        default=True,
        description="ch19 Level 0/1 pre-emption by a critical unit lasts until the receiver holds that "
        "belief; afterwards the critical unit competes on value per bit (it keeps its high mission "
        "value). Unbounded pre-emption lets one fast-moving critical belief hold the whole link.",
    )
    receiver_relative_value: bool = Field(
        default=True,
        description="BAAC only: score an increment by the information the receiver GAINS, discounting what "
        "it already holds about that belief (``stale_view_credit``), instead of by the unit's absolute "
        "information content.",
    )
    stale_view_credit: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="fraction of a belief's F1 information a receiver still holds when its view is at an "
        "older revision (ENGINEERING_ESTIMATE; the same modelling assumption the communication oracle "
        "makes for a stale view)",
    )
    high_aleatoric: float = Field(default=0.5, ge=0)
    high_epistemic: float = Field(default=0.5, ge=0)
    high_contradiction: float = Field(default=0.5, ge=0)
    queue_capacity_bits: int = Field(default=8_000_000_000, gt=0, description="onboard store (1 GB)")
    energy_budget_j_per_step: float | None = Field(default=None, gt=0)
    default_deadline_s: float | None = Field(default=None, gt=0)
