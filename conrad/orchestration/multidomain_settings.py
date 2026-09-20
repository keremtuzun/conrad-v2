"""Settings of the gate I6 multi-domain pieces (``conrad.orchestration.multidomain``). DEPLOYMENT PLANE.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from pydantic import Field

from conrad.schemas.base import ConradModel


class MultiDomainSettings(ConradModel):
    """All numbers are ENGINEERING_ESTIMATE deployment settings, fixed on development worlds (I6 config)."""

    enabled: bool = False
    turbidity_field: str = Field(default="turbidity", description="Model2E field whose belief is required")
    turbidity_property: str = "turbidity.mean"
    turbidity_sd_property: str = "turbidity.sd"
    sensing_consequence: float = Field(
        default=0.2, ge=0, le=1, description="below DecisionConfig.consequence_matters_above: context only"
    )
    inspection_range_m: float = Field(
        default=1.0, gt=0, description="closest feasible inspection range (MCBR min range + clearance)"
    )
    turbidity_sd_multiplier: float = Field(
        default=2.0,
        ge=0,
        description="the gate uses the upper credible turbidity mean + k * sd (a deferral is cheap, a blind "
        "inspection is not); k = 0 uses the mean",
    )
    min_visibility: float = Field(
        default=0.5,
        gt=0,
        lt=1,
        description="beam transmission exp(-c r) below which an inspection is deferred",
    )
