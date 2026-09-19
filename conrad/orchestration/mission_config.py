"""Deployment-side configuration of the MissionRuntime. DEPLOYMENT PLANE.

Cadences, module switches, planner choice, link profile and model overrides. Everything is config; the
runtime hard-codes no threshold. All link numbers are SYNTHETIC_ONLY simulation settings.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from conrad.schemas.base import ConradModel


class LinkConfig(ConradModel):
    name: str = "acoustic"
    bandwidth_bps: float = Field(default=1200.0, ge=0)
    latency_s: float = Field(default=1.0, ge=0)
    packet_loss: float = Field(default=0.05, ge=0, le=1)
    bit_error_rate: float = Field(default=1e-6, ge=0, le=1)
    energy_per_bit_j: float = Field(default=1e-4, ge=0)
    packet_bits: int = Field(default=1024, gt=0)
    outages_s: tuple[tuple[float, float], ...] = ()


class AssociationSettings(ConradModel):
    gate_distance_m: float = Field(default=0.6, gt=0)
    gate_sigma_multiplier: float = Field(default=3.0, ge=0)


class ModuleFaults(ConradModel):
    """Deterministic deployment-module crash injection (tests, INT benchmarks). ``None`` = never."""

    model2e_crash_at_s: float | None = None
    model2s_crash_at_s: float | None = None


class MissionRuntimeConfig(ConradModel):
    duration_s: float = Field(default=90.0, gt=0)
    control_period_s: float = Field(default=0.05, gt=0)
    decision_period_s: float = Field(default=2.0, gt=0)
    decision_history_s: float = Field(default=20.0, gt=0, description="decision history H_t window")
    comms_period_s: float = Field(default=1.0, gt=0)
    planner: str = "A-B10_mcbr_full"
    mcbr: dict[str, Any] = Field(default_factory=lambda: {"n_azimuth": 8, "elevations_rad": [0.0, 0.35]})
    decision: dict[str, Any] = Field(default_factory=dict)
    model2s: dict[str, Any] = Field(
        default_factory=lambda: {
            "refinement": {"enabled": False},
            "grid": {"base_voxel_m": 0.35, "refinement_voxels_m": [0.175]},
        }
    )
    model2t: dict[str, Any] = Field(
        default_factory=lambda: {
            "context": {
                # 2E publishes these names (conrad.domains.ecological.context); NTU is not an obscuration
                # level in [0, 1], so turbidity reaches 2T through the visibility fraction instead.
                "biofouling_keys": ["biofouling_cover", "surface_biofouling_cover"],
                "turbidity_keys": [],
                "visibility_keys": ["visibility", "visibility_fraction_at_5m"],
            }
        }
    )
    model2t_mode: str = "TCDP"
    model2e_enabled: bool = True
    model2e: dict[str, Any] = Field(default_factory=dict)
    baac: dict[str, Any] = Field(default_factory=dict)
    link: LinkConfig = LinkConfig()
    association: AssociationSettings = AssociationSettings()
    reoffer_interval_s: float = Field(default=15.0, gt=0, description="min BAAC re-offer interval per belief")
    critical_consequence: float = Field(default=0.9, ge=0, le=1)
    routine_consequence: float = Field(default=0.2, ge=0, le=1)
    critical_finding_value: float = Field(default=0.9, ge=0)
    routine_value: dict[str, float] = Field(
        default_factory=lambda: {"TECHNICAL": 0.5, "ECOLOGICAL": 0.3, "SPATIAL": 0.05}
    )
    inspection_station_half_m: float = Field(default=0.5, gt=0, description="mid-span station half length")
    inspection_dwell_s: float = Field(default=6.0, gt=0)
    inspection_timeout_s: float = Field(default=60.0, gt=0)
    mcbr_unknown_block_probability: float = Field(
        default=0.05,
        ge=0,
        le=1,
        description="per-cell blocking of UNKNOWN cells in MCBR view prediction (ENGINEERING_ESTIMATE)",
    )
    planner_inflation_m: float = Field(default=0.5, ge=0)
    candidate_clearance_m: float = Field(default=0.8, ge=0)
    min_altitude_m: float = Field(default=0.5, ge=0)
    cruise_speed_mps: float = Field(default=0.6, gt=0, description="planner travel-time model")
    cruise_speed_fraction: float = Field(default=0.8, gt=0, le=1, description="of RobotConfig max speed")
    max_plans_per_need: int = Field(default=4, gt=0)
    module_faults: ModuleFaults = ModuleFaults()


def runtime_config(raw: dict[str, Any] | None) -> MissionRuntimeConfig:
    return MissionRuntimeConfig.model_validate(raw or {})
