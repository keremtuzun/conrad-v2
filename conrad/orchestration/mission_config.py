"""Deployment-side configuration of the MissionRuntime. DEPLOYMENT PLANE.

Cadences, module switches, planner choice, link profile and model overrides. Everything is config; the
runtime hard-codes no threshold. All link numbers are SYNTHETIC_ONLY simulation settings.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from conrad.active.production import PRODUCTION, production_view_execution
from conrad.orchestration.multidomain_settings import MultiDomainSettings
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
    # Finding-following outage (gate I7). A fixed window can miss the event it is meant to stress: see
    # conrad.communication.channel.LinkProfile and docs/audits/I7_COMMUNICATIONS.md.
    outage_follows_critical_finding: bool = False
    outage_hold_after_finding_s: float = Field(default=45.0, ge=0)
    outage_max_s: float = Field(default=120.0, gt=0)


class AssociationSettings(ConradModel):
    gate_distance_m: float = Field(default=0.6, gt=0)
    gate_sigma_multiplier: float = Field(default=3.0, ge=0)


class ModuleFaults(ConradModel):
    """Deterministic deployment-module crash injection (tests, INT benchmarks). ``None`` = never."""

    model2e_crash_at_s: float | None = None
    model2s_crash_at_s: float | None = None


class RouteSettings(ConradModel):
    """Planned-route publication for Model 1 (``MissionState.notes["planned_route"]``) and the replan detour.

    All values are ENGINEERING_ESTIMATE simulation settings, not measured vehicle properties.
    """

    enabled: bool = True
    lookahead_m: float = Field(default=6.0, gt=0, description="route ahead of the estimated pose to publish")
    leg_length_m: float = Field(default=1.0, gt=0, description="length of one published corridor leg")
    corridor_half_m: float = Field(default=0.45, gt=0, description="corridor half width around the path")
    skip_near_m: float = Field(
        default=0.6, ge=0, description="path closer than this to the robot is not a leg"
    )
    min_occupied_cells: int = Field(
        default=2, ge=1, description="OBSERVED occupied belief cells to block a leg"
    )
    occupied_probability: float = Field(default=0.7, ge=0, le=1)
    design_clearance_m: float = Field(
        default=0.3,
        ge=0,
        description="cells this close to the surveyed design or charted seabed are known structure",
    )
    detour_climb_m: float = Field(default=1.6, gt=0, description="REPLAN detour: climb over a blocked leg")


class ViewExecutionSettings(ConradModel):
    """MCBR V4 execution protocol. Every switch is OFF by default, so the recorded V3 behaviour is unchanged.

    The V3 diagnostics measured that an accepted view is frequently never flown and that nothing told the
    planner so. These settings close that loop inside the existing contracts: MCBR still only outputs an
    ObservationPlan and navigation still moves the robot.
    """

    enabled: bool = Field(default=False, description="master switch for the V4 execution protocol")
    protect_active_view: bool = Field(
        default=True,
        description="a running inspection view is not abandoned by REPLAN(ROUTE_BLOCKED); the running view "
        "IS the attempt, exactly as the existing busy guard already treats MCBR and NAVIGATION requests",
    )
    belief_map_navigation: bool = Field(
        default=True,
        description="the navigation stack's global and local planners see Model2S OBSERVED occupancy, not "
        "only the surveyed registry design, so the approach routes around discovered structure",
    )
    refund_abandoned_attempt: bool = Field(
        default=True,
        description="a view that was accepted and never flown does not spend a planning attempt on the need",
    )
    drop_abandoned_prior_view: bool = Field(
        default=True,
        description="a view that was never flown is not a prior view: it delivered no observation",
    )
    declare_budgets: bool = Field(
        default=True,
        description="the PlanningRequest carries the remaining mission time and energy, so the feasibility "
        "filter can refuse a view that cannot be reached and dwelled on in the time that is left",
    )
    budget_from_mission_duration: bool = Field(
        default=False,
        description="with no operator time budget declared, use the mission's own remaining duration as the "
        "planning horizon; a view that cannot be reached and dwelled on before the mission ends is then "
        "refused by the feasibility filter instead of being commanded and abandoned at the clock",
    )
    max_view_attempts: int = Field(
        default=8, gt=0, description="hard cap on accepted view goals per need, whatever the refunds"
    )
    route_aware_navigation_cost: bool = Field(
        default=False,
        description="the planner's navigation-cost estimate charges a candidate whose believed corridor "
        "crosses OBSERVED occupied cells; it never refuses one, because on this family the only view that "
        "resolves the defect is reached through a gap between the discovered panels",
    )
    blocked_cost_multiplier: float = Field(default=2.0, ge=1.0)


class MissionRuntimeConfig(ConradModel):
    duration_s: float = Field(default=90.0, gt=0)
    control_period_s: float = Field(default=0.05, gt=0)
    decision_period_s: float = Field(default=2.0, gt=0)
    decision_history_s: float = Field(default=20.0, gt=0, description="decision history H_t window")
    comms_period_s: float = Field(default=1.0, gt=0)
    planner: str = Field(
        default="PRODUCTION",
        description="PRODUCTION = frozen validation-selected planner (configs/active/mcbr_frozen.yaml); "
        "or any conrad.active.make_planners name (baselines)",
    )
    mcbr: dict[str, Any] = Field(default_factory=lambda: {"n_azimuth": 8, "elevations_rad": [0.0, 0.35]})
    v4: dict[str, Any] = Field(
        default_factory=dict, description="conrad.active.v4.V4Config fields when planner is V4"
    )
    view_execution: ViewExecutionSettings = ViewExecutionSettings()
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
    model2t_mode: str = "NONE"  # ADR-0009: no relational propagation in production (2T-TCDP FAIL)
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
    view_position_tolerance_m: float = Field(
        default=0.35,
        gt=0,
        description="declared pose tolerance of an accepted MCBR view; a view is credited as FLOWN only "
        "inside it (conrad.orchestration.view_execution)",
    )
    view_orientation_tolerance_rad: float = Field(
        default=0.3, gt=0, description="declared aim tolerance of an accepted MCBR view"
    )
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
    time_budget_s: float | None = Field(
        default=None,
        gt=0,
        description="operator mission time budget; None = MissionSpec.time_budget_s (feeds "
        "ResourceState.time_remaining_s)",
    )
    route: RouteSettings = RouteSettings()
    # Gate I6 (conrad.orchestration.multidomain): 2E imaging-conditions requirement + inspection gate. Off by default.
    multidomain: MultiDomainSettings = MultiDomainSettings()


def runtime_config(raw: dict[str, Any] | None) -> MissionRuntimeConfig:
    """Resolve the runtime configuration, then adopt the FROZEN view execution protocol for PRODUCTION.

    The V4 selection is a mission protocol, not a ranking rule, so freezing it means the production planner
    and the protocol it was selected with travel together. An explicit ``view_execution`` block in the
    mission configuration still wins, which is what lets an experiment run the incumbent protocol as a
    control arm.
    """
    cfg = MissionRuntimeConfig.model_validate(raw or {})
    if cfg.planner != PRODUCTION or "view_execution" in cfg.model_fields_set:
        return cfg
    frozen = production_view_execution()
    return (
        cfg if frozen is None else cfg.model_copy(update={"view_execution": ViewExecutionSettings(**frozen)})
    )
