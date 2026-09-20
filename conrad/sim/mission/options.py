"""Truth-side options of the integrated mission world (Phases 5-11). TRUTH PLANE. SYNTHETIC_ONLY.

Every number here is a simulation setting (``SYNTHETIC_ONLY``), never a measured physical value. The
deployment side never sees this object: it receives the RHI, the mission context and its own config.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from conrad.schemas.base import ConradModel


class DefectOptions(ConradModel):
    """Hidden defect on the far (+Y) side of the target segment: corrosion wall loss plus a crack."""

    corrosion_depth_m: float = Field(default=0.006, ge=0)
    crack_length_m: float = Field(default=0.08, ge=0)
    patch_half_angle_deg: float = Field(default=50.0, gt=0, le=90)
    patch_tilt_deg: float = Field(
        default=15.0, ge=-60, le=60, description="patch axis above the +Y horizontal"
    )
    patch_axial_fraction: float = Field(default=0.3, gt=0, le=1)
    patch_samples: int = Field(default=48, gt=4)
    side: str = Field(default="far", pattern="^(far|near)$", description="far = +Y (hidden from the lane)")
    pristine_rest: bool = Field(
        default=False,
        description="the target's non-defect surface starts with no corrosion and no crack (default: sampled "
        "from the population priors); set only by I5-NOMINAL-READABLE",
    )


class StructuralSensorOptions(ConradModel):
    """Forward-looking structural inspection payload (Twin2T T0 measurements through a Twin2S view)."""

    hfov_deg: float = 100.0
    vfov_deg: float = 80.0
    min_range_m: float = 0.3
    max_range_m: float = 4.0
    mount_position_m: tuple[float, float, float] = (0.0, 0.15, 0.0)
    mount_yaw_deg: float = Field(default=90.0, description="side-looking (+Y body) inspection payload")
    range_sigma_m: float = Field(default=0.05, ge=0)
    angle_sigma_rad: float = Field(default=0.02, ge=0)
    surface_samples: int = Field(default=40, gt=4)
    degradation: dict[str, float] = Field(default_factory=dict, description="Twin2T quality keys")
    turbidity_ntu_full_scale: float = Field(default=20.0, gt=0)
    region_tiles: tuple[int, int] | None = Field(
        default=None,
        description="(axial, circumferential) tiles of the target's non-defect surface, each read on its own, "
        "so one view yields a reading per visible tile instead of one averaged point; None = one region "
        "(default, every scenario before I5-NOMINAL-READABLE)",
    )


class ContradictionOptions(ConradModel):
    """Optional credible contradiction: a second, degraded structural sensor with a biased reading."""

    enabled: bool = False
    start_s: float = 0.0
    contradiction: float = Field(default=-0.8, ge=-1, le=1)
    sensor_bias_m: float = -0.004
    corruption: float = Field(default=0.3, ge=0, le=1)


class FixOptions(ConradModel):
    period_s: float = Field(default=1.0, gt=0)
    sigma_m: float = Field(default=0.10, gt=0)
    outages: tuple[tuple[float, float], ...] = ()


class FaultSpec(ConradModel):
    """Kernel fault (FaultType name) or FIX_OUTAGE. ``trigger`` INSPECTION_GOAL fires ``t_s`` seconds after the
    runtime publicly logs its first accepted INSPECT goal (scenario scripting on the public event log)."""

    t_s: float = Field(ge=0)
    type: str
    trigger: str = Field(default="TIME", pattern="^(TIME|INSPECTION_GOAL)$")
    target: str | None = None
    magnitude: float = 0.0
    duration_s: float | None = None


class EcoEventSpec(ConradModel):
    t_s: float = Field(ge=0)
    event_type: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class LaneObstacleOptions(ConradModel):
    """Unregistered box on the transit lane (not in the asset registry): the I5 blocked-route scenario.

    Placed on the lane polyline at ``lane_fraction`` of its length (0 = start, 1 = end), centred on the lane
    height. Truth side only; the runtime can learn about it only through its sensors.
    """

    lane_fraction: float = Field(default=0.55, ge=0, le=1)
    half_extent_m: tuple[float, float, float] = (0.5, 0.7, 0.5)
    material_id: str = "rock"


class MissionWorldOptions(ConradModel):
    family: str = "straight_pipeline"
    target_segment_index: int = Field(default=1, ge=0)
    lane_offset_m: float = Field(default=2.0, gt=0, description="-Y transit lane distance from the pipe axis")
    lane_height_m: float = Field(default=0.6, description="lane height above the pipe axis")
    lane_margin_m: float = Field(default=1.0, ge=0, description="lane extension past the pipe ends")
    survey_sigma_m: float = Field(default=0.05, ge=0, description="registry design-geometry survey noise")
    seabed_chart_sigma_m: float = Field(default=0.05, ge=0)
    launch_sigma_m: float = Field(default=0.05, ge=0)
    sensor_overrides: dict[str, dict[str, Any]] = Field(
        default_factory=lambda: {
            "DEPTH_RANGE": {"width_px": 24, "height_px": 18, "max_range_m": 6.0},
            "SONAR": {"n_beams": 20, "n_range_bins": 32, "max_range_m": 6.0},
        }
    )
    side_looking_geometric: tuple[str, ...] = Field(default=("DEPTH_RANGE",), description="yawed +90 deg")
    geometric_period_s: float = Field(default=3.0, gt=0)
    geometric_degradation: dict[str, float] = Field(default_factory=dict, description="Twin2S keys")
    geometric_dropout: tuple[tuple[float, float], ...] = Field(default=(), description="modality dropout")
    structural_period_s: float = Field(default=1.0, gt=0)
    structural: StructuralSensorOptions = StructuralSensorOptions()
    contradiction: ContradictionOptions = ContradictionOptions()
    defect: DefectOptions = DefectOptions()
    ecological_enabled: bool = True
    environmental_period_s: float = Field(default=5.0, gt=0)
    survey_period_s: float = Field(default=5.0, gt=0)
    environment: dict[str, Any] = Field(
        default_factory=lambda: {
            "temperature": {"value": 11.0, "units": "degC"},
            "turbidity": {"value": 2.0, "units": "NTU"},
            "current": {"value": [0.02, 0.0, 0.0], "units": "m s-1"},
            "light": {"value": 300.0, "units": "W m-2"},
        }
    )
    eco_events: tuple[EcoEventSpec, ...] = ()
    fix: FixOptions = FixOptions()
    faults: tuple[FaultSpec, ...] = ()
    physics_dt_s: float = Field(default=0.025, gt=0)
    lane_obstacle: LaneObstacleOptions | None = None
    current_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)


def world_options(raw: dict[str, Any] | None) -> MissionWorldOptions:
    return MissionWorldOptions.model_validate(raw or {})
