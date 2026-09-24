"""Truth-side options of the integrated mission world (Phases 5-11). TRUTH PLANE. SYNTHETIC_ONLY.

Every number here is a simulation setting (``SYNTHETIC_ONLY``), never a measured physical value. The
deployment side never sees this object: it receives the RHI, the mission context and its own config.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from conrad.schemas.base import ConradModel
from conrad.schemas.structural_sensor import StructuralSensorModelV2


class SpatialCellOptions(ConradModel):
    corrosion_depth_m: float = Field(default=0.0, ge=0)
    crack_length_m: float = Field(default=0.0, ge=0)
    crack_depth_m: float = Field(default=0.0, ge=0)


class SpatialRateOptions(ConradModel):
    corrosion_m_per_s: float = Field(default=0.0, ge=0)
    crack_m_per_s: float = Field(default=0.0, ge=0)
    crack_depth_m_per_s: float = Field(default=0.0, ge=0)


class SpatialPatchOptions(ConradModel):
    axial_start_fraction: float = Field(ge=0, lt=1)
    axial_end_fraction: float = Field(gt=0, le=1)
    angle_start_rad: float = Field(ge=0, lt=6.283185307179586)
    angle_end_rad: float = Field(gt=0, le=6.283185307179586)
    state: SpatialCellOptions
    rate: SpatialRateOptions = SpatialRateOptions()

    @model_validator(mode="after")
    def _valid(self) -> SpatialPatchOptions:
        if self.axial_end_fraction <= self.axial_start_fraction or self.angle_end_rad <= self.angle_start_rad:
            raise ValueError("spatial patch must have positive non-wrapping extent")
        return self


class SpatialTruthOptions(ConradModel):
    axial_cells: int = Field(gt=0)
    sectors: int = Field(gt=0)
    base: SpatialCellOptions = SpatialCellOptions()
    cell_states: tuple[SpatialCellOptions, ...] | None = None
    base_rate: SpatialRateOptions = SpatialRateOptions()
    cell_rates: tuple[SpatialRateOptions, ...] | None = None
    patches: tuple[SpatialPatchOptions, ...] = ()

    @model_validator(mode="after")
    def _valid(self) -> SpatialTruthOptions:
        n = self.axial_cells * self.sectors
        if self.cell_states is not None and len(self.cell_states) != n:
            raise ValueError("spatial cell_states length must match grid")
        if self.cell_rates is not None and len(self.cell_rates) != n:
            raise ValueError("spatial cell_rates length must match grid")
        return self


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
    side: str = Field(
        default="far",
        pattern="^(far|near|off_lane|lane)$",
        description="far/near = the WORLD +Y normal of the pipe heading (historical; whether that is the "
        "lane side depends on the sampled heading). off_lane/lane are defined against the transit lane the "
        "mission actually flies: off_lane is genuinely hidden from the nominal route in every world "
        "(ACTIVE_INSPECTION_OCCLUDED_V1, docs/audits/I4_WORLD_FAMILY.md)",
    )
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


class DefectVariationOptions(ConradModel):
    """Per-world sampling of the hidden defect (ACTIVE_INSPECTION_OCCLUDED_V1). SYNTHETIC_ONLY.

    Off by default: every existing scenario keeps the single fixed defect it has today. Ranges are declared
    simulation settings, never measured values, and the draw uses its own seeded stream so enabling it does
    not disturb any other random stream of the world.
    """

    enabled: bool = False
    corrosion_depth_m: tuple[float, float] = (0.0035, 0.0095)
    crack_length_m: tuple[float, float] = (0.030, 0.140)
    patch_tilt_deg: tuple[float, float] = Field(
        default=(0.0, 45.0),
        description="patch axis above the horizontal. Not negative: the shared candidate generator samples "
        "only non-negative elevations, so a patch on the underside is unreachable for every planner",
    )
    patch_half_angle_deg: tuple[float, float] = (35.0, 55.0)
    patch_axial_fraction: tuple[float, float] = (0.22, 0.38)


class ViewOcclusionOptions(ConradModel):
    """One unregistered structure that hides part of the target's critical surface. SYNTHETIC_ONLY.

    Twin2S geometry only: no registry entry, no Twin2T/Twin2E state, no Observation names it, so the
    deployment can learn about it only through its own geometric sensors. Placement is sampled per world in
    the frame of the target segment and of the defect patch (``conrad.sim.mission.occlusion``).
    """

    enabled: bool = False
    azimuth_offset_deg: tuple[float, float] = Field(
        default=(-40.0, 40.0),
        description="angle of the WINDOW between the two panels, around the pipe axis, measured from the "
        "patch direction",
    )
    window_half_deg: tuple[float, float] = Field(
        default=(16.0, 34.0), description="angular half-width of the window left between the two panels"
    )
    standoff_m: tuple[float, float] = Field(
        default=(0.30, 0.95), description="radial gap between the pipe surface and a panel's near face"
    )
    half_width_m: tuple[float, float] = Field(default=(0.30, 0.55), description="tangential half-width")
    half_length_fraction: tuple[float, float] = Field(
        default=(0.35, 0.75), description="axial half-length as a fraction of the target segment length"
    )
    axial_shift_fraction: tuple[float, float] = Field(
        default=(-0.10, 0.10), description="axial offset of the panel centres, as a fraction of the length"
    )
    min_window_z: float = Field(
        default=-0.05,
        ge=-1,
        le=1,
        description="reachability guard: the window direction's z component may not fall below this, "
        "otherwise the rotation sign is flipped. A window aimed into the seabed is unreachable for every "
        "planner and would make the world measure nothing",
    )
    thickness_m: float = Field(default=0.06, gt=0)
    posts: bool = Field(default=False, description="two vertical posts from the window ends to the seabed")
    post_half_m: float = Field(default=0.07, gt=0)
    post_inset_m: float = Field(default=0.10, ge=0)
    semantic_class: str = "frame"
    post_semantic_class: str = "support"
    material_id: str = "steel_generic"


class MissionWorldOptions(ConradModel):
    twin2t_truth_model: Literal["legacy", "spatial_v1"] = "legacy"
    spatial_truth: SpatialTruthOptions | None = None
    spatial_sensor_model: StructuralSensorModelV2 | None = None
    family: str = "straight_pipeline"
    target_segment_index: int = Field(default=1, ge=0)
    lane_offset_m: float = Field(default=2.0, gt=0, description="-Y transit lane distance from the pipe axis")
    lane_height_m: float = Field(default=0.6, description="lane height above the pipe axis")
    lane_margin_m: float = Field(default=1.0, ge=0, description="lane extension past the pipe ends")
    survey_sigma_m: float = Field(default=0.05, ge=0, description="registry design-geometry survey noise")
    survey_endpoint_bound_m: float | None = Field(
        default=None,
        gt=0,
        description="optional hard endpoint error bound for a clipped synthetic design survey",
    )
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
    defect_variation: DefectVariationOptions = DefectVariationOptions()
    occlusion: ViewOcclusionOptions = ViewOcclusionOptions()
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

    @model_validator(mode="after")
    def _spatial_contract(self) -> MissionWorldOptions:
        if self.twin2t_truth_model == "spatial_v1":
            if self.spatial_truth is None or self.spatial_sensor_model is None:
                raise ValueError("spatial_v1 requires spatial_truth and spatial_sensor_model")
            if self.defect.pristine_rest or self.structural.region_tiles is not None:
                raise ValueError("spatial_v1 cannot use legacy pristine-rest or region tiles")
        elif self.spatial_truth is not None or self.spatial_sensor_model is not None:
            raise ValueError("spatial configuration requires twin2t_truth_model=spatial_v1")
        return self


def world_options(raw: dict[str, Any] | None) -> MissionWorldOptions:
    return MissionWorldOptions.model_validate(raw or {})
