"""Twin2E observation generator (ch12 levels E0/E1, partial surveys, sparse sensors). TRUTH PLANE.

* ENVIRONMENTAL sensors: one Observation per field channel at the sensor's TRUE position, with
  inline_values + inline_units; truth goes only to TwinSample.supervision.
* Ecological surveys (RGB/STRUCTURED sensors): cover estimates for entities inside the camera
  footprint, with detection probability and noise driven by turbidity-dependent visibility.
  E0 = noisy abstract cover; E1 = state-conditioned feature vector. E2 (pixels) belongs to the
  renderer and is refused rather than faked.
* Each Observation carries exactly ctx.timestamp: streams are never re-timed to look synchronous.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from conrad.schemas.frames import quat_to_matrix
from conrad.schemas.observation import Modality, Observation
from conrad.schemas.truth import SupervisionLabel
from conrad.schemas.world import Domain
from conrad.twins.base import SensingContext, TwinSample
from conrad.twins.twin2e.config import ObservationLevel
from conrad.twins.twin2e.ecology import SESSILE, EntityState
from conrad.twins.twin2e.fields import FIELD_UNITS
from conrad.twins.twin2e.priors import KIND_BENTHIC, KIND_BIOFOULING, KIND_MOBILE

if TYPE_CHECKING:
    from conrad.twins.twin2e.twin import Twin2E

SURVEY_MODALITIES = {Modality.RGB.value, Modality.STRUCTURED.value}
_KINDS = (KIND_BIOFOULING, KIND_BENTHIC, KIND_MOBILE)


class Twin2EObservationLevelError(NotImplementedError):
    """E2 sensor-level imagery is produced by the renderer (Unity/Twin2S), not by Twin2E."""


class ObservationGenerator:
    def __init__(self, twin: Twin2E, rng: np.random.Generator, feature_rng: np.random.Generator) -> None:
        self.twin = twin
        self.rng = rng
        self.projection = feature_rng.standard_normal((twin.cfg.observation.feature_dim, 10)) / math.sqrt(10)

    def _lineage(self) -> str:
        s = self.twin._require()
        return f"{s.scenario_id}/{s.metadata.get('family', 'unknown')}/{self.twin.seed}"

    def _obs(
        self, ctx: SensingContext, values: list[float], units: str, context: dict[str, Any]
    ) -> Observation:
        modality = Modality.ENVIRONMENTAL if context.get("kind") == "field_sample" else Modality.STRUCTURED
        return Observation(
            observation_id=self.twin.ids.new(),
            mission_id=ctx.mission_id,
            run_id=ctx.run_id,
            trace_id=ctx.trace_id,
            sensor_id=ctx.sensor.sensor_id,
            modality=modality,
            timestamp=ctx.timestamp,
            sensor_frame=ctx.sensor.frame_id,
            robot_pose_estimate=ctx.estimated_pose,
            inline_values=tuple(float(v) for v in values),
            inline_units=units,
            calibration_ref=ctx.sensor.calibration_ref,
            sensor_context=context,
        )

    def generate(self, ctx: SensingContext) -> list[TwinSample]:
        modality = ctx.sensor.modality
        if modality == Modality.ENVIRONMENTAL.value:
            return self._environmental(ctx)
        if modality in SURVEY_MODALITIES:
            level = ObservationLevel(ctx.sensor.parameters.get("level", self.twin.cfg.observation.level))
            if level is ObservationLevel.E2_SENSOR:
                raise Twin2EObservationLevelError("E2 imagery is renderer-owned; request E0/E1 from Twin2E")
            return self._survey(ctx, level)
        return []  # modality not produced by Twin2E: nothing sensed here, nothing fabricated

    # ------------------------------------------------------------------ environmental point sensor
    def _environmental(self, ctx: SensingContext) -> list[TwinSample]:
        self.twin.fields.local_grid.check_frame(ctx.true_pose.frame_id)
        pos = np.asarray(ctx.true_pose.position_m, dtype=np.float64)[None]
        names = list(ctx.sensor.parameters.get("fields", list(FIELD_UNITS)))
        scale = float(ctx.degradation.get("noise_scale", 1.0))
        out = []
        for name in names:
            if name not in FIELD_UNITS:
                raise KeyError(f"sensor requests unknown field {name!r}")
            true = np.atleast_1d(self.twin.fields.sample(name, pos)[0])
            sigma = self.twin.cfg.observation.sensor_sigma[name] * scale
            meas = true + sigma * self.rng.standard_normal(true.shape)
            if name in ("turbidity", "light"):
                meas = np.clip(meas, 0.0, None)
            obs = self._obs(
                ctx,
                list(meas),
                FIELD_UNITS[name],
                {"kind": "field_sample", "channel": name, "noise_sigma": sigma},
            )
            sup = SupervisionLabel(
                observation_id=obs.observation_id,
                domain=Domain.ECOLOGICAL,
                targets={name: [float(v) for v in true], "units": FIELD_UNITS[name]},
                target_masks={name: True},
                lineage=self._lineage(),
            )
            out.append(TwinSample(observation=obs, supervision=sup))
        return out

    # ------------------------------------------------------------------ ecological survey
    def _visibility(self, e: EntityState, sensor_pos: np.ndarray) -> tuple[float, float]:
        d = float(np.linalg.norm(e.position_m - sensor_pos))
        mid = 0.5 * (e.position_m + sensor_pos)
        turb = float(self.twin.fields.sample("turbidity", mid[None])[0])
        o = self.twin.cfg.observation
        c = o.beam_attenuation_clear_per_m + o.beam_attenuation_per_m_per_ntu * turb
        return d, math.exp(-c * d)

    def _survey(self, ctx: SensingContext, level: ObservationLevel) -> list[TwinSample]:
        self.twin.fields.local_grid.check_frame(ctx.true_pose.frame_id)
        o = self.twin.cfg.observation
        rng_m = float(ctx.sensor.parameters.get("footprint_radius_m", o.survey_range_m))
        scale = float(ctx.degradation.get("noise_scale", 1.0))
        pos = np.asarray(ctx.true_pose.position_m, dtype=np.float64)
        r_world_from_sensor = quat_to_matrix(ctx.true_pose.orientation_wxyz)
        out = []
        for eid in sorted(self.twin.entities, key=str):
            e = self.twin.entities[eid]
            if e.kind not in _KINDS:
                continue
            d, vis = self._visibility(e, pos)
            if d - e.radius_m > rng_m:
                continue
            u = float(self.rng.uniform())
            if e.kind == KIND_MOBILE and not e.presence:
                continue  # absent groups are not seen; false positives are not modelled
            if u >= o.detection_p0 * vis:
                continue  # missed detection (partial survey)
            offset = r_world_from_sensor.T @ (
                e.position_m - pos
            ) + o.range_sigma_m * scale * self.rng.standard_normal(3)
            noise = o.cover_sigma * scale / max(vis, 0.05)
            if level is ObservationLevel.E0_ABSTRACT:
                if e.kind == KIND_MOBILE:
                    vals, units, ch = [1.0, *offset], "detection[1];offset_m[m,m,m]", "mobile_group_detection"
                else:
                    est = min(1.0, max(0.0, e.cover + noise * float(self.rng.standard_normal())))
                    vals, units, ch = [est, *offset], "cover_fraction[1];offset_m[m,m,m]", "cover_estimate"
            else:
                vals, units, ch = (
                    [*self._features(e, noise), *offset],
                    "feature[1];offset_m[m,m,m]",
                    "ecological_feature",
                )
            obs = self._obs(
                ctx,
                vals,
                units,
                {
                    "kind": "ecological_survey",
                    "level": level.value,
                    "channel": ch,
                    "source_modality": ctx.sensor.modality,
                    "footprint_radius_m": rng_m,
                },
            )
            sessile = e.kind in SESSILE
            sup = SupervisionLabel(
                observation_id=obs.observation_id,
                true_world_entity_id=eid,
                domain=Domain.ECOLOGICAL,
                targets={
                    "entity_kind": e.kind,
                    "cover": e.cover if sessile else None,
                    "condition": e.condition if sessile else None,
                    "presence": e.presence if e.kind == KIND_MOBILE else None,
                },
                target_masks={"cover": sessile, "condition": sessile, "presence": e.kind == KIND_MOBILE},
                lineage=self._lineage(),
            )
            out.append(TwinSample(observation=obs, supervision=sup))
        return out

    def _features(self, e: EntityState, noise: float) -> list[float]:
        env = self.twin.local_env(e)
        onehot = [1.0 if e.kind == k else 0.0 for k in _KINDS]
        x = np.array(
            [
                e.cover,
                e.condition,
                float(e.presence),
                *onehot,
                env["temperature"] / 30.0,
                env["turbidity"] / 10.0,
                env["light"] / 1000.0,
                env["current_speed"],
            ]
        )
        f = self.projection @ x + noise * self.rng.standard_normal(self.projection.shape[0])
        return [float(v) for v in f]
