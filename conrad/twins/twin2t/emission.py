"""Build contract-valid Observation + separate SupervisionLabel for one visible component.

The Observation holds only sensor-shaped values (noisy measurements, units, measurement names, fidelity);
no world-entity id and no exact truth value. Truth goes to ``TwinSample.supervision`` only.

TRUTH PLANE. implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from typing import Any

import numpy as np

from conrad.persistence.object_store import ObjectStore
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Modality, Observation
from conrad.schemas.truth import SupervisionLabel
from conrad.schemas.world import Domain
from conrad.twins.base import SensingContext, TwinSample
from conrad.twins.twin2t.observation import (
    FidelityLevel,
    ObservationModel,
    SensingQuality,
    SensorLevelHookMissing,
    SensorLevelRenderer,
    surface_appearance_of,
)
from conrad.twins.twin2t.state import STATE_DIMENSIONS, ComponentRuntime, status_of


def emit_sample(
    *,
    ids: IdFactory,
    store: ObjectStore,
    model: ObservationModel,
    rng: np.random.Generator,
    renderers: dict[Modality, SensorLevelRenderer],
    rt: ComponentRuntime,
    ctx: SensingContext,
    quality: SensingQuality,
    fidelity: FidelityLevel,
    visibility: float,
    lineage: str,
) -> TwinSample:
    oid = ids.new()
    common: dict[str, Any] = {
        "observation_id": oid,
        "mission_id": ctx.mission_id,
        "run_id": ctx.run_id,
        "trace_id": ctx.trace_id,
        "sensor_id": ctx.sensor.sensor_id,
        "timestamp": ctx.timestamp,
        "sensor_frame": ctx.sensor.frame_id,
        "robot_pose_estimate": ctx.estimated_pose,
        "calibration_ref": ctx.sensor.calibration_ref,
    }
    gain = model.noise_gain(quality)
    if fidelity is FidelityLevel.T2:
        modality = Modality(ctx.sensor.modality)
        renderer = renderers.get(modality)
        if renderer is None:
            raise SensorLevelHookMissing(f"no sensor-level renderer registered for {modality.value}")
        ref = store.put_array(np.asarray(renderer(surface_appearance_of(rt), rng)))
        obs = Observation(
            **common,
            modality=modality,
            payload_ref=ref,
            sensor_health=model.health(quality, gain),
            sensor_context={"twin2t_fidelity": fidelity.value},
        )
    else:
        draft = model.t0(rt, quality, rng) if fidelity is FidelityLevel.T0 else model.t1(rt, quality, rng)
        obs = Observation(
            **common,
            modality=Modality.STRUCTURED,
            inline_values=draft.values,
            inline_units=",".join(draft.units),
            sensor_health=draft.health,
            sensor_context={
                "twin2t_fidelity": fidelity.value,
                "measurements": list(draft.names),
                "units": list(draft.units),
            },
        )
    s = rt.state
    targets: dict[str, Any] = {d: getattr(s, d) for d in STATE_DIMENSIONS}
    targets.update(
        status=status_of(s, rt.effective_wall_m).value,
        fidelity=fidelity.value,
        visibility=visibility,
        noise_gain=gain,
        contradiction_injected=quality.contradiction != 0.0,
        contradiction=quality.contradiction,
        sensor_bias_m=quality.sensor_bias_m,
    )
    label = SupervisionLabel(
        observation_id=oid,
        true_world_entity_id=rt.component.entity_id,
        domain=Domain.TECHNICAL,
        targets=targets,
        target_masks=dict(s.validity),
        lineage=lineage,
    )
    return TwinSample(observation=obs, supervision=label)
