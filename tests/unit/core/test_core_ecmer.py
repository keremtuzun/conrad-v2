"""ECMER: encoders, fusion with missing modalities, quality head, Observation -> Evidence service."""

import hashlib
import math

import cv2
import numpy as np
import pytest
import torch

from conrad.core.ecmer import (
    MODALITIES,
    EcmerEncoder,
    EcmerModel,
    EventBatch,
    ReprObjective,
    build_fusion_baseline,
    image_quality,
    info_nce,
    sample_modality_dropout,
    sonar_quality,
    stage_e1_representation,
    stage_e2_degradation,
    stage_e3_quality,
    stage_e4_correspondence,
)
from conrad.core.ecmer.encoders import ResNet18Encoder, ScalarSensorEncoder, VoxelPointEncoder
from conrad.domains.technical.spatial_local import LocalCondition, LocalThresholds, SpatialModel2T
from conrad.schemas.capsule_surface import CapsuleSurfaceGrid, SurfaceRect
from conrad.schemas.frames import Pose
from conrad.schemas.observation import (
    EntityCandidate,
    EvidenceValidity,
    Modality,
    Observation,
    PayloadRef,
    SensorHealth,
)
from conrad.schemas.provenance import SourceType
from conrad.schemas.structural_sensor import StructuralSensorModel
from conrad.schemas.structural_support import CapsuleSurfaceSupport, ParameterAuthority
from conrad.schemas.timebase import stamp
from conrad.twins.twin2t.spatial_emission import emit_spatial_observation
from conrad.twins.twin2t.spatial_field import LocalStructuralState, SpatialStructuralTruth


def _present(value: float | None) -> float:
    assert value is not None, "quality feature was not computed"
    return value


def _batch(cfg, b=3, avail=None):
    s = cfg.ecmer.image_size
    avail = avail or {n: torch.ones(b, dtype=torch.bool) for n in MODALITIES}
    return EventBatch(
        batch_size=b,
        context_raw=torch.randn(b, 14),
        sensor_idx=torch.zeros(b, dtype=torch.long),
        frame_idx=torch.zeros(b, dtype=torch.long),
        raw_quality=torch.rand(b, cfg.ecmer.quality_feature_dim),
        rgb=torch.rand(b, 3, s, s),
        sonar=torch.rand(b, 1, s, s),
        points=torch.randn(b, 20, 3),
        points_mask=torch.ones(b, 20, dtype=torch.bool),
        scalars=torch.randn(b, 3),
        scalar_present=torch.tensor([[1, 0, 1]] * b, dtype=torch.bool),
        avail=avail,
    )


def test_encoder_token_shapes(cfg):
    img = ResNet18Encoder(cfg, 3)(torch.rand(2, 3, 32, 32))
    assert img.shape == (2, 1 + cfg.ecmer.patch_grid**2, cfg.evidence_dim)
    assert VoxelPointEncoder(cfg)(torch.randn(2, 10, 3), torch.ones(2, 10, dtype=torch.bool)).shape == (
        2,
        1,
        cfg.evidence_dim,
    )
    assert ScalarSensorEncoder(cfg)(torch.randn(2, 3), torch.ones(2, 3)).shape == (2, 1, cfg.evidence_dim)


def test_rgb_and_sonar_encoders_do_not_share_weights(cfg):
    m = EcmerModel(cfg)
    assert m.rgb.stem[0].weight.data_ptr() != m.sonar.stem[0].weight.data_ptr()
    assert m.rgb.stem[0].in_channels == 3 and m.sonar.stem[0].in_channels == 1


def test_fusion_quality_head_ranges_and_backward(cfg):
    out = EcmerModel(cfg)(_batch(cfg))
    assert out.event.shape == (3, cfg.evidence_dim)
    assert ((out.reliability >= 0) & (out.reliability <= 1)).all()
    assert (out.log_var_a >= cfg.ecmer.log_var_min).all() and (out.log_var_a <= cfg.ecmer.log_var_max).all()
    (out.event.sum() + out.reliability.sum() + out.ood_logit.sum()).backward()


def test_missing_modality_is_normal_input(cfg):
    m = EcmerModel(cfg).eval()
    avail = {n: torch.tensor([True, False]) for n in MODALITIES}
    avail["scalar"] = torch.tensor([True, True])
    out = m(_batch(cfg, b=2, avail=avail))
    assert torch.isfinite(out.event).all()
    assert out.availability[1].tolist() == [0.0, 0.0, 0.0, 1.0]
    none = {n: torch.zeros(1, dtype=torch.bool) for n in MODALITIES}
    assert torch.isfinite(m(_batch(cfg, b=1, avail=none)).event).all()  # [EVENT] alone is still valid


def test_modality_dropout_never_drops_everything():
    avail = torch.ones(50, 4, dtype=torch.bool)
    drop = sample_modality_dropout(avail, 0.99, torch.Generator().manual_seed(0))
    assert ((avail & ~drop).sum(-1) >= 1).all()


@pytest.mark.parametrize("flag", ["use_reliability", "use_modality_id", "use_alignment", "use_disagreement"])
def test_ablation_switches(cfg, flag):
    c = cfg.model_copy(update={"ecmer": cfg.ecmer.model_copy(update={flag: False})})
    out = EcmerModel(c)(_batch(c))
    if flag == "use_reliability":
        assert torch.equal(out.reliability, torch.ones(3))
    assert torch.isfinite(out.event).all()


@pytest.mark.parametrize("name", ["single_rgb", "concat", "early", "late", "cross_attention"])
def test_fusion_baselines(cfg, name):
    out = EcmerModel(cfg)(_batch(cfg))
    fused = build_fusion_baseline(cfg, name)(out.modality_tokens, out.availability)
    assert fused.shape == (3, cfg.evidence_dim)


def test_handcrafted_quality_features_respond():
    rng = np.random.default_rng(0)
    sharp = (rng.random((64, 64)) > 0.5).astype(np.float32)
    blurred = cv2.GaussianBlur(sharp, (9, 9), 3)
    assert _present(image_quality(blurred).blur) > _present(image_quality(sharp).blur)
    assert _present(image_quality(np.full((8, 8), 0.9)).brightness) > _present(
        image_quality(np.full((8, 8), 0.1)).brightness
    )
    clean = np.vstack([np.zeros((4, 32)) + rng.normal(0, 0.01, (4, 32)), np.ones((28, 32))])
    noisy = clean + rng.normal(0, 0.5, clean.shape)
    assert _present(sonar_quality(clean).snr_db) > _present(sonar_quality(noisy).snr_db)
    assert image_quality(np.full((4, 4), np.nan)).valid_fraction == 0.0


def test_ssl_objective_and_stages_run(cfg):
    model, obj = EcmerModel(cfg), ReprObjective(cfg)
    loss, den = info_nce(torch.randn(4, 8), torch.randn(4, 8), 0.1)
    assert torch.isfinite(loss) and float(den) == 4.0
    fn = lambda step: _batch(cfg, b=4)  # noqa: E731
    for hist in (
        stage_e1_representation(model, obj, fn, 1, 0),
        stage_e2_degradation(model, fn, 1, 0),
        stage_e3_quality(model, fn, 1, 0),
        stage_e4_correspondence(model, obj, fn, 1, 0),
    ):
        assert len(hist) == 1 and np.isfinite(hist[0])
    assert not any(
        p is q for p in obj.projection.parameters() for q in model.parameters()
    )  # head is discardable


def test_service_observation_to_evidence(cfg, ids):
    store = {}

    def ref(arr):
        data = arr.tobytes()
        d = hashlib.sha256(data).hexdigest()
        store[d] = arr
        return PayloadRef(
            uri=f"mem://{d}",
            digest=d,
            media_type="application/x-npy",
            shape=arr.shape,
            dtype=str(arr.dtype),
            byte_length=len(data),
        )

    def obs(mod, **kw):
        return Observation(
            observation_id=ids.new(),
            mission_id=ids.new(),
            run_id=ids.new(),
            trace_id=ids.new(),
            sensor_id=ids.new(),
            modality=mod,
            timestamp=stamp(2.0, "c"),
            sensor_frame="cam",
            robot_pose_estimate=Pose(frame_id="WORLD", position_m=(1.0, 2.0, 3.0)),
            **kw,
        )

    rng = np.random.default_rng(0)
    observations = [
        obs(Modality.RGB, payload_ref=ref(rng.random((40, 40, 3)))),
        obs(Modality.SONAR, payload_ref=ref(rng.random((20, 30))), sensor_health=SensorHealth.FAULT),
        obs(
            Modality.ENVIRONMENTAL,
            inline_values=(12.0, 35.0),
            inline_units="SI",
            sensor_context={"measurement_names": ["temperature_c", "salinity_psu"]},
        ),
    ]
    enc = EcmerEncoder(cfg, ids, loader=lambda r: store[r.digest])
    out = enc.encode(observations)
    assert [o.evidence.modality for o in out] == [Modality.RGB, Modality.SONAR, Modality.ENVIRONMENTAL]
    for o, src in zip(out, observations, strict=True):
        ev = o.evidence
        assert ev.source_observation_id == src.observation_id and ev.timestamp == src.timestamp
        assert (
            o.provenance.source_type is SourceType.DIRECT_OBSERVATION
            and ev.provenance_id == o.provenance.record_id
        )
        assert len(ev.embedding) == cfg.evidence_dim and 0 <= ev.reliability <= 1
        assert ev.independence_group == str(src.observation_id)
        assert ev.spatial_support is not None
        assert ev.spatial_support.frame_id == "WORLD"
    assert out[1].evidence.validity is EvidenceValidity.INVALID
    assert out[2].evidence.measurements == {"temperature_c": 12.0, "salinity_psu": 35.0}
    assert out[0].evidence.measurements == {}  # nothing is invented for image payloads
    with pytest.raises(ValueError):
        EcmerEncoder(cfg, ids).encode(observations[:1])  # payload_ref without an injected loader


def test_spatial_truth_observation_evidence_local_belief(cfg, ids):
    grid = CapsuleSurfaceGrid(1.0, 1.0, 1, 1)
    sensor = StructuralSensorModel(
        footprint_width_m=1.0,
        footprint_height_m=2 * math.pi,
        minimum_resolvable_corrosion_m=0.1,
        minimum_resolvable_crack_m=0.1,
        range_min_m=0.3,
        range_max_m=4.0,
        noise_sigma_m=0,
        position_uncertainty_m=0,
        orientation_uncertainty_rad=0,
        footprint_uncertainty_m=0,
        aggregation_kernel="LOCAL_MAX",
        authority=ParameterAuthority.ENGINEERING_ESTIMATE,
    )
    truth = SpatialStructuralTruth(grid, (LocalStructuralState(),))
    rect = SurfaceRect(0, 1, 0, 2 * math.pi)
    support = CapsuleSurfaceSupport(
        sensor_model_version=sensor.version,
        sensor_config_digest=sensor.digest,
        frame_id="CAPSULE_DESIGN",
        axial_start_m=rect.x0,
        axial_end_m=rect.x1,
        angle_start_rad=rect.a0,
        angle_end_rad=rect.a1,
        axial_uncertainty_m=0,
        angular_uncertainty_rad=0,
        aggregation_kernel="LOCAL_MAX",
    )
    observation = emit_spatial_observation(
        truth,
        sensor,
        support,
        ids=ids,
        rng=np.random.default_rng(1),
        mission_id=ids.new(),
        run_id=ids.new(),
        trace_id=ids.new(),
        sensor_id=ids.new(),
        sensor_frame="SENSOR",
        timestamp=stamp(2.0, "c"),
        estimated_pose=Pose(frame_id="WORLD", position_m=(1, 2, 3)),
        measured_range_m=1,
        measured_bearing_rad=0,
        measured_elevation_rad=0,
        range_sigma_m=0,
        angle_sigma_rad=0,
        independence_group="capture-1",
    )
    assert observation.structural_support == support
    assert "world_id" not in observation.canonical_json()
    ev = EcmerEncoder(cfg, ids).encode([observation])[0].evidence
    assert ev.structural_support == support
    assert ev.measurements == {
        "apparent_wall_loss": 0.0,
        "crack_indication_length": 0.0,
        "crack_indication_depth": 0.0,
    }
    assert ev.independence_group == "capture-1"
    thresholds = LocalThresholds(0.002, 0.006, 0.012, 0.003, 0.01, 0.03)
    rid = ids.new()
    belief = SpatialModel2T(grid, sensor, thresholds, rid)
    associated = ev.model_copy(
        update={"entity_candidates": (EntityCandidate(registry_entity_id=rid, score=1),)}
    )
    assert belief.ingest(associated)
    assert belief.condition() is LocalCondition.OBSERVED_INTACT
