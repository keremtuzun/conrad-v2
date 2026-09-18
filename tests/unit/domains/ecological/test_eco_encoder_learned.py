from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from conrad.domains.ecological import (
    BASELINES,
    EcologicalEncoder,
    Model2E,
    Model2EConfig,
    UnitError,
    evidence_kind,
    make_baseline,
)
from conrad.domains.ecological.learned import (
    LearnedCEFD,
    cefd_loss,
    run_forward,
    synthetic_batch,
    tiny_learned_config,
    train_learned_cefd,
)
from conrad.domains.ecological.units import split_compound
from conrad.schemas.frames import Pose
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Modality, Observation
from conrad.schemas.timebase import stamp

PKG = Path(__file__).resolve().parents[4] / "conrad" / "domains" / "ecological"
FORBIDDEN = ("conrad.twins", "conrad.schemas.truth", "conrad.sim", "conrad.evaluation", "conrad.training")


def _obs(ids, modality, values, units, ctx, pose=(1.0, 2.0, -5.0)):
    return Observation(
        observation_id=ids.new(),
        mission_id=ids.new(),
        run_id=ids.new(),
        trace_id=ids.new(),
        sensor_id=ids.new(),
        modality=modality,
        timestamp=stamp(12.5, "SIM"),
        sensor_frame="sensor",
        robot_pose_estimate=None if pose is None else Pose(frame_id="WORLD", position_m=pose),
        inline_values=values,
        inline_units=units,
        sensor_context=ctx,
    )


def test_no_truth_plane_imports_in_belief_package():
    offenders = []
    for path in PKG.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            offenders += [(path.name, n) for n in names if n.startswith(FORBIDDEN)]
    assert offenders == []


def test_encoder_keeps_units_timestamp_and_position():
    ids = IdFactory(1).child("enc")
    enc = EcologicalEncoder(Model2EConfig(), ids)
    obs = _obs(
        ids, Modality.ENVIRONMENTAL, (14.2,), "degC", {"kind": "field_sample", "channel": "temperature"}
    )
    ev, prov = enc.encode(obs, created_time_ns=obs.timestamp.time_ns + 5)
    assert ev.timestamp == obs.timestamp and ev.measurements == {"temperature": 14.2}
    assert ev.measurement_units == {"temperature": "degC"}
    assert ev.spatial_support.center_m == (1.0, 2.0, -5.0) and prov.source_ids == (obs.observation_id,)
    assert evidence_kind(ev, Model2EConfig()) == "FIELD"
    cur = _obs(
        ids, Modality.ENVIRONMENTAL, (0.1, -0.2, 0.0), "m s-1", {"kind": "field_sample", "channel": "current"}
    )
    assert set(enc.encode(cur, 0)[0].measurements) == {"current.0", "current.1", "current.2"}
    bad = _obs(ids, Modality.ENVIRONMENTAL, (14.2,), "K", {"kind": "field_sample", "channel": "temperature"})
    with pytest.raises(UnitError):
        enc.encode(bad, 0)
    survey = _obs(
        ids,
        Modality.STRUCTURED,
        (0.4, 1.0, 0.0, -2.0),
        "cover_fraction[1];offset_m[m,m,m]",
        {"kind": "ecological_survey", "channel": "cover_estimate"},
    )
    sev, _ = enc.encode(survey, 0)
    assert sev.measurements == {"cover_fraction": 0.4}
    assert sev.spatial_support.center_m == (2.0, 2.0, -7.0)
    assert sev.sensor_context.range_m == pytest.approx(5**0.5)
    blind = _obs(
        ids,
        Modality.STRUCTURED,
        (1.0, 0.0, 0.0, 1.0),
        "detection[1];offset_m[m,m,m]",
        {"kind": "ecological_survey", "channel": "mobile_group_detection"},
        pose=None,
    )
    bev, _ = enc.encode(blind, 0)
    assert bev.spatial_support is None and bev.validity.value == "DEGRADED"


def test_compound_units_parse_and_fail_closed():
    parts = split_compound((0.1, 0.2, 0.3, 1.0, 2.0, 3.0), "feature[1];offset_m[m,m,m]")
    assert parts["feature"][0] == (0.1, 0.2, 0.3) and parts["offset_m"][1] == ("m", "m", "m")
    with pytest.raises(UnitError):
        split_compound((0.1, 1.0), "cover_fraction[1];offset_m[m,m,m]")
    with pytest.raises(UnitError):
        split_compound((0.1,), "cover_fraction")


def test_all_baselines_share_one_interface():
    for name in BASELINES:
        m = make_baseline(name, IdFactory(2).child(name))
        assert isinstance(m, Model2E)
        m.initialize({"clock_domain": "SIM"})
        assert m.availability().value == "AVAILABLE"
    with pytest.raises(KeyError):
        make_baseline("magic", IdFactory(2))


def test_learned_cefd_forward_backward_and_training():
    torch.manual_seed(0)
    cfg = tiny_learned_config()
    model = LearnedCEFD(cfg)
    g = torch.Generator().manual_seed(4)
    batch = synthetic_batch(cfg, g)
    out = run_forward(model, batch)
    assert out.entity_out.shape == (2, 3, cfg.entity_heads_out)
    assert out.field_out.shape == (2, cfg.field_out_channels, 4, 4, 4)
    assert out.gate.shape == (2, 3, 1) and bool(((out.gate >= 0) & (out.gate <= 1)).all())
    losses = cefd_loss(out, batch, cfg)
    losses["total"].backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(gr is not None and torch.isfinite(gr).all() for gr in grads)
    assert model.gate_mlp[0].weight.grad.abs().sum() > 0  # the coupling gate learns
    hist = train_learned_cefd(model, lambda gen: synthetic_batch(cfg, gen), steps=15, generator=g)
    assert len(hist) == 15 and all(torch.isfinite(torch.tensor(hist)))
    assert sum(hist[-5:]) < sum(hist[:5])


def test_learned_cefd_padded_entities_do_not_couple():
    cfg = tiny_learned_config()
    model = LearnedCEFD(cfg).eval()
    batch = synthetic_batch(cfg, torch.Generator().manual_seed(1))
    batch.entity_mask[:, -1] = False
    out = run_forward(model, batch)
    assert torch.all(out.gate[:, -1] == 0)
