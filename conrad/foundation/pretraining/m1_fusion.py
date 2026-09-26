"""OS-FM M1 multimodal fusion smoke training path."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from conrad.foundation.data.pairing import PairEdge, PairGraph, PairRelation
from conrad.foundation.encoders.geometry import GeometryEncoderConfig, GeometryGroupedEncoder
from conrad.foundation.encoders.range import RangeEncoderConfig, RangeViTP8Encoder
from conrad.foundation.encoders.rgb import RGBEncoderConfig, RGBViTS14Encoder
from conrad.foundation.encoders.sonar import SonarEncoderConfig, SonarViTS14Encoder
from conrad.foundation.fusion.scene_fusion import ModalityTokenSet, SceneFusionConfig, SceneFusionOutput, SceneFusionTransformer
from conrad.foundation.pretraining.ema import EMASchedule, update_ema_teacher
from conrad.foundation.pretraining.losses import ObjectiveResult, ObjectiveRouter, ObjectiveStatus
from conrad.foundation.pretraining.smoke import split_hash_for_plan
from conrad.foundation.pretraining.u1_geometry import synthetic_geometry_views
from conrad.foundation.pretraining.u1_range import synthetic_range_views
from conrad.foundation.pretraining.u1_rgb import RepresentationHealth, parameter_count, synthetic_rgb_views
from conrad.foundation.pretraining.u1_rgb import representation_health as _representation_health
from conrad.foundation.pretraining.u1_sonar import synthetic_sonar_views
from conrad.settings import REPO_ROOT
from conrad.training.checkpoint import load_checkpoint, save_checkpoint
from conrad.training.checkpoint_meta import CompatibilityTuple, config_digest
from conrad.training.run_dir import RunDirectory

MODALITIES = ("rgb", "sonar", "range", "geometry")
EXPECTED_PARENT_DIGESTS = {
    "rgb": "7909a94586f244b9e7ec6f71cbbf274c047b979112d41f8347b7cb8250cb9fe1",
    "sonar": "9e7e68cbdcd63d9816323c2b2925862ecc2e142edd378070211df2f3e70955a0",
    "range": "c797695601b8ee5875165d9c3c9d7da4059726c3d300b50890cdbedd658e9256",
    "geometry": "cf82dede076700d68ece32be888234dc34ef008f7d18e6cd22adc1ce99cc4583",
}


@dataclass(frozen=True)
class M1Fixture:
    token_sets: tuple[ModalityTokenSet, ...]
    pair_graphs: tuple[PairGraph, ...]
    sample_ids: tuple[str, ...]
    relation_matrix: tuple[tuple[PairRelation, ...], ...]
    metric_capable: torch.Tensor
    geometry_capable: torch.Tensor
    artificial_dropout_plan: dict[str, list[bool]]


@dataclass(frozen=True)
class M1FusionStepOutput:
    loss: torch.Tensor
    results: tuple[ObjectiveResult, ...]
    student: SceneFusionOutput
    teacher: SceneFusionOutput
    health: RepresentationHealth


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def parent_checkpoint_status() -> dict[str, dict[str, str | bool | None]]:
    paths = {
        "rgb": REPO_ROOT / "artifacts/runs/train-osfm_u1_rgb_smoke-1790410824000679000/checkpoints/osfm_u1_rgb_smoke.pt",
        "sonar": REPO_ROOT / "artifacts/runs/train-osfm_u1_sonar_smoke-1790411261663354000/checkpoints/osfm_u1_sonar_smoke.pt",
        "range": REPO_ROOT / "artifacts/runs/train-osfm_u1_range_smoke-1790411749668941000/checkpoints/osfm_u1_range_smoke.pt",
        "geometry": REPO_ROOT / "artifacts/runs/train-osfm_u1_geometry_smoke-1790412183310131000/checkpoints/osfm_u1_geometry_smoke.pt",
    }
    status: dict[str, dict[str, str | bool | None]] = {}
    for modality, path in paths.items():
        if not path.is_file():
            status[modality] = {
                "path": str(path),
                "expected_digest": EXPECTED_PARENT_DIGESTS[modality],
                "actual_digest": None,
                "available": False,
                "digest_matches": False,
                "load_status": "UNAVAILABLE_FOR_LOAD",
            }
            continue
        digest = _sha256_file(path)
        status[modality] = {
            "path": str(path),
            "expected_digest": EXPECTED_PARENT_DIGESTS[modality],
            "actual_digest": digest,
            "available": True,
            "digest_matches": digest == EXPECTED_PARENT_DIGESTS[modality],
            "load_status": "VERIFIED_AVAILABLE_NOT_LOADED_INTO_M1_SMOKE",
        }
    return status


def synthetic_m1_fixture(job: dict[str, Any], seed: int) -> M1Fixture:
    torch.manual_seed(seed)
    batch = int(job.get("batch_size", 4))
    if batch != 4:
        raise ValueError("M1 smoke fixture is fixed at batch_size=4 to cover pair/missingness cases")
    rgb = synthetic_rgb_views({"batch_size": batch, "image_size": 28}, seed)
    sonar = synthetic_sonar_views({"batch_size": batch, "image_size": 28}, seed + 1)
    range_views = synthetic_range_views({"batch_size": batch, "image_size": 32}, seed + 2)
    geometry = synthetic_geometry_views(
        {
            "batch_size": batch,
            "points_per_sample": int(job.get("points_per_sample", 64)),
            "num_groups": int(job.get("num_groups", 4)),
            "group_size": int(job.get("group_size", 8)),
        },
        seed + 3,
    )
    encoders = {
        "rgb": RGBViTS14Encoder(RGBEncoderConfig(image_size=28, depth=int(job.get("encoder_depth", 1)))),
        "sonar": SonarViTS14Encoder(SonarEncoderConfig(image_size=28, depth=int(job.get("encoder_depth", 1)))),
        "range": RangeViTP8Encoder(RangeEncoderConfig(image_size=32, depth=int(job.get("encoder_depth", 1)))),
        "geometry": GeometryGroupedEncoder(
            GeometryEncoderConfig(
                num_groups=int(job.get("num_groups", 4)),
                group_size=int(job.get("group_size", 8)),
            )
        ),
    }
    for encoder in encoders.values():
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad_(False)
    with torch.no_grad():
        rgb_out = encoders["rgb"](rgb.teacher_rgb)
        sonar_out = encoders["sonar"](sonar.teacher_sonar)
        range_out = encoders["range"](range_views.teacher_range, range_views.teacher_validity_mask)
        geo_out = encoders["geometry"](geometry.teacher_points, geometry.teacher_validity_mask)

    natural = {
        "rgb": torch.tensor([False, False, True, False]),
        "sonar": torch.tensor([False, False, False, True]),
        "range": torch.tensor([False, True, True, False]),
        "geometry": torch.tensor([False, False, True, True]),
    }
    artificial = {
        "rgb": torch.tensor([False, False, False, False]),
        "sonar": torch.tensor([False, True, False, False]),
        "range": torch.tensor([False, False, False, False]),
        "geometry": torch.tensor([False, False, False, False]),
    }
    padded = {m: torch.zeros(batch, dtype=torch.bool) for m in MODALITIES}
    token_sets = (
        ModalityTokenSet("rgb", rgb_out.patch_tokens, rgb_out.visible_mask, natural["rgb"], artificial["rgb"], padded["rgb"]),
        ModalityTokenSet(
            "sonar",
            sonar_out.patch_tokens,
            sonar_out.visible_mask,
            natural["sonar"],
            artificial["sonar"],
            padded["sonar"],
        ),
        ModalityTokenSet(
            "range",
            range_out.patch_tokens,
            range_out.visible_mask,
            natural["range"],
            artificial["range"],
            padded["range"],
        ),
        ModalityTokenSet(
            "geometry",
            geo_out.group_tokens,
            geo_out.visible_mask,
            natural["geometry"],
            artificial["geometry"],
            padded["geometry"],
        ),
    )
    pair_graphs = (
        PairGraph(
            nodes=("rgb", "sonar", "range", "geometry"),
            edges=(
                PairEdge("rgb", "sonar", PairRelation.SAME_WINDOW, 4),
                PairEdge("rgb", "range", PairRelation.SAME_WINDOW, 7),
                PairEdge("range", "geometry", PairRelation.SAME_WINDOW, 3),
            ),
        ),
        PairGraph(nodes=("rgb", "sonar", "geometry"), edges=(PairEdge("rgb", "geometry", PairRelation.SAME_WINDOW, 8),)),
        PairGraph(nodes=("sonar",), edges=()),
        PairGraph(nodes=("rgb", "range"), edges=(PairEdge("rgb", "range", PairRelation.UNPAIRED, 0),)),
    )
    relation_rows: list[tuple[PairRelation, ...]] = []
    for graph in pair_graphs:
        relation_rows.append(tuple(graph.relation(a, b) for a, b in (("rgb", "sonar"), ("rgb", "range"), ("range", "geometry"))))
    return M1Fixture(
        token_sets=token_sets,
        pair_graphs=pair_graphs,
        sample_ids=tuple(f"m1-smoke-{idx}" for idx in range(batch)),
        relation_matrix=tuple(relation_rows),
        metric_capable=torch.tensor([True, False, False, True]),
        geometry_capable=torch.tensor([True, True, False, False]),
        artificial_dropout_plan={k: v.tolist() for k, v in artificial.items()},
    )


class M1FusionTrainingBundle(nn.Module):
    def __init__(self, fusion: SceneFusionTransformer, *, ema_schedule: EMASchedule | None = None) -> None:
        super().__init__()
        self.fusion = fusion
        self.teacher = copy.deepcopy(fusion)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad_(False)
        d_f = fusion.config.d_f
        self.masked_latent_head = nn.Linear(d_f, d_f)
        self.global_projector = nn.Sequential(nn.LayerNorm(d_f), nn.Linear(d_f, d_f))
        self.modality_decoders = nn.ModuleDict({m: nn.Linear(d_f, d_f) for m in MODALITIES})
        self.degradation_head = nn.Linear(d_f, 2)
        self.geometry_head = nn.Linear(d_f, d_f)
        self.metric_head = nn.Linear(d_f, d_f)
        self.router = ObjectiveRouter(
            {
                "m1_masked_latent_prediction": 1.0,
                "m1_global_consistency": 0.5,
                "m1_cross_modal_consistency": 0.5,
                "m1_missing_modality": 0.5,
                "m1_degradation": 0.25,
                "m1_geometry_consistency": 0.25,
                "m1_metric_reconstruction": 0.25,
            }
        )
        self.ema_schedule = ema_schedule or EMASchedule(start=0.996, end=0.9999, total_steps=100)

    def _eligible_pair_mask(self, fixture: M1Fixture, output: SceneFusionOutput, pairs: tuple[tuple[str, str], ...]) -> torch.Tensor:
        idx = {name: pos for pos, name in enumerate(output.modality_names)}
        eligible = torch.zeros(output.global_repr.shape[0], dtype=torch.bool, device=output.global_repr.device)
        for row, graph in enumerate(fixture.pair_graphs):
            for left, right in pairs:
                if left not in idx or right not in idx:
                    continue
                present = bool(output.modality_presence[row, idx[left]] and output.modality_presence[row, idx[right]])
                if present and graph.relation(left, right) is PairRelation.SAME_WINDOW:
                    eligible[row] = True
        return eligible

    def forward(self, fixture: M1Fixture) -> M1FusionStepOutput:
        student = self.fusion(fixture.token_sets)
        with torch.no_grad():
            teacher = self.teacher(fixture.token_sets)
        device = student.global_repr.device
        latent_target = teacher.scene_latents.roll(shifts=1, dims=1).detach()
        latent_mask = torch.zeros(student.scene_latents.shape[:2], dtype=torch.bool, device=device)
        latent_mask[:, ::3] = True
        mask_loss = self.router._result(
            "m1_masked_latent_prediction",
            (F.mse_loss(self.masked_latent_head(student.scene_latents), latent_target, reduction="none").mean(dim=-1) * latent_mask).sum(),
            latent_mask.float().sum(),
        )
        global_loss = self.router._result(
            "m1_global_consistency",
            F.mse_loss(self.global_projector(student.global_repr), teacher.global_repr.detach(), reduction="sum")
            / student.global_repr.shape[-1],
            torch.tensor(float(student.global_repr.shape[0]), device=device),
        )
        xm_mask = self._eligible_pair_mask(fixture, student, (("rgb", "sonar"), ("rgb", "range"), ("range", "geometry")))
        xm_loss = self.router._result(
            "m1_cross_modal_consistency",
            (student.global_repr[xm_mask] - teacher.global_repr[xm_mask].detach()).pow(2).sum() / student.global_repr.shape[-1]
            if bool(xm_mask.any())
            else student.global_repr.sum() * 0.0,
            xm_mask.float().sum(),
        )
        missing_rows = student.natural_missing_mask.any(dim=1) | student.artificial_dropout_mask.any(dim=1)
        decoded = torch.stack([head(student.global_repr) for head in self.modality_decoders.values()], dim=1)
        missing_targets = teacher.global_repr.detach().unsqueeze(1).expand_as(decoded)
        missing_slots = (student.natural_missing_mask | student.artificial_dropout_mask) & ~student.padding_mask
        missing_loss = self.router._result(
            "m1_missing_modality",
            (F.mse_loss(decoded, missing_targets, reduction="none").mean(dim=-1) * missing_slots).sum(),
            missing_slots.float().sum(),
        )
        logits = self.degradation_head(student.global_repr)
        degradation_loss = self.router._result(
            "m1_degradation",
            F.cross_entropy(logits, missing_rows.long(), reduction="sum"),
            torch.tensor(float(logits.shape[0]), device=device),
        )
        geo_mask = self._eligible_pair_mask(fixture, student, (("range", "geometry"),)) & fixture.geometry_capable.to(device)
        geo_loss = self.router._result(
            "m1_geometry_consistency",
            F.mse_loss(self.geometry_head(student.global_repr[geo_mask]), teacher.global_repr[geo_mask].detach(), reduction="sum")
            / student.global_repr.shape[-1]
            if bool(geo_mask.any())
            else student.global_repr.sum() * 0.0,
            geo_mask.float().sum(),
        )
        metric_mask = fixture.metric_capable.to(device) & student.modality_presence.any(dim=1)
        metric_loss = self.router._result(
            "m1_metric_reconstruction",
            F.mse_loss(self.metric_head(student.global_repr[metric_mask]), teacher.global_repr[metric_mask].detach(), reduction="sum")
            / student.global_repr.shape[-1]
            if bool(metric_mask.any())
            else student.global_repr.sum() * 0.0,
            metric_mask.float().sum(),
        )
        results = (mask_loss, global_loss, xm_loss, missing_loss, degradation_loss, geo_loss, metric_loss)
        return M1FusionStepOutput(
            loss=self.router.total(results),
            results=results,
            student=student,
            teacher=teacher,
            health=_representation_health(student.global_repr),
        )

    @torch.no_grad()
    def update_teacher_after_optimizer(self, step: int) -> float:
        momentum = self.ema_schedule.value(step)
        update_ema_teacher(self.fusion, self.teacher, momentum)
        self.teacher.eval()
        return momentum


def _objective_metrics(results: tuple[ObjectiveResult, ...]) -> dict[str, float | str]:
    metrics: dict[str, float | str] = {}
    for result in results:
        for key, value in result.as_metrics().items():
            metrics[key.replace("/", "_")] = value
    return metrics


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_m1_fusion_smoke(run: RunDirectory, job: dict[str, Any], seed: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    fixture = synthetic_m1_fixture(job, seed)
    fusion_config = SceneFusionConfig()
    bundle = M1FusionTrainingBundle(SceneFusionTransformer(fusion_config))
    optimizer = torch.optim.AdamW((p for p in bundle.parameters() if p.requires_grad), lr=float(job.get("lr", 2e-4)))
    bundle.train()
    out = bundle(fixture)
    out.loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    ema = bundle.update_teacher_after_optimizer(1)
    bundle.eval()
    with torch.no_grad():
        post = bundle(fixture)
    replay = synthetic_m1_fixture(job, seed)
    split_hash = split_hash_for_plan(
        {
            "sample_ids": fixture.sample_ids,
            "relations": [[r.value for r in row] for row in fixture.relation_matrix],
            "artificial_dropout": fixture.artificial_dropout_plan,
            "seed": seed,
        }
    )
    metrics: dict[str, float | str] = {
        "val/osfm_m1_fusion_loss": float(post.loss.detach().cpu()),
        "optimizer_steps": 1.0,
        "ema_updates": 1.0,
        "ema_momentum": float(ema),
        "scene_latents_finite": float(torch.isfinite(post.student.scene_latents).all().detach().cpu()),
        "global_repr_finite": float(torch.isfinite(post.student.global_repr).all().detach().cpu()),
        "representation_variance_mean": post.health.variance_mean,
        "representation_collapse_score": post.health.collapse_score,
        "representation_effective_rank": post.health.effective_rank,
        "formal_rank_guard_numeric": {"PASS": 1.0, "FAIL": 0.0, "NOT_EVALUABLE": -1.0}[post.health.formal_rank_guard],
        "fusion_parameter_count": float(parameter_count(bundle.fusion)),
        "m1a_trainable_parameter_count": float(trainable_parameter_count(bundle)),
        "memory_allocated_bytes": float(torch.cuda.memory_allocated() if torch.cuda.is_available() else 0),
    }
    metrics.update({k: v for k, v in _objective_metrics(post.results).items() if isinstance(v, float)})
    manifest_digest = hashlib.sha256(b"SYNTHETIC:osfm-m1-fusion-smoke:v1").hexdigest()
    ckpt = run.path / "checkpoints" / "osfm_m1_fusion_smoke.pt"
    meta = save_checkpoint(
        ckpt,
        model=bundle,
        component="foundation.osfm.m1_fusion",
        config=job,
        manifest_digests=(manifest_digest,),
        split_hash=split_hash,
        seeds={"torch": seed, "fixture": seed},
        metrics={k: float(v) for k, v in metrics.items() if isinstance(v, float)},
        epoch=0,
        step=1,
        created_time_ns=time.time_ns(),
        optimizer=optimizer,
        trainer_state={"cuda_rng_state": [], "amp_scaler": None, "sample_ids": fixture.sample_ids},
        is_encoder=True,
        representation_pretraining_id="OSFM-M1-FUSION-SMOKE-001",
        selection_metric="val/osfm_m1_fusion_loss",
        extra_compatibility={"osfm_stage": "M1-FUSION", "p4_smoke": "true"},
    )
    fresh = M1FusionTrainingBundle(SceneFusionTransformer(fusion_config))
    load_checkpoint(
        ckpt,
        expected=CompatibilityTuple(
            config_digest=config_digest(job),
            manifest_digests=(manifest_digest,),
            split_hash=split_hash,
            component="foundation.osfm.m1_fusion",
            representation_pretraining_id="OSFM-M1-FUSION-SMOKE-001",
            extra={"osfm_stage": "M1-FUSION", "p4_smoke": "true"},
        ),
        model=fresh,
    )
    fresh.eval()
    with torch.no_grad():
        reloaded = fresh(fixture)
    reload_max_abs_diff = float((post.student.global_repr - reloaded.student.global_repr).abs().max().detach().cpu())
    replay_same = (
        fixture.sample_ids == replay.sample_ids
        and fixture.artificial_dropout_plan == replay.artificial_dropout_plan
        and fixture.relation_matrix == replay.relation_matrix
    )
    wall_s = time.perf_counter() - t0
    result = {
        "component": "foundation.osfm.m1_fusion",
        "experiment_id": "OSFM-M1-FUSION-SMOKE-001",
        "architecture": {
            "fusion": "SceneFusionTransformer",
            "d_f": fusion_config.d_f,
            "scene_latents": [fusion_config.num_scene_latents, fusion_config.d_f],
            "cross_attention_blocks": fusion_config.cross_attention_blocks,
            "latent_self_attention_blocks": fusion_config.latent_self_attention_blocks,
            "heads": fusion_config.heads,
            "mlp_expansion": fusion_config.mlp_ratio,
            "outputs": {"scene_latents": list(post.student.scene_latents.shape), "global_repr": list(post.student.global_repr.shape)},
            "fusion_parameter_count": parameter_count(bundle.fusion),
            "m1a_trainable_parameter_count": trainable_parameter_count(bundle),
        },
        "parent_checkpoint_ancestry": parent_checkpoint_status(),
        "encoder_policy": {
            "phase": "M1-A",
            "modality_encoders_frozen": True,
            "trainable_components": "fusion/projections/objective_heads_only",
            "m1_b_implemented": False,
        },
        "fixture": {
            "sample_ids": list(fixture.sample_ids),
            "coverage": [
                "row0 all_four legitimate paired multimodal with range+geometry metric/geometric support",
                "row1 natural range missing plus artificial sonar dropout; rgb+geometry survives",
                "row2 sonar-only single-modality sample",
                "row3 rgb+range unpaired sample; geometry naturally missing",
            ],
            "relations": [[r.value for r in row] for row in fixture.relation_matrix],
            "modality_presence": post.student.modality_presence.to(torch.int).tolist(),
            "natural_missing": post.student.natural_missing_mask.to(torch.int).tolist(),
            "artificial_dropout": post.student.artificial_dropout_mask.to(torch.int).tolist(),
            "padding": post.student.padding_mask.to(torch.int).tolist(),
        },
        "checkpoint": str(ckpt),
        "checkpoint_id": meta.checkpoint_id,
        "split_hash": split_hash,
        "metrics": metrics,
        "objectives": {r.objective_id: r.as_metrics() for r in post.results},
        "ema": {"updates": 1, "momentum": ema, "teacher_eval": not bundle.teacher.training},
        "reload": {"max_abs_diff": reload_max_abs_diff, "matches": reload_max_abs_diff <= 1e-6},
        "replay": {
            "deterministic": replay_same,
            "fixture_hash": hashlib.sha256(
                json.dumps(
                    {
                        "sample_ids": fixture.sample_ids,
                        "relations": [[r.value for r in row] for row in fixture.relation_matrix],
                        "dropout": fixture.artificial_dropout_plan,
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest(),
        },
        "representation_health": post.health.__dict__,
        "provenance_leakage": {"truth_state_used": False, "downstream_state_inference": False, "temp_objective_active": False},
        "compute": {
            "device": "cpu",
            "peak_memory_bytes": int(metrics["memory_allocated_bytes"]),
            "wall_clock_s": wall_s,
            "throughput_samples_per_s": len(fixture.sample_ids) / wall_s,
        },
    }
    run.log_metrics({"step": 1, **metrics})
    run.log_event(
        "OSFM_M1_FUSION_SMOKE_STEP",
        time.time_ns(),
        {
            "sample_ids": fixture.sample_ids,
            "objective_status": {r.objective_id: r.status.value for r in post.results},
            "formal_rank_guard": post.health.formal_rank_guard,
        },
    )
    run.write_artifact("reports", "m1_fusion_smoke_report.json", json.dumps(result, indent=2, default=str))
    run.write_artifact("reports", "checkpoint_index.json", json.dumps({"latest": str(ckpt), "checkpoint_id": meta.checkpoint_id}, indent=2))
    return result
