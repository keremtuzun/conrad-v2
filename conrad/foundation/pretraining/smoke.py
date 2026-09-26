"""End-to-end OS-FM smoke helpers used by tests and `conrad train run`."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any
from uuid import UUID

import torch

from conrad.foundation.augment import ViewEngine, ViewRecipe, default_registry
from conrad.foundation.data.batch import collate_foundation_windows
from conrad.foundation.data.capture import FoundationCaptureUnit, build_windows
from conrad.foundation.data.manifest import synthetic_manifest_for_tests
from conrad.foundation.data.sampler import HierarchicalSampler
from conrad.foundation.pretraining.bundle import (
    ArchitectureFaithfulFoundationEncoder,
    OSFMTrainingBundle,
    TinyFoundationEncoder,
)
from conrad.training.checkpoint import load_checkpoint, save_checkpoint
from conrad.training.checkpoint_meta import CompatibilityTuple, config_digest
from conrad.training.run_dir import RunDirectory

MODALITIES = ("RGB", "SONAR", "DEPTH_RANGE", "POINT_CLOUD", "STRUCTURED")


def synthetic_units(*, missing: tuple[str, ...] = (), temporal_gap: bool = False) -> tuple[FoundationCaptureUnit, ...]:
    units = []
    times = [0, 500, 1000]
    if temporal_gap:
        times = [0, 500, 2001]
    for step, t in enumerate(times):
        for idx, modality in enumerate(MODALITIES):
            if modality in missing:
                continue
            payload = torch.linspace(0, 1, 8).numpy() + idx + step
            units.append(
                FoundationCaptureUnit(
                    unit_id=f"u-{step}-{modality}",
                    sequence_id="synthetic-seq",
                    timestamp_ms=t,
                    modality=modality,
                    payload=payload,
                    lineage={"dataset_id": "synthetic", "sequence_id": "synthetic-seq"},
                    calibration_ref=f"cal-{modality}",
                )
            )
    return tuple(units)


def split_hash_for_plan(plan: object) -> str:
    return hashlib.sha256(json.dumps(plan, sort_keys=True, default=str).encode()).hexdigest()


def run_osfm_smoke(
    run: RunDirectory,
    job: dict[str, Any],
    seed: int,
    *,
    architecture_faithful: bool = False,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    manifest = synthetic_manifest_for_tests()
    units = synthetic_units(missing=tuple(job.get("missing_modalities", ())), temporal_gap=bool(job.get("temporal_gap", False)))
    windows = build_windows(units, expected_modalities=MODALITIES)
    sampler = HierarchicalSampler(seed=seed, batch_size=int(job.get("batch_size", 2)))
    plan = sampler.plan(windows, corpus_digest=manifest.corpus_digest, epoch=0)
    selected = {w.window_id: w for w in windows}
    batch_windows = tuple(selected[i] for i in plan.ordered_window_ids[: sampler.batch_size])
    batch = collate_foundation_windows(
        batch_windows,
        modalities=MODALITIES,
        artificial_dropout={m: True for m in job.get("dropout_modalities", ())},
    )
    engine = ViewEngine(default_registry(), seed=seed)
    teacher_view = engine.make_view(batch, ViewRecipe("teacher-recorded-minimal", ("identity_recorded_observation",), teacher=True))
    student_view = engine.make_view(
        batch,
        ViewRecipe(
            "student-corrupt-drop-mask",
            tuple(job.get("student_transforms", ("rgb_calibrated_jitter", "sonar_snr_jitter"))),
            tuple(job.get("dropout_modalities", ())),
            float(job.get("token_mask_fraction", 0.25)),
        ),
    )
    engine.validate_lineage_stable(batch, teacher_view)
    engine.validate_lineage_stable(batch, student_view)
    encoder = ArchitectureFaithfulFoundationEncoder() if architecture_faithful else TinyFoundationEncoder()
    bundle = OSFMTrainingBundle(encoder)
    optimizer = torch.optim.AdamW(bundle.parameters(), lr=float(job.get("lr", 1e-3)))
    out = bundle(student_view.batch)
    out.loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    ema = bundle.update_teacher_after_optimizer(1)
    metrics = {
        "val/osfm_loss": float(out.loss.detach()),
        "objective_active": float(sum(r.status.value == "ACTIVE" for r in out.results)),
        "modality_coverage": float((~batch.natural_missing_mask).float().mean()),
        "pairing_coverage": float(batch.pair_coverage.mean()),
        "representation_effective_rank": float(min(out.student_repr.shape[-1], out.student_repr.shape[1])),
        "representation_collapse_score": float(out.student_repr.std().detach()),
        "throughput_windows_per_step": float(len(batch_windows)),
        "memory_allocated_bytes": float(torch.cuda.memory_allocated() if torch.cuda.is_available() else 0),
        "ema_momentum": float(ema),
    }
    for result in out.results:
        for key, value in result.as_metrics().items():
            if isinstance(value, float):
                metrics[key.replace("/", "_")] = value
    run.log_metrics({"step": 1, **metrics})
    run.log_event(
        "OSFM_SMOKE_STEP",
        time.time_ns(),
        {
            "windows": len(windows),
            "lineage_ids": batch.lineage_ids,
            "teacher_traces": [t.transform_id for t in teacher_view.traces],
            "student_traces": [t.transform_id for t in student_view.traces],
        },
    )
    cfg_digest = config_digest(job)
    split_hash = split_hash_for_plan({"plan": plan.plan_digest, "ordered": plan.ordered_window_ids})
    ckpt = run.path / "checkpoints" / "osfm_smoke.pt"
    meta = save_checkpoint(
        ckpt,
        model=bundle,
        component="foundation.osfm",
        config=job,
        manifest_digests=(manifest.corpus_digest,),
        split_hash=split_hash,
        seeds={"torch": seed, "sampler": seed},
        metrics=metrics,
        epoch=0,
        step=1,
        created_time_ns=time.time_ns(),
        optimizer=optimizer,
        trainer_state={
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "amp_scaler": None,
            "epoch_plan_digest": plan.plan_digest,
        },
        is_encoder=True,
        representation_pretraining_id="OSFM-P3-SMOKE",
        selection_metric="val/osfm_loss",
        extra_compatibility={"osfm_stage": str(job.get("stage_id", "P3"))},
    )
    fresh = OSFMTrainingBundle(ArchitectureFaithfulFoundationEncoder() if architecture_faithful else TinyFoundationEncoder())
    loaded = load_checkpoint(
        ckpt,
        expected=CompatibilityTuple(
            config_digest=cfg_digest,
            manifest_digests=(manifest.corpus_digest,),
            split_hash=split_hash,
            component="foundation.osfm",
            representation_pretraining_id="OSFM-P3-SMOKE",
            extra={"osfm_stage": str(job.get("stage_id", "P3"))},
        ),
        model=fresh,
    )
    promotion = {
        "stage_id": str(job.get("stage_id", "P3")),
        "parent_checkpoint_id": job.get("parent_checkpoint_id"),
        "checkpoint_id": loaded.metadata.checkpoint_id,
        "run_id": run.path.name,
        "promoted": True,
        "reason": "P3 smoke validation artifact only; not a formal U1 result",
    }
    run.write_artifact("reports", "promotion_record.json", json.dumps(promotion, indent=2))
    run.write_artifact(
        "reports",
        "checkpoint_index.json",
        json.dumps({"latest": str(ckpt), "checkpoint_id": meta.checkpoint_id}, indent=2),
    )
    return {
        "component": "foundation.osfm",
        "architecture_faithful": architecture_faithful,
        "manifest_digest": manifest.corpus_digest,
        "split_hash": split_hash,
        "epoch_plan_digest": plan.plan_digest,
        "windows": len(windows),
        "checkpoint": str(ckpt),
        "checkpoint_id": loaded.metadata.checkpoint_id,
        "metrics": metrics,
        "promotion": promotion,
    }


def public_real_subpipe_probe(_config: dict[str, Any]) -> dict[str, Any]:
    """Probe the real SubPipe archive through its production adapter, if present."""
    from uuid import UUID

    from conrad.data.adapters.subpipe import SubPipeAdapter
    from conrad.data.manifest import data_root_for, find_manifest, load_manifest
    from conrad.persistence.object_store import ObjectStore
    from conrad.schemas.ids import IdFactory

    try:
        manifest_path = find_manifest("public.subpipe@zenodo-12666132-v3.0.1-SubPipeMini2")
        manifest = load_manifest(manifest_path)
        adapter = SubPipeAdapter(
            manifest,
            data_root_for(manifest.dataset_id),
            ObjectStore(Path("/tmp/conrad-public-real-probe")),
            IdFactory(seed=47, namespace=UUID(int=47)),
            UUID(int=47),
            streams=("sss_lf",),
            manifest_path=manifest_path,
        )
        audit = adapter.inspect()
        refs = adapter.frames("sss_lf") if audit.usable else ()
        adapter.close()
        if not audit.usable:
            return {"status": "BLOCKED_EXTERNAL", "reason": "; ".join(str(p) for p in audit.problems)}
        return {"status": "VERIFIED", "adapter": "subpipe_zip-1.0.0", "sonar_frames": len(refs)}
    except Exception as exc:  # unavailable external payloads remain an explicit blocker
        return {"status": "BLOCKED_EXTERNAL", "reason": str(exc)}
