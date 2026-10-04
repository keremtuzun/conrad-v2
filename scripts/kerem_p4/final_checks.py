"""Final evidence for a finished Kerem-trainer run: strict reload of best/last and held-out health per source.

Usage (inside conrad-v2, GPU or CPU): uv run python scripts/kerem_p4/final_checks.py artifacts/runs/<run_id> <out_dir>
Held-out health = effective rank on up to 256 distinct clean held-out frames, and the temporal-neighbour hit rate
(nearest neighbour of a held-out frame is within +-3 frames of it in the same sequence) on sequential sources.
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

from conrad.foundation.pretraining.u1_rgb import rank_diversity_loss
from conrad.foundation.universal_v11 import trainer as T
from conrad.foundation.universal_v11.core import FamilyEncoderConfig, UniversalOSFMV11

run, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
variant = json.loads((run / "reports" / "trainer_variant.json").read_text())
anchor = bool(variant.get("sonar_anchor_encoder"))
os.environ["KEREM_SONAR_ANCHOR"] = "1" if anchor else "0"
T.SONAR_ANCHOR = anchor
dims = {m: (384 if m == "imaging_sonar" and anchor else T._patch_dim(s["channels"])) for m, s in T.MODALITIES.items()}


def build(path):
    m = UniversalOSFMV11(input_dims=dims, family_encoder_config=FamilyEncoderConfig(depth=int(variant.get("family_depth", 6))), include_v1_bank=True)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model"], strict=True)
    finite = all(torch.isfinite(v).all().item() for v in ck["model"].values() if torch.is_tensor(v) and v.is_floating_point())
    return m.to(dev).eval(), {"step": ck["trainer_state"]["step"], "strict_load": True, "all_weights_finite": finite,
                              "parameters": sum(p.numel() for p in m.parameters()), "has_optimizer_state": ck.get("optimizer") is not None}


reload = {}
for name in ("best", "last"):
    p = run / "checkpoints" / f"{name}.pt"
    if p.is_file():
        t = time.time()
        _, info = build(p)
        reload[name] = {**info, "load_seconds": round(time.time() - t, 1)}
(out / "checkpoint_reload_report.json").write_text(json.dumps(reload, indent=2))
print(json.dumps(reload, indent=2))

extra = T.verify_extra_sources()
b = T._MultiSourceBatcher(archive_sha256=T.verify_subpipe_full()["actual_sha256"], extra_camera=extra, batch_size=256, device=dev, seed=1)


def encode(model, modality, x):
    zs = []
    for i in range(0, x.shape[0], 256):
        chunk = x[i : i + 256]
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            payload = T._clean_view(chunk, modality, model)
            o = model((T._make_input(modality, payload, 1),), reference_time_s=torch.ones(chunk.shape[0], device=dev))
        zs.append(o.fusion.global_repr.float())
    return torch.cat(zs)


health = {}
for name in reload:
    model, _ = build(run / "checkpoints" / f"{name}.pt")
    for modality in T.MODALITIES:
        imgs, _ = b.batch(modality, "val", 0, distinct=True)
        health[f"{name}/{modality}/pooled_rank"] = round(float(rank_diversity_loss(encode(model, modality, imgs), target=75.0)[1]), 1)
        for src in b.sources(modality):
            si, _ = b.batch(modality, "val", 0, distinct=True, source=src)
            health[f"{name}/{modality}/{src}/rank"] = round(float(rank_diversity_loss(encode(model, modality, si), target=75.0)[1]), 1)
        x, labels, pos = b.heldout_sequences(modality)
        health[f"{name}/{modality}/nn_temporal_hit_sequential"] = round(T._temporal_nn_hit(encode(model, modality, x), labels, pos), 3)
        pix = T._patchify(T._standardize(T._clean_images(x))).flatten(1)
        health[f"pixels/{modality}/nn_temporal_hit_sequential"] = round(T._temporal_nn_hit(pix, labels, pos), 3)
(out / "representation_health_heldout.json").write_text(json.dumps(health, indent=2))
print(json.dumps(health, indent=2))
