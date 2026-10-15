"""How high can each source's held-out rank honestly go? Rank of raw pixels, of an untrained V1.1 and of given
checkpoints, on the exact held-out frames final_checks.py uses (up to 256 distinct clean frames per source),
plus the same at a fixed n=100 frames so sources with few held-out frames can be compared.

Usage (inside conrad-v2): uv run python scripts/kerem_p4/ceiling_check.py <out.json> [name=checkpoint.pt ...]
"""
import json
import sys

import torch

from conrad.foundation.pretraining.u1_rgb import rank_diversity_loss
from conrad.foundation.universal_v11 import trainer as T
from conrad.foundation.universal_v11.core import FamilyEncoderConfig, UniversalOSFMV11

out = sys.argv[1]
checkpoints = dict(arg.split("=", 1) for arg in sys.argv[2:])
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dims = {m: T._patch_dim(s["channels"]) for m, s in T.MODALITIES.items()}
FIXED_N = 100


def rank(z: torch.Tensor) -> float:
    return round(float(rank_diversity_loss(z.float(), target=75.0)[1]), 1)


def build(path: str | None) -> UniversalOSFMV11:
    torch.manual_seed(0)
    model = UniversalOSFMV11(input_dims=dims, family_encoder_config=FamilyEncoderConfig(depth=T.FAMILY_DEPTH), include_v1_bank=True)
    if path:
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"], strict=True)
    return model.to(dev).eval()


def encode(model: UniversalOSFMV11, modality: str, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        payload = T._clean_view(x, modality, model)
        o = model((T._make_input(modality, payload, 1),), reference_time_s=torch.ones(x.shape[0], device=dev))
    return o.fusion.global_repr.float()


b = T._MultiSourceBatcher(archive_sha256=T.verify_subpipe_full()["actual_sha256"], extra_camera=T.verify_extra_sources(), batch_size=256, device=dev, seed=1)
frames = {}
for modality in T.MODALITIES:
    frames[(modality, "pooled")] = b.batch(modality, "val", 0, distinct=True)[0]
    for src in b.sources(modality):
        frames[(modality, src)] = b.batch(modality, "val", 0, distinct=True, source=src)[0]

result: dict[str, dict[str, float | int]] = {}
for (modality, src), x in frames.items():
    pixels = T._standardize(T._clean_images(x)).flatten(1)
    key = f"{modality}/{src}"
    result[key] = {"n": int(x.shape[0]), "pixels": rank(pixels), f"pixels@n{FIXED_N}": rank(pixels[:FIXED_N])}
for name, path in [("untrained", None), *checkpoints.items()]:
    model = build(path)
    for (modality, src), x in frames.items():
        z = encode(model, modality, x)
        result[f"{modality}/{src}"][name] = rank(z)
        result[f"{modality}/{src}"][f"{name}@n{FIXED_N}"] = rank(z[:FIXED_N])
    del model
with open(out, "w") as handle:
    json.dump(result, handle, indent=2)
cols = list(next(iter(result.values())))
print("source".ljust(32) + "".join(c.rjust(16) for c in cols))
for key, row in result.items():
    print(key.ljust(32) + "".join(str(row[c]).rjust(16) for c in cols))
