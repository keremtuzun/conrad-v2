"""Per-modality clean-frame rank on train vs validation frames for a V1.1 Kerem-trainer checkpoint."""
import sys

import torch

from conrad.foundation.pretraining.u1_rgb import rank_diversity_loss
from conrad.foundation.universal_v11 import trainer as T
from conrad.foundation.universal_v11.core import FamilyEncoderConfig, UniversalOSFMV11

dev = torch.device("cuda")
b = T._SubPipeFullBatcher(archive_sha256=T.verify_subpipe_full()["actual_sha256"], batch_size=256, device=dev, seed=11)
model = UniversalOSFMV11(
    input_dims={m: T._patch_dim(s["channels"]) for m, s in T.MODALITIES.items()},
    family_encoder_config=FamilyEncoderConfig(depth=6),
    include_v1_bank=True,
).to(dev)
if len(sys.argv) > 1:
    model.load_state_dict(torch.load(sys.argv[1], map_location="cpu", weights_only=False)["model"])
model.eval()


def rank(z):
    _, r = rank_diversity_loss(z, target=75.0)
    s = torch.linalg.svdvals(z - z.mean(0, keepdim=True))
    return round(float(r), 1), round(float(s[:5].sum() / s.sum()), 3), round(float(z.std(0).mean()), 3)


for modality in T.MODALITIES:
    for split in ("train", "val"):
        imgs, _ = b.batch(modality, split, 999, distinct=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model((T._make_input(modality, T._clean_view(imgs), 1),), reference_time_s=torch.ones(imgs.shape[0], device=dev))
        z = out.fusion.global_repr.float()
        pix = T._clean_view(imgs).flatten(1)
        print(f"{modality:14s} {split:5s} n={imgs.shape[0]} repr(rank,top5,std)={rank(z)}  pixels(rank)={rank(pix)[0]}")
