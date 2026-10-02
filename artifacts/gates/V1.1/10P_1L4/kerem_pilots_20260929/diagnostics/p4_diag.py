"""Where does validation rank go? Compare clean-train, clean-val, augmented-val, per-stream, and random init."""
import sys

import torch

from conrad.foundation.pretraining.u1_rgb import rank_diversity_loss
from conrad.foundation.universal_v11 import trainer as T
from conrad.foundation.universal_v11.core import FamilyEncoderConfig, UniversalOSFMV11

ckpt_path = sys.argv[1]
dev = torch.device("cuda")
b = T._SubPipeBatcher(streams=T.SONAR_STREAMS, frame_stride=1, batch_size=256, device=dev, seed=7)


def build():
    return UniversalOSFMV11(
        input_dims={"imaging_sonar": T.PATCH_DIM}, family_encoder_config=FamilyEncoderConfig(depth=2), include_v1_bank=True
    ).to(dev).eval()


def enc(model, payload):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model((T._make_input("imaging_sonar", payload, 1),), reference_time_s=torch.ones(payload.shape[0], device=dev))
    return out.fusion.global_repr.float()


def rank(z):
    _, r = rank_diversity_loss(z, target=75.0)
    s = torch.linalg.svdvals(z - z.mean(0, keepdim=True))
    return round(float(r), 2), round(float((s[:5].sum() / s.sum())), 3), round(float(z.std(0).mean()), 4)


def imgs(split, n):
    ids = b.train_index if split == "train" else b.val_index
    refs = b.train_refs if split == "train" else b.val_refs
    g = torch.Generator().manual_seed(3)
    pick = torch.randperm(len(ids), generator=g)[:n].tolist()
    x = b.frames[ids[torch.tensor(pick, device=dev)]].unsqueeze(1).float() / 255.0
    streams = [refs[i][0] for i in pick]
    return x, streams


gen = torch.Generator(device=dev).manual_seed(5)
xt, _ = imgs("train", 206)
xv, sv = imgs("val", 206)
print("input pixel-space rank  train/val:", rank(T._clean_view(xt).flatten(1)), rank(T._clean_view(xv).flatten(1)))
for name, model in (("random_init", build()), ("trained", None)):
    if model is None:
        model = build()
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        model.eval()
    zt = enc(model, T._clean_view(xt))
    zv = enc(model, T._clean_view(xv))
    za = enc(model, T._train_view(xv, gen))
    lf = torch.tensor([s == "sss_lf" for s in sv], device=dev)
    print(f"[{name}] (rank, top5 sv share, mean per-dim std)")
    print("  clean train :", rank(zt))
    print("  clean val   :", rank(zv))
    print("  aug val     :", rank(za))
    print("  val lf only :", rank(zv[lf]), " hf only:", rank(zv[~lf]))
    print("  cos(mean_lf, mean_hf):", round(float(torch.nn.functional.cosine_similarity(zv[lf].mean(0), zv[~lf].mean(0), dim=0)), 4))
b.close()
