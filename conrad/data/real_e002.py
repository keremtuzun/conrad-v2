"""DATA-REAL-E002: ECMER on real SubPipe data (see datasets/experiments/data_real_e002.yaml).

(a1) camera quality-feature sanity on cam0 frames; (a2) logistic probe on ECMER evidence for the
side-scan "Pipeline" presence label, evaluated on held-out time blocks fixed in the config before
the first run. Labels come only from the adapter's PartialTruth channel and are used only to fit and
score the probe; the encoder never sees them. (b) Model2S integration is not run (config limitations).
"""

from __future__ import annotations

import gc
import json
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import torch

from conrad.core.config import CoreConfig
from conrad.core.ecmer.preprocess import raw_quality_for
from conrad.core.ecmer.quality_features import image_quality
from conrad.core.ecmer.service import EcmerEncoder
from conrad.data.adapters.subpipe import FrameRef, SubPipeAdapter
from conrad.data.manifest import LineageKeys, SplitUnit, data_root_for, find_manifest, load_manifest
from conrad.data.real_smoke import EXPECTED, _corrupt
from conrad.data.splits import SampleRef, assert_no_lineage_leakage
from conrad.persistence.object_store import ObjectStore
from conrad.schemas.ids import IdFactory
from conrad.schemas.observation import Observation

EXPERIMENT_ID = "DATA-REAL-E002"
FloatArr = np.ndarray


# ---------------------------------------------------------------------- split (fixed by config)
def time_block_split(n: int, cfg: Mapping[str, Any]) -> tuple[list[str | None], list[int]]:
    """Index -> split name (None = dropped by the boundary guard), and index -> block id."""
    n_blocks = int(cfg["n_blocks"])
    block = [i * n_blocks // n for i in range(n)]
    test, val = set(cfg["test_blocks"]), set(cfg["validation_blocks"])
    name = ["test" if b in test else "validation" if b in val else "train" for b in block]
    guard = int(cfg["boundary_guard_images"])
    out: list[str | None] = []
    for i in range(n):
        lo, hi = max(0, i - guard), min(n, i + guard + 1)
        out.append(name[i] if all(name[j] == name[i] for j in range(lo, hi)) else None)
    return out, block


# ---------------------------------------------------------------------- metrics
def balanced_accuracy(y: FloatArr, pred: FloatArr) -> float:
    pos, neg = y == 1, y == 0
    if not pos.any() or not neg.any():
        return float("nan")
    return float(0.5 * ((pred[pos] == 1).mean() + (pred[neg] == 0).mean()))


def auroc(y: FloatArr, score: FloatArr) -> float:
    pos, neg = score[y == 1], score[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(order.size)
    ranks[order] = np.arange(1, order.size + 1)
    return float((ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def bootstrap_ci(y: FloatArr, pred: FloatArr, resamples: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(resamples):
        idx = rng.integers(0, y.size, y.size)
        v = balanced_accuracy(y[idx], pred[idx])
        if np.isfinite(v):
            vals.append(v)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ---------------------------------------------------------------------- probe
def fit_logistic(x: FloatArr, y: FloatArr, l2: float, steps: int, lr: float, seed: int) -> FloatArr:
    torch.manual_seed(seed)
    xt = torch.from_numpy(np.asarray(x, dtype=np.float32))
    yt = torch.from_numpy(np.asarray(y, dtype=np.float32))
    w = torch.zeros(xt.shape[1] + 1, requires_grad=True)
    pos = float(yt.mean())
    weights = torch.where(yt > 0.5, 0.5 / max(pos, 1e-6), 0.5 / max(1 - pos, 1e-6))  # class-balanced
    opt = torch.optim.Adam([w], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        logits = xt @ w[:-1] + w[-1]
        loss = (
            weights * torch.nn.functional.binary_cross_entropy_with_logits(logits, yt, reduction="none")
        ).mean()
        loss = loss + l2 * (w[:-1] ** 2).sum()
        loss.backward()
        opt.step()
    return np.asarray(w.detach().numpy(), dtype=np.float64)


def _score(w: FloatArr, x: FloatArr) -> FloatArr:
    return np.asarray(x @ w[:-1] + w[-1], dtype=np.float64)


def evaluate_probe(
    feats: FloatArr, y: FloatArr, split: Sequence[str | None], cfg: Mapping[str, Any], seed: int
) -> dict[str, Any]:
    idx = {s: np.asarray([i for i, v in enumerate(split) if v == s]) for s in ("train", "validation", "test")}
    mu, sd = feats[idx["train"]].mean(0), feats[idx["train"]].std(0) + 1e-6
    z = (feats - mu) / sd
    pc = cfg["probe"]
    best: tuple[float, float] | None = None
    val_scores = {}
    for l2 in pc["l2_grid"]:
        w = fit_logistic(z[idx["train"]], y[idx["train"]], float(l2), int(pc["steps"]), float(pc["lr"]), seed)
        v = balanced_accuracy(y[idx["validation"]], (_score(w, z[idx["validation"]]) > 0).astype(int))
        val_scores[str(l2)] = v
        if best is None or v > best[0]:
            best = (v, float(l2))
    assert best is not None
    w = fit_logistic(z[idx["train"]], y[idx["train"]], best[1], int(pc["steps"]), float(pc["lr"]), seed)
    s = _score(w, z[idx["test"]])
    yt, pred = y[idx["test"]], (s > 0).astype(int)
    lo, hi = bootstrap_ci(yt, pred, int(cfg["pass"]["bootstrap_resamples"]), seed)
    return {
        "l2_selected_on_validation": best[1],
        "validation_balanced_accuracy_by_l2": val_scores,
        "test_balanced_accuracy": balanced_accuracy(yt, pred),
        "test_balanced_accuracy_ci95": [lo, hi],
        "test_auroc": auroc(yt, s),
    }


# ---------------------------------------------------------------------- run
def _drop(store: ObjectStore, observations: Sequence[Observation]) -> None:
    for o in observations:
        if o.payload_ref is not None:
            store.path_for(o.payload_ref.digest).unlink(missing_ok=True)


def run(config: Mapping[str, Any], seeds: Sequence[int], out_dir: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="conrad-e002-") as tmp:
        return _run(config, seeds, Path(out_dir), Path(tmp))


def _encoders(seeds: Sequence[int], store: ObjectStore) -> dict[int, EcmerEncoder]:
    out = {}
    for seed in seeds:
        torch.manual_seed(int(seed))
        out[int(seed)] = EcmerEncoder(
            CoreConfig(), IdFactory(seed=int(seed), namespace=f"{EXPERIMENT_ID}-enc"), loader=store.get_array
        )
    return out


def _camera(
    adapter: SubPipeAdapter, cfg: Mapping[str, Any], seeds: Sequence[int], store: ObjectStore
) -> dict[str, Any]:
    refs = adapter.frames(str(cfg["camera_stream"]))[:: int(cfg["camera_stride"])][
        : int(cfg["camera_max_frames"])
    ]
    rng = np.random.default_rng(int(seeds[0]))
    clean = []
    corrupted: dict[str, list[Any]] = {k: [] for k in EXPECTED}
    shape: list[int] = []
    for r in refs:  # one frame in memory at a time (cam0 frames are 2704x1520x3)
        im = adapter.load(r)
        shape = list(im.shape)
        clean.append(image_quality(im))
        for kind in EXPECTED:
            corrupted[kind].append(image_quality(_corrupt(im, kind, cfg["corruptions"], rng)))
    response: dict[str, Any] = {}
    fractions = []
    for kind, feats in EXPECTED.items():
        for feat, sign in feats.items():
            d = np.asarray(
                [getattr(c, feat) - getattr(q, feat) for c, q in zip(corrupted[kind], clean, strict=True)]
            )
            frac = float((np.sign(d) == sign).mean())
            fractions.append(frac)
            response[f"{kind}.{feat}"] = {"mean_delta": float(d.mean()), "fraction_expected_direction": frac}
    finite = True
    rel: dict[str, float] = {}
    for seed, enc in _encoders(seeds, store).items():
        for start in range(0, len(refs), 8):
            obs = [adapter.normalize_observation(r) for r in refs[start : start + 8]]
            ev = enc.encode(obs)
            finite &= all(np.isfinite(e.evidence.embedding).all() for e in ev)
            rel.setdefault(f"seed_{seed}", 0.0)
            rel[f"seed_{seed}"] += sum(e.evidence.reliability for e in ev) / len(refs)
            _drop(store, obs)
            gc.collect()

    def stat(name: str) -> dict[str, float]:
        v = np.asarray([getattr(q, name) for q in clean], dtype=np.float64)
        return {"mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()), "max": float(v.max())}

    return {
        "n_frames": len(refs),
        "time_span_s": (refs[-1].time_ns - refs[0].time_ns) / 1e9 if refs else 0.0,
        "frame_shape": shape,
        "calibration": adapter.calibration(str(cfg["camera_stream"])),
        "quality": {k: stat(k) for k in ("blur", "brightness", "contrast", "snr_db")},
        "corruption_response": response,
        "min_fraction_expected_direction": min(fractions),
        "all_embeddings_finite": bool(finite),
        "reliability_mean_untrained": rel,
        "check": "PASS" if min(fractions) >= float(cfg["camera_pass_fraction"]) and finite else "FAIL",
    }


def _run(config: Mapping[str, Any], seeds: Sequence[int], out_dir: Path, tmp: Path) -> dict[str, Any]:
    manifest_path = find_manifest(str(config["manifest_id"]))
    manifest = load_manifest(manifest_path)
    store = ObjectStore(tmp / "store")
    stream = str(config["sonar_stream"])
    adapter = SubPipeAdapter(
        manifest,
        data_root_for(manifest.dataset_id),
        store,
        IdFactory(seed=int(seeds[0]), namespace=EXPERIMENT_ID),
        UUID(int=int(seeds[0])),
        streams=(stream, str(config["camera_stream"])),
        manifest_path=manifest_path,
    )
    audit = adapter.inspect()
    if not audit.usable:
        raise RuntimeError(f"manifest does not verify: {[str(p) for p in audit.problems]}")

    # ---- labels from the evaluation-side channel only; split fixed by the config
    all_refs = adapter.frames(stream)
    truth = adapter.map_labels((stream, all_refs))
    assert truth is not None
    listed = truth.availability_masks["pipeline_present"]
    refs: list[FrameRef] = [r for r, ok in zip(all_refs, listed, strict=True) if ok]
    y = truth.targets["pipeline_present"][listed].astype(np.int64)
    split, block = time_block_split(len(refs), config["split"])
    samples: dict[str, list[SampleRef]] = {"train": [], "validation": [], "test": []}
    for r, s, b in zip(refs, split, block, strict=True):
        if s is not None:
            samples[s].append(SampleRef(sample_id=r.name, lineage=LineageKeys(trajectory=f"block{b}")))
    assert_no_lineage_leakage(samples, [SplitUnit.TRAJECTORY])

    # ---- one decoding pass: engineered quality + every seed's ECMER embedding
    encoders = _encoders(seeds, store)
    quality = np.zeros((len(refs), CoreConfig().ecmer.quality_feature_dim))
    emb: dict[int, list[list[float]]] = {s: [] for s in encoders}
    reliability: dict[int, list[float]] = {s: [] for s in encoders}
    batch = int(config.get("batch_size", 8))
    for start in range(0, len(refs), batch):
        obs = [adapter.normalize_observation(r) for r in refs[start : start + batch]]
        for k, o in enumerate(obs):
            assert o.payload_ref is not None
            q = raw_quality_for(o, store.get_array(o.payload_ref))
            quality[start + k] = q.vector(quality.shape[1])
        for seed, enc in encoders.items():
            for e in enc.encode(obs):
                emb[seed].append(list(e.evidence.embedding))
                reliability[seed].append(e.evidence.reliability)
        _drop(store, obs)
        del obs
        gc.collect()
        if start % 200 < batch:
            print(f"{EXPERIMENT_ID}: sonar {start + batch}/{len(refs)}", file=sys.stderr, flush=True)

    counts = {s: int(sum(1 for v in split if v == s)) for s in ("train", "validation", "test")}
    pos = {
        s: int(sum(int(y[i]) for i, v in enumerate(split) if v == s)) for s in ("train", "validation", "test")
    }
    majority = int(pos["train"] * 2 >= counts["train"])
    test_idx = np.asarray([i for i, v in enumerate(split) if v == "test"])
    maj_ba = balanced_accuracy(y[test_idx], np.full(test_idx.size, majority))
    pc = config["pass"]
    per_seed: dict[str, Any] = {}
    passes = []
    for seed in encoders:
        x = np.asarray(emb[seed], dtype=np.float64)
        ecmer = evaluate_probe(x, y, split, config, seed)
        qual = evaluate_probe(quality, y, split, config, seed)
        ok = (
            ecmer["test_balanced_accuracy"] >= float(pc["min_balanced_accuracy"])
            and ecmer["test_balanced_accuracy"] - maj_ba >= float(pc["min_margin_over_majority"])
            and ecmer["test_balanced_accuracy_ci95"][0] > float(pc["require_bootstrap_ci_low_above"])
        )
        passes.append(ok)
        per_seed[f"seed_{seed}"] = {
            "ecmer_probe": ecmer,
            "engineered_quality_probe": qual,
            "reliability_mean_untrained": float(np.mean(reliability[seed])),
            "embeddings_finite": bool(np.isfinite(x).all()),
            "pass": ok,
        }
    camera = _camera(adapter, config, seeds, store)
    adapter.close()
    ecmer_ba = [v["ecmer_probe"]["test_balanced_accuracy"] for v in per_seed.values()]
    result: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "dataset": {
            "manifest_id": config["manifest_id"],
            "manifest_digest": manifest.manifest_digest(),
            "archive_sha256": adapter.archive.sha256,
            "license": manifest.license,
            "adapter": f"{adapter.adapter_name}@{adapter.adapter_version}",
        },
        "sonar": {
            "stream": stream,
            "n_listed_images": len(refs),
            "n_by_split": counts,
            "positives_by_split": pos,
            "dropped_by_boundary_guard": int(sum(1 for v in split if v is None)),
            "time_span_s": (refs[-1].time_ns - refs[0].time_ns) / 1e9,
            "majority_class_train": majority,
            "majority_test_balanced_accuracy": maj_ba,
            "per_seed": per_seed,
            "ecmer_probe_test_balanced_accuracy_mean": float(np.mean(ecmer_ba)),
            "check": "PASS" if all(passes) else "FAIL",
        },
        "camera": camera,
        "model2s": "NOT_RUN: see config limitations",
        "limitations": list(config.get("limitations", [])),
    }
    (out_dir / f"{EXPERIMENT_ID}_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
