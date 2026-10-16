# V1.1 P4 on 1x L4: final training report

Run `v11-p4-kerem-long-a1-20260930T001158Z`. Finished 2026-10-02 18:44 Turkey time.

## Decision: NO-GO on the rank-75 gate (training valid and complete)

| Gate | Result |
|---|---|
| Steps completed | **775,000 / 775,000** (about 6.5P; 10P = 1.2M does not fit 72 h on one L4) |
| Wall clock | 63.5 h (limit 72 h) |
| NaN/Inf, variance collapse, stalls, GPU loss | none in 776 validations |
| Meaning guard (held-out temporal nearest-neighbour hit >= 0.3) | passed at every validation; minimum 0.62 |
| Checkpoint write and strict reload | passed (best and last) |
| **Rank >= 75 for every family** | **failed: camera 94.2 passes, sonar 61.4 does not** |

The weakest family (sonar) peaks at 61.4, so the trainer's own report says **NO-GO**. This is a complete,
healthy, reproducible P4 run, but it is **not a promotable 10P checkpoint** by the project's gate.

## What was trained

- Model: Universal OS-FM V1.1 at the project's full depth (FamilyEncoderConfig depth 6), **308.1M parameters**,
  384-D interface, 15 families, 829-entry registry.
- Sonar front end: the upstream P4.8 sonar encoder (OSFM-S-PRETRAIN-V1, sha256 `b2dfc162...`). Its teacher weights
  were loaded 150/150 into V1.1's V1 compatibility bank and kept frozen (config stage A: protect the imported
  anchor). Its CLS + patch tokens are the imaging_sonar input.
- Camera: 16 RGB patch tokens per frame.
- Objective: plain VICReg. One view is the clean frame; the other is a crop with photometric noise and patch dropout.
  bf16, AdamW, lr 5e-5 with 2k warm-up and cosine decay, weight decay 0.05. Odd steps train sonar, even steps camera.
- Trainer: **Kerem variant, `reviewed_trainer: false`**. It was launched through
  `conrad train osfm-v11-10p-launch --kerem-override-unreviewed-trainer`, which records the readiness override
  (blockers 2/4/8 only: not the reviewed trainer or branch) in the run manifests. Code is on local branch
  `kerem/v11-p4-patch-vicreg-trainer` @ `8d82f01`, based on the pinned `8885291`. **Not pushed.**

## Results on held-out data

Best checkpoint (`best.pt`, step 416,000). It was selected after warm-up, only if every family's meaning score was >= 0.3, then by the highest weakest-family rank:

| | Rank (gate 75) | Meaning score |
|---|---|---|
| imaging_sonar | 61.4 | 0.76 |
| rgb_camera, pooled over 3 sources | **94.2** | 0.65 |
| camera, SeaClear only | 71.1 | |
| camera, UVVID only | 54.8 | |
| camera, SubPipe only | 20.0 | |

Trajectory (sonar rank / meaning | camera rank / meaning):

| Step | Sonar | Camera | Train loss |
|---|---|---|---|
| 1 | 60.9 / 0.52 | 15.8 / 0.56 | 47.6 |
| 5k | 36.6 / 0.69 | 74.0 / 0.71 | 19.9 |
| 50k | 42.1 / 0.68 | 93.6 / 0.67 | 11.5 |
| 200k | 47.9 / 0.73 | 94.1 / 0.65 | 8.0 |
| 400k | 51.1 / 0.74 | 101.5 / 0.69 | 6.4 |
| 600k | 53.2 / 0.77 | 98.6 / 0.65 | 5.8 |
| 775k | 55.6 / 0.77 | 100.2 / 0.65 | 5.0 |

### Independent representation-health check (`representation_health_heldout.json`)

This check runs over **all** held-out frames: 981 sonar and 5,585 camera.

| | Meaning score | Rank (random 256) |
|---|---|---|
| sonar: pixels / P4.8 anchor alone | 0.81 / 0.84 | |
| sonar: best.pt / last.pt | 0.69 / 0.69 | 59.1 / 59.0 |
| camera: pixels | 0.79 | |
| camera: best.pt / last.pt | 0.41 / 0.41 | 48.6 / 52.2 |

Honest reading:
1. **Camera's pooled 94 is driven largely by differences between sites.** On a random sample of all held-out
   camera frames, 83% of which are SubPipe, rank is 49-52, and SubPipe-only camera rank is 20.
   SubPipe camera frames have little pixel-level diversity to begin with (pixel rank about 34).
2. **Sonar is limited by its frozen anchor.** The P4.8 encoder itself scores about 63 on this held-out split,
   and V1.1 reaches 59-61 on top of it. Passing 75 needs sonar recordings from other missions.
3. On the temporal-neighbour test, the learned representations keep less frame-to-frame detail than raw pixels.
   That is expected for augmentation-invariant features, but it means this test shows **no evidence of
   superiority over the input**. It shows only that the features are structured and not noise.

## Checkpoint reload (`checkpoint_reload_report.json`)

| | Step | Strict load | Identical after load | All finite | Params | Optimizer state |
|---|---|---|---|---|---|---|
| best.pt | 416,000 | yes | yes | yes | 308,104,961 | yes |
| last.pt | 775,000 | yes | yes | yes | 308,104,961 | yes |

The checkpoints are copied to the Mac at `artifacts/runs/v11-p4-kerem-long-a1-20260930T001158Z/checkpoints/`:
- best.pt sha256 `bff713372d2cdfbd875b6bd676d00a3ca925c5c52fad3a5719113560903f5123`
- last.pt sha256 `808944a3d48fa93fb2e22ccf232d45ddb48c021b528c798483c261ae368887f8`

## Semantic modality coverage

| Modality (family) | Sources | Train frames | Held-out frames |
|---|---|---|---|
| imaging_sonar (active_acoustic) | SubPipe side-scan LF + HF, chunks 0-4 | 6,868 | 981 |
| rgb_camera (visual_image) | SubPipe cam0 (every 2nd frame) + cam1 | 32,279 | 4,611 |
| | SeaClear, 5 sites (11 site/camera groups) | 6,021 | 856 |
| | UVVID ROV_GoPro_1-8, 2 frames/s | 836 | 118 |
| | **camera total** | **39,136** | **5,585** |

All sources are CC-BY-4.0, rights CLEARED and training_allowed. Each archive or video is hash-verified against its
manifest (SubPipe.zip `7723d93d...`; SeaClear rar `2a053f74...`; UVVID 1-8 recorded in uvvid.manifest.yaml).

## Registered but inactive

**827 of 829** registry entries received no training data. They remain interface coverage only. No semantic
claim is made for them, or for 13 of the 15 encoder families.

## P4 V1 non-contamination

- The upstream checkpoint was read-only: its sha256 is still `b2dfc162...` after training.
- The trained commit changes only `conrad/foundation/universal_v11/trainer.py`, `conrad/cli/commands.py` (one
  opt-in flag) and 3 dataset manifests. **0 files changed** in `conrad/oceansense` (P5-P10),
  `conrad/foundation/pretraining`, `conrad/foundation/encoders` or `configs/train/osfm`.
- No old V1 configs or tensor contracts were used. `osfm-universal-v1.1` on GitHub is untouched (still `8885291`).

## Cost report (TRY, live GCP SKU prices)

| Item | TRY |
|---|---|
| VM 1 (us-central1-b): benchmark of the reviewed trainer | about 50 |
| VM 2 (us-east4-a): pilots on SubPipe | about 155 |
| VM 3 (us-central1-a): setup, pilots, first long run (stopped for rank gaming) | about 290 |
| VM 3: final long run | about 2,575 |
| CPU helpers, disks, egress | about 40 |
| **Total** | **about 3,100** (ceiling 8,500; within the free trial) |

## Things found along the way (for Burak)

1. The reviewed trainer used 32 row-band means per frame and every 16th frame (6 validation frames). Its rank
   could never pass, and it built depth 2 (201M), not the 307M V1.1.
2. `load_v1_checkpoint_for_v11` loads 0 tensors from the P4.8 checkpoint. The weights sit under `ck["model"]` with
   `teacher.`/`student.` prefixes. Also, V1.1 builds `self.v1` but never calls it in `forward`.
3. The explicit rank term can be gamed: a first long run hit "rank 154" while the meaning score fell to chance.
   A rank gate alone is not evidence of learning, and an untrained V1.1 already scores 82-98.
4. Readiness writes to a tracked file, and the acceptance RunDirectory then refuses the dirty tree.
5. p7zip cannot extract SeaClear's RAR5 (it silently writes empty files); use bsdtar.

## Evidence

`final_v11-p4-kerem-long-a1/` (next to this file) holds:
- config snapshot (`config.resolved.yaml`) and environment, git and manifests
- `metrics.jsonl` (776 validations) and events
- `reports/` (training report, trainer_variant, data_coverage, upstream_import)
- checkpoint metadata, reload report, held-out health check and checkpoint hashes
- readiness output (the recorded override)
- orchestrator, watchdog and trainer logs, plus GPU utilisation samples

Earlier attempts are in `kerem_pilots_20260929/` and `kerem_vm_benchmark_20260929/`. Hourly progress is in `PROGRESS_LOG.md`.
