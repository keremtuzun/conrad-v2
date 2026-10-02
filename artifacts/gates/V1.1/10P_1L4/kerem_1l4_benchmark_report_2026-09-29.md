# V1.1 P4 10P, 1-L4 fallback: benchmark result and launch decision

Date: 2026-09-29 (UTC)
Decision: **NO-GO. The 1.2M-step run was not launched.**

## Setup (all verified)

| Check | Result |
|---|---|
| Branch / commit | osfm-universal-v1.1 @ 8885291572591d0a953e9c613caa3f604ca492fb, tracked tree clean |
| GCP project | <gcp-project> (paid billing, <billing-account>) |
| Quota | GPUS_ALL_REGIONS=1, NVIDIA_L4_GPUS=1 (us-central1) |
| VM | conrad-p4-l4, us-central1-b, g2-standard-12, 1x NVIDIA L4 23 GB, driver 580, torch 2.6.0+cu124, cuda device_count=1 |
| Checkpoint payload | sha256 b2dfc162...5696 OK (GitHub release osfm-v11-p4-handoff) |
| Dataset payload | SubPipeMini2.zip from Zenodo 12666132, sha256 a3068be2...188f OK, md5 7e0d925f OK |
| Manifest metadata | 8 derived members extracted from the zip, all sha256 OK, verify_manifest_by_id = [] |
| V1.1 unit tests on VM | 33 passed, 1 failed (test_readiness, before payloads were placed) |
| Readiness | READY FOR KEREM, 0 blockers |

## Benchmark (non-promotable)

Run: `bench1000-nonpromotable-20260929T153414Z`, started through the guarded
`osfm-v11-10p-launch` command with `--max-steps 1000`.

| Metric | Value | Gate | Pass |
|---|---|---|---|
| Steps completed | 1000 | 1000 | yes |
| Train loss | 0.964 -> 0.401, finite | finite | yes |
| Val effective rank | 3.81 (step 1) -> 4.06 (step 1000) | >= 75 | **NO** |
| Collapse score | 1.000 -> 1.006 | > 1e-6 | yes |
| Throughput | ~2.6 steps/s in-loop (2.29 including setup and saves) | >= 4.63 for 1.2M in 72 h | **NO** |
| GPU util | mean 91% while busy, 78% over whole window, 12.8 GB used | >= 70% | yes |
| Checkpoint write | best.pt + last.pt, ~1.0 GB each | written | yes |
| Checkpoint reload | torch.load(last.pt) OK in 0.8 s, keys format/model/optimizer/scheduler/trainer_state | loads | yes |
| Trainer's own report | decision NO-GO (best_rank 4.06 < 75) | | |

## Projection

- 1.2M steps / 2.6 steps/s = about 128 h. That is over the 72 h limit.
- The most that fits in 72 h is about 670k steps (about 5.6P). This is below the config's own
  `candidate_steps_min: 700000`, so 10P is infeasible and 12.5P is out of the question on 1 L4.
- Cost is not the binding limit. The live price is about 49.5 TRY/h, so 72 h would be about 3,570 TRY.

## Why not launch anyway

Two hard stop conditions in the brief are already met:
1. Rank remains below the gate after validation: 4.06 vs 75. Loss falls while rank barely moves,
   so the representation is not spreading out.
2. The projected runtime for the minimum 10P target exceeds 72 h.

Even a shortened run would end with the trainer's own gate returning NO-GO (`best_rank < rank_floor`),
so it would spend about 3.5k TRY of trial credit to produce a non-promotable checkpoint.

## Findings for Burak

1. Rank of about 4 on 256-sample validation batches. The trainer feeds each frame as `_frame_features(tokens=4, dim=8)`,
   which is 32 numbers per frame, into a 384-D interface. It also uses one modality (imaging_sonar),
   a 1000-step cosine schedule in the benchmark, and a noise-only augmentation (`+0.01 * randn`).
   The rank term does not overcome this. A likely structural bottleneck, not a tuning issue.
2. The upstream checkpoint is hash-verified but never loaded (`load_checkpoint` is only used for `--resume`).
   The model starts from random init.
3. The trainer does not enforce 72 h or TRY limits and does not stop on rank below the floor.
4. Readiness writes to the tracked `readiness.json`, which makes the tree dirty, and then the
   trainer's acceptance RunDirectory refuses with `DirtyGitError`. Workaround used here: the committed file was restored
   and readiness was written to an ignored `readiness_vm_8885291.json`.
5. The SubPipe loader needs 8 metadata members extracted from the zip under `extracted/`. Nothing in
   the handoff says so; the missing files cause `DatasetNotAvailableError`.
6. `size_bytes` in required_payloads.json is wrong for both files (the hashes are right).
7. The VM needs `libgl1` for OpenCV.

## Cost

- VM running: about 14:42Z to 15:43Z, roughly 1.0 h x 49.5 TRY = **about 50 TRY**, plus two failed restarts (not billed).
- VM, helper and disk deleted 2026-09-29; the project now has no instances, disks, snapshots or IPs, so nothing is billing.

## Evidence location

- Full bundle (metrics, events, reports, config snapshot, environment, git, manifests, checkpoint meta,
  run log, GPU samples, reload check, both readiness reports): `kerem_vm_benchmark_20260929/` next to this report.
  It was copied off the disk via a small CPU helper VM after L4 restarts hit a stockout.
- The benchmark checkpoints (~1 GB each, NOT promotable) were deleted with the VM on 2026-09-29.

