# V1.1 P4 on 1x L4: training attempts, 2026-09-29

Decision: **NO-GO. No long run was kept, and no promotable checkpoint exists.**
Spend: about **205 TRY** in total (VM 1 about 50 TRY; VM 2 about 101 + 54 TRY). VM 2 is stopped, so only its 150 GB disk is billing (about 1.1 TRY/h).

## What was run (all on one NVIDIA L4, us-east4-a, g2-standard-8)

All runs use the pinned repo 8885291 plus the Kerem trainer commits on the local branch
`kerem/v11-p4-patch-vicreg-trainer` (not pushed). Every run is launched through
`conrad train osfm-v11-10p-launch --kerem-override-unreviewed-trainer`, which records the readiness
override (blockers 2/4/8: not the reviewed trainer or branch) in the run manifests.
Payload hashes were verified in every run.

Validation effective rank on distinct held-out SubPipe frames (gate: 75):

| Run | Change | Step 1 | 1k | 2k | 3k | 4k | 5k |
|---|---|---|---|---|---|---|---|
| reviewed trainer (VM 1) | as delivered, 52 frames | 3.8 | 4.1 | | | | |
| all-frames run | 16 real patches, VICReg, 2,066 frames, depth 2 | 97.6 | 16.1 | 13.0 | | | |
| pilot 1 | + GPU random-resized crops | 86.8 | 22.5 | 11.4 | 11.0 | 9.7 | 9.0 |
| pilot 2 | + per-frame contrast normalisation, light/strong views | 88.8 | 25.0 | 11.7 | | | |
| pilot 3a | full 307M depth, lr 3e-5, wd 0.1 | 81.6 | 58.7 | 22.7 | 16.0 | 14.5 | |
| pilot 3b | full 307M depth, big blocks frozen (stage-A style) | 83.9 | 44.5 | 20.8 | | | |

In every variant, validation loss falls briefly and then rises after warm-up, while training loss keeps falling.

## Findings

1. **The reviewed trainer could not pass.** It fed 32 row-band means per frame and used every 16th frame
   (46 train and 6 validation frames), so a 256-sample validation batch held only 6 distinct frames.
2. **The reviewed trainer trains a 201M model, not the 307M V1.1.** It uses `FamilyEncoderConfig(depth=2)`;
   the project default is depth 6 (307.8M).
3. **An untrained V1.1 already passes the rank gate** (rank 82-98 on held-out frames). Training is what
   lowers it. The rank-75 gate therefore rewards "not destroying random diversity" more than learning.
4. **Root cause: far too little data.** SubPipeMini2 has 2,066 side-scan frames (1,445 train) from one
   mission. 10P is 1.2M steps x 256 samples, about 150,000 passes over the training frames. Every variant
   memorises the training block and folds the later, lower-contrast held-out block
   (pixel std 0.022 vs 0.060) onto a few directions. Diagnostic at pilot 1 step 5k: clean-train rank 45.7,
   clean-val rank 9.0.
5. **The upstream checkpoint import loads 0 tensors.** Its key names don't match V1.1.
6. **Throughput is not the blocker.** Depth 6 runs at about 3.5-3.7 steps/s with bf16 and GPU crops.
   1.2M steps would take about 95 h, which still exceeds 72 h on 1 L4.

## What would actually fix it

More data-backed ocean data, not more steps. The repo already has manifests for candidates:
- `underwater_caves_sonar` (Zenodo 7828405, sonar, training_allowed: true, not yet downloaded or verified)
- `uvvid` (underwater visual/visual-inertial, training_allowed: true, VERIFIED)
- `seaclear` (training_allowed: true, not verified)
- The full SubPipe release (28 GB, adds the camera streams)

Each needs the project's procurement and hash-verification step and a loader, and it adds encoder families
beyond active acoustic. That is the curriculum the 10P config describes.


## Round 2: the full SubPipe release (sonar + camera)

On Kerem's approval, the full `SubPipe.zip` was downloaded from the same Zenodo record and licence
(CC-BY-4.0, 28,011,923,314 bytes). Its md5 `0af8d8231a4c09b2b3303e44765ae8d6` matches Zenodo, and its
sha256 `7723d93d71a8eca8cdd354fc07f95f433a7cdbd42143d38ca3e362ceab85ae3e` is recorded in the new
`datasets/public/subpipe_full.manifest.yaml`. Data used:
- imaging_sonar: 6,868 train and 981 validation frames (sss_lf + sss_hf, chunks 0-4)
- rgb_camera: 32,279 train and 4,611 validation frames (cam0 at stride 2, plus cam1)

Both runs use the full 307M model (depth 6), all parameters trainable, lr 1e-4. Odd steps train sonar and
even steps train camera. Every family is gated at validation rank 75.

| Run | Change | Step | 1 | 1k | 2k | 3k | 4k | 5k | 6k |
|---|---|---|---|---|---|---|---|---|---|
| pilot 4 | sonar + camera | sonar rank | 42.3 | 29.9 | 27.5 | 25.2 | | | |
| | | camera rank | 15.6 | 26.3 | 30.1 | 32.0 | | | |
| pilot 5 | + clean anchor view | sonar rank | 47.9 | 40.3 | 35.8 | 32.0 | 33.3 | 32.6 | |
| | | camera rank | 15.3 | 24.9 | 30.9 | 29.0 | 32.0 | 31.0 | 32.0 (final best) |

Pilot 5's own report: NO-GO, best rank 31.98.

### Why it fails: a generalisation gap, not a training bug

Clean-frame effective rank on 256 distinct frames (`diagnostics/p4_diag2.py`):

| | Pixels | Untrained model | Pilot 5, step 5000 |
|---|---|---|---|
| sonar, train frames | 200.0 | 70.0 | **77.9** (passes 75) |
| sonar, held-out frames | 152.9 | 50.7 | 34.2 |
| camera, train frames | 49.1 | 14.7 | **85.8** (passes 75, above its pixel rank = memorisation) |
| camera, held-out frames | 34.3 | 16.4 | 32.0 |

The objective works on the frames the model trains on. Held-out frames come from a later, contiguous
block of the single SubPipe mission, and on those the representation stays near 32-34. Longer training
widens the gap: sonar validation loss rose from 46 to 90 over pilot 5. Camera held-out frames only carry
pixel-space rank about 34, so a camera rank of 75 on this data would mean inventing diversity that isn't there.
An interleaved (random-frame) validation split would pass the gate only through near-duplicate leakage,
so it was not used.

### Conclusion

On the data available (one AUV mission), no honest variant of the training reaches the project's held-out
rank-75 gate for any family, and a 60-72 h run would only deepen memorisation. **Decision: NO-GO for
the long run.** Reaching the gate needs data from other missions and sites: independent sonar and
camera recordings, not more frames of the same dive.

## Evidence

`kerem_pilots_20260929/` next to this file holds, for every run (both rounds), the metrics, events, reports
(including trainer_variant.json, data_coverage.json and upstream_import.json), the resolved config,
git.json, manifests, checkpoint metadata, the launch/probe logs and the watchdog logs.
