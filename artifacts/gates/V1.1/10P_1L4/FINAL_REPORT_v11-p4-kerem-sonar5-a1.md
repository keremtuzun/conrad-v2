# V1.1 P4 on 1x L4, round 3: final training report

Run `v11-p4-kerem-sonar5-a1-20261004T165839Z`. Finished 2026-10-07 08:46 Turkey time.

## Decision: pooled rank gate PASSED; per-source sonar NOT passed

| Gate | Result |
|---|---|
| Steps completed | **730,000 / 730,000** (about 6P; 10P = 1.2M does not fit 72 h on one L4) |
| Wall clock | 60.8 h (limit 72 h) |
| NaN/Inf, variance collapse, stalls, GPU loss, cost or time limit | none in 731 validations; the watchdog's only action was "trainer exited" |
| Meaning guard (held-out temporal nearest-neighbour hit >= 0.3 after warm-up) | passed at every validation; minimum 0.35 |
| Checkpoint write and strict reload | passed (best and last) |
| **Rank >= 75 for every family, pooled held-out (the trainer's gate)** | **passed: sonar 109.4, camera 111.0 at best.pt** |
| Rank >= 75 for every sonar source on its own (not part of the gate) | **failed: only China offshore (97-99) passes** |

The trainer's own report says **CONDITIONAL-GO** (`v11_10p_training_report.json`). It is conditional because the
trainer is the Kerem variant (`reviewed_trainer: false`), launched with the recorded readiness override.

Round 2 failed the same gate with sonar at 61.4, so this round passes the gate as written. **The gate is passed
by the pooled number, and the pooled number is mostly earned by telling the five sonar datasets apart.** Inside any
single dataset the representation is narrower than the untrained network's. Read the "Honest reading" section
before treating this as a promotable 10P checkpoint.

## What changed from round 2

- **Sonar data: 1 source -> 5.** SubPipe side-scan (as before) plus Catalunya SSS (split zip64 read in place,
  every 6th frame), China offshore SSS, AquaScan 1k and UATD. All are CC-BY-4.0 or compatible, hash-checked
  against the publishers' md5 and recorded in `datasets/public/*.manifest.yaml`.
- **Sonar front end: raw pixel patches** (pilot A). The frozen P4.8 anchor (pilot B) was tested in a 6k-step pilot
  and lost: its sonar meaning score finished at 0.28 (below the 0.3 floor) and SubPipe-only rank fell to 12. The
  P4.8 weights are still loaded 150/150 into `model.v1.sonar` but are not used as the front end.
- **Equal share per source** in every batch, per-recording contiguous train/validation splits, unreadable images
  skipped and recorded (1 corrupt AquaScan PNG), and a sha256 cache so the 434k-member archive is hashed once.
- Same model (depth 6, 308.1M parameters), objective (VICReg, clean anchor view, rank term weight 0), lr 5e-5
  with cosine decay, bf16.

## Results on held-out data

`best.pt` is step 451,000. It was selected after warm-up, only if every family's meaning score was >= 0.3, then by
the highest weakest-family rank.

| | best.pt (451k) | last.pt (730k) | untrained (step 1) |
|---|---|---|---|
| **sonar, pooled 5 sources** | **109.0** | 108.9 | 67.0 |
| sonar, China offshore only | 96.6 | 98.3 | 90.8 |
| sonar, AquaScan only | 46.4 | 45.6 | 39.6 |
| sonar, Catalunya only | 39.4 | 37.6 | 73.6 |
| sonar, UATD only | 33.1 | 32.5 | 64.5 |
| sonar, SubPipe only | 24.5 | 24.2 | 48.5 |
| sonar meaning score (sequential sources) | 0.373 | 0.372 | 0.19 |
| sonar meaning score, raw pixels | 0.357 | | |
| **camera, pooled 3 sources** | **96.8** | 95.9 | 18.0 |
| camera, SeaClear only | 70.9 | 68.5 | 37.5 |
| camera, UVVID only | 57.1 | 58.6 | 27.4 |
| camera, SubPipe only | 24.3 | 23.7 | 14.7 |
| camera meaning score | 0.695 | 0.682 | 0.60 |
| camera meaning score, raw pixels | 0.898 | | |

Per-source and pooled ranks above come from `final_checks.py` (`representation_health_heldout.json`); the
"untrained" column is the trainer's step-1 validation. During training the pooled validation rank at best.pt was
109.4 (sonar) and 111.0 (camera).

Trajectory (pooled sonar rank / SubPipe sonar rank / camera rank / sonar meaning score):

| Step | Sonar pooled | Sonar SubPipe | Camera pooled | Meaning | Train loss |
|---|---|---|---|---|---|
| 1 | 67.0 | 48.5 | 18.0 | 0.19 | 43.8 |
| 5k | 81.2 | 17.2 | 74.2 | 0.36 | 20.0 |
| 50k | 109.9 | 21.9 | 93.0 | 0.37 | 11.8 |
| 200k | 107.0 | 21.9 | 100.9 | 0.37 | 7.5 |
| 451k (best) | 109.4 | 24.3 | 111.0 | 0.36 | 6.3 |
| 600k | 111.4 | 24.5 | 104.7 | 0.36 | 5.7 |
| 730k (last) | 106.8 | 23.5 | 101.1 | 0.36 | 5.7 |

Everything was settled by about 50k steps. The other 680k steps moved no rank by more than a few points.

### Honest reading

1. **The sonar pass is a between-dataset pass.** Pooled sonar rank is 109, but four of the five datasets score
   25-46 on their own, and Catalunya, UATD and SubPipe score **lower than the untrained network** (74 -> 39,
   65 -> 33, 49 -> 24). Training made the features ignore within-recording variation (what VICReg's augmentation
   invariance asks for) while separating the datasets from each other. Only China offshore, the most varied set,
   passes 75 on its own.
2. **The meaning test shows no clear gain over raw pixels.** On sequential sonar, nearest-neighbour hit is 0.373
   for the model and 0.357 for raw pixels. On camera, the model (0.695) is worse than raw pixels (0.898). The
   features are structured, not noise, but this test does not show they are better than the input.
3. **More steps of this recipe will not fix it.** Ranks were flat from 50k to 730k. Passing 75 per source needs
   a different signal: more varied recordings per source, an objective that rewards within-recording detail
   (for example a masked-patch or temporal loss), or a gate defined on held-out downstream tasks.
4. Camera is unchanged from round 2 (pooled 94 -> 97; SubPipe-only about 24).

## Checkpoint reload (`checkpoint_reload_report.json`)

| | Step | Strict load | All finite | Params | Optimizer state |
|---|---|---|---|---|---|
| best.pt | 451,000 | yes | yes | 308,055,809 | yes |
| last.pt | 730,000 | yes | yes | 308,055,809 | yes |

Checkpoints are copied to the Mac at `artifacts/runs/v11-p4-kerem-sonar5-a1-20261004T165839Z/checkpoints/`:
- best.pt sha256 `edf74d95782e67e897cd6e60b26b4ac3f55a93ad0b609db6f0f7488117f16638`
- last.pt sha256 `b7b99a3d938b65af6e7f380e6f7b3d92db0c773372e2952cc8bb3eb066d16a7d`

## Semantic modality coverage

| Modality (family) | Source | Train frames | Held-out frames |
|---|---|---|---|
| imaging_sonar (active_acoustic) | SubPipe side-scan LF + HF | 6,868 | 981 |
| | Catalunya SSS (every 6th frame) | 50,265 | 6,726 |
| | China offshore SSS | 2,276 | 324 |
| | UATD | 6,449 | 921 |
| | AquaScan 1k (1 unreadable file skipped) | 722 | 103 |
| rgb_camera (visual_image) | SubPipe cam0 + cam1 | 32,279 | 4,611 |
| | SeaClear, 5 sites | 6,021 | 856 |
| | UVVID ROV_GoPro_1-8 | 836 | 118 |

Batches give each source an equal share, so small sources (AquaScan, UVVID) are heavily repeated.

**827 of 829** registry entries received no training data. They remain interface coverage only. No semantic claim
is made for them, or for 13 of the 15 encoder families.

## P4 V1 non-contamination

- The upstream P4.8 checkpoint was read-only: sha256 still `b2dfc162...` (`sha256.txt`).
- Trained commit `0f0f9fd` (clean tree) on branch `kerem/v11-p4-patch-vicreg-trainer`. Compared with round 2 it
  changes only `conrad/foundation/universal_v11/trainer.py`, `scripts/kerem_p4/*`,
  `scripts/make_sonar_manifests.py` and 4 new dataset manifests, plus gate evidence under `artifacts/`.
  **0 files changed** in `conrad/oceansense` (P5-P10),
  `conrad/foundation/pretraining`, `conrad/foundation/encoders` or `configs/train/osfm`.
- `osfm-universal-v1.1` on GitHub is untouched.

## Cost report (TRY, estimated from live GCP SKU prices)

| Item | TRY |
|---|---|
| Rounds 1-2 (earlier VMs, final long run, helpers) | about 3,100 |
| Round 3: data download and hashing, pilots A and B | about 160 |
| Round 3: long run (60.8 h) | about 2,675 |
| Round 3: 1 h idle before power-off, CPU helper, disks, egress | about 60 |
| **Round 3 total** | **about 2,900** (per-run ceiling 8,500) |
| **All rounds** | **about 6,000** (cap 12,000) |

## Evidence

`sonar5_a1_20261004/` (next to this file) holds `final_bundle.tgz` and the unpacked
`final_v11-p4-kerem-sonar5-a1-20261004T165839Z/`:
- config snapshot (`config.resolved.yaml`), environment, git and manifests
- `metrics.jsonl` (731 validations, with per-source ranks) and events
- `reports/` (training report, trainer_variant, data_coverage with skipped files, upstream_import)
- checkpoint metadata, reload report, held-out health check and checkpoint hashes
- readiness output (the recorded override), pilot A and B metrics
- orchestrator, watchdog and trainer logs, plus GPU utilisation samples

Hourly progress is in `PROGRESS_LOG.md`. The round 2 report is `FINAL_REPORT_v11-p4-kerem-long-a1.md`.
