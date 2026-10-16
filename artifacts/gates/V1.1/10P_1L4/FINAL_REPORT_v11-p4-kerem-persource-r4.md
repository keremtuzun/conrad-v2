# V1.1 P4 on 1x L4, round 4: per-source sonar

Final run `v11-p4-kerem-persource-a3-20261016T001909Z`. It finished 2026-10-16 at 10:00 Turkey time.

## Decision

**NO-GO on "every sonar source >= 75 on its own"**, because SubPipe is at 56. The other measures:
- 2 sources pass clearly: Catalunya and China offshore.
- UATD passes at 78.8.
- AquaScan is at 59 against a ceiling of 70, which raw pixels cannot exceed on its 103 held-out frames.
- **The pooled gate still passes**: sonar 152.6 and camera 97.5.
- Training is valid: no NaN, no stalls and no watchdog stops. The meaning score stayed at 0.37 or above at every validation, and both checkpoints reload strictly.

The trainer's own report says **NO-GO**. With `KEREM_SELECT=per_source`, its gate is the weakest sonar source (SubPipe 56) and the pooled families, so this round fails its own gate. AquaScan is left out of the gate and reported separately (below).

## Per-source result against the ceiling check

All values are effective rank on the same up-to-256 distinct held-out frames used by `final_checks.py`.

| Sonar source | Raw pixels (ceiling) | Pixels blurred to 16x16 | Untrained V1.1 | Round 3 best | **Round 4 best** | Round 4 last |
|---|---|---|---|---|---|---|
| Catalunya | 184.2 | 99.1 | 77.8 | 39.4 | **133.9** | 141.6 |
| China offshore | 233.6 | 171.7 | 91.9 | 96.6 | **128.5** | 135.9 |
| UATD | 155.2 | 77.4 | 58.3 | 33.1 | **78.8** | 78.1 |
| AquaScan (103 frames) | 70.2 | 58.6 | 38.8 | 46.4 | **58.7** | 65.0 |
| SubPipe (LF+HF mixed) | 155.3 | 73.5 | 45.8 | 24.5 | **56.3** | 56.7 |
| SubPipe, LF stream only | 166.3 | 73.9 | 50.7 | | 31.3 | 32.1 |
| SubPipe, HF stream only | 142.5 | 61.9 | 41.9 | | 38.2 | 37.7 |
| pooled sonar | 161.9 | | 61.0 | 109.0 | **152.6** | 162.1 |
| pooled camera | 77.7 | | 16.7 | 96.8 | **97.5** | 95.5 |

Meaning score (held-out temporal nearest-neighbour hit, sequential sources):
- Sonar: 0.385 for best.pt, against 0.357 for raw pixels. Round 3 was 0.373.
- Camera: 0.682, against 0.898 for raw pixels. Unchanged from round 3.

Sources: `ceiling_before.json`, `ceiling_after.json`, `representation_health_heldout.json`, `subpipe_diag.json` and
`denoise_diag.json` (blurred-pixel ceilings).

## What was wrong, and what fixed it

1. **Mixed-source sonar batches (round 3).** VICReg's variance and covariance terms were satisfied by telling the five datasets apart. Three sources ended *below* the untrained network.
   - Fix: `KEREM_SONAR_BATCH=single`. Each sonar step draws its whole batch from one source, round robin.
   - In 16k steps, Catalunya went 39 -> 126 and UATD 33 -> 62.
2. **Large crops on sonar.** `KEREM_SONAR_VIEWS=sonar` keeps at least 65 % of the frame and uses stronger speckle.
   - It added +8 Catalunya, +7 UATD, +6 SubPipe and +3 AquaScan over fix 1 (pilot P2).
3. **SubPipe is two sonars.** At about step 13k, its mixed rank of 46.7 came only from separating LF from HF.
   - Inside each stream it was 27.5 (LF) and 36.2 (HF), below the untrained network (50.7 and 41.9).
   - Fix: `KEREM_STREAM_BATCH_SOURCES=subpipe` (one stream per batch) and `KEREM_SONAR_WEIGHTS=subpipe:2`.
   - SubPipe rose 47 -> 54 in an 8k-step pilot, and the within-stream ranks to 31 and 38.

**Tried and not kept:**
- *Temporal neighbour pairs* (`KEREM_TEMPORAL_PAIRS=3`, pilot P3). It gave the best meaning score (0.42) but lower ranks.
- *SubPipe-only gentle crops and less speckle at weight 4* (`KEREM_SOURCE_VIEWS=subpipe=0.85/0.10`, pilot T). It gave +2 only, below the +3 bar.

## Why SubPipe stops near 56

- **Full-resolution pixels overstate SubPipe's content.** Its pixels reach 155, but most of that is fine speckle and texture. Blurred to 16x16, the held-out frames reach only 73.5, and the HF stream alone reaches 61.9. At the 64x64 input the model sees, SubPipe has barely 75 worth of coarse content. A model at 75+ would have to encode fine speckle, which would raise the number without meaning.
- **The model is still compressed inside each stream** (31 LF, 38 HF). Over 80k further steps, SubPipe stayed flat between 47 and 58.
- **Honest next options (not done):**
  - A 128x128 input. This is a new input dimension, so there is no warm start.
  - A masked-patch reconstruction term that rewards within-frame detail.
  - Either is a fresh run of about a day and roughly TRY 1,000-1,500, with no guarantee.

## AquaScan

AquaScan has 1,033 images, split per recording into 723 train / 103 validation / the rest held back as test. On the 103 validation frames even raw pixels score 70.2. The test split was **not** used to enlarge the evaluation.
- AquaScan is therefore excluded from best.pt selection (`KEREM_SELECT_EXCLUDE=aquascan`).
- It is reported against its own ceiling: 58.7 at best.pt and 65.0 at last.pt, against 70.2.

## Checkpoint lineage and reload

The run chain:
1. Round 3 best.pt (step 451k, `edf74d95...`).
2. Pilot P2 (16k steps).
3. a1, stopped at 14k for the SubPipe fix.
4. Pilot S (8k).
5. a2, stopped at 12k because SubPipe was flat. Its best.pt is step 3k, `0ec3e518...`.
6. Pilot T, not used.
7. **a3, 80k steps from a2 best.pt.** Its best.pt is at step 24k.

Each warm start loads model weights only (strict), with a fresh optimizer; the source checkpoint's sha256 is in `reports/init_from.json`.

| | Step | Strict load | All finite | Params | Optimizer state |
|---|---|---|---|---|---|
| best.pt | 24,000 | yes | yes | 308,055,809 | yes |
| last.pt | 80,000 | yes | yes | 308,055,809 | yes |

The checkpoints are on the Mac in `artifacts/runs/v11-p4-kerem-persource-a3-20261016T001909Z/checkpoints/` (not pushed):
- best.pt sha256 `a2992555f01538250aade9c0c4658bae01a6b66451c679eeb1727b5afb5f2ab3`
- last.pt sha256 `47a747e96ab0eb20f2dfc5cf1ff8008931635b6ce1e82720488f269bfbe17db6`

## Non-contamination

- The trained commit is `a7b3d66` (clean tree) on `kerem/v11-p4-patch-vicreg-trainer`.
- Changes since round 3:
  - `conrad/foundation/universal_v11/trainer.py`, with opt-in env switches only; the defaults reproduce round 3.
  - `scripts/kerem_p4/` (ceiling_check.py, orchestrate_persource.sh, orchestrate_stage2.sh, vm_setup_r4.sh, final_checks.py).
- 0 files changed in `conrad/oceansense` (P5-P10), `conrad/foundation/pretraining`, `conrad/foundation/encoders` or `configs/train/osfm`.
- The P4.8 upstream checkpoint is unchanged (`b2dfc162...`).
- `osfm-universal-v1.1` is untouched.
- The trainer is still the unreviewed Kerem variant, launched with the recorded readiness override.

## Cost (TRY, estimated)

| Item | TRY |
|---|---|
| VM conrad-p4-l4d (g2-standard-8 + L4, asia-northeast1-a), about 17.5 h at about 52/h | about 910 |
| Disks and egress (3.2 GB of checkpoints to the Mac) | about 40 |
| **Round 4 total** | **about 950** |
| **All rounds** | **about 6,950** (cap 12,000) |

L4s were stocked out in 14 US, EU and Taiwan zones, so the VM ran in Tokyo.

## Evidence

`persource_r4_20261015/` holds `final_bundle.tgz` and the unpacked folder:
- Final run: metrics, reports, config, git and environment.
- Pilots P1-P3, S and T: metrics and reports.
- Stopped runs a1 and a2: metrics and reports.
- The ceiling checks before and after, the SubPipe diagnostics, the orchestrator log, watchdog logs and GPU samples.

Hourly progress is in `PROGRESS_LOG.md`.
