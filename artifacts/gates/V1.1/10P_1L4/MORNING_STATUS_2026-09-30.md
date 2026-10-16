# V1.1 P4 long run: overnight status (2026-09-30)

**A training run is running.** No action is needed from you.

| | |
|---|---|
| Run | `v11-p4-kerem-long-a1-20260930T001158Z` |
| Machine | `conrad-p4-l4b`, GCP us-central1-a, 1x NVIDIA L4 (g2-standard-8) |
| Started | 2026-09-30 03:12 Turkey time |
| Planned length | 775,000 steps (about 6.5P), learning rate 5e-5 |
| Speed | about 3.3 steps/s, GPU at 90%+ |
| Projected finish | about **Thu 2 Oct, ~20:00-22:00 Turkey time** |
| Projected cost | about 2,800 TRY for this run. About 425 TRY has been spent on everything before it. The total stays within 8,500 TRY and the free trial. |
| When it ends | The machine powers itself off, so billing stops |

## What happened overnight

1. **More ocean data from other sites** (CC-BY, rights-cleared, hash-verified against their manifests):
   full SubPipe (sonar + cameras), SeaClear (8,610 images from 5 sites), and UVVID ROV videos 1-8.
2. **The P4.8 sonar encoder is actually used.** The upstream checkpoint's 150 tensors are now loaded into
   V1.1's V1 compatibility bank, which is kept frozen, and sonar runs through it (config stage A).
3. **A first long run was started and then stopped at step ~17k** because it was gaming the rank metric.
   Held-out sonar rank jumped 50 -> 154 while the meaning check fell to chance. The meaning check asks
   whether a held-out frame's nearest neighbour is one of its adjacent frames in time: 0.61 -> 0.06.
   Its "best" checkpoint was that broken step. It is not used.
4. **The fix, now running:**
   - plain VICReg without the rank term
   - half the learning rate
   - the meaning check runs at every validation
   - a checkpoint can only become `best.pt` if every family scores >= 0.3 on it (chance is about 0.05)
   - the watchdog treats a meaning score below 0.15 on 3 validations in a row as representation collapse,
     and the orchestrator then relaunches once at half the learning rate. You won't wake up to nothing running.

## The new run so far (held-out data)

| Step | Sonar rank | Sonar meaning | Camera rank | Camera meaning |
|---|---|---|---|---|
| 1 (untrained) | 60.9 | 0.52 | 15.8 | 0.56 |
| 1,000 | 27.5 | 0.74 | 44.6 | 0.73 |
| 2,000 | 28.7 | 0.69 | 56.7 | 0.75 |
| 3,000 | 31.5 | 0.70 | 63.9 | 0.72 |
| 4,000 | 37.2 | 0.68 | 68.5 | 0.70 |
| 5,000 | 36.6 | 0.69 | 74.0 | 0.71 |

Both ranks are climbing while the meaning score stays well above the untrained model's. That is the
healthy pattern.

## Honest expectations

- **Camera** is close to the rank-75 gate. Caveat: the pooled camera validation mixes 3 sites, and part of its
  rank comes from differences between sites. A random sample of only SubPipe-dominated frames scores lower (~45).
- **Sonar** will most likely stay under 75. The frozen P4.8 encoder scores about 63 on this held-out
  split, and sonar recordings from other missions would be needed to go higher.
- The rank-75 gate is reported, not a kill switch. NaN/Inf, collapse (variance or meaning), stalls, GPU loss,
  72 h and cost all still stop the run.
- Final decision: CONDITIONAL-GO only if every family reaches 75 with a meaningful checkpoint; otherwise
  NO-GO on the rank gate, with the per-family numbers stated. Scale: about 6.5P, not 10P.
- Trainer: Kerem variant (`reviewed_trainer: false`, recorded owner override). Code is on the local branch
  `kerem/v11-p4-patch-vicreg-trainer`, not pushed.

## Where to look

- On the VM: `~/p4logs/longrun.log` (orchestrator), `~/p4logs/watchdog-<run>.jsonl`, and
  `~/conrad-v2/artifacts/runs/<run>/metrics.jsonl`
- In this folder: `kerem_training_attempts_report_2026-09-29.md` and `kerem_pilots_20260929/`
