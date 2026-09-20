# Conrad V2 handoff (for the next agent or session)

Written 2026-09-20 at commit `df0dc78`. Read this first, then verify everything against the repository.
Nothing here is a substitute for inspecting the actual state.

## 0. Before you do anything

```
cd C:\Users\Kerem\Kerem\conrad-v2
git status --short
git log --oneline -20
python -m uv run conrad gates status
```

Background jobs from the previous session are dead. Their finished work may be **uncommitted in the working
tree**, and their in-flight work may be **half written**. Recover what is valid, discard what is incomplete,
and never assume a killed job finished. `python -m uv` is how Python runs here; `uv` is not on PATH.

The repository is `C:\Users\Kerem\Kerem\conrad-v2`. Do not create another repository, do not restart the
project, and do not modify the legacy repository at `C:\Users\Kerem\OneDrive\Documents\ChatGPT\conrad`
(HEAD `6bf01ee`, must stay at 0 modified files).

## 1. The rules that make results mean anything

1. A formal PASS needs the exact ch25/ch26 source criterion, on the formal execution path (Unity for I1-I7),
   on held-out data never used to design or tune the mechanism.
2. A Python-kernel result is SURROGATE evidence. It never promotes a gate (ADR-0008).
3. A green test suite is not a passed gate. A test encoding a currently failing gate is a strict xfail, and
   the Unity recorder reads an xfail as FAIL.
4. Once a final partition's results have been seen, those seeds are SPENT. Declare a new digest-pinned final
   partition, freeze the design, run once.
5. Never change a threshold, a baseline, a test or a world distribution after seeing a final outcome in order
   to manufacture a PASS. A failed research mechanism is an acceptable result; a fabricated PASS is not.
6. Truth never reaches the runtime planes. `tests/leakage` enforces it.
7. Never fabricate a physical measurement. Unmeasured values stay OPEN or SYNTHETIC_ONLY.
8. Only one Unity player at a time, run sequentially. Parallel players caused a bridge timeout.
9. No em dashes in written deliverables (the user's standing preference).

## 2. Gate state at `df0dc78`

| Gate | Official | Note |
|---|---|---|
| P0, I0, C1, 2S-FIRST, U0 | PASS | formal |
| I1 | PASS | Unity, world 7800000, re-recorded after the safety fix |
| I2 | PASS | Unity, NAV seeds 7400001-7400006, 7 criteria incl. fault safe states |
| 2T | PASS | FINAL-3 seeds 6500000-6500059 |
| I3 | PASS | Unity, world 7800001 |
| 2E | PASS | FINAL-3 seeds 6600000-6600019 |
| 2T-TCDP | FAIL | killed; propagation off in production (ADR-0009) |
| 2E-CEFD | FAIL | killed; coupling off (ADR-0007). The redesign removed unsupported claims (30/80 -> 0/80) |
| I4 | FAIL | Unity, 12 worlds, 48 flights: beats random and coverage, not fixed views |
| I5 | BLOCKED_UPSTREAM, surrogate FAIL | nominal "continue" unreachable: Model2T coverage 0.6-0.7 vs 0.8 needed |
| I6 | BLOCKED_UPSTREAM, surrogate PASS | Unity harness written; needs a player rebuild and one command |
| I7 | BLOCKED_UPSTREAM, surrogate PASS | all 4 criteria pass in the kernel; needs Unity missions |
| I8, I9 | BLOCKED_EXTERNAL | onboard computer; physical vehicle. Not software-passable |

Health at `df0dc78`: ruff, ruff format and `mypy conrad tests` clean; full suite 1139 passed, 1 skipped,
3 strict xfails (the I4 and I5 gate failures).

## 3. What was in flight when the session ended

Two workstreams were running and their state must be checked before continuing:

1. **Model2T iteration 4** (coverage per view, intact-band calibration, per-cell/locus sizing, stop the
   re-report loop). Look for `conrad/domains/technical/*` changes, `tests/unit/domains/technical/test_m2t_iteration4.py`,
   `configs/eval/2t_e00*_r4*.yaml` and an "Iteration 4" section in `docs/audits/MODEL2T_REPAIR.md`.
   The declared fresh 2T final range is **6700000-6700059**. If its final run did not complete, redo it once
   after freezing. Any Model2T change invalidates 2T and I3 evidence: re-run 2T and re-record Unity I3.
2. **I4 preparation**: a predeclared occluded world family (`ACTIVE_INSPECTION_OCCLUDED_V1` style), its
   digest-pinned splits (range 8000000+), and the missing planner baselines (geometric NBV, entropy NBV,
   conventional EIG). Look for `docs/audits/I4_WORLD_FAMILY.md`, a new `configs/eval/partitions_*.yaml`, and
   changes under `conrad/active/`. The final run was deliberately NOT started.

Also uncommitted at handoff: TCDP's rejected-ablation code in `conrad/domains/technical/{tcdp,config}.py`
and the iteration-4 doc sections. It was held back only because Model2T was editing the same package.

## 4. The plan, in dependency order

1. **Finish Model2T iteration 4.** Fix coverage per view, the intact band and per-cell sizing on DEV and
   VALIDATION worlds; freeze; run 2T E001-E004 once on 6700000-6700059; re-record 2T formal; re-record Unity
   I3 (`python -m uv run python scripts/record_unity_gate_evidence.py I3`). Report the I5 development-seed
   check (7500000-7500009): does the nominal continue warrant become reachable?
2. **I4.** Finish the occluded family and the planner set; select on DEV/VALIDATION only; freeze the winner
   (if a conventional EIG or NBV planner wins, make it the production MCBR: MCBR is a contract, not one
   ranker); then run the final once, in Unity, on the pinned held-out worlds, under matched budgets against
   fixed, random and coverage-only. Report paired CIs. Directionality must be positive to claim a win.
3. **I5 formal**, once I4 passes: the action matrix already passes formally; the integrated-mission criteria
   need Unity missions. Its fresh final seeds were 7900000-7900009 (SPENT: declare new ones).
4. **I6 formal**: rebuild the player, then `record_unity_gate_evidence.py I6`. The harness exists.
5. **I7 formal**: the same missions through Unity across 100/50/10/1/0.1 % bandwidth plus outage and
   reconnection, against raw, FIFO, fixed priority and value-per-bit. A reduced sweep is acceptable only if
   declared as reduced.
6. **Finish the report**: `docs/reports/CONRAD_V2_REMEDIATION_REPORT.md` has three placeholders left,
   `PENDING_CI`, `PENDING_FLAGSHIP` and `PENDING_VERDICT`. Fill them from artifacts, not from memory.
   The flagship is done: see `docs/audits/FLAGSHIP_UNITY.md`.

## 5. Spent and contaminated seeds (never reuse as independent evidence)

| Range | Owner | Status |
|---|---|---|
| 2026201 | MCBR | CONTAMINATED (tuned on it) |
| 5300000-5300059 | mission final_test | SPENT (2T-R2, MCBR-E003, COM-I7-E001/E002) |
| 5500000-5500004 | I7 surrogate v2 (`configs/eval/partitions_i7_v2.yaml`) | declared 2026-09-20 for COM-I7-E003/E004 after the BAAC scheduling repair |
| 6300000-6300011, 6400000-6400019, 6600000-6600019 | 2E | SPENT |
| 6500000-6500059 | 2T FINAL-3 | SPENT |
| 6700000-6700059 | 2T FINAL-4 | declared, check whether it was consumed |
| 7300001-7300006 | old I2 NAV | CONTAMINATED |
| 7400001-7400006 | NAV final | used by the current I2 evidence |
| 7600000-7600009, 7900000-7900009 | I5 finals | SPENT |
| 7800000, 7800001 | I1, I3 | current evidence |
| 7800002-7800013 | I4 | SPENT (the failed formal run) |
| 7800014-7800016 | I6 surrogate | used |
| 7800017 | flagship | used |
| 8000000+ | I4 occluded family | reserved by the in-flight work |

Partition files are digest-pinned in `conrad/evaluation/partitions.py`; a changed file is refused. Add a new
versioned file rather than editing an existing one.

## 6. Commands

```
python -m uv run ruff format --check .
python -m uv run ruff check .
python -m uv run mypy conrad tests
python -m uv run pytest -q                      # add --ignore=tests/unity_live to skip the player
python -m uv run conrad gates status
python -m uv run python scripts/record_gate_evidence.py <GATE> [--surrogate]
python -m uv run python scripts/record_unity_gate_evidence.py I1 I2 I3 I4 I6
python -m uv run python scripts/build_requirements_ledger.py
python -m uv run python scripts/falsify_truth_removal.py <bundle>
python -m uv run python scripts/run_flagship_unity.py
```

Unity player build (only when the C# changed):

```
"C:/Program Files/Unity/Hub/Editor/6000.5.9f1/Editor/Unity.exe" -batchmode -nographics -quit \
  -projectPath unity/ConradUnityV2 \
  -executeMethod Conrad.UnityV2.Editor.BuildScript.BuildWindows64Player \
  -logFile unity/ConradUnityV2/Logs/build.log
```

## 7. Where the evidence lives

- Gate evidence: `artifacts/gates/<gate>/evidence_{formal,surrogate}.json`, the source of truth for status.
- Experiments: `artifacts/experiments/<ID>/`, registry `artifacts/registry/experiments.jsonl`.
- Unity gate bundles: `artifacts/unity/gate_runs/` (gitignored, local only).
- Audits: `docs/audits/` (remediation audit, per-subsystem repairs, flagship, falsification, data procurement).
- Report: `docs/reports/CONRAD_V2_REMEDIATION_REPORT.md`.

## 8. Known open items

- Zero thrust is not a hover: the vehicle is positively buoyant and rises at about 0.045 m/s with thrusters
  off. A depth-holding safe-hold variant that keeps only the vertical loop is the better answer, and is OPEN.
- `same_player_binary` is false for gate bundles recorded before the player rebuild; I1 and I2 were
  re-recorded, I3 and I4 were not.
- The ECMER quality head is untrained, so no reliability claim is allowed.
- Real data: only smoke-level evidence (UVVID, SubPipe). No D6 anywhere.
- External blockers: see section N of the report.
