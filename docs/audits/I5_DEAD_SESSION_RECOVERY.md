# I5 dead-session recovery (2026-09-22)

This note records the state recovered before any checkout, reset, cleanup, E005 rerun, Unity rerun, or I5
semantic repair. The recovery target is the local primary repository at `C:\Users\Kerem\Kerem\conrad-v2`.
The legacy repository was not modified.

## Repository state

- Local branch: `main`.
- Local HEAD: `8f210f1288edb323d1f2eca83372482bd3f073c8` (`Legacy independence re-executed at the end of the pass: PASS`).
- Fetched `origin/main`: `8f210f1288edb323d1f2eca83372482bd3f073c8`.
- Divergence after fetch: 0 commits local-only, 0 commits remote-only.
- Staged files: none.
- Local commits not pushed: none.
- Running Python/Unity experiment processes: none. The only matching process during inspection was the recovery PowerShell itself.

The dead session left ten tracked files modified:

- `artifacts/gates/I5/evidence_formal.json`
- `artifacts/gates/I5/evidence_surrogate.json`
- `artifacts/gates/I5/unity_measured.json`
- `artifacts/registry/experiments.jsonl`
- `configs/eval/i5_unity.yaml`
- `docs/audits/I5_ACTION_MATRIX.md`
- `scripts/check_i5_unity_harness.py`
- `scripts/record_gate_evidence.py`
- `scripts/record_unity_gate_evidence.py`
- `tests/acceptance/test_i5_integrated_missions.py`

It also left untracked E005, Unity-development, and pytest artifacts:

- `artifacts/experiments/M1-ACTION-E005/`
- `artifacts/experiments/I5-UNITY-HARNESS-CHECK/unity_development.json`
- `artifacts/experiments/I5-UNITY-HARNESS-CHECK/unity_development_s7710100_nominal.json`
- `artifacts/gates/I5/unity_pytest.txt`
- `artifacts/gates/I5/unity_pytest.xml`

The Unity I5 run bundles under `artifacts/unity/gate_runs/I5/` are ignored runtime artifacts and therefore do
not appear in `git status`, but they are material recovery evidence and were inspected in place.

## Recovered E005 final run

`artifacts/experiments/M1-ACTION-E005/dispatch_record.json` and
`artifacts/experiments/M1-ACTION-E005/m1_action_e005.json` show a completed 3904.4-second run of all declared
final seeds `7720000` through `7720009`, all eight scenarios, and all three arms (240 missions). The result was
written and then read into the modified surrogate evidence, acceptance test, registry, and chronological I5
audit. File SHA-256 values are:

- `dispatch_record.json`: `a60aa1ecf6136c30a9efaed9d54f46102d5532319a039aba1a0b275d0e19b5bd`
- `m1_action_e005.json`: `ebb310fc09317c2396d095c519a218f25a21ae67fe3d97370ed080fa51228a81`

The final result is a valid retained record of iteration 4 at pre-repair HEAD, but it is not a clean partition for
post-repair evaluation. The result reports correct-given-warrant 69/69, route-blocked 10/10, 3910/3910
traceable decisions, UIR 0 over 680 relied claims, zero hard-constraint violations, and competitive pooled
outcomes. It also reports I5-NOMINAL 0/10 and two nominal over-escalations. Accordingly:

- E005 output salvaged: yes, as a spent pre-repair surrogate result.
- Final seeds executed: yes, all `7720000-7720009`.
- Final outcomes written and observed: yes.
- Partition status for any post-repair final: **SPENT**. It must not be rerun or represented as independent.

## Recovered Unity development work

The dead session left a full 21-flight development sweep for world `7710101` and a single-scenario development
artifact for world `7710100`. The chronological audit records the quiet-host `7710101` sweep as complete, with
six of seven EGDC scenarios scoring 1.0, zero over-escalations, zero violations, 376/376 traceable decisions,
UIR 0 over 101 relied claims, replay success, and a zero-hit runtime truth-leakage scan. These are development
diagnostics only and cannot promote I5.

## Recovered formal Unity flight

A formal Unity I5 run did occur. It did **not** complete.

The declared grid was 2 final worlds x 7 Unity scenarios x 3 arms = 42 sequential flights. Recovered state:

- World `7710002`: all 21 flight bundles completed.
- World `7710003`: 13 flight bundles completed, one flight started and remained incomplete, and seven flights
  never started.
- Completed bundles with both `reports/i5_score.json` and `bundle_manifest.json`: 34.
- Incomplete bundle: `I5-UNITY-I5-ROUTE-BLOCKED-s7710003-rule_fsm`.
- Never completed/started after that point: route-blocked/naive, all three time-reserve arms, and all three
  comms-outage arms for world `7710003`.

The incomplete flight failed while requesting `GET_GROUND_TRUTH` from the player:
`UnityBridgeTimeout: no reply from Unity within 30000 ms`. The retained pytest transcript reports 7 action-matrix
tests passed and 6 integrated-mission tests errored after 3776.90 seconds. It does not support a formal I5 verdict.
`artifacts/gates/I5/unity_i5_results.json` was never produced.

The wiring record identifies the player used for the recovered flights as Unity `6000.5.9f1`, build output
`Builds/Win64/ConradSim.exe`, SHA-256
`36c5c9f13481406382a8e9ef8fc0ea7cdf055c43bb12fc8fd545b07c199cd277`, successful build start
`2026-09-20T03:51:10.1367628Z`. The executable is no longer present at that path, so this identity is retained
metadata rather than a current-player verification.

The 34 complete bundles are salvaged as historical, spent partial-grid evidence. The incomplete route-blocked
bundle is retained for failure diagnosis but rejected as a scored flight. The modified `evidence_formal.json`
is likewise retained as a dead-session recorder attempt, not accepted as formal closure: its final three FAIL
rows reflect missing aggregate results after a fixture error, not a complete 42-flight comparison.

Both `7710002` and `7710003` were executed and their outcomes were observable. Therefore both are **SPENT** and
must not be reused as fresh post-repair formal worlds. A later formal I5 evaluation requires a new, disjoint,
versioned and digest-pinned partition frozen before any result is read.

## Recovery disposition

Salvaged:

- Complete E005 surrogate artifact and dispatch record, as spent pre-repair evidence.
- E005 registry/evidence/test/audit edits, pending code review and a focused commit.
- Unity development diagnostics, as development-only evidence.
- 34 complete formal Unity bundles, as historical partial-grid evidence only.
- Pytest text/XML and the partial formal-evidence recorder output, as failure/recovery records.

Rejected as complete evidence:

- The incomplete `I5-UNITY-I5-ROUTE-BLOCKED-s7710003-rule_fsm` bundle.
- Any claim that the 42-flight formal grid completed.
- Any formal I5 PASS or FAIL inferred from the dead session's incomplete grid.
- Reuse of `7720000-7720009`, `7710002`, or `7710003` as fresh post-repair evidence.

No recovered file or process was deleted. No experiment was rerun during recovery.
