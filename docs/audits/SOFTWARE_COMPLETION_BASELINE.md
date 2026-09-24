# Software completion baseline recovery

Branch `kerem/software-completion` starts from `4b2d31b`. The dirty main
checkout and legacy repository were not edited. Historical gate evidence was
not rewritten, and no spent final seed was run.

## Historical I5 acceptance artifact

`tests/acceptance/test_i5_integrated_missions.py` depends on the gitignored
`artifacts/experiments/M1-ACTION-E008/m1_action_e008.json`. A clean checkout
does not contain that record. The tracked I5 gate summaries remain historical
records, but they do not contain the complete per-run payload those tests
inspect. Reconstructing it from summaries or rerunning spent final seeds would
fabricate a verification result.

The artifact-dependent tests now report explicit `historical I5 surrogate
evidence unavailable` skips when the file is absent. If the artifact is
present, the original assertions still run. Setting
`CONRAD_REQUIRE_HISTORICAL_I5=1` makes absence fail with a clear message for a
strict historical-evidence verification. A clean checkout produced 1 passed,
11 skipped; the strict single-test check failed as intended. Neither result
changes the stored I5 surrogate or formal verdict.

## Twin2T seed 401

The reported seed was reproduced with `CONFIGURED`, a 4,919,074 s tick
containing `REINFORCEMENT` then `IMPACT`, and a subsequent 1 s tick. The
component's original wall was 0.009283234150824882 m and the reinforced wall
was 0.014283234150824883 m. Impact left a 0.00928323715758941 m crack depth.
The next fatigue tick clipped that depth to the original wall, a decrease of
about 3.0e-9 m without intervention. This was a real legacy mechanism defect,
not a change caused by Spatial V1.

Fatigue now receives the runtime's effective wall thickness. Its maximum
crack-length clamp also retains an existing crack if the configured maximum
is below the existing state. A deterministic seed-401 regression and the
mechanistic property suite pass.

## Baseline verification

- Technical, Twin2T unit/property, contract and leakage suites: 296 passed.
- I4 cost calibration fixture after type correction: 3 passed.
- I5 historical acceptance module in a clean checkout: 1 passed, 11 explicit
  evidence-unavailable skips.
- Repository Ruff format: 746 files formatted; Ruff check: passed.
- Mypy: 635 source files checked, no issues.

Four preexisting source files were reformatted to satisfy the repository-wide
format gate. The dynamic I4 cost test fixture was explicitly typed as a
partial test double. No gate outcome is inferred from these regression tests.
