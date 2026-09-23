# I7 criterion 3 failure analysis and final software-gate decision (2026-09-23)

## Scope and decision rule

This pass touched only I7. It did not alter I4, I5, the I7 criteria, the baseline set, the
bandwidth levels, the outage construction, the retention oracle, or any acceptance threshold.
The operative criterion remains strict: BAAC must retain more mission-relevant information
than raw, FIFO, and fixed priority on every development seed at every declared non-zero
bandwidth level before a repaired mechanism can be frozen for a fresh surrogate final.

The starting authority was `artifacts/gates/I7/evidence_surrogate.json`, backed by
COM-I7-E005/E006 on spent final seeds 5600000-5600004. It records 3 of 4 surrogate criteria
passing. Criterion 3 fails on all-policy zero-retention ties at 1% and 0.1%, and on a genuine
BAAC loss to FIFO at outage 10% on seed 5600001 (0.108130 versus 0.113821).

## Failure decomposition

The two observed classes are materially different.

1. **Genuine scheduler loss after a long outage.** BAAC always serves the critical F0/F1 first.
   On the failing seed it then starts an increment that cannot complete before mission end,
   while FIFO completes two smaller routine increments. The selected BAAC unit ended only
   216 bits short of completion. The queue contained smaller mission-relevant units that could
   have completed. This was a policy defect, not a scoring tie.
2. **Physical-floor zero ties.** A 240 s mission at 1% of the 1200 bps reference link has only
   2880 nominal bits; at 0.1% it has 288. The compact F0 is 184 bits, while the smallest current
   F1 observed in this diagnosis is roughly 3912 bits. Fast-revising beliefs also make the F0
   stale by mission end. Consequently no policy can deliver a scored current update on most
   development worlds at those levels. Under the frozen strict rule, a zero tie is a failure.

## Rejected and retained repairs

A schema-dictionary DEFLATE prototype was screened on development worlds and rejected. It
helped the baselines as well as BAAC and did not remove the failure class. It was fully reverted.

The retained candidate has two policy-local mechanisms:

- completion feasibility: BAAC does not start an increment whose remaining receiver-side bits
  cannot arrive before the deployment-known mission horizon; a nearly complete increment is
  evaluated from its remaining bits and stays protected;
- receiver-aligned fidelity value: the present receiver records `evidence_support` at F1, while
  F2 repeats the same evidence IDs and stores no additional embedding state. F2 therefore has
  real wire cost but zero marginal receiver value, and the scheduler no longer spends scarce
  capacity on that nominal-only step.

The horizon comes from declared mission duration, not evaluation truth. Raw, FIFO, fixed
priority, and value-per-bit plans are unchanged by the completion-feasibility flag. Diagnostic
fields now expose useful receiver bits, obsolete-on-arrival bits, unique beliefs advanced,
coalesced queued bits, pending unit sizes, and partial progress.

## Development evidence

All results below use mission `development` seeds only: 5100000, 5100001, 5100002, 5100003,
5100007, and 5100008. They are diagnostic evidence, not gate evidence.

`I7-CRIT3-DEV-FULL-OUTAGE` shows the genuine loss is repaired. BAAC is strictly greater than
raw, FIFO, and fixed priority on every seed at both declared outage levels:

| Outage level | BAAC mean | raw mean | FIFO mean | fixed mean | worst BAAC-FIFO |
|---|---:|---:|---:|---:|---:|
| 100% | 0.595935 | 0.221951 | 0.132791 | 0.028455 | +0.368293 |
| 10% | 0.182114 | 0.093902 | 0.104336 | 0.000000 | +0.011382 |

At outage 10%, BAAC delivers the critical item first on every seed, with mean alert latency
47.533 s and mean critical-delta latency 98.803 s. Duplicate contributions are zero.

`I7-CRIT3-DEV-FULL-BANDWIDTH` shows no mid/high-bandwidth regression and the unresolved strict
failure:

| Bandwidth | BAAC mean | raw mean | FIFO mean | fixed mean | strict every-seed win? |
|---|---:|---:|---:|---:|---|
| 100% | 0.595935 | 0.221951 | 0.221951 | 0.202981 | yes |
| 50% | 0.591192 | 0.221951 | 0.221951 | 0.189702 | yes |
| 10% | 0.262602 | 0.221951 | 0.221951 | 0.023713 | yes |
| 1% | 0.007317 | 0.000000 | 0.000000 | 0.000000 | **no** |
| 0.1% | 0.007317 | 0.000000 | 0.000000 | 0.000000 | **no** |

At both 1% and 0.1%, seeds 5100001, 5100002, 5100003, 5100007, and 5100008 tie all three
required baselines at 0.000. Seed 5100000 scores above zero, which accounts for the positive
mean, but the frozen criterion is every seed rather than mean-only.

## Stop decision

The candidate is not frozen as a gate candidate because it does not satisfy the predeclared
development selection rule. Therefore no new final partition was declared, no fresh surrogate
final was run, and no Unity mission was launched. Reusing COM-I7-E005/E006 or changing the rule
after observing these ties would manufacture a pass.

The existing surrogate record remains authoritative: criterion 3 FAIL, overall surrogate 3 of
4. Formal I7 was not run and cannot be claimed. I4 and I5 remain untouched.

## Repository health and exact gate status

The final working tree passed `ruff format --check .` (715 files), `ruff check .`, `mypy conrad
tests` (623 source files), the secret scan (0 problems in 1430 tracked files), and the focused I7
suite (84 passed, 1 expected strict xfail). The repository-wide non-Unity suite completed with
1390 passed, 3 skipped, 3 expected xfailed, and 11 setup errors. All 11 errors are the same
pre-existing I5 fixture problem: the isolated checkout does not contain the gitignored
`artifacts/experiments/M1-ACTION-E008/m1_action_e008.json`. It was not regenerated because this
pass is forbidden from rerunning or repairing I5.

`conrad gates status` reports I7 as `official=BLOCKED_UPSTREAM`, `formal=NOT_RUN`, and
`surrogate=FAIL`. There is no `artifacts/gates/I7/evidence_formal.json`. For the requested final
software-gate decision, I7 is fail-closed: `I7 FORMAL = FAIL` because the mandatory surrogate
prerequisite did not clear criterion 3, so formal execution was correctly not attempted. This is
not a claim that a Unity measurement failed; the formal evidence status remains NOT_RUN.
