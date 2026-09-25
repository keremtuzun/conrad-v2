# I7 compact critical summary v4 validation

Status: **FAIL; FINAL UNOPENED**. The frozen candidate was evaluated once on
the five validation seeds 8400000-8400004 at source commit
`c86799b572fcab21757d37d012d581ef1fedc33e`. Historical I7 surrogate FAIL 3/4
and formal NOT_RUN remain authoritative.

## Result

| Criterion | Result | Evidence |
|---|---|---|
| Full matrices and no duplicate contribution | PASS | 30 bandwidth runs, 10 outage shadow runs, 20 outage closed-loop runs; zero duplicate contributions |
| Finding-following outage delivery | PASS | 10/10 findings while down and 10/10 BAAC critical deltas delivered first after reconnection |
| Strict retention over raw/FIFO/fixed | **FAIL** | one 0.1% zero tie and three outage-10% comparison losses |
| Latency and sync comparison present | PASS | every arm and declared level reported the required fields |

Bandwidth means for BAAC versus the strongest required-baseline mean were:
100% 0.61958 vs 0.22650; 50% 0.59385 vs 0.22650; 10% 0.26799 vs
0.22650; 1% 0.03163 vs 0; and 0.1% 0.02724 vs 0. The mean does not override
the frozen every-seed rule. On seed 8400004 at 0.1%, BAAC, raw, FIFO, and
fixed priority all retained exactly zero.

At outage 10%, seed 8400000 gave BAAC 0.108130 versus FIFO 0.113821. Seed
8400004 gave BAAC 0.108130 versus raw 0.142276 and FIFO 0.113821. These are
real strict-comparison losses, not setup errors. The remaining cells passed.

The machine result is
`artifacts/experiments/I7-COMPACT-SUMMARY-V4-VALIDATION/assessment.json`.
Complete mission results and traces are in `COM-I7-V4-VAL-BW` and
`COM-I7-V4-VAL-OUTAGE` under `artifacts/experiments`.

## Stop decision

The candidate is rejected for the gate because validation did not satisfy
criterion 3. No threshold, baseline, retention rule, seed, or result was
changed after observation. `COM-I7-E007` and `COM-I7-E008` were not run and
their final seeds 8400100-8400104 remain unopened. Under the predeclared rule,
no formal Unity I7 run is permitted.
