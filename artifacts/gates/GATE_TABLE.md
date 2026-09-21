| gate | official | formal | criteria | level | split | commit | evidence | blocker |
|---|---|---|---|---|---|---|---|---|
| P0 | PASS | PASS | 8/8 pass | FORMAL |  | 28871b4a | artifacts/gates/P0/ |  |
| I0 | PASS | PASS | 8/8 pass | FORMAL |  | 28871b4a | artifacts/gates/I0/ |  |
| C1 | PASS | PASS | 9/9 pass | FORMAL |  | 28871b4a | artifacts/gates/C1/ |  |
| 2S-FIRST | PASS | PASS | 2/2 pass | FORMAL |  | 28871b4a | artifacts/gates/2S-FIRST/ |  |
| U0 | PASS | PASS | 6/6 pass | FORMAL |  | 6824356b | artifacts/gates/U0/ |  |
| I1 | PASS | PASS | 9/9 pass | FORMAL | configs/eval/partitions_unity_gates.yaml final_test | 025b302f | artifacts/gates/I1/ |  |
| I2 | PASS | PASS | 7/7 pass | FORMAL |  | 025b302f | artifacts/gates/I2/ |  |
| 2T | PASS | PASS | 1/1 pass | FORMAL |  | 0a7179bc | artifacts/gates/2T/ |  |
| 2T-TCDP | FAIL | FAIL | 0/1 pass, 1 fail | FORMAL |  | 0a7179bc | artifacts/gates/2T-TCDP/ | TCDP improves hidden-state reconstruction vs generic/no propagation without excessive contamination |
| I3 | PASS | PASS | 3/3 pass | FORMAL | configs/eval/partitions_unity_gates.yaml final_test | 0a7179bc | artifacts/gates/I3/ |  |
| I4 | FAIL | FAIL | 4/5 pass, 1 fail | FORMAL | configs/eval/partitions_i4_mcbr_v4.yaml final_test first 30 (declared in i4_v4_unity_final.yaml) | 9280b5bb | artifacts/gates/I4/ | beats simple views on information/time/energy |
| I5 | BLOCKED_UPSTREAM | NOT_RUN | 7/10 pass | FORMAL |  | e0b44237 | artifacts/gates/I5/ | blocked by I4 |
| 2E | PASS | PASS | 3/3 pass | FORMAL |  | 1d2b15d8 | artifacts/gates/2E/ |  |
| 2E-CEFD | FAIL | FAIL | 0/1 pass, 1 fail | FORMAL |  | 1d2b15d8 | artifacts/gates/2E-CEFD/ | CEFD beats uncoupled baselines without unsupported ecological claims |
| I6 | BLOCKED_UPSTREAM | PASS | 3/3 pass | FORMAL | configs/eval/partitions_unity_gates.yaml final_test (declared in i6_multidomain.yaml) | 3e1470e1 | artifacts/gates/I6/ | blocked by I5 |
| I7 | BLOCKED_UPSTREAM | NOT_RUN | 0/4 pass | SURROGATE |  | 6e7913d9 | artifacts/gates/I7/ | blocked by I6 |
| I8 | BLOCKED_EXTERNAL | NOT_RUN | 0/9 pass | NONE |  |  | none | blocked by I7 |
| I9 | BLOCKED_EXTERNAL | NOT_RUN | 0/1 pass | NONE |  |  | none | blocked by I8 |
