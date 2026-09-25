# I5 Spatial V1 cycle

Status: **DEVELOPMENT DECLARED; NO VALIDATION OR FINAL WORLD OPENED**.

The historical I5 formal result remains **FAIL, 9/10 criteria**. This is a
new versioned cycle after Spatial Structural Architecture V1 passed A-J and
was frozen. It does not rewrite or relabel any earlier evidence.

## Fixed architecture and semantics

The cycle uses the frozen `spatial_v1` Twin2T truth,
`RESOLUTION_CELL_SAMPLES`, pose-derived conservative visibility support,
`model2t-spatial-v1`, and spatial MCBR prediction. The sensor file is pinned
by SHA-256 `c7f68b08a008135d5a86961135c05930170c19f86d772b10094926f207877efe`;
the parsed sensor digest is
`034f949a14ded197ecca8cb1a8ab714b9872c98fac94a099414e77645fdd2104`.
All parameters remain `ENGINEERING_ESTIMATE` and `SYNTHETIC_ONLY`.

The legacy `I5-NOMINAL-READABLE` control is excluded and explicitly refused:
it uses `pristine_rest` and duplicated tile truth, which Spatial V1 forbids.
Ordinary `I5-NOMINAL` instead has an explicit healthy 2 by 4 local truth map.
Critical-finding and outage cases use a heterogeneous covered resolvable
corrosion map; other scenarios use healthy local truth. The warrant is
unchanged: critical component `OBSERVED INTACT` and nothing pending.

The required inspection domain and condition thresholds are the frozen
Spatial V1 values. Nominal duration is 480 simulated seconds because the
architecture acceptance mission first reached qualified intact at 291.3 s;
all other I5 scenarios retain 120 s. The longer duration is declared before
opening development and is not a threshold or warrant change.

## Partition and order

`configs/eval/partitions_i5_v8.yaml` is digest-pinned as
`f0a7007deda3470a742a13427ab3618a1c237d1712fcc11df48d55aa15752d29`.
Development is `8500000..8500009`, validation is `8500100..8500109`, and the
sealed final pool is `8500200..8500239`. The legacy fake-tile scenario is not
in the partition. Validation and final are not to be opened during mechanism
development.

The development screen starts with seeds 8500000 and 8500001 across the seven
legitimate action scenarios and all three frozen arms. It must demonstrate:

- frozen Spatial V1 truth and belief backends on every mission;
- at least one legitimate nominal intact warrant without nominal escalation;
- zero false intact results when a resolvable defect cell has measured support;
- no hard-constraint violation, complete traceability, and UIR zero; and
- no regression that makes another I5 action class structurally untestable.

The harness records an evaluation-only local truth map, support count and
digest, local coverage/beliefs, component condition, knowledge status, first
intact time, and defect-cell coverage. None of this truth data enters the
runtime decision path.

Validation, its selection rule, the warrant-incidence power calculation, final
N, minimum warrant count, and all final thresholds will be committed before
validation or final worlds are opened. If validation fails, final remains
sealed. If the one-shot surrogate final fails, no formal Unity run is allowed.

## Development iteration 1: failed attempt-budget reproduction

Seeds 8500000 and 8500001 completed all 42 missions (seven scenarios, three
arms) in 537.5 wall seconds. Spatial truth/backend selection and the frozen
sensor digest matched in every row. The six non-nominal action scenarios were
2/2 correct given warrant, hard violations were zero, traceability was 1.0,
UIR was 0, nominal over-escalation was zero, and every covered resolvable
defect stayed non-intact. Ordinary nominal nevertheless reached zero intact
warrants: both 480 s worlds ended `UNKNOWN` with incomplete required-cell
coverage. This is a **failed development hypothesis**, not gate evidence.

Fresh diagnostic development seed 8500002 reproduced the cause. It executed
only four views, then the default `max_plans_per_need=4` marked acquisition
unavailable while required cells remained incomplete. The architecture's
recorded successful 480 s development mission used fifteen views after its
earlier investigation had declared a larger Model1/MCBR attempt budget. The
I5 profile had accidentally inherited the generic four-plan default instead
of carrying that known spatial mission requirement forward.

Iteration 2 therefore changes only finite software acquisition capacity:
`max_plans_per_need`, Model1 `max_information_attempts`, and
`max_view_attempts` are each prospectively set to 24. Sensor, support,
detectability, truth maps, required inspection domain, condition thresholds,
warrant, baselines, mission durations, and scoring are unchanged. Fresh
development seeds 8500003 and 8500004 are used; iteration-1 seeds are not
rerun. Validation and final remain unopened.

The immutable R1 result is
`artifacts/experiments/M1-ACTION-SPATIAL-V1-DEV/m1_action_spatial_v1_dev_development.json`
with SHA-256
`5bf6b83bf80e1b6fc0bd645174246d24fc9c3f2bdd9c47c035844ae82a534b0b`.
