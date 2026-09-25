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
