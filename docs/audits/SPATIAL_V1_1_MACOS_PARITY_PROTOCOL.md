# Spatial V1.1 native-macOS Unity parity supplement

Status: **PASS (16/16); PLATFORM-BOUNDED**

Protocol ID: `SPATIAL-V1.1-MACOS-PARITY-1`

This is a separately declared, platform-specific supplement to
`SPATIAL_V1_1_REVALIDATION_PROTOCOL.md`. It does not edit, relabel, or rerun the
Windows-player protocol pinned there. That protocol's Unity worlds at seeds
2001 and 2002 remain unopened and its outcome remains **NOT RUN**.

The supplement exists because the only available host is Apple silicon. After
the Unity license blocker was cleared, the exact required editor compiled the
project, built a native macOS player, and that player passed one development
smoke case at seed 1999. None of those steps opened the evaluation worlds
declared below.

## Frozen inputs

- Unity Editor: `6000.5.9f1 (b57deb96f08d)`.
- Host: macOS 26.6.2 (25G83) on Apple silicon.
- Player: Mach-O universal executable containing `x86_64` and `arm64`, run
  natively on the arm64 host.
- Player executable SHA-256:
  `7e42f64199b043204d62737d06aabc6b6ceb1f3c1c699f158a612b938ea87ac6`.
- Fresh Unity world seeds: 2003 and 2004.
- Runner: `scripts/validate_spatial_v1_1_macos_unity_parity.py`.

Before this declaration, seeds 2003 and 2004 were not used by a committed
Conrad spatial protocol, configuration, test, or audit, and neither was opened
by the native player. The output directory must be absent on first execution.
Only one Unity player may run at a time.

## Frozen matrix and acceptance rule

For each seed, run the complete cross product of:

- clear and occluded view;
- deterministic and noisy measurement; and
- exact and bounded survey registration.

This is 16 cases total. All 16 must pass. Every case must have exactly matched
kernel/Unity structural supports and inline measurements, matched association,
credited-cell coverage within absolute tolerance `1e-12`, matched cell and
component condition, zero kernel-visible/Unity-hidden rows, positive bounded
coverage, and an actually blocked view in occluded cases. A missing player,
hash mismatch, skipped case, interruption, or any failed assertion is a
failure. Thresholds and case selection are identical to the earlier frozen
parity matrix; only the declared player, platform, and fresh seeds differ.

This supplement can qualify Spatial V1.1 only for the exact macOS player hash
above. It cannot be reported as execution of the historical Windows binary or
as physical, ROS, hardware, or deployment validation.

## Result recorded 2026-09-26

The sealed matrix completed 16/16 cases with zero failures or interruptions:
8/8 for seed 2003 and 8/8 for seed 2004. All declared clear/occluded,
deterministic/noisy, and exact/bounded combinations ran once. The retained
result files are:

| File | SHA-256 |
|---|---|
| `protocol.json` | `2ca446bf5045049cc9bb8cbac8b2b9a01b7989eb8ffd10aa352ea56c167b2e3f` |
| `results.jsonl` | `d606993f9bb50814fbf8090dba6fcd894c5f2a46f82d3bd9bcc2a92d6eb0c393` |
| `summary.json` | `45fd75debe10fa8d950fc88edb2bb3d2f6174696aced5ed37a6873683392a23b` |

The full per-case Unity input documents and player logs are retained below
`artifacts/experiments/SPATIAL-V1-1-MACOS-PARITY-1/`. This PASS establishes
kernel/Unity semantic parity only for the exact player hash and platform scope
declared above; the Windows-pinned protocol remains **NOT RUN**.
