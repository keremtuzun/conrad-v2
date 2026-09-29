# Kerem Handoff: OceanSense Universal V1.1 P4 10P

This handoff is fail-closed. Do not start V1.1 10P training from prose notes alone.

Run the readiness gate first:

```bash
uv run conrad train osfm-v11-10p-readiness --output artifacts/gates/V1.1/10P_KEREM_HANDOFF/readiness.json
```

The gate must print `READY FOR KEREM` before any formal 10P launch. If it prints `NOT READY FOR KEREM`, preserve the JSON report and fix the listed blockers.

## Current Expected State

- Repository branch: `osfm-universal-v1.1`.
- Architecture: `UniversalOSFMV11`, 307M-class, 15 encoder families, 384-D interface, 64 scene latents.
- Registry: 829 concrete modality/type entries. This is interface/routing coverage, not proof that all 829 modalities have semantic training data.
- Candidate config: `configs/train/osfm/v11_p4_10p_829_semantic_candidate.yaml`.
- Primary protocol: `configs/train/osfm/v11_p4_10p_3am_launch_protocol.yaml`.
- Fallback protocol: `configs/train/osfm/v11_p4_1l4_10p_fallback_launch_protocol.yaml`.
- Approved launch mode: 1-L4 fallback, by user approval in `artifacts/gates/V1.1/10P_1L4/user_approved_fallback.json`.
- Qualified upstream metadata: `artifacts/gates/OSFM_S_PRETRAIN_V1/qualified_checkpoint_metadata.json`.

## Hard Rules

- Do not use old V1 assumptions, tensor contracts, or legacy 20P configs.
- Do not claim all 829 modalities are semantically pretrained unless each claimed modality has a usable manifest.
- Do not launch if the readiness report has any `BLOCKER`.
- Do not launch if the tree is dirty unless the dirty artifacts are intentionally part of the handoff package.
- Do not launch if cloud quota/cost/projection gates fail.
- Do not call a benchmark, rehearsal, or microbenchmark a promotable 10P checkpoint.

## Known Blockers The Gate Checks

- The guarded launch command exists, but it refuses unless the readiness gate is already `READY FOR KEREM`.
- The original 8-L4 protocol is not viable under current quota evidence. The approved path is now the 1-L4 fallback.
- Existing 1-L4 benchmark evidence is not a clean GO; this is a user-approved fallback, not a claim that the fallback benchmark fully passed.
- The data-backed semantic subset is smaller than the 829 registry. Registry coverage is not semantic data coverage.

## Non-Training Checks

These commands are safe before launch:

```bash
uv run pytest tests/unit/foundation/universal_v11 -q
uv run conrad train osfm-v11-pillar-benchmark --fast-probe --output artifacts/gates/V1.1/local_fast_pillar_benchmark.json
uv run conrad train osfm-v11-10p-readiness --output artifacts/gates/V1.1/10P_KEREM_HANDOFF/readiness.json
```

## Launch Command

The guarded launch surface is:

```bash
uv run conrad train osfm-v11-10p-launch --config configs/train/osfm/v11_p4_10p_829_semantic_candidate.yaml --readiness artifacts/gates/V1.1/10P_KEREM_HANDOFF/readiness.json
```

It refuses to proceed unless the readiness report says `READY FOR KEREM`. If it refuses, do not bypass it with an
ad hoc training command.

During the 1-L4 fallback run, Kerem must monitor rank, loss finiteness, checkpoint writes, projected runtime, and
projected cost. Terminate immediately on the fail-closed conditions in the fallback protocol.
