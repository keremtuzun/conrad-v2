# Kerem Handoff: OceanSense Universal V1.1 P4 10P

This handoff is fail-closed. Do not start V1.1 10P training from prose notes alone.

Run the readiness gate first:

```bash
uv run conrad train osfm-v11-10p-readiness --output artifacts/gates/V1.1/10P_KEREM_HANDOFF/readiness.json
```

The gate must print `READY FOR KEREM` before any formal 10P launch. If it prints `NOT READY FOR KEREM`, preserve the JSON report and fix the listed blockers.

Current state: this repository contains the V1.1 handoff metadata, payload manifest, readiness gates, and reviewed trainer surface. A clean clone is still not sufficient to start paid training until Kerem also has the ignored large payloads and the readiness gate is green.

## Current Expected State

- Repository branch: `osfm-universal-v1.1`.
- Architecture: `UniversalOSFMV11`, 307M-class, 15 encoder families, 384-D interface, 64 scene latents.
- Registry: 829 concrete modality/type entries. This is interface/routing coverage, not proof that all 829 modalities have semantic training data.
- Candidate config: `configs/train/osfm/v11_p4_10p_829_semantic_candidate.yaml`.
- Primary protocol: `configs/train/osfm/v11_p4_10p_3am_launch_protocol.yaml`.
- Fallback protocol: `configs/train/osfm/v11_p4_1l4_10p_fallback_launch_protocol.yaml`.
- Approved launch mode: 1-L4 fallback, by user approval in `artifacts/gates/V1.1/10P_1L4/user_approved_fallback.json`.
- Qualified upstream metadata: `artifacts/gates/OSFM_S_PRETRAIN_V1/qualified_checkpoint_metadata.json`.
- Required payload manifest: `artifacts/gates/V1.1/10P_KEREM_HANDOFF/required_payloads.json`.
- Current blocker note: `docs/KEREM_P4_10P_OPEN_BLOCKERS.md`.
- Required upstream checkpoint payload: `artifacts/runs/train-osfm_u1_sonar_research-1790575109765916596/checkpoints/osfm_u1_sonar_research_full.pt`, SHA-256 `b2dfc162c1d72e4363643b516b6566eb0f11ca4a874f9674fc5b06f1ddf85696`.
- Required SubPipe payload: `artifacts/data/public.subpipe/raw/SubPipeMini2.zip`, SHA-256 `a3068be28471786c726cd6100e0b1d92d1c17615a4dcfe7f5544ba758821188f`.

## Hard Rules

- Do not use old V1 assumptions, tensor contracts, or legacy 20P configs.
- Do not claim all 829 modalities are semantically pretrained unless each claimed modality has a usable manifest.
- Do not launch if the readiness report has any `BLOCKER`.
- Do not launch if the tree is dirty unless the dirty artifacts are intentionally part of the handoff package.
- Do not launch if cloud quota/cost/projection gates fail.
- Do not launch if the upstream checkpoint file is missing or its SHA-256 does not match the metadata.
- Do not launch if the SubPipe dataset payload is missing or its SHA-256 does not match this handoff.
- Do not launch from stale readiness output. `osfm-v11-10p-launch` must be run only after a fresh `READY FOR KEREM` report.
- Do not call a benchmark, rehearsal, or microbenchmark a promotable 10P checkpoint.

## Known Blockers The Gate Checks

- The guarded launch command exists, but it refuses unless the readiness gate is already `READY FOR KEREM`.
- The original 8-L4 protocol is not viable under current quota evidence. The approved path is now the 1-L4 fallback.
- Existing 1-L4 benchmark evidence is not a clean GO; this is a user-approved fallback, not a claim that the fallback benchmark fully passed.
- The data-backed semantic subset is smaller than the 829 registry. Registry coverage is not semantic data coverage.
- The large upstream checkpoint and SubPipe dataset are ignored by Git. Kerem needs those bytes supplied separately, then must verify their hashes.
- The reviewed 10P trainer is intentionally limited to data-backed SubPipe active-acoustic training and must not be described as full-829 semantic mastery.

## Non-Training Checks

These commands are safe before launch:

```bash
uv run pytest tests/unit/foundation/universal_v11 -q
uv run conrad train osfm-v11-pillar-benchmark --fast-probe --output artifacts/gates/V1.1/local_fast_pillar_benchmark.json
uv run conrad train osfm-v11-10p-readiness --output artifacts/gates/V1.1/10P_KEREM_HANDOFF/readiness.json
```

## Launch Command

The guarded preflight surface is:

```bash
uv run conrad train osfm-v11-10p-launch --config configs/train/osfm/v11_p4_10p_829_semantic_candidate.yaml --readiness artifacts/gates/V1.1/10P_KEREM_HANDOFF/readiness.json
```

It refuses unless the readiness report says `READY FOR KEREM`. If it refuses, do not bypass it with an ad hoc training command.

During the 1-L4 fallback run, Kerem must monitor rank, loss finiteness, checkpoint writes, projected runtime, and
projected cost. Terminate immediately on the fail-closed conditions in the fallback protocol.
