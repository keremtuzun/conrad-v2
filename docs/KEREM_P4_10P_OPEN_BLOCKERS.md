# Kerem P4 10P Open Blockers

This note exists to prevent accidental paid training from an incomplete handoff.

## Solved Payload Requirements

The required large binary payloads are identified and hash-pinned:

- Upstream checkpoint:
  `artifacts/runs/train-osfm_u1_sonar_research-1790575109765916596/checkpoints/osfm_u1_sonar_research_full.pt`
  SHA-256: `b2dfc162c1d72e4363643b516b6566eb0f11ca4a874f9674fc5b06f1ddf85696`
  Public download: `https://github.com/keremtuzun/conrad-v2/releases/download/osfm-v11-p4-handoff/osfm_u1_sonar_research_full.pt`
- SubPipe dataset:
  `artifacts/data/public.subpipe/raw/SubPipeMini2.zip`
  SHA-256: `a3068be28471786c726cd6100e0b1d92d1c17615a4dcfe7f5544ba758821188f`

The machine-readable payload manifest is:

```text
artifacts/gates/V1.1/10P_KEREM_HANDOFF/required_payloads.json
```

These files are not tracked by Git. Kerem must receive them separately and run the manifest's verify command before readiness.

## Trainer Closure

The reviewed formal V1.1 10P trainer/resume command is registered through:

The current command:

```bash
uv run conrad train osfm-v11-10p-launch --config configs/train/osfm/v11_p4_10p_829_semantic_candidate.yaml --readiness artifacts/gates/V1.1/10P_KEREM_HANDOFF/readiness.json
```

It now invokes the reviewed V1.1 10P trainer after the readiness gate is `READY FOR KEREM`.

This trainer is intentionally narrower than a full-829 semantic claim: it trains the current Universal V1.1 path on verified SubPipe rendered side-scan sonar and reports the 829 registry as interface coverage, not as semantic mastery.

## Required Runtime Checks

The reviewed command must continue to:

- loads the current V1.1 architecture and 829 registry,
- verifies the upstream checkpoint payload by SHA-256,
- verifies the SubPipe payload by SHA-256,
- loads only data-backed semantic modalities,
- keeps no-data registry entries interface-covered but not semantically claimed,
- writes an immutable `RunDirectory`,
- supports checkpoint cadence and exact resume,
- reports rank, loss, cost, checkpoint reload, semantic coverage, inactive modalities, exact-evidence preservation, and P4 V1 non-contamination,
- enforces 1-L4 fallback limits of 72 hours and 8500 TRY,
- emit a final GO / NO-GO / CONDITIONAL-GO package.
