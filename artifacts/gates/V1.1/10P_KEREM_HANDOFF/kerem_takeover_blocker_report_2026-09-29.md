# V1.1 P4 10P takeover: NO-GO (not launched)

Date: 2026-09-29
Branch: osfm-universal-v1.1
Commit: 366b59ea87827a5b5eca89b0d25a662a446754ab (matches expected)
Clone: /Users/kerem/Kerem/conrad-v2 (clean tree)

## Decision

NO-GO. No VM was created, no GPU time was bought, no training step ran. Cost so far: 0 TRY.

## Blockers

1. Readiness gate says NOT READY FOR KEREM on a fresh clone (blocker_count=1).
   Item 6 BLOCKER: upstream checkpoint missing.
   `artifacts/runs/train-osfm_u1_sonar_research-1790575109765916596/checkpoints/osfm_u1_sonar_research_full.pt`
   is not in git (`artifacts/*` is gitignored) and is not anywhere on this Mac.
   Expected sha256: b2dfc162c1d72e4363643b516b6566eb0f11ca4a874f9674fc5b06f1ddf85696.
   The committed readiness.json saying READY was produced on Burak's machine
   (commit c5044270, 14 dirty entries), where the file exists.
   Item 5 WARNING: `artifacts/data/public.subpipe/raw/SubPipeMini2.zip` is also missing.
   Unit tests: 32 passed, 1 failed (test_readiness, same root cause).

2. There is no formal trainer to launch.
   `conrad train osfm-v11-10p-launch` only prints `READY_TO_LAUNCH` and
   "Formal training implementation must be invoked by the reviewed launch runner for this config."
   (conrad/cli/commands.py:310-319). No code consumes
   `v11_p4_10p_829_semantic_candidate.yaml`, and that config sets `formal_training_allowed: false`.
   Readiness item 8 is marked PASS but its own evidence reads "no formal trainer/resume
   implementation is registered". Training would need an ad hoc runner, which the brief
   forbids without Burak's approval.

3. Budget mismatch to resolve: the brief says 8,500 TRY ceiling; the fallback protocol
   hard-limits 8,000 TRY (max_total_cost_try) and 111.11 TRY/h. The stricter repo value should win
   unless Burak updates the protocol.

4. The fallback protocol also requires a bounded 1000-step 1-L4 benchmark with
   effective_rank >= 75 before launch; the existing 1-L4 microbenchmark is NO-GO with
   `rank_below_75`. Even with blockers 1 and 2 fixed, that gate must pass on real hardware.

5. Google Cloud: no gcloud CLI or authenticated account on this Mac. Creating a GCP account,
   accepting its terms and adding a payment card have to be done by Kerem.

## To unblock

- Burak: upload the checkpoint (hash above) and SubPipeMini2.zip somewhere Kerem can fetch
  (GCS bucket or Drive), or commit them via LFS.
- Burak: land the reviewed V1.1 10P trainer (RunDirectory, checkpoint/resume, rank validation)
  and flip `formal_training_allowed` through a reviewed commit.
- Burak: confirm 8,000 vs 8,500 TRY.
- Kerem: create/sign in to the GCP project with billing, request L4 quota (1 GPU, g2-standard-12),
  then `gcloud auth login` on this Mac.
- Then: readiness -> local preflight -> cloud preflight -> 1000-step benchmark -> projection gate -> launch.
