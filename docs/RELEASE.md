# Release lanes

`conrad/runtime/release.py` defines three release lanes and a `ReleaseManifest`. A checkpoint is never promoted
because it is newest.

| Lane | Meaning | Required by `build_release_manifest` |
|---|---|---|
| `dev` | Noncomparable development | nothing |
| `candidate` | Reproducible sim/HIL evidence from a clean commit | clean tracked tree, and at least one gate with status PASS and an evidence artifact |
| `physical` | A candidate that passed the physical gates | clean tracked tree, and gates `I8` and `I9` both PASS with evidence artifacts |

`build_release_manifest(lane, robot_config_digest, gates, checkpoint_ids, unity_adapter_version, git)` records git
SHA and dirty flag, the `uv.lock` sha256, the migration head, the schema version, architecture and stack IDs, the
Unity adapter version, the RobotConfig digest, checkpoint IDs and gate evidence (`GateEvidence`: gate_id, status
PASS / FAIL / NOT_EVALUABLE / BLOCKED_EXTERNAL, evidence_artifact). Refusal reasons go into `problems`;
`releasable` is true only when there are none.

Tests: `tests/replay/test_bundle_backup_restore.py::test_release_lanes`.

## Status

- No candidate or physical release has been built. The gates with evidence so far are development-tier
  experiments, all INCONCLUSIVE.
- I8 (HIL on target compute) is BLOCKED_EXTERNAL (EXT-HIL-01, EXT-UNITY-01). I9 (physical) is BLOCKED_EXTERNAL
  (EXT-HW-01..05). A physical release is therefore impossible today.
- Related: execution lanes in `conrad/settings.py` (`dev`, `simulation`, `hil`, `physical`) govern command
  authority; run purposes in `conrad/training/run_dir.py` require a clean tree for BENCHMARK, ACCEPTANCE and
  PHYSICAL runs.
