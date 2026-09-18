# Replay

Replay means a stored run can be verified byte for byte and re-executed to the same event signature.

## Event log (`conrad/runtime/event_log.py`)

- `EventLog.emit` writes append-only JSONL: payload redacted (`conrad.settings.redact`), monotonic `sequence`,
  `payload_digest`, canonical JSON, flushed per line. Every `RuntimeEvent` carries `architecture_id`, `stack_id`
  and a `MessageEnvelope` (run, mission, clock domain, measurement and created time).
- `event_signature(events)` = list of (sequence, event type, module, payload digest). This is what a replay must
  reproduce.
- `read_events(path)` reads a log back.

## Run bundles (`conrad/persistence/replay_store.py`)

- `write_bundle_manifest(run_dir, run_id, replay_inputs, object_digests)` writes `bundle_manifest.json` with the
  sha256 of every file in the run directory (except `notes/` and the manifest) and every referenced object
  digest. It refuses replay inputs missing any of `REQUIRED_REPLAY_KEYS`: scenario_seed, scenario_version,
  twin_versions, model_versions, robot_config_digest, sensor_configuration, config_digest, git_commit, seeds,
  architecture_id, stack_id.
- `verify_bundle(run_dir, store)` fails closed before any processing and names every missing or corrupt file or
  object (`ReplayIntegrityError`). Missing evidence is never synthesized.
- `backup(...)` uses the SQLite online backup API and copies referenced objects; `restore(...)` is valid only if
  database hash, object hashes, `PRAGMA integrity_check` and migration state all check out.

The integrated-mission bundle layout (`config.resolved.yaml`, `events.jsonl`, SQLite, `mission/`, `truth/`,
`reports/metrics.json`, `bundle_manifest.json`) is specified in
[development/INTEGRATION_BRIEF.md](development/INTEGRATION_BRIEF.md) and was still being implemented at the time
of writing.

## Determinism

Seeded `IdFactory` streams, seeded RNGs and injected clocks make runs repeatable. Evidence:

- `tests/integration/test_i0_fake_full_system.py::test_i0_deterministic_replay`: same seed gives the same event
  signature, belief revisions and decisions; a different seed gives a different signature; the stored JSONL
  reproduces the in-memory signature.
- `tests/replay/test_bundle_backup_restore.py`: SS-04 fail-closed verification, tampered event log detection,
  complete replay inputs, SS-08 backup/restore then replay, corrupt backup rejection.
- `tests/property/decision/test_egdc_properties.py::test_decisions_replay_deterministically`.

## Command-line replay

`scripts/replay.sh` calls `conrad replay run --run <id>`. At the time of writing the `replay` command group has no
commands, so that script does not work yet. Use the Python API above, or the read-only console
([CONSOLE.md](CONSOLE.md)), which verifies bundle digests before rendering.

Tests: `uv run pytest tests/replay tests/integration -q`.
