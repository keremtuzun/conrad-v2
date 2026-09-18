# Security

## Private binding, no public control endpoint

- `RuntimeSettings.bind_address` defaults to `127.0.0.1`; the validator refuses `0.0.0.0` and `::`
  (`conrad/settings.py`).
- Unity bridge endpoints must be `tcp://<literal IP>:<port>`, the IP must be loopback, private or link-local,
  wildcards are refused, and the host must be in `sim.unity.bridge.allowed_peers`
  (`conrad/adapters/unity/transport.py`, `validate_private_endpoint`). The player's simulator ID must be in
  `allowed_simulator_ids`.
- The console binds only to `127.0.0.1`, refuses other addresses (`BindRefusedError`), only serves loopback host
  names, is read-only and has no command capability (`conrad/console/server.py`, [CONSOLE.md](CONSOLE.md)).
- There is no network control endpoint for commands. Commands reach hardware only through the in-process
  `CommandGateway`.

## Unknown peer rejection

`CommandGateway.validate` adds `UNKNOWN_PEER` when the submitting peer is not in `runtime.allowed_peers`
(default `["127.0.0.1"]`), and rejects the command. Tests: `tests/contract/test_command_gateway.py`.

## Command authority

- Hardware commands need command_mode `hardware`, `hardware_enable: true` (machine-local, default false), the hil
  or physical lane, a HIL evidence reference and a physically grounded RobotConfig. `ConradSettings`,
  `conrad doctor`, `RuntimeSupervisor.start(hardware=True)` and the gateway each enforce this.
- `configs/runtime/physical_example.yaml` fails `conrad doctor` by design.
- No CI job sets command_mode `hardware`.

## Secrets

- Secrets are referenced by name only: `runtime.required_secrets` lists environment variable names, and
  `conrad doctor` reports missing names, never values.
- `redact` replaces values whose key contains password, secret, token, apikey, api_key, private_key or credential
  before config snapshots and event payloads are written.
- `scripts/secret_scan.py` scans `git ls-files` for private keys, AWS keys, GitHub tokens, `sk-` keys, Slack tokens
  and quoted secret assignments, and fails on tracked `.env*`, `*.pem`, `*.key`, `id_rsa`, `credentials.json`, or
  anything under `artifacts/runs/`, `artifacts/objects/`, `mlruns/`. It runs in CI; its detector is tested in
  `tests/contract/test_ss09_ci_hygiene.py`.

```
uv run python scripts/secret_scan.py
```

## Data integrity

Object store reads re-hash content; bundles, backups, sealed run directories, the experiment registry and the gate
ledger are digest-verified and fail closed. Checkpoints load with `weights_only=True`.

## Limits

No authentication or encryption exists on the ZeroMQ bridge beyond address allow-lists; it is meant for a private
link. Physical network, hardware safety and operator procedures are OPEN until EXT-HW-01 and EXT-HW-05.
