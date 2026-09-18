# Operator and Replay Console

The console (spec ch37) is a local, read-only page for looking over one replayable run bundle. It
**explains** a run and nothing more. It is not a source of truth, not a decision authority, and it has no
command capability. The run bundle stays authoritative.

## Usage

```python
from conrad.console import build_console, serve_console

# One self-contained HTML file: inline CSS/JS, charts as inline SVG, no network, no CDN.
build_console("artifacts/runs/<run_id>", "artifacts/runs/<run_id>/notes/console.html")

# Read-only stdlib server on the loopback address. port=0 picks a free port.
server = serve_console("artifacts/runs/<run_id>", host="127.0.0.1", port=0)
print(server.url)
server.wait()  # or server.close()

# Evaluation mode: also show the truth/ record (clearly labelled). Default is False.
build_console("artifacts/runs/<run_id>", "eval.html", include_truth=True)
```

`include_truth` exists on `build_console`, `serve_console`, `render_run` and `open_bundle`, and defaults to
`False`.

Both functions accept an optional `object_store=` path. Without one, the store is looked up in this order:
`<run>/objects`, the `object_store` key in `config.resolved.yaml`, then `artifacts/objects`.

Write the HTML somewhere other than a digest-covered path in the run directory (for example `notes/`,
which `write_bundle_manifest` skips, or anywhere outside the run).

## What it reads

| Bundle part | Required | Used for |
|---|---|---|
| `bundle_manifest.json` | yes | verified with `verify_bundle` before anything else is read |
| `events.jsonl` | no | runtime timeline, command outcomes, faults, BAAC events, replay scrubber |
| SQLite DB (path from `config.resolved.yaml`, else `*.sqlite` in the run dir) | no | belief revisions, claims, uncertainty, provenance, observations and pose |
| `mission/*.json` | no | MCBR plans and candidate tables, needs, decisions, trajectories, transmissions and receiver state |
| `truth/*.json` | no | evaluation mode only (`include_truth=True`): the **TRUTH (evaluation only)** section and the labelled truth series on the pose chart. By default only its existence is noted |
| `reports/metrics.json` | no | metrics table |

Any optional part that is absent is shown as "not present". The database is copied to a private temp
directory before it is opened, so the console never writes into the run directory.

## Views

Run bundle summary · replay controls (event scrubber by sequence index) · robot health and runtime-state
timeline · pose track (estimated, with truth only when a `truth/` record exists) · 2S/2T/2E belief status per
claim over revisions (OBSERVED / INFERRED / PREDICTED / UNKNOWN) · uncertainty decomposition U_A / U_E / U_C /
U_O per belief over time · provenance explorer (select a claim, belief revision, decision, plan or command and
walk the DAG down to raw observations through `Repository.provenance_closure`; the last command's chain is
pre-rendered) · active contradictions · InformationNeeds and MCBR plans with rejected reasons and the
selected observation · navigation goals and trajectories · safety and command outcomes with reason codes ·
BAAC transmissions and receiver state · faults · metrics · truth record (evaluation mode only).

## Limits

- **Explanatory only.** The console recomputes nothing and owns no belief state. It only displays what the
  bundle recorded. Contradictions, for example, are the belief heads with U_C > 0 and the `CONTRADICTS`
  claim edges that were stored. The console does not judge them.
- **Fails closed.** If `verify_bundle` reports any missing or corrupt digest, or the manifest is missing, the
  page shows only an integrity-error list and nothing from the bundle.
- **Truth is hidden by default (ch37 UI-08).** With the default `include_truth=False`, `truth/` is not even
  parsed and no truth-derived value appears anywhere in the HTML. The page only notes that an evaluation-only
  truth record exists and requires evaluation mode.
- **Evaluation mode.** With `include_truth=True`, the page title and a banner read "EVALUATION MODE", and
  `truth/` is shown in its own dashed "TRUTH (evaluation only)" section plus a labelled dashed series on the
  pose chart. It is never mixed into belief, uncertainty or provenance views. The robot never saw it. Do not
  hand an evaluation-mode page to an operator as a mission view.
- **No command path.** `conrad/console` does not import the command gateway and never calls `send`
  (a static test checks this). The server answers 405 to every non-GET method, has no write endpoint, binds to
  `127.0.0.1` only and refuses any other bind address. It also rejects non-loopback `Host` headers.
- **Raw actuator payloads are hidden.** `thruster_commands` values are redacted from displayed events.
- **Snapshot, not live.** The page is rendered once from the bundle. There is no live event stream.
- **Simplified stack.** Spec ch37 describes a React/FastAPI console with operator requests and fault
  controls. Neither is implemented here: this is a static page plus a stdlib server, and it has no operator
  actions and no fault injection.
- `mission/` and `truth/` file shapes are not frozen yet. They are read tolerantly and unknown shapes are
  shown as generic tables.
