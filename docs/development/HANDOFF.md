# Conrad V2 handoff

Updated 2026-09-26 on branch `kerem/software-completion`. Verify the current
HEAD and remote before changing anything.

## Read first

- `docs/reports/SOFTWARE_COMPLETION_2026_09_26.md`
- `docs/reports/CONRAD_V2_GATE_TABLE.md`
- `docs/architecture/SPATIAL_V1_1_FREEZE.md`
- `docs/audits/I5_SPATIAL_V1_1_UNITY_PROTOCOL.md`
- `docs/hardware/PENDING_PHYSICAL_CALIBRATION.md`

## Current state

- Spatial V1.1 A-J: PASS for synthetic software; physical calibration pending.
- I4: negative formal record preserved; no new development candidate qualified.
- I5 historical: 9/10 formal FAIL, immutable.
- I5 Spatial V1.1: surrogate final PASS, Unity formal 7/10 FAIL, immutable.
- Post-formal Unity `LOW_POWER` software correction: development PASS only;
  exact player SHA is documented in the correction audit.
- I6: formal 3/3 PASS but official status is blocked upstream by I5.
- I7: surrogate 3/4 FAIL, formal NOT_RUN; compact-summary validation also
  failed the unchanged ultra-low-bandwidth rule.
- I8/I9: require target hardware and a physical robot.

Do not retry a stopped I4/I7 mechanism or rerun the I5 formal cycle as if its
negative result did not exist. A future cycle needs a new version, a prospective
rule, and entirely fresh partitions.

## Verification snapshot

Both the integrated checkout and a fresh remote clone passed:

```text
ruff format --check: PASS (827 files)
ruff check: PASS
mypy conrad tests: PASS (659 source files)
secret scan: PASS (0 findings)
pytest: 1519 passed, 15 skipped, 139 deselected, 3 xfailed
```

The xfails are the immutable negative I4/I7 gate checks. The skips have explicit
artifact, dataset, or environment reasons; there are no opaque setup errors.

## Evidence integrity rules

1. Historical and separately versioned negative evidence is immutable.
2. Never regenerate a missing historical artifact from spent final seeds.
3. Python-kernel results are surrogate evidence, never formal Unity evidence.
4. Only one Unity player may run at a time.
5. Truth is allowed only in simulation/evaluation generation, never runtime
   belief, decision, MCBR, or communication inputs.
6. Engineering estimates remain labelled synthetic until physically measured.

## Useful commands

```text
.venv/bin/ruff format --check .
.venv/bin/ruff check .
.venv/bin/mypy conrad tests
.venv/bin/pytest -q -p no:cacheprovider
.venv/bin/python scripts/secret_scan.py
.venv/bin/python scripts/build_requirements_ledger.py
.venv/bin/python scripts/gate_report_table.py --markdown docs/reports/CONRAD_V2_GATE_TABLE.md
```

## Remaining work boundary

Only physical calibration, target-computer HIL, real underwater communication
measurement, pool/open-water validation, and owner deployment authorization
remain. Do not classify Unity, integration, Model2T, MCBR, replay, schemas, or
tests as hardware blockers.
