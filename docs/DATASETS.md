# Datasets

`conrad/data` holds the tooling that makes a dataset usable: manifests, verification, lineage splits, the SSL
corpus report and the sim/real ledger. The manifests on disk and their status are listed in
[DATASET_REGISTER.md](DATASET_REGISTER.md); this page describes the code.

Two real public datasets are downloaded and verified:

- two UVVID videos (CC BY 4.0), used by DATA-REAL-SMOKE-E001;
- SubPipeMini2.zip (CC BY 4.0, 4.9 GB), downloaded after user approval and used by DATA-REAL-E002.

Every other experiment uses synthetic generators.

## Manifests (`manifest_model.py`, `manifest.py`)

- `DatasetManifest` records source, release, license (default `REVIEW_REQUIRED`) and evidence, rights,
  `training_allowed` / `deployment_allowed` / `redistribution_allowed` (default False), modalities, labels, units,
  frames, timestamps, split strategy, transformations, adapter version and a file inventory with sha256.
- `check_manifest_declarations` flags STATUS_NOT_USABLE, LICENSE_MISSING, RIGHTS_NOT_CLEARED,
  TRAINING_NOT_PERMITTED, ADAPTER_VERSION_MISSING, SPLIT_STRATEGY_MISSING, NO_FILES.
- `verify_manifest(path, data_root)` also checks files and checksums and fails closed.
- `find_manifest("id[@version]")` searches `datasets/**/*.manifest.yaml`; data lives at
  `$CONRAD_DATA_ROOT/<id>` or `artifacts/data/<id>`.

CLI:

```
uv run conrad data verify --manifest subpipe            # id, id@version, or a .yaml path
```

It prints each problem and `RESULT: OK` or `RESULT: FAIL (n problems)`. Only `public.uvvid` and `public.subpipe` pass today. Use the id
form: given a `.yaml` path, the CLI takes the manifest's own folder as the data root. `conrad doctor` verifies every ID in `data.manifest_ids`.

## Splits (`splits.py`)

`build_splits` groups samples into lineage groups (connected components over shared split-unit values such as
dive, site, asset or trajectory) so related samples never cross splits. Splits: train, validation, test,
test_ood. `compute_split_hash` is recorded in checkpoints; `assert_no_lineage_leakage` raises
`LineageLeakageError`. No experiment uses these splits yet.

## Other modules

- `ssl_corpus.py`: DATA-CONRAD-SSL-01 acceptance report. An empty real partition is `EMPTY_BLOCKED_EXTERNAL`.
- `simreal_ledger.py`: SIMREAL-LEDGER-01. `datasets/simreal_ledger.yaml` (revision `0.1.0-unmeasured`) marks
  every row unmeasured; UNMEASURED parameters are never treated as calibrated.
- `inventory.py`: hashes files into manifest rows.
- `adapters/`: `ImageFolderSequenceAdapter` (bytes unchanged, no labels), `UvvidVideoAdapter` (UVVID MP4 to RGB
  observations with container-time ns; pose, calibration and labels are None), `PublicDatasetAdapter` (every access
  raises `DatasetNotAvailableError`), `SyntheticTwinAdapter`.
- `real_smoke.py`: DATA-REAL-SMOKE-E001 runs ECMER quality features and encoding on the real UVVID frames.
- `adapters/subpipe.py`: `SubPipeAdapter` reads SubPipeMini2 camera and side-scan images from the verified zip. Its
  side-scan "Pipeline" presence labels come only through `map_labels`.
- `real_e002.py`: DATA-REAL-E002, a probe on ECMER evidence for side-scan pipeline presence over held-out time blocks, plus a
  camera quality check.

## Forbidden data

The legacy Model 2 datasets must never be used: their observation confidence was computed from the true severity
([migration/DIGITAL_TWIN_SALVAGE_REPORT.md](migration/DIGITAL_TWIN_SALVAGE_REPORT.md)).

## Blocked

Per-dataset verdicts are in [audits/PUBLIC_DATA_PROCUREMENT.md](audits/PUBLIC_DATA_PROCUREMENT.md). SubPipeMini2 gives
real side-scan pipeline-presence labels from one mission. SeaClear needs a download-budget decision, and five candidates need
human rights review.
Conrad-owned captures are still external. Tests: `uv run pytest tests/unit/data tests/property/data -q`.
