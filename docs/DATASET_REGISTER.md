# Dataset register

The data manifests on disk. The source is spec ch24 (Dataset Production Registry) and ch35/36.
A listed candidate is a procurement task. It does not mean the data has been downloaded,
licensed, labelled or accepted.

## How a dataset becomes usable

1. The DATA-VERIFY task fills the manifest: source URL, pinned release, asset inventory with a
   sha256 for every file, license evidence, asset-level rights, label audit, calibration and
   split manifest.
2. `conrad.data.manifest.verify_manifest(path, data_root)` must return no problems. It fails
   closed on each of these: missing license or evidence, rights not CLEARED, training not
   permitted, adapter version not pinned, no split lineage, missing or changed files, an
   undeclared transformation, a derived file with no transformation, or an inventory checksum
   mismatch.
3. Local data lives outside git at `$CONRAD_DATA_ROOT/<dataset_id>`, or at
   `artifacts/data/<dataset_id>` when the variable is unset. The runtime doctor calls
   `verify_manifest_by_id("<dataset_id>[@version]")`.
4. Splits come from `conrad.data.splits.build_splits`, which keeps lineage together. Its
   `split_hash` is recorded in every checkpoint and run.

## Public candidates (all `UNVERIFIED_NOT_DOWNLOADED`, license `REVIEW_REQUIRED`)

| manifest | candidate | spec status |
|---|---|---|
| datasets/public/subpipe.manifest.yaml | SubPipe | source pointer checked 2026-09-17; audit open |
| datasets/public/aqualoc.manifest.yaml | AQUALOC | source pointer checked 2026-09-17; audit open |
| datasets/public/benthicnet.manifest.yaml | BenthicNet (seafloor imagery compilation) | source pointer checked 2026-09-17; audit open |
| datasets/public/fathomnet.manifest.yaml | FathomNet | source pointer checked 2026-09-17; rights vary per asset |
| datasets/public/uvvid.manifest.yaml | UVVID | OPEN source audit |
| datasets/public/underwater_caves_sonar.manifest.yaml | Underwater Caves sonar | OPEN source audit |
| datasets/public/uw_slam.manifest.yaml | UW-SLAM | OPEN source audit |
| datasets/public/seaview.manifest.yaml | Seaview | OPEN source audit |
| datasets/public/suim.manifest.yaml | SUIM | OPEN source audit |
| datasets/public/seaclear.manifest.yaml | SeaClear | OPEN source audit |

The ch24 dedicated 2T search (corrosion, crack, material loss, NDT/SHM, fatigue) and the 2E
search (oceanographic time series) have no named candidates yet, so they have no manifests.

`conrad.data.adapters.public_adapter(name)` returns an adapter that raises
`DatasetNotAvailableError` from every data method. The error names the manifest and each
missing artefact.

## Conrad-owned corpus: DATA-CONRAD-SSL-01

Capture-unit rows are `conrad.data.ssl_corpus.CaptureUnit`. The acceptance report comes from
`build_corpus_acceptance_report`. No Conrad real capture exists yet, so `ssl_real_train` is
reported as `EMPTY_BLOCKED_EXTERNAL`.

## Sim-to-real ledger: SIMREAL-LEDGER-01

`datasets/simreal_ledger.yaml` has 18 randomized parameters. Every range is an
ENGINEERING_ESTIMATE and every row is UNMEASURED. `conrad.data.simreal_ledger.load_ledger`
rejects any range that has no physical explanation.
