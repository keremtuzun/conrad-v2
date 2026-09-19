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

## Public candidates (audit 2026-09-19, [audits/PUBLIC_DATA_PROCUREMENT.md](audits/PUBLIC_DATA_PROCUREMENT.md))

Each manifest carries `procurement_status`, `verified_on`, `source_urls`, `conrad_tasks`, `restrictions` and
`forbidden_uses`. A PUBLIC_REAL manifest may list files only when `procurement_status` is APPROVED or
APPROVED_WITH_RESTRICTIONS.

| manifest | candidate | procurement_status | licence | on disk |
|---|---|---|---|---|
| datasets/public/uvvid.manifest.yaml | UVVID (DTU, v5) | APPROVED_WITH_RESTRICTIONS | CC-BY-4.0 | 2 MP4s, 10.8 MB, VERIFIED |
| datasets/public/subpipe.manifest.yaml | SubPipe (Zenodo v3.0.1) | APPROVED_WITH_RESTRICTIONS | CC-BY-4.0 | no (smallest file 4.9 GB) |
| datasets/public/seaclear.manifest.yaml | SeaClear (4TU v1) | APPROVED_WITH_RESTRICTIONS | CC-BY-4.0 | no (1.7 GB) |
| datasets/public/underwater_caves_sonar.manifest.yaml | Underwater Caves (Zenodo) | APPROVED_WITH_RESTRICTIONS | CC-BY-NC-SA-4.0 | no |
| datasets/public/aqualoc.manifest.yaml | AQUALOC | NEEDS_HUMAN_RIGHTS_REVIEW | none stated | no |
| datasets/public/benthicnet.manifest.yaml | BenthicNet | NEEDS_HUMAN_RIGHTS_REVIEW | per source dataset | no |
| datasets/public/fathomnet.manifest.yaml | FathomNet | NEEDS_HUMAN_RIGHTS_REVIEW | per asset | no |
| datasets/public/seaview.manifest.yaml | Seaview | NEEDS_HUMAN_RIGHTS_REVIEW | unreadable (403) | no |
| datasets/public/suim.manifest.yaml | SUIM | NEEDS_HUMAN_RIGHTS_REVIEW | none found | no |
| datasets/public/uw_slam.manifest.yaml | UW-SLAM | UNAVAILABLE | n/a | no |

The ch24 dedicated 2T search (corrosion, crack, material loss, NDT/SHM, fatigue) and the 2E search
(oceanographic time series) name no dataset, so they have no manifests.

`conrad.data.adapters.UvvidVideoAdapter` reads the UVVID sample. Verify it with
`conrad data verify --manifest public.uvvid`. `public_adapter(name)` still raises
`DatasetNotAvailableError` for every other candidate, naming the manifest and each missing artefact.

## Conrad-owned corpus: DATA-CONRAD-SSL-01

Capture-unit rows are `conrad.data.ssl_corpus.CaptureUnit`. The acceptance report comes from
`build_corpus_acceptance_report`. No Conrad real capture exists yet, so `ssl_real_train` is
reported as `EMPTY_BLOCKED_EXTERNAL`.

## Sim-to-real ledger: SIMREAL-LEDGER-01

`datasets/simreal_ledger.yaml` has 18 randomized parameters. Every range is an
ENGINEERING_ESTIMATE and every row is UNMEASURED. `conrad.data.simreal_ledger.load_ledger`
rejects any range that has no physical explanation.
