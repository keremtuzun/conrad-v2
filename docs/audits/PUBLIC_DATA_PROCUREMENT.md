# Public data procurement audit (2026-09-19)

This audit replaces the old blanket "EXT-DATA-01 blocked" with one verdict per candidate. The
candidates are the ten that spec ch24 names, which are also the ten manifests in `datasets/public/`.
Every licence below was read from the cited page on 2026-09-19. When no page could be read, the
licence stays `REVIEW_REQUIRED` and the manifest says what a human has to check.

Status meanings:

- **APPROVED_WITH_RESTRICTIONS**: the licence text is explicit (CC BY 4.0 or CC BY-NC-SA 4.0) and allows research use. The restriction column says what still applies.
- **NEEDS_HUMAN_RIGHTS_REVIEW**: there is no licence, it is per asset, or no page could be read.
- **UNAVAILABLE**: no release can be identified under the name the spec uses.

No candidate is plain APPROVED, because every licence found requires at least attribution. None is
REJECTED, because no licence found forbids research use.

## Verdicts

| Candidate | Status | Licence (evidence) | Official source | Conrad task | Downloaded |
|---|---|---|---|---|---|
| UVVID | APPROVED_WITH_RESTRICTIONS | CC BY 4.0 (figshare API `license`, article 27694068 v5) | https://data.dtu.dk/articles/dataset/Underwater_Visual_and_Visual-Inertial_Datasets_UVVID_/27694068 | ECMER quality/representation; 2S smoke only | yes, 10,787,881 B |
| SubPipe | APPROVED (user approved download 2026-09-19) | CC BY 4.0 (Zenodo `license.id=cc-by-4.0`) | https://zenodo.org/records/12666132 | ECMER, side-scan pipeline presence; 2S limited (no nav truth) | yes, SubPipeMini2.zip only, 4,945,761,374 B |
| SeaClear Marine Debris | APPROVED_WITH_RESTRICTIONS | CC BY 4.0 (4TU.ResearchData API `license`) | https://data.4tu.nl/datasets/4f1dff25-e157-4399-a5d4-478055461689/1 | ECMER object-perception baseline | no, a single 1.7 GB rar |
| Underwater Caves sonar | APPROVED_WITH_RESTRICTIONS | CC BY-NC-SA 4.0 (Zenodo `license.id`). The CIRS page gives no licence. | https://zenodo.org/records/7828405 | 2S sonar geometry | no, see note |
| AQUALOC | NEEDS_HUMAN_RIGHTS_REVIEW | none stated | https://www.lirmm.fr/aqualoc/ | 2S | no |
| BenthicNet | NEEDS_HUMAN_RIGHTS_REVIEW | metadata CC BY 4.0; images licensed per source dataset, some NC/ND | https://www.nature.com/articles/s41597-025-04491-1 | 2E, ECMER-SSL via manifest | no |
| FathomNet | NEEDS_HUMAN_RIGHTS_REVIEW | per asset: CC0 / CC-BY / CC-BY-NC / CC-BY-NC-ND (https://www.fathomnet.org/datause) | https://www.fathomnet.org/ | 2E | no |
| Seaview | NEEDS_HUMAN_RIGHTS_REVIEW | unreadable: HTTP 403 on eSpace and RDA | https://espace.library.uq.edu.au/view/UQ:734799 | 2E | no |
| SUIM | NEEDS_HUMAN_RIGHTS_REVIEW | none found (GitHub `NOASSERTION`, dataset page 403) | https://github.com/xahidbuffon/SUIM | ECMER segmentation | no |
| UW-SLAM | UNAVAILABLE | n/a | no release has this exact name | 2S | no |

## Per candidate

**UVVID** (DTU Data, doi:10.11583/DTU.27694068.v5, version 5, published 2026-08-21).
- Content: videos of roughly planar walls and seabottom. The visual subset is `visual/*.mp4`. The visual-inertial subset adds IMU-stabilisation CSVs, `calibration.yaml` and `time_offsets.yml`.
- Timestamps: per-video container time only.
- Calibration: the visual-subset README says "No calibration data or IMU data is available".
- Labels: none.
- Lineage: one video is one sequence. The recording site is not documented, so two videos may come from the same site.
- Restrictions:
  - Attribution is required.
  - Only UVVID-native files are approved. UVVID also redistributes processed AQUALOC and Underwater Caves data, and those files keep their upstream terms. AQUALOC has no licence, so its files are excluded.
- Forbidden uses: supervising any head, localisation ground truth, and cross-video pairing.

**SubPipe** (Zenodo 12666132, version 3.0.1, 2024-07-05).
- Attribution: Alvarez-Tunon, Marnet, Antal, Aubard, Costa, Brodskiy (2024), SubPipe, doi:10.5281/zenodo.12666132. The data belongs to Oceanscan-MST and was acquired in the H2020 REMARO project.
- Approval: the user approved downloading one file, SubPipeMini2.zip, on 2026-09-19. SubPipeMini.zip (6,078,303,615 B) and SubPipe.zip (28,011,923,314 B) were not downloaded.

Download:

| file | bytes | Zenodo md5 (matched) | sha256 |
|---|---|---|---|
| raw/SubPipeMini2.zip | 4,945,761,374 | 7e0d925f93a89bc0e8715e4f6f7caecb | a3068be28471786c726cd6100e0b1d92d1c17615a4dcfe7f5544ba758821188f |

It came from `https://zenodo.org/api/records/12666132/files/SubPipeMini2.zip/content` (resumable curl). Both `conrad data verify --manifest public.subpipe` and `strict_inventory=True` return no problems.

Inventory. The archive root is `SubPipeMiniSSS/`: one mission and 20,089 zip entries.

| stream | files | span | cadence | notes |
|---|---|---|---|---|
| Cam0_images (GoPro) | 16,200 JPEG, 3,993 MB | 540.6 s | 33 ms | 2704x1520 RGB; intrinsics + radtan distortion in config.yaml |
| Cam1_images (grey) | 430 JPEG, 44 MB | 107.2 s | 242 ms | 1936x1216; mount offset only, no intrinsics |
| SSS_LF_images (455 kHz) | 1,055 | 4,173 s | 1.0 s | binary PPM (`P6`) with a `.pbm` name, 2500x500, colormapped |
| SSS_HF_images (900 kHz) | 1,011 (903 `.pbm` + 108 `.bpm`) | 4,172 s | 1.0 s | binary PPM, 5000x500, colormapped |
| CSV (EstimatedState, Altitude, Depth, ForwardDistance, Acceleration, AngularVelocity, Pressure, Rpm, Temperature, WaterVelocity) | 16,200 rows each | cam0 span | per cam0 frame | `image`, `timestamp` columns; the image paths say `Chunk0` |

- Labels: the COCO file per sonar frequency has one category, "Pipeline", with boxes only.
  - LF: 1,055 images listed, 726 boxes, 696 images with at least one box.
  - HF: 903 images listed, 593 boxes, 566 images with at least one box.
  - YOLO copies are also included; the file `d.860.txt` does not follow the stamp naming.
- No RGB segmentation masks: this archive has no `Segmentation` folder.
- Timestamps: UNIX epoch seconds with millisecond resolution in the file names. The adapter converts them to integer ns in clock domain `subpipe:vehicle_unix`, with a time uncertainty of 1 ms.
- Pose: `EstimatedState.csv` is the vehicle's own navigation solution. Its frame conventions are undocumented and it is not ground truth, so the adapter's `robot_pose_estimate` is None.

Manifest (`datasets/public/subpipe.manifest.yaml`):
- It lists the archive plus 8 metadata members extracted losslessly (config.yaml, both COCO files, classes.txt, EstimatedState, Altitude, Depth and ForwardDistance CSVs) as derived files, under transformation `subpipe-zip-extract-v1`.
- Images are never extracted. `SubPipeAdapter` reads them from the verified zip, and zipfile checks each member's CRC-32.

Adapter (`conrad.data.adapters.SubPipeAdapter`, `subpipe_zip-1.0.0`):
- Streams: cam0, cam1, sss_lf, sss_hf.
- Frames: `subpipe.cam0_optical`, `subpipe.cam1_optical`, `subpipe.sss_lf`, `subpipe.sss_hf`.
- `calibration_ref` is set for cam0 only.
- Labels come only through `map_labels((stream, refs))`, which returns `PartialTruth` with `pipeline_present` and `pipeline_box_count`. Images not listed in COCO are masked out. No label value is written to `sensor_context`.

Forbidden uses:
- Pipeline boxes must not supervise 2T material loss, corrosion or fatigue.
- EstimatedState must not be scored as localisation truth.
- One mission is not independent-site evidence.

### DATA-REAL-E002 (SubPipe)

Pre-registration: the hypothesis, split and thresholds were written in `datasets/experiments/data_real_e002.yaml` before the first run.
- Split: the LF sonar images, sorted by time, form 10 equal-count blocks. Blocks 2 and 7 are test, block 4 is validation, the rest is train. 3 images are dropped on each side of a split boundary.
- Probe: a class-balanced L2 logistic probe on the 256-d ECMER event embedding, using a randomly initialised encoder, one per seed.
- Pass rule: test balanced accuracy of at least 0.60, at least 0.10 above the majority baseline, and a 95% bootstrap CI whose lower bound is above 0.5, for every seed.

The run went through `run_experiment` and is recorded in the registry once. Wall time was 2004 s on CPU. Two earlier attempts died with MemoryError while another agent's worker pool held about 56 GB of commit, and recorded nothing.

Data: 1,055 labelled LF images. Train 720 (443 positive), validation 100 (100 positive), test 199 (129 positive), and 36 dropped by the boundary guard.

| seed | ECMER probe test BA [95% CI] | AUROC | quality-only probe BA | quality AUROC | majority BA |
|---|---|---|---|---|---|
| 2026201 | 0.839 [0.783, 0.890] | 0.906 | 0.500 | 0.791 | 0.500 |
| 2026202 | 0.811 [0.751, 0.869] | 0.906 | 0.500 | 0.791 | 0.500 |
| 2026203 | 0.810 [0.747, 0.863] | 0.894 | 0.500 | 0.791 | 0.500 |

(a2) PASS: the mean ECMER-probe balanced accuracy was 0.820. Caveats:
- The fixed validation block turned out to contain only positives, so validation balanced accuracy is undefined. The L2 strength therefore defaulted to the first grid value (0.001). Nothing was tuned on the test blocks, and the split was not changed after seeing the result.
- The engineered-quality probe ranks reasonably (AUROC 0.791), but at the fixed threshold it predicts a single class on the test blocks.
- The encoder is untrained. The result shows that ECMER evidence keeps enough image information for a linear probe to recover the human "Pipeline" presence label on held-out time blocks of the same mission. It is not a trained detector, and it is not evidence of cross-site generalisation.

(a1) camera, 120 cam0 frames at 1 Hz (119 s), PASS on the pre-registered rule. Every corruption moved its feature in the expected direction on 100% of frames, and all embeddings were finite. The feature scaling is not informative on these frames:
- `blur` averages 0.998 (range 0.990 to 0.999), so blurring can only add +0.001.
- `snr_db` averages 112 dB. This is consistent with a near-zero MAD noise estimate on smooth, turbid 4 MP frames, but it was not checked frame by frame.
- The ENGINEERING_ESTIMATE constants in `quality_features.py` need recalibrating on real data before these features can carry weight.

(b) Model2S: not run.
- EstimatedState is not ground truth, and its frame is undocumented.
- The DVL, echo-sounder and side-scan mount geometry needed for range integration is not published.
- Running it would require inventing extrinsics.

**SeaClear** (4TU doi:10.4121/4f1dff25-e157-4399-a5d4-478055461689.v1, 2024-01-08; paper doi:10.1038/s41597-024-03759-2).
- Files: one rar of 1,711,829,309 B (MD5 1cfcf0c2fa3ef0dc219a66f063c2fe99).
- Content: 8,610 ROV images with 40 categories (debris, animals, plants, robot parts), from Bistrina, Jakljan, Lokrum and Slano (Croatia) and Marseille (France). The sites are the lineage unit.
- Forbidden uses: debris or annotation availability must never act as structural-condition supervision.

**Underwater Caves sonar** (Zenodo 7828405; paper doi:10.1177/0278364917732838).
- Sensors: two mechanically scanned sonars, a DVL, two IMUs, a depth sensor and a downward camera. The camera gives ground truth only at specific points.
- Files: `full_dataset.zip` (CSV) is 38,144,710 B and fits the budget. It was not downloaded because no sonar-CSV adapter exists yet; the next step is to download it and write that adapter.
- Restrictions:
  - NC: non-commercial use only.
  - SA: a human must decide whether trained weights count as adaptations.
  - The cave setting is outside Conrad's operating domain.

**AQUALOC** (IJRR 2019, doi:10.1177/0278364919883346).
- Data: hosted on LIRMM Seafile. The page has no licence statement.
- To clear: a human has to ask the corresponding author for written terms.

**BenthicNet** (Scientific Data, doi:10.1038/s41597-025-04491-1).
- Licence: the paper says metadata and models are CC BY 4.0. Images are CC BY 4.0 "except where the original licenses of individual datasets indicate limitations".
- Files: the FRDR files could not be reached from this host.
- To clear: read the licence column of the metadata CSV and keep only CC BY sources.
- Forbidden uses: static imagery gives no longitudinal dynamics.

**FathomNet**.
- Licence: the Data Use Policy has each contributor choose a licence per image (CC0, CC-BY, CC-BY-NC or CC-BY-NC-ND) and per annotation (CC0, CC-BY or CC-BY-NC). For URL-submitted images the policy states availability only, not a licence.
- Why review: this is only approvable per asset, after reading each asset's licence from the API.

**Seaview** (Seaview Survey Photo-quadrat and Image Classification Dataset, UQ eSpace UQ:734799).
- Access: every licence page returned HTTP 403 from this host.
- To clear: a human opens eSpace in a browser and records the licence and version.

**SUIM** (arXiv:2004.01241).
- Licence: the README names no data licence, GitHub reports `NOASSERTION`, and the IRVLab page returned 403. A repository code licence is not a data licence.

**UW-SLAM**.
- Search results: the name matches only code repositories (MecatronicaUSB/uw-slam, chintha/UW-SLAM) and a differently named "UWslam dataset" (Billings and Johnson-Roberson, U. Michigan Deep Blue). That dataset's GitHub README states CC0 1.0, but the Deep Blue record returned 403.
- Decision: spec ch24 forbids silent substitution, so this stays UNAVAILABLE until a human confirms which dataset was meant.

**Spec categories with no named dataset.** Ch24 also lists "Corrosion/crack/material-loss", "NDT/SHM/fatigue" and "Oceanographic field/time-series" as OPEN search tasks. Ch12 mentions "Moorea-type imagery". None of these names a dataset, so there is nothing to audit and no manifest was created. The 2T and 2E searches are still open.

## Download (official sources only)

The files are in `artifacts/data/public.uvvid/raw/` (gitignored), fetched from `https://ndownloader.figshare.com/files/<id>`:

| file | figshare id | bytes | sha256 | MD5 matches figshare |
|---|---|---|---|---|
| ROV_GoPro_1.mp4 | 50436333 | 6,511,081 | fbdfbc39d7c2dc42e0ebf07b4298b5ccd01508bc0a3c9ff221f15c0462d810a2 | yes |
| ROV_GoPro_4.mp4 | 50436318 | 4,276,800 | e364f2faf8c2cc1ed3798fd00e69b8d23548e912e2f90f2d6dd39e0f88b62ace | yes |
| README.md (visual) | 50436324 | 180 | fe05c847746e0b7f14da826ca249c386d728376bae45de8fa0f66c02be1ffa26 | yes |
| manifest.txt | 50981490 | 4,324 | fd99ef0c9fa8777b1a6c20826279f8422eae3385bfe173c746c8abfb977d0c36 | yes |

The total is 10,792,385 bytes. The two MP4s are the inventory in `datasets/public/uvvid.manifest.yaml`, whose files digest is `737c88d4...86b21`. `conrad data verify --manifest public.uvvid` prints `RESULT: OK`.

The CLI treats a `.yaml` path argument differently: it uses the manifest's own directory as the data root, so the path form reports MISSING_FILE. Use the id form.

## Adapter

`conrad.data.adapters.UvvidVideoAdapter` (`uvvid_video-1.0.0`) turns each approved MP4 into a `SequenceSample` of RGB `Observation`s.

- Decoding is the declared transformation `uvvid-decode-v1`.
- Timestamps are container presentation times in ns. Each video has its own clock domain and is never synchronised with another.
- Pose, calibration and labels are `None`. `map_labels` refuses raw labels.
- A changed or missing file makes the adapter fail closed.

## DATA-REAL-SMOKE-E001

Setup:
- Run with: `conrad.evaluation.dispatch.run_experiment("DATA-REAL-SMOKE-E001")`.
- Config: `datasets/experiments/data_real_smoke_e001.yaml`.
- Data: seeds 2026201, 2026202 and 2026203; 2 videos, one frame per second (stride 15), 24 frames each, so 48 real frames of 240x432.
- Wall time: 57.5 s on CPU.

Real-frame statistics (engineered ECMER quality features):

| sequence | blur | brightness | contrast | SNR dB |
|---|---|---|---|---|
| ROV_GoPro_1 | 0.953 | 0.305 | 0.036 | 40.05 |
| ROV_GoPro_4 | 0.564 | 0.507 | 0.067 | 40.93 |

Corruption response (mean delta; the fraction of frames moving in the expected direction was 1.00 for every seed):

| corruption | feature | mean delta |
|---|---|---|
| blur | blur | +0.238 |
| noise | SNR | -21.56 dB |
| darken | brightness | -0.244 |
| darken | contrast | -0.031 |

The check passed: the minimum fraction was 1.0 against a threshold of 0.9, which is an ENGINEERING_ESTIMATE.

Encoding:
- Every seed turned all 48 real observations into finite 256-d Evidence with DIRECT_OBSERVATION provenance.
- Validity with the untrained random weights: all VALID for seeds 2026201 and 2026203, and all DEGRADED for seed 2026202 (its usable score fell below the threshold).
- Under corruption, the learned reliability moved by at most 0.002 (for example 0.516 clean and 0.516 blurred). This finding matters: the engineered features respond to real degradation, but the untrained quality head does not. That is why training stage E3 on a rights-cleared corpus is needed before any claim is made about reliability.

Limitations: 2 short clips; no truth; synthetic corruptions of real frames; untrained ECMER.

The registry record written by `dispatch.run_experiment` hard-codes the limitation "synthetic data only". That text is wrong for this experiment. `dispatch.py` was outside the edit scope, so it is listed as a gap.

## Remaining human actions

1. SeaClear (1.7 GB): decide whether to spend the budget; the licence is already clear. SubPipeMini2 is downloaded. RGB masks would need another SubPipe archive, which is not approved.
2. Underwater Caves: accept the NC-SA terms, then download `full_dataset.zip` (38 MB) and write a sonar adapter.
3. AQUALOC and SUIM: request written terms from the authors.
4. Seaview, UWslam (Deep Blue) and BenthicNet (FRDR): read the licence in a browser, since this host got 403 or no access.
5. FathomNet: decide whether CC-BY-NC-ND assets may be used for training. Otherwise use only CC0 and CC-BY assets, with a per-asset licence lookup.
6. UW-SLAM: confirm which dataset the design discussion meant.
