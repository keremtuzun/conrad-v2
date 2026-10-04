"""Write manifests for the extra sonar datasets from a hashes file (md5 sha256 size path per line).

Run inside conrad-v2: uv run python scripts/make_sonar_manifests.py <hashes4.txt>. Each md5 is compared with the repository's
published checksum before its sha256 is recorded; a mismatch aborts.
"""
import sys
from pathlib import Path

import yaml

BASE = yaml.safe_load(Path("datasets/public/subpipe_full.manifest.yaml").read_text())
DATASETS = {
    "sss_catalunya": {
        "title": "A Large Scale Side-Scan Sonar Dataset of Seafloor Sediments for Self-Supervised Pretraining",
        "source": "https://zenodo.org/records/10209445", "release": "doi:10.5281/zenodo.10209445 (version 1.0)",
        "license": "CC-BY-4.0", "evidence": "Zenodo record metadata license.id=cc-by-4.0 (https://zenodo.org/api/records/10209445), read 2026-10-04",
        "md5": {"sss_ssl_dataset_N713_384.z01": "6a41f1196f12c17491e28c7a84672194", "sss_ssl_dataset_N713_384.z02": "0b90b20d98ef48e0eabf168d9071d193",
                "sss_ssl_dataset_N713_384.zip": "1ba31d40f0cde4dc374a78378a88df99"},
        "modalities": ["SONAR"], "role": "434,164 side-scan sonar patches (384 px TIFF) from Catalan coastal surveys; self-supervised imaging_sonar pretraining.",
    },
    "china_offshore_sss": {
        "title": "China Offshore SSS-AI: multi-region side-scan sonar image dataset for cross-regional seafloor target recognition",
        "source": "https://zenodo.org/records/20048164", "release": "doi:10.5281/zenodo.20048164",
        "license": "CC-BY-4.0", "evidence": "Zenodo record metadata license.id=cc-by-4.0 (https://zenodo.org/api/records/20048164), read 2026-10-04",
        "md5": {"China-Offshore-SSS-AI_Zenodo_public_upload.zip": "e01811b3067043eefa745b70c965df67"},
        "modalities": ["SONAR"], "role": "3,255 side-scan target chips from four sea regions (Shenzhen, Dongying, Quanzhou, Yantai); images only, labels unused.",
    },
    "aquascan_1k": {
        "title": "AquaScan-1K Side Scan Sonar Dataset", "source": "https://zenodo.org/records/18771165", "release": "doi:10.5281/zenodo.18771165",
        "license": "MIT", "evidence": "Zenodo record metadata license.id=mit-license (https://zenodo.org/api/records/18771165), read 2026-10-04",
        "md5": {"AquaScan-1K.zip": "db568d8add83bf00fe81b4122aa851d7"},
        "modalities": ["SONAR"], "role": "1,033 lakebed side-scan sonar screenshots; images only, labels unused.",
    },
    "uatd": {
        "title": "Underwater acoustic target detection (UATD) dataset", "source": "https://figshare.com/articles/dataset/21331143",
        "release": "doi:10.6084/m9.figshare.21331143; paper doi:10.1038/s41597-022-01854-w", "license": "CC-BY-4.0",
        "evidence": "figshare API license 'CC BY 4.0' for article 21331143, read 2026-10-04",
        "md5": {"UATD_Training.zip": "891b146871edcda313cdbd8cd6252f39", "UATD_Test_1.zip": "267818131310873c05950879bfec3e6f",
                "UATD_Test_2.zip": "979479864b52925edca49074a09c1b3d", "UATD_OpenSLT.zip": "d62542fa284a4cc1119687bfa9141754"},
        "modalities": ["SONAR"], "role": "About 9,200 multibeam forward-looking sonar images (Tritech Gemini 1200ik), lake and shallow water; images only.",
    },
}

rows = {}
for line in Path(sys.argv[1]).read_text().splitlines():
    parts = line.split(maxsplit=3)
    if len(parts) == 4 and len(parts[0]) == 32:
        md5, sha, size, path = parts
        rows[Path(path).name] = (md5, sha, int(size))

for key, d in DATASETS.items():
    files = []
    for name, md5 in d["md5"].items():
        got = rows.get(name)
        if got is None:
            raise SystemExit(f"{key}: {name} missing from hashes")
        if got[0] != md5:
            raise SystemExit(f"{key}: {name} md5 {got[0]} != published {md5}")
        files.append({"path": f"raw/{name}", "sha256": got[1], "byte_length": got[2], "stream": "archive", "modality": "STRUCTURED",
                      "lineage": {"sequence": f"{key}/all"}})
    m = dict(BASE)
    m.update({
        "dataset_id": f"public.{key}", "version": d["release"], "status": "APPROVED", "source": d["source"], "release_id": d["release"],
        "retrieved_at": "2026-10-04", "license": d["license"], "license_evidence_ref": d["evidence"],
        "usage_rights": f"{d['license']}: research, adaptation and training with attribution (no non-commercial or share-alike term)",
        "rights_by_asset_ref": "single record; all files share the licence", "rights_review_status": "CLEARED",
        "training_allowed": True, "redistribution_allowed": False, "deployment_allowed": False,
        "checksum": files[0]["sha256"], "modalities": d["modalities"], "labels": [], "label_schema_ref": "UNUSED (self-supervised)",
        "frames": {}, "units": {}, "truth_availability": "NONE", "files": files,
        "intended_role": d["role"], "permitted_tasks": ["self-supervised imaging_sonar representation pretraining (V1.1 P4)"],
        "required_files_note": "Only the archives are required; images are read from them by the V1.1 Kerem trainer.",
        "split_strategy": ["sequence"], "split_manifest_ref": "per-group contiguous PRETRAIN_REAL/VALIDATION blocks (trainer _split_groups)",
        "duplicate_audit_ref": "OPEN", "remaining_verification": ["frame-level duplicate audit"],
    })
    header = (f"# {d['title']}\n# Downloaded 2026-10-04 for V1.1 P4 sonar pretraining (Kerem trainer); every file's md5 matched the "
              f"repository's published checksum before its sha256 was recorded.\n")
    Path(f"datasets/public/{key}.manifest.yaml").write_text(header + yaml.safe_dump(m, sort_keys=False, width=110))
    print("wrote", key, len(files), "files")
