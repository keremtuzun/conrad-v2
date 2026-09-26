"""DATA-OSFM public-real corpus assembly checks for P4.7B."""

from __future__ import annotations

import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any
from uuid import UUID

import yaml

from conrad.data.adapters.subpipe import STREAMS, SubPipeAdapter
from conrad.data.manifest import data_root_for, load_manifest
from conrad.foundation.data.manifest import CorpusPartition, digest_mapping
from conrad.persistence.object_store import ObjectStore
from conrad.schemas.ids import IdFactory
from conrad.settings import REPO_ROOT

SUBPIPE_MANIFEST = REPO_ROOT / "datasets" / "public" / "subpipe.manifest.yaml"
SUBPIPE_SOURCE_ID = "public.subpipe@zenodo-12666132-v3.0.1-SubPipeMini2"
SUBPIPE_STREAMS = ("sss_lf", "sss_hf")
PARTITION_ORDER = (
    CorpusPartition.PRETRAIN_REAL,
    CorpusPartition.VALIDATION,
    CorpusPartition.FINAL_TEST,
    CorpusPartition.OOD_TEST,
)
PARTITION_FRACTIONS = {
    CorpusPartition.PRETRAIN_REAL: 0.70,
    CorpusPartition.VALIDATION: 0.10,
    CorpusPartition.FINAL_TEST: 0.10,
    CorpusPartition.OOD_TEST: 0.10,
}
NORMALIZATION_STAT_PARTITIONS = (CorpusPartition.PRETRAIN_REAL,)


def _partition_ranges(n: int) -> dict[CorpusPartition, tuple[int, int]]:
    train_end = int(n * PARTITION_FRACTIONS[CorpusPartition.PRETRAIN_REAL])
    validation_end = train_end + int(n * PARTITION_FRACTIONS[CorpusPartition.VALIDATION])
    final_end = validation_end + int(n * PARTITION_FRACTIONS[CorpusPartition.FINAL_TEST])
    return {
        CorpusPartition.PRETRAIN_REAL: (0, train_end),
        CorpusPartition.VALIDATION: (train_end, validation_end),
        CorpusPartition.FINAL_TEST: (validation_end, final_end),
        CorpusPartition.OOD_TEST: (final_end, n),
    }


def build_subpipe_osfm_evidence(*, frame_stride: int = 1, smoke_load: bool = True) -> dict[str, Any]:
    """Inspect verified SubPipe through the production adapter and return small corpus evidence."""
    manifest = load_manifest(SUBPIPE_MANIFEST)
    root = data_root_for(manifest.dataset_id)
    with tempfile.TemporaryDirectory(prefix="conrad-osfm-subpipe-") as tmp:
        adapter = SubPipeAdapter(
            manifest,
            root,
            ObjectStore(Path(tmp) / "objects"),
            IdFactory(seed=2026092601),
            UUID("00000000-0000-7000-8000-00000000047b"),
            streams=SUBPIPE_STREAMS,
            frame_stride=frame_stride,
            manifest_path=SUBPIPE_MANIFEST,
        )
        audit = adapter.inspect()
        stream_reports: dict[str, Any] = {}
        partition_members: dict[str, set[str]] = {p.value: set() for p in PARTITION_ORDER}
        lineage_members: dict[str, set[str]] = {p.value: set() for p in PARTITION_ORDER}
        source_frames = 0
        dropped_invalid_items = 0
        first_payload: dict[str, Any] | None = None
        try:
            for stream in SUBPIPE_STREAMS:
                refs = adapter.frames(stream)[::frame_stride]
                source_frames += len(refs)
                ranges = _partition_ranges(len(refs))
                stream_counts: dict[str, int] = {}
                time_ranges: dict[str, dict[str, int | None]] = {}
                sequence_id = f"subpipe/mini_sss/{stream}"
                for partition, (start, end) in ranges.items():
                    selected = refs[start:end]
                    stream_counts[partition.value] = len(selected)
                    time_ranges[partition.value] = {
                        "start_time_ns": selected[0].time_ns if selected else None,
                        "end_time_ns": selected[-1].time_ns if selected else None,
                    }
                    for ref in selected:
                        partition_members[partition.value].add(f"{stream}:{ref.member}")
                    if selected:
                        lineage_members[partition.value].add(sequence_id)
                stream_reports[stream] = {
                    "frame_count": len(refs),
                    "modality": STREAMS[stream].modality.value,
                    "lineage_unit": sequence_id,
                    "partition_counts": stream_counts,
                    "partition_time_ranges_ns": time_ranges,
                }
                if smoke_load and first_payload is None and refs:
                    obs = adapter.normalize_observation(refs[0])
                    first_payload = {
                        "stream": stream,
                        "archive_member": refs[0].member,
                        "shape": list(obs.payload_ref.shape),
                        "dtype": obs.payload_ref.dtype,
                        "sonar_rendering": obs.sensor_context.get("sonar_rendering"),
                        "license": obs.sensor_context.get("license"),
                    }
        finally:
            adapter.close()

    overlaps: dict[str, int] = {}
    parts = list(partition_members)
    for idx, left in enumerate(parts):
        for right in parts[idx + 1 :]:
            overlaps[f"{left}__{right}"] = len(partition_members[left] & partition_members[right])
    partition_counts = {name: len(values) for name, values in partition_members.items()}
    lineage_counts = {name: len(values) for name, values in lineage_members.items()}
    partition_digests = {
        name: digest_mapping({"members": sorted(values), "partition": name})
        for name, values in partition_members.items()
    }
    train_only_normalization = {
        "fit_partitions": [p.value for p in NORMALIZATION_STAT_PARTITIONS],
        "excluded_partitions": [
            CorpusPartition.VALIDATION.value,
            CorpusPartition.FINAL_TEST.value,
            CorpusPartition.OOD_TEST.value,
        ],
        "source_frame_count": partition_counts[CorpusPartition.PRETRAIN_REAL.value],
        "source_digest": partition_digests[CorpusPartition.PRETRAIN_REAL.value],
        "policy": "fit image normalization/statistics on PRETRAIN_REAL only; never on VALIDATION, FINAL_TEST, or OOD_TEST",
    }
    evidence = {
        "schema_version": "1.0.0",
        "evidence_id": "DATA-OSFM-P4.7B-SUBPIPE-PRETRAIN-REAL",
        "dataset_id": manifest.dataset_id,
        "source_id": SUBPIPE_SOURCE_ID,
        "manifest_path": str(SUBPIPE_MANIFEST.relative_to(REPO_ROOT)),
        "manifest_digest": manifest.manifest_digest(),
        "source_digest": manifest.files_digest(),
        "adapter": {"name": SubPipeAdapter.adapter_name, "version": SubPipeAdapter.adapter_version},
        "verification": {
            "adapter_usable": audit.usable,
            "problem_count": len(audit.problems),
            "problems": [str(p) for p in audit.problems],
            "smoke_loaded_first_payload": first_payload is not None,
        },
        "license": manifest.license,
        "rights_review_status": manifest.rights_review_status.value,
        "training_allowed": manifest.training_allowed,
        "deployment_allowed": manifest.deployment_allowed,
        "source_category": manifest.source_category.value,
        "sonar_limitations": (
            "SubPipe sonar is rendered side-scan imagery (colormapped PPM), not raw acoustic backscatter."
        ),
        "lineage_strategy": {
            "highest_available_lineage": "one mission/sequence in SubPipeMiniSSS; stream plus contiguous time block",
            "split_unit": "contiguous time block within each sonar stream",
            "limitation": (
                "SubPipeMini2 metadata available here does not expose independent site, asset, dive, "
                "trajectory, or multi-sequence hierarchy beyond the single mission/sequence."
            ),
            "random_frame_split": False,
        },
        "streams": stream_reports,
        "source_frame_count": source_frames,
        "partition_counts": partition_counts,
        "lineage_counts": lineage_counts,
        "partition_overlap_counts": overlaps,
        "dropped_invalid_items": dropped_invalid_items,
        "partition_digests": partition_digests,
        "train_only_normalization": train_only_normalization,
        "first_payload": first_payload,
    }
    evidence["evidence_digest"] = digest_mapping({k: v for k, v in evidence.items() if k != "evidence_digest"})
    return evidence


def write_subpipe_osfm_evidence(output: str | Path) -> Path:
    path = Path(output)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    evidence = build_subpipe_osfm_evidence()
    path.write_text(json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_subpipe_osfm_evidence(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{p} must contain a JSON object")
    return data


def validate_subpipe_osfm_evidence(evidence: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    try:
        live = build_subpipe_osfm_evidence()
    except Exception as exc:  # fail closed for readiness
        return [f"SubPipe evidence could not be rebuilt through the adapter: {exc}"]
    checks = (
        "dataset_id",
        "manifest_digest",
        "source_digest",
        "source_frame_count",
        "partition_counts",
        "lineage_counts",
        "partition_overlap_counts",
        "partition_digests",
        "train_only_normalization",
        "license",
        "rights_review_status",
        "training_allowed",
    )
    for key in checks:
        if evidence.get(key) != live.get(key):
            problems.append(f"{key} does not match live SubPipe adapter evidence")
    if not evidence.get("verification", {}).get("adapter_usable"):
        problems.append("SubPipe adapter is not usable")
    if evidence.get("verification", {}).get("problem_count") != 0:
        problems.append("SubPipe adapter reported verification problems")
    if not evidence.get("verification", {}).get("smoke_loaded_first_payload"):
        problems.append("SubPipe first-payload smoke did not run")
    if evidence.get("license") != "CC-BY-4.0":
        problems.append("SubPipe license gate is not CC-BY-4.0")
    if evidence.get("rights_review_status") != "CLEARED":
        problems.append("SubPipe rights review is not CLEARED")
    if evidence.get("training_allowed") is not True:
        problems.append("SubPipe training_allowed is not true")
    overlaps = Counter(evidence.get("partition_overlap_counts") or {})
    leaking = {k: v for k, v in overlaps.items() if v}
    if leaking:
        problems.append(f"partition overlaps are non-zero: {leaking}")
    norm = evidence.get("train_only_normalization") or {}
    if norm.get("fit_partitions") != [CorpusPartition.PRETRAIN_REAL.value]:
        problems.append("normalization/statistics are not fit on PRETRAIN_REAL only")
    for partition in (CorpusPartition.VALIDATION, CorpusPartition.FINAL_TEST, CorpusPartition.OOD_TEST):
        if partition.value not in set(norm.get("excluded_partitions") or []):
            problems.append(f"{partition.value} is not excluded from normalization/statistics fitting")
    first = evidence.get("first_payload") or {}
    if first.get("sonar_rendering") != "colormapped PPM, not raw backscatter":
        problems.append("rendered side-scan sonar limitation was not preserved in adapter metadata")
    return problems


def corpus_sources_from_subpipe_evidence(evidence_path: str | Path) -> list[dict[str, Any]]:
    evidence = load_subpipe_osfm_evidence(evidence_path)
    manifest_digest = evidence["manifest_digest"]
    partition_digests = evidence["partition_digests"]
    return [
        {
            "source_id": f"{SUBPIPE_SOURCE_ID}:{partition.value.lower()}",
            "manifest_digest": manifest_digest,
            "source_digest": partition_digests[partition.value],
            "partition": partition.value,
            "first_party": False,
        }
        for partition in PARTITION_ORDER
    ]


def write_subpipe_osfm_corpus(output: str | Path, evidence_path: str | Path) -> Path:
    path = Path(output)
    if not path.is_absolute():
        path = REPO_ROOT / path
    evidence_ref = Path(evidence_path)
    evidence = load_subpipe_osfm_evidence(evidence_ref)
    synthetic_payload = {
        "partition": CorpusPartition.PRETRAIN_SYNTHETIC.value,
        "kind": "approved-non-promotable-synthetic-staging",
    }
    corpus = {
        "corpus_id": "DATA-OSFM-01",
        "readiness": "READY",
        "sources": [
            *corpus_sources_from_subpipe_evidence(evidence_ref),
            {
                "source_id": "synthetic-pretrain-synthetic-staging",
                "manifest_digest": digest_mapping(synthetic_payload),
                "source_digest": digest_mapping(synthetic_payload),
                "partition": CorpusPartition.PRETRAIN_SYNTHETIC.value,
                "first_party": False,
            },
        ],
        "required_partitions": [p.value for p in CorpusPartition],
        "notes": [
            "P4.7B public-real DATA-OSFM corpus assembly for P4.8+ readiness.",
            f"public-real evidence: {evidence_ref.as_posix()}",
            "PRETRAIN_REAL uses verified local SubPipeMini2 rendered side-scan imagery only.",
            "SubPipe sonar is rendered side-scan imagery, not raw acoustic backscatter.",
            "SubPipeMini2 has one mission/sequence; partitions are contiguous time blocks per sonar stream.",
            "normalization/statistics fit on PRETRAIN_REAL only; validation/final/OOD excluded.",
            f"evidence_digest: {evidence['evidence_digest']}",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(corpus, sort_keys=False), encoding="utf-8")
    return path
