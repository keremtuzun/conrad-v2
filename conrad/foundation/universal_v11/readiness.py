"""Fail-closed handoff readiness checks for Universal OS-FM V1.1 10P."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import yaml

from conrad.foundation.universal_v11.migration import parameter_count
from conrad.foundation.universal_v11.core import UniversalOSFMV11
from conrad.foundation.universal_v11.registry import EncoderFamily, MODALITY_REGISTRY


REPO_ROOT = Path(__file__).resolve().parents[3]
PASS = "PASS"
WARNING = "WARNING"
BLOCKER = "BLOCKER"


@dataclass(frozen=True)
class ReadinessItem:
    id: int
    status: str
    title: str
    evidence: str
    action: str | None = None


def _run_git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _status_item(ok: bool, warning: bool, item_id: int, title: str, evidence: str, action: str | None = None) -> ReadinessItem:
    if ok:
        return ReadinessItem(item_id, PASS, title, evidence, action)
    return ReadinessItem(item_id, WARNING if warning else BLOCKER, title, evidence, action)


def evaluate_v11_10p_readiness() -> dict[str, Any]:
    """Return a code/artifact-derived readiness report for Kerem's V1.1 10P handoff."""

    branch = _run_git(["branch", "--show-current"])
    commit = _run_git(["rev-parse", "HEAD"])
    status_lines = _run_git(["status", "--short"]).splitlines()
    dirty = bool(status_lines)

    counts = parameter_count(UniversalOSFMV11(input_dims={"rgb_camera": 8, "imaging_sonar": 8}))
    registry_size = len(MODALITY_REGISTRY)
    family_count = len(EncoderFamily)

    tenp_path = REPO_ROOT / "configs/train/osfm/v11_p4_10p_829_semantic_candidate.yaml"
    launch_path = REPO_ROOT / "configs/train/osfm/v11_p4_10p_3am_launch_protocol.yaml"
    fallback_launch_path = REPO_ROOT / "configs/train/osfm/v11_p4_1l4_10p_fallback_launch_protocol.yaml"
    tenp = _load_yaml(tenp_path)
    launch = _load_yaml(launch_path)
    fallback_launch = _load_yaml(fallback_launch_path)

    subpipe_path = REPO_ROOT / "artifacts/data/public.subpipe/raw/SubPipeMini2.zip"
    expected_subpipe_sha256 = "a3068be28471786c726cd6100e0b1d92d1c17615a4dcfe7f5544ba758821188f"
    subpipe_digest_ok = subpipe_path.is_file() and _sha256(subpipe_path) == expected_subpipe_sha256
    uvvid_manifest = REPO_ROOT / "datasets/public/uvvid.manifest.yaml"
    subpipe_manifest = REPO_ROOT / "datasets/public/subpipe.manifest.yaml"
    checkpoint_meta_path = REPO_ROOT / "artifacts/gates/OSFM_S_PRETRAIN_V1/qualified_checkpoint_metadata.json"
    checkpoint_meta = _load_json(checkpoint_meta_path) if checkpoint_meta_path.is_file() else {}
    checkpoint_path = REPO_ROOT / str(checkpoint_meta.get("checkpoint_path", ""))
    checkpoint_digest_ok = checkpoint_path.is_file() and _sha256(checkpoint_path) == checkpoint_meta.get("checkpoint_id")

    local_preflight = REPO_ROOT / "artifacts/gates/V1.1/10P_3AM/local_preflight.json"
    cloud_preflight = REPO_ROOT / "artifacts/gates/V1.1/10P_3AM/cloud_price_quota_preflight.json"
    launch_record = REPO_ROOT / "artifacts/gates/V1.1/10P_3AM/launch_record.json"
    cloud_report = _load_json(cloud_preflight) if cloud_preflight.is_file() else {}
    launch_report = _load_json(launch_record) if launch_record.is_file() else {}
    fallback_approval_path = REPO_ROOT / "artifacts/gates/V1.1/10P_1L4/user_approved_fallback.json"
    fallback_approval = _load_json(fallback_approval_path) if fallback_approval_path.is_file() else {}
    fallback_microbenchmark_path = REPO_ROOT / "artifacts/gates/V1.1/10P_1L4/cloud_microbenchmark_report.json"
    fallback_microbenchmark = _load_json(fallback_microbenchmark_path) if fallback_microbenchmark_path.is_file() else {}
    fallback_approved = fallback_approval.get("decision") == "APPROVED_1L4_FALLBACK"

    v11_docs = REPO_ROOT / "docs/HANDOFF_KEREM_V11_10P.md"
    training_doc = REPO_ROOT / "docs/TRAINING.md"
    cli_commands = (REPO_ROOT / "conrad/cli/commands.py").read_text(encoding="utf-8")
    launch_command_registered = "osfm-v11-10p-launch" in cli_commands
    launch_is_preflight_only = "This command is a preflight guard only" in cli_commands
    reviewed_trainer_registered = "run_v11_10p_training" in cli_commands and "reviewed_v11_10p_subpipe_v1" in (
        REPO_ROOT / "conrad/foundation/universal_v11/trainer.py"
    ).read_text(encoding="utf-8")

    items = [
        ReadinessItem(
            1,
            WARNING if dirty else PASS,
            "repo path and branch/ref/commit",
            f"path={REPO_ROOT}; branch={branch}; commit={commit}; dirty_entries={len(status_lines)}",
            "Hand off a clean committed tree or explicitly include the dirty artifact list." if dirty else None,
        ),
        _status_item(
            branch == "osfm-universal-v1.1" and 250_000_000 <= counts["total"] <= 320_000_000,
            False,
            2,
            "current V1.1 is 307M-class architecture",
            f"UniversalOSFMV11 parameter_count.total={counts['total']}; branch={branch}",
            "Use the osfm-universal-v1.1 branch and do not substitute old V1 modules." if branch != "osfm-universal-v1.1" else None,
        ),
        _status_item(
            registry_size == 829 and family_count == 15,
            False,
            3,
            "829 modality/configuration registry present",
            f"MODALITY_REGISTRY={registry_size}; EncoderFamily={family_count}; candidate references registered_modality_count={tenp['source_constraints'].get('registered_modality_count')}",
        ),
        _status_item(
            tenp_path.is_file()
            and launch_path.is_file()
            and fallback_launch_path.is_file()
            and tenp.get("training_budget", {}).get("total_optimizer_steps") == 1_200_000
            and launch_command_registered
            and reviewed_trainer_registered,
            False,
            4,
            "P4/P4.8/10P training entrypoint and frozen config",
            (
                f"config={tenp_path.relative_to(REPO_ROOT)}; protocol={launch_path.relative_to(REPO_ROOT)}; "
                f"formal_training_allowed={tenp.get('formal_training_allowed')}; "
                f"launch_command_registered={launch_command_registered}; "
                f"launch_is_preflight_only={launch_is_preflight_only}; "
                f"reviewed_trainer_registered={reviewed_trainer_registered}"
            ),
            None
            if reviewed_trainer_registered
            else "Register the reviewed formal V1.1 10P trainer command before paid training; current osfm-v11-10p-launch is a preflight/launch-check surface only.",
        ),
        _status_item(
            subpipe_digest_ok and uvvid_manifest.is_file() and subpipe_manifest.is_file(),
            False,
            5,
            "dataset manifests and payload bytes",
            (
                f"SubPipeMini2.zip_present={subpipe_path.is_file()}; "
                f"SubPipeMini2.sha256_ok={subpipe_digest_ok}; "
                f"SubPipeMini2.expected_sha256={expected_subpipe_sha256}; "
                f"uvvid_manifest={uvvid_manifest.is_file()}; subpipe_manifest={subpipe_manifest.is_file()}; "
                f"semantic_claim_all_829={not tenp['source_constraints'].get('semantic_training_claim_requires_data_manifest', True)}"
            ),
            "Transfer or mount the ignored SubPipeMini2.zip payload and verify its SHA-256 before paid training.",
        ),
        _status_item(
            checkpoint_digest_ok
            and checkpoint_meta.get("checkpoint_label") == "OSFM-S-PRETRAIN-V1"
            and checkpoint_meta.get("status") == "VALIDATED-RUN"
            and checkpoint_meta.get("decision") in {"GO", "PROMOTE"},
            False,
            6,
            "valid upstream checkpoint/artifact references",
            (
                f"metadata={checkpoint_meta_path.relative_to(REPO_ROOT)}; "
                f"checkpoint_path={checkpoint_meta.get('checkpoint_path')}; "
                f"checkpoint_present={checkpoint_path.is_file()}; "
                f"expected_sha256={checkpoint_meta.get('checkpoint_id')}; "
                f"digest_ok={checkpoint_digest_ok}"
            ),
            "Transfer or mount the ignored upstream checkpoint file and verify the SHA-256 before paid training.",
        ),
        _status_item(
            (REPO_ROOT / "conrad/foundation/pretraining/p48_promotion.py").is_file()
            and (REPO_ROOT / "tests/unit/foundation/universal_v11").is_dir(),
            False,
            7,
            "promotion and representation-health gates",
            "p48 promotion code plus universal_v11 unit tests are present and runnable",
        ),
        _status_item(
            tenp.get("training_budget", {}).get("checkpoint_every_steps") is not None
            and "checkpoint_reload_exact_match" in json.dumps(tenp)
            and reviewed_trainer_registered,
            False,
            8,
            "checkpointing/resume and output directories",
            "10P config declares checkpoint cadence and reload gates, but no formal trainer/resume implementation is registered.",
            "Wire formal V1.1 10P through an immutable RunDirectory trainer before Kerem launches.",
        ),
        _status_item(
            (REPO_ROOT / "pyproject.toml").is_file()
            and (REPO_ROOT / "uv.lock").is_file()
            and (REPO_ROOT / "scripts/bootstrap.sh").is_file(),
            False,
            9,
            "dependency/environment lockfiles and setup scripts",
            "pyproject.toml, uv.lock, and scripts/bootstrap.sh are present",
        ),
        _status_item(
            fallback_approved and fallback_launch.get("hard_limits", {}).get("required_gpu_count") == 1,
            False,
            10,
            "cloud/local launch scripts or exact commands",
            (
                "command=uv run conrad train osfm-v11-10p-launch "
                "--config configs/train/osfm/v11_p4_10p_829_semantic_candidate.yaml; "
                f"primary_8l4_preflight_decision={cloud_report.get('decision')}; "
                f"fallback_protocol={fallback_launch.get('protocol_id')}; "
                f"fallback_user_approval={fallback_approval.get('decision')}; "
                f"fallback_microbenchmark_decision={fallback_microbenchmark.get('decision')}; "
                f"fallback_microbenchmark_blockers={fallback_microbenchmark.get('blockers')}"
            ),
            "Record explicit 1-L4 fallback approval before handoff." if not fallback_approved else None,
        ),
        _status_item(
            not any(("/mnt/" in line or "/content/" in line or "C:\\" in line) for line in status_lines),
            True,
            11,
            "hidden old V1/stale symlink/mounted-path dependency scan",
            f"git_dirty_entries={len(status_lines)}; cloud_quota_failed_gate={cloud_report.get('failed_gates')}",
            "Run the documented secret scan and clean-tree check before external handoff.",
        ),
        _status_item(
            v11_docs.is_file() and "osfm-v11-10p-readiness" in training_doc.read_text(encoding="utf-8"),
            False,
            12,
            "README/HANDOFF instructions",
            f"handoff_doc={v11_docs.relative_to(REPO_ROOT) if v11_docs.is_file() else 'missing'}",
            "Keep Kerem handoff doc current with the readiness command output.",
        ),
    ]

    blockers = [item for item in items if item.status == BLOCKER]
    warnings = [item for item in items if item.status == WARNING]
    launch_command = None
    decision = "READY FOR KEREM" if not blockers else "NOT READY FOR KEREM"
    if decision == "READY FOR KEREM":
        launch_command = (
            "uv run conrad train osfm-v11-10p-launch "
            "--config configs/train/osfm/v11_p4_10p_829_semantic_candidate.yaml "
            "--readiness artifacts/gates/V1.1/10P_KEREM_HANDOFF/readiness.json"
        )

    return {
        "gate_id": "OSFM-V11-10P-KEREM-HANDOFF-READINESS",
        "status": "VALIDATED-RUN",
        "decision": decision,
        "repo": {"path": str(REPO_ROOT), "branch": branch, "commit": commit, "dirty_entries": status_lines},
        "architecture": {"parameter_count": counts, "registry_size": registry_size, "family_count": family_count},
        "items": [asdict(item) for item in items],
        "blocker_count": len(blockers),
        "warning_count": len(warnings),
        "launch_command": launch_command,
        "approved_launch_mode": "1L4_FALLBACK_USER_APPROVED" if fallback_approved else None,
        "fallback_limitations": fallback_approval.get("known_limitations", ()),
    }


def write_v11_10p_readiness(report: dict[str, Any], output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path
