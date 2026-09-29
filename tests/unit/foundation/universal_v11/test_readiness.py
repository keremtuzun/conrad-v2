import json
from pathlib import Path

from conrad.foundation.universal_v11.readiness import evaluate_v11_10p_readiness


def test_v11_10p_readiness_is_code_backed_and_fail_closed() -> None:
    report = evaluate_v11_10p_readiness()
    items = {item["id"]: item for item in report["items"]}

    assert report["gate_id"] == "OSFM-V11-10P-KEREM-HANDOFF-READINESS"
    assert report["status"] == "VALIDATED-RUN"
    assert report["architecture"]["registry_size"] == 829
    assert 250_000_000 <= report["architecture"]["parameter_count"]["total"] <= 320_000_000
    assert items[2]["status"] == "PASS"
    assert items[3]["status"] == "PASS"
    assert items[4]["status"] == "PASS"
    assert "formal_training_allowed=False" in items[4]["evidence"]
    assert "launch_command_registered=True" in items[4]["evidence"]
    assert "reviewed_trainer_registered=True" in items[4]["evidence"]
    assert items[5]["status"] == "PASS"
    assert "SubPipeMini2.sha256_ok=True" in items[5]["evidence"]
    assert items[6]["status"] == "PASS"
    assert "digest_ok=True" in items[6]["evidence"]
    assert items[8]["status"] == "PASS"
    assert items[10]["status"] == "PASS"
    assert "fallback_user_approval=APPROVED_1L4_FALLBACK" in items[10]["evidence"]
    assert "fallback_microbenchmark_blockers=['rank_below_75']" in items[10]["evidence"]
    assert report["decision"] == "READY FOR KEREM"
    assert report["approved_launch_mode"] == "1L4_FALLBACK_USER_APPROVED"
    assert report["launch_command"] is not None


def test_v11_10p_readiness_has_all_requested_rows() -> None:
    report = evaluate_v11_10p_readiness()

    assert [item["id"] for item in report["items"]] == list(range(1, 13))
    assert all(item["status"] in {"PASS", "WARNING", "BLOCKER"} for item in report["items"])


def test_v11_10p_required_payload_manifest_is_hash_pinned() -> None:
    manifest = json.loads(Path("artifacts/gates/V1.1/10P_KEREM_HANDOFF/required_payloads.json").read_text())
    payloads = {payload["id"]: payload for payload in manifest["payloads"]}

    assert payloads["upstream_checkpoint"]["sha256"] == (
        "b2dfc162c1d72e4363643b516b6566eb0f11ca4a874f9674fc5b06f1ddf85696"
    )
    assert payloads["subpipe_dataset"]["sha256"] == (
        "a3068be28471786c726cd6100e0b1d92d1c17615a4dcfe7f5544ba758821188f"
    )
    assert all(payload["git_tracked"] is False for payload in payloads.values())
