from __future__ import annotations

import json
from typing import Any

import pytest
import torch

from conrad.training.checkpoint import (
    CheckpointError,
    CompatibilityTuple,
    ReasonCode,
    config_digest,
    load_checkpoint,
    meta_path,
    read_metadata,
    save_checkpoint,
    verify_checkpoint_metadata,
)

CONFIG = {"lr": 3e-4, "dims": 8}
MANIFESTS = ("m" * 64,)
SPLIT = "s" * 64


def make(tmp_path, **kw):
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 2)
    opt = torch.optim.AdamW(model.parameters())
    args: dict[str, Any] = {
        "model": model,
        "optimizer": opt,
        "component": "buo",
        "config": CONFIG,
        "manifest_digests": MANIFESTS,
        "split_hash": SPLIT,
        "seeds": {"torch": 1},
        "metrics": {"val_loss": 0.5},
        "epoch": 1,
        "step": 4,
        "created_time_ns": 10,
        "extra_compatibility": {"twin_adapter_version": "t1"},
    }
    args.update(kw)
    path = tmp_path / "ck.pt"
    save_checkpoint(path, **args)
    return path, model


def expected(**kw) -> CompatibilityTuple:
    base = {
        "config_digest": config_digest(CONFIG),
        "manifest_digests": MANIFESTS,
        "split_hash": SPLIT,
        "component": "buo",
    }
    base.update(kw)
    return CompatibilityTuple(**base)


def test_roundtrip_restores_weights_and_optimizer(tmp_path):
    path, model = make(tmp_path)
    fresh = torch.nn.Linear(3, 2)
    loaded = load_checkpoint(path, expected=expected(), model=fresh)
    assert torch.equal(fresh.weight, model.weight)
    assert loaded.optimizer_state is not None and loaded.metadata.step == 4
    assert loaded.metadata.device_info["device"] == "cpu"


@pytest.mark.parametrize(
    ("save_kw", "expect_kw", "code"),
    [
        ({"architecture_id": "other_arch"}, {}, ReasonCode.ARCHITECTURE_ID_MISMATCH),
        ({"stack_id": "other_stack"}, {}, ReasonCode.STACK_ID_MISMATCH),
        ({}, {"schema_version": "2.0.0"}, ReasonCode.SCHEMA_VERSION_INCOMPATIBLE),
        ({"config": {"lr": 1.0}}, {}, ReasonCode.CONFIG_DIGEST_MISMATCH),
        ({"manifest_digests": ("x" * 64,)}, {}, ReasonCode.MANIFEST_DIGEST_MISMATCH),
        ({"split_hash": "other"}, {}, ReasonCode.SPLIT_HASH_MISMATCH),
        ({"component": "rbp"}, {}, ReasonCode.COMPONENT_MISMATCH),
        ({}, {"representation_pretraining_id": "ssl-1"}, ReasonCode.REPRESENTATION_PRETRAINING_MISMATCH),
        ({}, {"extra": {"robot_config_digest": "r"}}, ReasonCode.EXTRA_ELEMENT_UNKNOWN),
        ({}, {"extra": {"twin_adapter_version": "t2"}}, ReasonCode.EXTRA_ELEMENT_MISMATCH),
    ],
)
def test_each_tuple_element_mismatch_fails_closed(tmp_path, save_kw, expect_kw, code):
    path, _ = make(tmp_path, **save_kw)
    with pytest.raises(CheckpointError) as info:
        load_checkpoint(path, expected=expected(**expect_kw))
    assert code in info.value.codes


def test_payload_tamper_is_detected_before_deserialization(tmp_path):
    path, _ = make(tmp_path)
    path.write_bytes(path.read_bytes() + b"\x00")
    with pytest.raises(CheckpointError) as info:
        load_checkpoint(path, expected=expected())
    assert info.value.codes == (ReasonCode.CONTENT_DIGEST_MISMATCH,)


def test_missing_metadata_and_wrong_model_shape(tmp_path):
    path, _ = make(tmp_path)
    with pytest.raises(CheckpointError) as info:
        load_checkpoint(path, expected=expected(), model=torch.nn.Linear(3, 5))
    assert ReasonCode.MODEL_STATE_MISMATCH in info.value.codes
    meta_path(path).unlink()
    with pytest.raises(CheckpointError) as info:
        read_metadata(path)
    assert info.value.codes == (ReasonCode.METADATA_MISSING,)


def test_verify_checkpoint_metadata(tmp_path):
    path, _ = make(tmp_path)
    assert verify_checkpoint_metadata(path).component == "buo"
    meta = json.loads(meta_path(path).read_text())
    meta["stack_id"] = "foreign"
    meta_path(path).write_text(json.dumps(meta))
    with pytest.raises(CheckpointError) as info:
        verify_checkpoint_metadata(path)
    assert ReasonCode.STACK_ID_MISMATCH in info.value.codes


def test_encoder_needs_pretraining_id_and_selection_is_validation_only(tmp_path):
    with pytest.raises(ValueError):
        make(tmp_path, is_encoder=True)
    with pytest.raises(ValueError):
        make(tmp_path, selection_metric="test_loss")
    path, _ = make(
        tmp_path, is_encoder=True, representation_pretraining_id="ssl-1", selection_metric="val_loss"
    )
    assert load_checkpoint(path, expected=expected(representation_pretraining_id="ssl-1")).metadata.is_encoder


def test_payload_with_arbitrary_objects_is_refused(tmp_path):
    class Opaque:
        pass

    with pytest.raises(CheckpointError) as info:
        make(tmp_path, trainer_state={"obj": Opaque()})
    assert info.value.codes == (ReasonCode.UNSAFE_PAYLOAD,)
