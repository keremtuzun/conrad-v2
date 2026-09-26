from pathlib import Path

import pytest
import torch

from conrad.foundation.encoders.rgb import RGBEncoderConfig, RGBViTS14Encoder
from conrad.foundation.pretraining.dinov2 import (
    DINOV2_VITS14_BYTES,
    DINOV2_VITS14_SHA256,
    import_dinov2_vits14,
    verify_dinov2_checkpoint,
)


CHECKPOINT = Path("artifacts/external/dinov2/dinov2_vits14.pth")


def test_official_checkpoint_hash_and_size_are_verified() -> None:
    result = verify_dinov2_checkpoint(CHECKPOINT)
    assert result == {"path": str(CHECKPOINT), "sha256": DINOV2_VITS14_SHA256, "byte_length": DINOV2_VITS14_BYTES}


def test_wrong_checkpoint_fails_closed(tmp_path: Path) -> None:
    wrong = tmp_path / "wrong.pth"
    wrong.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_dinov2_checkpoint(wrong)


def test_official_checkpoint_maps_to_rgb_encoder() -> None:
    encoder = RGBViTS14Encoder(RGBEncoderConfig(image_size=28))
    mapped = import_dinov2_vits14(CHECKPOINT, encoder)
    incompatible, missing = encoder.load_state_dict(mapped, strict=False)
    assert incompatible == []
    assert missing == []
    assert mapped["pos_embed"].shape == (1, 5, 384)
