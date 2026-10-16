"""Kerem-variant V1.1 trainer: readiness override scope, splits and the per-source sonar batching paths."""

import pytest
import torch

from conrad.foundation.universal_v11 import trainer as T

SOURCES = ("subpipe", "catalunya", "china_offshore", "aquascan", "uatd")


def _batcher(refs_by_source: dict[str, list[str]], batch_size: int = 64) -> T._MultiSourceBatcher:
    """A batcher holding synthetic frames whose pixel value is the frame's position in its training list."""
    b = T._MultiSourceBatcher.__new__(T._MultiSourceBatcher)
    b.batch_size, b.device, b.seed = batch_size, torch.device("cpu"), 1
    b.data = {"imaging_sonar": {}}
    for source, refs in refs_by_source.items():
        frames = (
            torch.arange(len(refs), dtype=torch.uint8)[:, None, None, None].expand(-1, 1, 4, 4).contiguous()
        )
        b.data["imaging_sonar"][source] = {
            "train": tuple(refs),
            "val": (),
            "frames": frames,
            "train_index": torch.arange(len(refs)),
            "sequential": True,
        }
    return b


def _positions(images: torch.Tensor) -> torch.Tensor:
    return (images[:, 0, 0, 0] * 255).round().long()


def test_readiness_override_only_covers_unreviewed_trainer_blockers() -> None:
    def readiness(ids: list[int]) -> dict:
        items = [{"id": i, "status": "BLOCKER", "title": f"b{i}", "evidence": "e"} for i in ids]
        return {"decision": "NOT READY FOR KEREM", "items": items}

    record = T.readiness_override_record(readiness([2, 4, 8]))
    assert [b["id"] for b in record["overridden_blockers"]] == [2, 4, 8]
    with pytest.raises(RuntimeError):
        T.readiness_override_record(readiness([2, 4, 5]))
    with pytest.raises(RuntimeError):
        T.readiness_override_record(readiness([]))


def test_split_groups_keeps_train_before_validation_inside_every_recording() -> None:
    groups = {"a": [f"a{i:03d}" for i in range(100)], "b": [f"b{i:03d}" for i in range(50)]}
    train, val, counts = T._split_groups(groups)
    assert not set(train) & set(val)
    for name, refs in groups.items():
        last_train = max(refs.index(r) for r in train if r.startswith(name))
        first_val = min(refs.index(r) for r in val if r.startswith(name))
        assert last_train < first_val
        assert counts[name]["train"] + counts[name]["validation"] <= counts[name]["all"]


def test_defaults_reproduce_earlier_rounds() -> None:
    # Every round-4 option is opt-in; without the env switches the trainer behaves as in round 3.
    assert T.SONAR_BATCH == "mixed"
    assert T.SONAR_VIEWS == "default"
    assert T.TEMPORAL_PAIRS == 0
    assert T.STREAM_BATCH_SOURCES == ()
    assert T.SONAR_WEIGHTS == {}
    assert T.SOURCE_VIEWS == {}
    assert T.INIT_FROM == ""
    assert T.SELECT == "family"
    assert T.SELECT_EXCLUDE == ()


def test_sonar_rotation_visits_every_source_with_weights() -> None:
    plain = [T._sonar_rotation(SOURCES, step, {})[0] for step in range(1, 21, 2)]
    assert plain == list(SOURCES) * 2
    weighted = [T._sonar_rotation(SOURCES, step, {"subpipe": 2}) for step in range(1, 25, 2)]
    assert [s for s, _ in weighted].count("subpipe") == 4
    assert [occ for s, occ in weighted if s == "subpipe"] == [0, 1, 2, 3]


def test_batch_pairs_stay_inside_one_recording_and_window() -> None:
    refs = [f"/x/extracted/UATD_Training/{i:05d}.png" for i in range(30)]
    refs += [f"/x/extracted/UATD_Test_1/{i:05d}.png" for i in range(20)]
    b = _batcher({"uatd": refs})
    for step in range(1, 200, 2):
        images, partners, ids = b.batch_pairs("imaging_sonar", "uatd", step, 3)
        a, p = _positions(images), _positions(partners)
        assert len(ids) == 64
        assert torch.equal(a >= 30, p >= 30)
        assert int((a - p).abs().max()) <= 3


def test_batch_group_draws_one_subpipe_stream_at_a_time() -> None:
    lf = [f"SubPipe/DATA/Chunk0/SSS_LF_images/Image/{i:05d}.pbm" for i in range(40)]
    hf = [f"SubPipe/DATA/Chunk0/SSS_HF_images/Image/{i:05d}.pbm" for i in range(30)]
    b = _batcher({"subpipe": lf + hf})
    seen = []
    for occurrence in range(4):
        images, _ = b.batch_group("imaging_sonar", "subpipe", 2 * occurrence + 1, occurrence)
        pos = _positions(images)
        assert bool((pos < 40).all()) or bool((pos >= 40).all())
        seen.append("LF" if bool((pos < 40).all()) else "HF")
    assert sorted(set(seen)) == ["HF", "LF"]
    assert seen[0] != seen[1]
