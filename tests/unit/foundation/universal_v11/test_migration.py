import torch

from conrad.foundation.universal_v11 import UniversalOSFMV11, load_v1_checkpoint_for_v11, parameter_count


def test_v1_migration_fails_closed_without_final_p412_metadata(tmp_path) -> None:
    model = UniversalOSFMV11(input_dims={"rgb_camera": 8})
    path = tmp_path / "bad.pt"
    torch.save({"metadata": {"label": "OSFM-S-PRETRAIN-V1", "status": "IMPLEMENTED", "decision": "CONDITIONAL-GO"}, "model_state_dict": {}}, path)
    try:
        load_v1_checkpoint_for_v11(model, path)
    except ValueError as exc:
        assert "P4.12-qualified" in str(exc)
    else:
        raise AssertionError("unqualified V1 checkpoint must not import for V1.1 formal training")


def test_v1_migration_loads_compatible_keys_and_reports_new_modules(tmp_path) -> None:
    model = UniversalOSFMV11(input_dims={"rgb_camera": 8})
    state = model.state_dict()
    compatible_key = next(iter(state))
    path = tmp_path / "good.pt"
    torch.save(
        {
            "metadata": {"label": "OSFM-S-PRETRAIN-V1", "status": "VALIDATED-RUN", "decision": "GO"},
            "model_state_dict": {compatible_key: state[compatible_key].clone(), "legacy.unused": torch.zeros(1)},
        },
        path,
    )
    report = load_v1_checkpoint_for_v11(model, path)
    assert report.qualified_for_training
    assert compatible_key in report.loaded_keys
    assert "legacy.unused" in report.skipped_keys
    assert report.missing_v11_keys


def test_parameter_count_reports_total_and_modules() -> None:
    counts = parameter_count(UniversalOSFMV11(input_dims={"rgb_camera": 8}))
    assert counts["total"] > 0
    assert counts["fusion"] > 0
    assert counts["temporal"] > 0
