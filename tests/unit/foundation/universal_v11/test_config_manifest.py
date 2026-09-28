from conrad.foundation.universal_v11 import EncoderFamily, ModalityState, UniversalObservationManifest, UniversalV11Config


def test_config_save_load_roundtrip(tmp_path) -> None:
    cfg = UniversalV11Config(active_modalities=("rgb_camera", "imaging_sonar"))
    path = tmp_path / "v11.json"
    cfg.to_json(path)
    loaded = UniversalV11Config.from_json(path)
    assert loaded.active_modalities == ("rgb_camera", "imaging_sonar")
    assert loaded.global_fusion.scene_latents == 64
    assert loaded.temporal.memory_tokens == 16


def test_manifest_schema_accepts_optional_universal_metadata() -> None:
    manifest = UniversalObservationManifest(
        asset_id="asset-1",
        scene_id="scene-1",
        observation_id="obs-1",
        session_id="session-1",
        platform="host",
        modality_family=EncoderFamily.ULTRASONIC_NDT,
        concrete_modality="paut_scan",
        timestamp_s=1.0,
        timing_uncertainty_s=0.01,
        data_uri="sha256:abc",
        units="a.u.",
        calibration="cal-1",
        quality=0.9,
        modality_state=ModalityState.AVAILABLE,
        provenance="unit-test",
        source_license="synthetic-test",
        exact_measurements={"wall_thickness_mm": 8.27},
        corruption_metadata={"noise": "synthetic"},
        correspondence_ids=("weld-1",),
    )
    manifest.validate_minimum()


def test_manifest_requires_payload_or_exact_measurements() -> None:
    manifest = UniversalObservationManifest(
        asset_id="asset-1",
        scene_id="scene-1",
        observation_id="obs-1",
        session_id="session-1",
        platform="host",
        modality_family=EncoderFamily.CHEMICAL_ELECTROCHEMICAL_SPECTRAL,
        concrete_modality="methane",
        timestamp_s=1.0,
    )
    try:
        manifest.validate_minimum()
    except ValueError as exc:
        assert "data_uri or exact_measurements" in str(exc)
    else:
        raise AssertionError("manifest with no payload/evidence must fail")
