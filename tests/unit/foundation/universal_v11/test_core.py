import torch

from conrad.foundation.universal_v11 import (
    CorrespondenceGraph,
    EncoderFamily,
    ExactEvidenceRecord,
    FamilyEncoderConfig,
    ModalityState,
    SampleLabResult,
    UniversalAdapter,
    UniversalModalityInput,
    UniversalOSFMV11,
    parameter_count,
)


def _input(name: str, state: ModalityState = ModalityState.AVAILABLE, *, timestamp: float = 0.0) -> UniversalModalityInput:
    return UniversalModalityInput(
        name=name,
        payload=torch.randn(2, 4, 8),
        state=state,
        timestamp_s=torch.full((2,), timestamp),
    )


def test_universal_adapter_maps_family_outputs_to_384_tokens() -> None:
    adapter = UniversalAdapter(input_dim=17)
    tokens, mask = adapter(torch.randn(3, 5, 17))
    assert tokens.shape == (3, 5, 384)
    assert mask.shape == (3, 5)
    assert mask.all()


def test_universal_model_smoke_preserves_exact_evidence_and_lab_results() -> None:
    model = UniversalOSFMV11(input_dims={"rgb_camera": 8, "imaging_sonar": 8, "ctd_profile": 8})
    exact = (ExactEvidenceRecord("ctd_profile", {"temperature_c": 8.2, "salinity_psu": 34.1}),)
    lab = (SampleLabResult("edna_sequence", {"species_reads": 42}, sample_id="S1", method="metabarcoding"),)
    out = model(
        (_input("rgb_camera"), _input("imaging_sonar"), _input("ctd_profile", ModalityState.DEGRADED)),
        reference_time_s=torch.zeros(2),
        exact_evidence=exact,
        sample_lab_results=lab,
    )
    assert out.fusion.scene_latents.shape == (2, 64, 384)
    assert out.fusion.global_repr.shape == (2, 384)
    assert out.exact_evidence == exact
    assert out.sample_lab_results == lab
    assert out.router_weights.shape == (2, 3)


def test_unsupported_failed_invalid_states_are_not_fused() -> None:
    model = UniversalOSFMV11(input_dims={"rgb_camera": 8, "imaging_sonar": 8, "gamma_spectrometer": 8})
    out = model(
        (
            _input("rgb_camera"),
            _input("imaging_sonar", ModalityState.UNSUPPORTED),
            _input("gamma_spectrometer", ModalityState.INVALID),
        ),
        reference_time_s=torch.zeros(2),
    )
    assert out.fusion.modality_names == ("visual_image_family",)
    assert torch.allclose(out.router_weights[:, 1:], torch.zeros_like(out.router_weights[:, 1:]))


def test_async_alignment_marks_stale_and_masks_skewed_tokens() -> None:
    model = UniversalOSFMV11(input_dims={"rgb_camera": 8, "imaging_sonar": 8})
    token_sets = model.tokenize((_input("rgb_camera", timestamp=-2.0), _input("imaging_sonar", timestamp=0.0)))
    aligned = model.aligner.align(token_sets, torch.zeros(2))
    assert aligned[0].state == ModalityState.STALE
    assert not aligned[0].valid_token_mask.any()
    assert aligned[1].valid_token_mask.all()


def test_temporal_smoke_uses_existing_bounded_memory_contract() -> None:
    model = UniversalOSFMV11(input_dims={"rgb_camera": 8, "imaging_sonar": 8})
    out = model(
        (_input("rgb_camera"), _input("imaging_sonar")),
        reference_time_s=torch.zeros(2),
        temporal_timestamps_s=torch.zeros(2, 1),
        temporal_valid_mask=torch.ones(2, 1, dtype=torch.bool),
    )
    assert out.temporal is not None
    assert out.temporal.window_repr.shape == (2, 1, 384)


def test_correspondence_graph_allows_only_meaningful_pairs() -> None:
    graph = CorrespondenceGraph()
    assert graph.eligible(EncoderFamily.VISUAL_IMAGE, EncoderFamily.ACTIVE_ACOUSTIC)
    assert graph.eligible(EncoderFamily.CHEMICAL_ELECTROCHEMICAL_SPECTRAL, EncoderFamily.BIOLOGICAL_MOLECULAR)
    assert not graph.eligible(EncoderFamily.RADIOLOGICAL, EncoderFamily.ENGINEERING_DOCUMENT_CONTEXT)


def test_all_required_states_are_explicit_and_route_or_fail_cleanly() -> None:
    assert {state.name for state in ModalityState} == {
        "AVAILABLE",
        "MISSING",
        "UNSUPPORTED",
        "FAILED",
        "DEGRADED",
        "STALE",
        "INVALID",
        "UNCALIBRATED",
        "SATURATED",
    }
    model = UniversalOSFMV11(
        input_dims={"rgb_camera": 8},
        family_encoder_config=FamilyEncoderConfig(depth=1),
        include_v1_bank=False,
    )
    for state in (ModalityState.DEGRADED, ModalityState.UNCALIBRATED, ModalityState.SATURATED):
        out = model((_input("rgb_camera", state),), reference_time_s=torch.zeros(2))
        assert out.router_weights.shape == (2, 1)
    try:
        model((_input("rgb_camera", ModalityState.FAILED),), reference_time_s=torch.zeros(2))
    except ValueError as exc:
        assert "no routeable" in str(exc)
    else:
        raise AssertionError("all-unrouteable inputs must fail explicitly")


def test_each_universal_family_encoder_produces_384d_tokens() -> None:
    model = UniversalOSFMV11(family_encoder_config=FamilyEncoderConfig(depth=1), include_v1_bank=False)
    for family in EncoderFamily:
        encoder = model.family_encoders[family.value]
        tokens, mask = encoder(torch.randn(2, 3, 384), torch.ones(2, 3, dtype=torch.bool))
        assert tokens.shape == (2, 3, 384)
        assert mask.all()


def test_default_parameter_scale_matches_universal_v11_expectation() -> None:
    counts = parameter_count(UniversalOSFMV11(input_dims={"rgb_camera": 8, "imaging_sonar": 8}))
    assert 250_000_000 <= counts["total"] <= 320_000_000
    assert counts["family_encoders"] > counts["fusion"]
    assert counts["v1"] > 0
