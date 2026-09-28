import torch

from conrad.foundation.universal_v11 import (
    CorrespondenceGraph,
    EncoderFamily,
    ExactEvidenceRecord,
    ModalityState,
    SampleLabResult,
    UniversalAdapter,
    UniversalModalityInput,
    UniversalOSFMV11,
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
    assert out.fusion.modality_names == ("rgb_camera",)
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
