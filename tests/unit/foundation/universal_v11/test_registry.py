from conrad.foundation.universal_v11 import EncoderFamily, MODALITY_REGISTRY, inactive_modalities
from conrad.foundation.universal_v11.registry import validate_registry


def test_registry_covers_all_universal_encoder_families() -> None:
    validate_registry()
    families = {spec.family for spec in MODALITY_REGISTRY}
    assert families == set(EncoderFamily)
    assert len(MODALITY_REGISTRY) >= 45


def test_registry_preserves_rare_inactive_modalities() -> None:
    inactive = set(inactive_modalities())
    for name in {
        "synthetic_aperture_sonar",
        "paut_scan",
        "tofd_scan",
        "tfm_scan",
        "em_ndt_eddy_current",
        "edna_sequence",
        "gamma_spectrometer",
        "seismic_reflection",
    }:
        assert name in inactive


def test_exact_and_lab_paths_are_declared_separately() -> None:
    by_name = {spec.name: spec for spec in MODALITY_REGISTRY}
    assert by_name["ph_probe"].exact_evidence is True
    assert by_name["ph_probe"].sample_or_lab_result is False
    assert by_name["edna_sequence"].sample_or_lab_result is True
    assert by_name["edna_sequence"].exact_evidence is False
