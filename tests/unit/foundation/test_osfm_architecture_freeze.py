from __future__ import annotations

from pathlib import Path


FREEZE_DOC = Path("docs/OSFM_ARCHITECTURE_FREEZE.md")


def test_osfm_architecture_freeze_records_product_boundary_and_phase_split() -> None:
    text = FREEZE_DOC.read_text(encoding="utf-8")
    assert "OSFM-P4-CONTEXT-R02" in text
    assert "host-independent underwater asset intelligence" in text
    assert "The host remains final authority for execution" in text
    assert "| Capability | Current status | Phase | Pre-P4.8 required?" in text
    assert "Model2O | Optional operational/process belief family when real data exists" in text
    assert "Third-party host | Executes supported safe actions" in text


def test_osfm_architecture_freeze_preserves_truth_and_exact_evidence_boundaries() -> None:
    text = FREEZE_DOC.read_text(encoding="utf-8")
    assert "exact physical values remain available through a" in text
    assert "parallel Model2 direct-evidence path" in text
    assert "simulator truth never enters runtime perception or belief as a" in text
    assert "U_A aleatoric" in text
    assert "U_E epistemic" in text
    assert "U_C contradiction" in text
    assert "U_O observational" in text
