"""Shared uncertainty contract U = (U_A, U_E, U_C, U_O) (ch2, ch6, ch28).

The four channels are NOT mutually exclusive probabilities and do not sum to one. There is
deliberately no ``total`` / ``confidence`` scalar on this object: collapsing the decomposition
requires a mapping calibrated for the consuming decision (ch28 Tensor and Mask Semantics).

implementation_status: FROZEN_CONTRACT
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field

from conrad.schemas.base import ConradModel


class UncertaintyRepresentation(str, Enum):
    SCALAR_CHANNELS = "SCALAR_CHANNELS"
    ENSEMBLE = "ENSEMBLE"
    EVIDENTIAL = "EVIDENTIAL"
    GAUSSIAN = "GAUSSIAN"
    ANALYTIC = "ANALYTIC"


class CalibrationMetadata(ConradModel):
    calibrated: bool = False
    method: str | None = None
    calibration_run_id: str | None = None
    units: str = "unitless_nonnegative"


class Uncertainty(ConradModel):
    aleatoric: float = Field(ge=0.0, description="U_A: irreducible sensing/measurement noise")
    epistemic: float = Field(ge=0.0, description="U_E: model ignorance / out-of-distribution")
    contradiction: float = Field(ge=0.0, description="U_C: credible evidence disagreement")
    observational: float = Field(ge=0.0, description="U_O: lack of coverage / observability")
    representation_type: UncertaintyRepresentation = UncertaintyRepresentation.SCALAR_CHANNELS
    calibration_metadata: CalibrationMetadata = CalibrationMetadata()

    def dominant_channel(self) -> str:
        channels = {
            "aleatoric": self.aleatoric,
            "epistemic": self.epistemic,
            "contradiction": self.contradiction,
            "observational": self.observational,
        }
        return max(channels, key=lambda k: channels[k])

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.aleatoric, self.epistemic, self.contradiction, self.observational)


def unknown_uncertainty() -> Uncertainty:
    """Prior for something never observed: coverage is absent, so U_O is maximal on a [0, 1] scale."""
    return Uncertainty(aleatoric=0.0, epistemic=1.0, contradiction=0.0, observational=1.0)
