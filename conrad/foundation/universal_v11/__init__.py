"""Universal OS-FM V1.1 additive architecture."""

from conrad.foundation.universal_v11.core import (
    AsyncTimestampAligner,
    CorrespondenceGraph,
    ExactEvidenceRecord,
    FamilyEncoderConfig,
    SampleLabResult,
    SensorMetadata,
    UniversalAdapter,
    UniversalModalityInput,
    UniversalOSFMV11,
)
from conrad.foundation.universal_v11.config import UniversalV11Config
from conrad.foundation.universal_v11.manifest import UniversalObservationManifest
from conrad.foundation.universal_v11.migration import MigrationReport, load_v1_checkpoint_for_v11, parameter_count
from conrad.foundation.universal_v11.readiness import evaluate_v11_10p_readiness, write_v11_10p_readiness
from conrad.foundation.universal_v11.registry import (
    MODALITY_REGISTRY,
    EncoderFamily,
    ModalitySpec,
    ModalityState,
    inactive_modalities,
)

__all__ = [
    "AsyncTimestampAligner",
    "CorrespondenceGraph",
    "EncoderFamily",
    "ExactEvidenceRecord",
    "FamilyEncoderConfig",
    "MODALITY_REGISTRY",
    "MigrationReport",
    "ModalitySpec",
    "ModalityState",
    "SampleLabResult",
    "SensorMetadata",
    "UniversalObservationManifest",
    "UniversalAdapter",
    "UniversalModalityInput",
    "UniversalOSFMV11",
    "UniversalV11Config",
    "inactive_modalities",
    "evaluate_v11_10p_readiness",
    "load_v1_checkpoint_for_v11",
    "parameter_count",
    "write_v11_10p_readiness",
]
