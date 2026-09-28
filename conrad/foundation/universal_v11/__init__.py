"""Universal OS-FM V1.1 additive architecture."""

from conrad.foundation.universal_v11.core import (
    AsyncTimestampAligner,
    CorrespondenceGraph,
    ExactEvidenceRecord,
    SampleLabResult,
    SensorMetadata,
    UniversalAdapter,
    UniversalModalityInput,
    UniversalOSFMV11,
)
from conrad.foundation.universal_v11.migration import MigrationReport, load_v1_checkpoint_for_v11, parameter_count
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
    "MODALITY_REGISTRY",
    "MigrationReport",
    "ModalitySpec",
    "ModalityState",
    "SampleLabResult",
    "SensorMetadata",
    "UniversalAdapter",
    "UniversalModalityInput",
    "UniversalOSFMV11",
    "inactive_modalities",
    "load_v1_checkpoint_for_v11",
    "parameter_count",
]
