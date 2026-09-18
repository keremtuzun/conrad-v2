"""Dataset adapters. Models never own dataset-specific loading (ch24 Manifest and Adapter Contract)."""

from conrad.data.adapters.base import (
    DatasetAdapter,
    DatasetAudit,
    DatasetContractError,
    DatasetNotAvailableError,
    PartialTruth,
    SequenceSample,
)
from conrad.data.adapters.image_folder import ImageFolderSequenceAdapter
from conrad.data.adapters.public import PUBLIC_CANDIDATES, PublicDatasetAdapter, public_adapter
from conrad.data.adapters.synthetic import SyntheticTwinAdapter, TwinSampleLike

__all__ = [
    "PUBLIC_CANDIDATES",
    "DatasetAdapter",
    "DatasetAudit",
    "DatasetContractError",
    "DatasetNotAvailableError",
    "ImageFolderSequenceAdapter",
    "PartialTruth",
    "PublicDatasetAdapter",
    "SequenceSample",
    "SyntheticTwinAdapter",
    "TwinSampleLike",
    "public_adapter",
]
