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
from conrad.data.adapters.subpipe import FrameRef, SubPipeAdapter
from conrad.data.adapters.synthetic import SyntheticTwinAdapter, TwinSampleLike
from conrad.data.adapters.uvvid import DecodedFrame, UvvidVideoAdapter

__all__ = [
    "PUBLIC_CANDIDATES",
    "DatasetAdapter",
    "DatasetAudit",
    "DatasetContractError",
    "DatasetNotAvailableError",
    "DecodedFrame",
    "FrameRef",
    "ImageFolderSequenceAdapter",
    "PartialTruth",
    "PublicDatasetAdapter",
    "SequenceSample",
    "SubPipeAdapter",
    "SyntheticTwinAdapter",
    "TwinSampleLike",
    "UvvidVideoAdapter",
    "public_adapter",
]
