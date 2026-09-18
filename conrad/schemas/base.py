"""Base classes and identity constants shared by every public Conrad V2 contract.

implementation_status: FROZEN_CONTRACT
source_sections: [ch2 Versioning, ch28, ch33 Status, ch34 Authority]
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

ARCHITECTURE_ID = "conrad_v2_impl_freeze_v0_2"
STACK_ID = "conrad_v2_stack_freeze_v0_2"
SCHEMA_VERSION = "1.0.0"
"""Interface Freeze V1. Breaking changes require a major bump plus a migration note."""


class ConradModel(BaseModel):
    """Immutable, strict base for all public messages. Unknown keys fail validation."""

    model_config = ConfigDict(
        frozen=True, extra="forbid", validate_assignment=True, ser_json_inf_nan="strings"
    )

    SCHEMA_NAME: ClassVar[str] = ""

    def canonical_json(self) -> str:
        """Deterministic JSON used for content digests and replay comparison."""
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    def content_digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


class VersionedModel(ConradModel):
    """A public message that carries its own schema version."""

    schema_version: str = SCHEMA_VERSION


class SchemaVersionError(ValueError):
    """Raised when a payload was produced by an incompatible schema major version."""


def check_schema_compatible(found: str, expected: str = SCHEMA_VERSION) -> None:
    """Same major version is compatible; anything else fails closed."""
    try:
        found_major = int(found.split(".")[0])
        expected_major = int(expected.split(".")[0])
    except (ValueError, IndexError) as exc:
        raise SchemaVersionError(f"unparseable schema version {found!r}") from exc
    if found_major != expected_major:
        raise SchemaVersionError(
            f"schema major {found_major} is incompatible with supported major {expected_major}"
        )


def digest_of(obj: Any) -> str:
    """sha256 over canonical JSON of plain data."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
