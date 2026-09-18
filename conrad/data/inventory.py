"""Hash existing local files into manifest rows (reads bytes only; never rewrites them)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from conrad.data.manifest_model import FileEntry, LineageKeys, ManifestError, sha256_file
from conrad.schemas.observation import Modality


def build_file_entries(
    data_root: str | Path,
    relative_paths: Iterable[str],
    *,
    stream: str,
    modality: Modality,
    lineage: LineageKeys,
) -> tuple[FileEntry, ...]:
    """Hash existing local files into manifest rows. Reads bytes only; never rewrites them."""
    root = Path(data_root)
    entries: list[FileEntry] = []
    for index, rel in enumerate(relative_paths):
        target = root / rel
        if not target.is_file():
            raise ManifestError(f"cannot inventory missing file {target}")
        entries.append(
            FileEntry(
                path=rel,
                sha256=sha256_file(target),
                byte_length=target.stat().st_size,
                stream=stream,
                modality=modality,
                lineage=lineage,
                sequence_index=index,
            )
        )
    return tuple(entries)
