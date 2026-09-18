"""Content-addressed local object store: ``<root>/sha256/<digest>`` (ch34 Persistence).

Objects are immutable. Reads verify the digest and fail closed (SS-04).

implementation_status: FROZEN_CONTRACT
"""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
from pathlib import Path

import numpy as np

from conrad.schemas.observation import PayloadRef


class ObjectStoreError(RuntimeError):
    pass


class MissingObjectError(ObjectStoreError):
    def __init__(self, digest: str) -> None:
        super().__init__(f"referenced object is missing: sha256:{digest}")
        self.digest = digest


class DigestMismatchError(ObjectStoreError):
    def __init__(self, digest: str, actual: str) -> None:
        super().__init__(f"object sha256:{digest} is corrupt (actual sha256:{actual})")
        self.digest = digest
        self.actual = actual


class ObjectStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        (self.root / "sha256").mkdir(parents=True, exist_ok=True)

    def path_for(self, digest: str) -> Path:
        return self.root / "sha256" / digest

    def put_bytes(
        self, data: bytes, media_type: str, shape: tuple[int, ...] = (), dtype: str | None = None
    ) -> PayloadRef:
        digest = hashlib.sha256(data).hexdigest()
        target = self.path_for(digest)
        if not target.exists():
            fd, tmp = tempfile.mkstemp(dir=str(self.root), prefix=".incoming-")
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp, target)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        return PayloadRef(
            uri=f"sha256:{digest}",
            digest=digest,
            media_type=media_type,
            shape=shape,
            dtype=dtype,
            byte_length=len(data),
        )

    def put_array(self, array: np.ndarray, media_type: str = "application/x-npy") -> PayloadRef:
        buf = io.BytesIO()
        np.save(buf, np.ascontiguousarray(array), allow_pickle=False)
        return self.put_bytes(
            buf.getvalue(), media_type, tuple(int(s) for s in array.shape), str(array.dtype)
        )

    def exists(self, digest: str) -> bool:
        return self.path_for(digest).exists()

    def get_bytes(self, ref: PayloadRef | str) -> bytes:
        digest = ref if isinstance(ref, str) else ref.digest
        path = self.path_for(digest)
        if not path.exists():
            raise MissingObjectError(digest)
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != digest:
            raise DigestMismatchError(digest, actual)
        return data

    def get_array(self, ref: PayloadRef | str) -> np.ndarray:
        return np.load(io.BytesIO(self.get_bytes(ref)), allow_pickle=False)

    def verify(self, digests: list[str]) -> list[str]:
        """Return a list of problems; empty means every digest exists and hashes correctly."""
        problems: list[str] = []
        for digest in digests:
            try:
                self.get_bytes(digest)
            except ObjectStoreError as exc:
                problems.append(str(exc))
        return problems

    def self_test(self) -> None:
        probe = os.urandom(32)
        ref = self.put_bytes(probe, "application/octet-stream")
        if self.get_bytes(ref) != probe:
            raise ObjectStoreError("object store write/read hash self-test failed")
        self.path_for(ref.digest).unlink(missing_ok=True)
