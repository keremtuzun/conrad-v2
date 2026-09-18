"""Integer voxel indexing for the sparse hierarchical map. BELIEF PLANE.

A cell at resolution ``res`` with integer index (i, j, k) covers [i*res, (i+1)*res) per axis in the map
frame. Indices pack into one int64 code (21 bits per axis, offset 2**20) so sets of cells are numpy arrays.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

IntArr = NDArray[np.int64]
FloatArr = NDArray[np.float64]

_BITS = 21
_OFF = 1 << (_BITS - 1)
_MASK = (1 << _BITS) - 1
INDEX_MIN = -_OFF
INDEX_MAX = _OFF - 1


def point_to_index(points: FloatArr, res: float) -> IntArr:
    """(N, 3) WORLD points -> (N, 3) integer indices at resolution ``res``."""
    idx = np.floor(np.atleast_2d(np.asarray(points, dtype=np.float64)) / res).astype(np.int64)
    if idx.size and (idx.min() < INDEX_MIN or idx.max() > INDEX_MAX):
        raise ValueError("point outside the representable map extent")
    return idx


def index_to_center(idx: IntArr, res: float) -> FloatArr:
    return (np.asarray(idx, dtype=np.float64) + 0.5) * res


def encode(idx: IntArr) -> IntArr:
    """(N, 3) indices -> (N,) int64 codes. Order-preserving per axis, collision-free inside the extent."""
    a = np.atleast_2d(np.asarray(idx, dtype=np.int64)) + _OFF
    if a.size and (a.min() < 0 or a.max() > _MASK):
        raise ValueError("index outside the representable map extent")
    return np.asarray((a[:, 0] << (2 * _BITS)) | (a[:, 1] << _BITS) | a[:, 2], dtype=np.int64)


def decode(codes: IntArr) -> IntArr:
    c = np.asarray(codes, dtype=np.int64).reshape(-1)
    out = np.stack([(c >> (2 * _BITS)) & _MASK, (c >> _BITS) & _MASK, c & _MASK], axis=1)
    return np.asarray(out - _OFF, dtype=np.int64)


def points_to_codes(points: FloatArr, res: float) -> IntArr:
    return encode(point_to_index(points, res))


def parent_index(idx: IntArr, levels_up: int = 1) -> IntArr:
    """Index of the enclosing cell ``levels_up`` halvings coarser (floor division keeps negatives right)."""
    return np.asarray(np.floor_divide(np.asarray(idx, dtype=np.int64), 1 << levels_up), dtype=np.int64)


def children_index(idx: IntArr) -> IntArr:
    """(N, 3) -> (N*8, 3) indices of the eight children one level finer."""
    base = np.asarray(idx, dtype=np.int64).reshape(-1, 1, 3) * 2
    offs = np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)], dtype=np.int64)
    return np.asarray((base + offs[None, :, :]).reshape(-1, 3), dtype=np.int64)


def cube_offsets(radius: int) -> IntArr:
    r = np.arange(-radius, radius + 1, dtype=np.int64)
    g = np.stack(np.meshgrid(r, r, r, indexing="ij"), axis=-1).reshape(-1, 3)
    return np.asarray(g, dtype=np.int64)


FACE_OFFSETS: IntArr = np.array(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.int64
)
LINE_DIRECTIONS: IntArr = np.array(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, -1, 0],
        [1, 0, 1],
        [1, 0, -1],
        [0, 1, 1],
        [0, 1, -1],
        [1, 1, 1],
        [1, 1, -1],
        [1, -1, 1],
        [1, -1, -1],
    ],
    dtype=np.int64,
)
"""The 13 undirected lattice lines through a cell (half of the 26-neighbourhood)."""


def region_indices(center: FloatArr, half_extent: FloatArr, res: float) -> IntArr:
    """Indices of every cell whose centre lies inside the axis-aligned box (at least the containing cell)."""
    c, h = np.asarray(center, dtype=np.float64), np.asarray(half_extent, dtype=np.float64)
    lo = np.ceil((c - h) / res - 0.5).astype(np.int64)
    hi = np.floor((c + h) / res - 0.5).astype(np.int64)
    thin = hi < lo  # box thinner than a cell along that axis: take the containing cell
    own = np.floor(c / res).astype(np.int64)
    lo, hi = np.where(thin, own, lo), np.where(thin, own, hi)
    axes = [np.arange(lo[k], hi[k] + 1, dtype=np.int64) for k in range(3)]
    return np.asarray(np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3), dtype=np.int64)
