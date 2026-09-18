"""NAV-GLOBAL-01: A* on a 3D lattice over an occupancy query, with a separate UNKNOWN-space policy.

The map answers "what is known" (``is_free`` / optional ``knowledge``); the safety policy - not the
map - decides whether UNKNOWN is traversable and at what cost (ch20 Knowledge-aware planning).
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections.abc import Callable
from enum import Enum

import numpy as np
from pydantic import Field

from conrad.schemas.base import ConradModel

IsFree = Callable[[np.ndarray], np.ndarray]
"""``is_free(points[N,3]) -> bool[N]``: True where the map says free (OBSERVED or INFERRED)."""
KnowledgeQuery = Callable[[np.ndarray], np.ndarray]
"""``knowledge(points[N,3]) -> str[N]`` in {OBSERVED, INFERRED, UNKNOWN}; absent means all OBSERVED."""
CostHook = Callable[[np.ndarray, np.ndarray], float]
"""Knowledge-aware extra cost ``hook(from_xyz, to_xyz) -> float >= 0`` (current, energy, risk...)."""


class UnknownSpacePolicy(str, Enum):
    FORBID = "FORBID"
    TRAVERSE_WITH_PENALTY = "TRAVERSE_WITH_PENALTY"


class PlannerConfig(ConradModel):
    resolution_m: float = Field(default=0.25, gt=0)
    unknown_policy: UnknownSpacePolicy = UnknownSpacePolicy.FORBID
    unknown_cost_multiplier: float = Field(default=4.0, ge=1)
    inferred_cost_multiplier: float = Field(default=1.5, ge=1)
    max_expansions: int = Field(default=60000, gt=0)
    bounds_margin_m: float = Field(default=3.0, ge=0)
    min_depth_m: float | None = Field(default=None, description="keep path below this depth (policy)")


class PlanningError(RuntimeError):
    """No path found under the active policy (reported, never silently replaced by a straight line)."""


_NEIGHBOURS = [d for d in itertools.product((-1, 0, 1), repeat=3) if d != (0, 0, 0)]


class AStarPlanner:
    name = "NAV-GLOBAL-01-astar-lattice"

    def __init__(
        self,
        is_free: IsFree,
        config: PlannerConfig | None = None,
        knowledge: KnowledgeQuery | None = None,
        cost_hook: CostHook | None = None,
    ) -> None:
        self.config = config or PlannerConfig()
        self._is_free = is_free
        self._knowledge = knowledge
        self._cost_hook = cost_hook

    def _traversable(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        free = np.asarray(self._is_free(pts), dtype=bool)
        mult = np.ones(len(pts))
        if self._knowledge is not None:
            k = np.asarray(self._knowledge(pts))
            unknown = k == "UNKNOWN"
            mult[k == "INFERRED"] = self.config.inferred_cost_multiplier
            if self.config.unknown_policy is UnknownSpacePolicy.FORBID:
                free = free & ~unknown
            else:
                free = free | unknown
                mult[unknown] = self.config.unknown_cost_multiplier
        if self.config.min_depth_m is not None:
            free = free & (pts[:, 2] <= -self.config.min_depth_m)
        return free, mult

    def segment_clear(self, a: np.ndarray, b: np.ndarray) -> bool:
        n = max(2, math.ceil(float(np.linalg.norm(b - a)) / (0.5 * self.config.resolution_m)) + 1)
        pts = a[None, :] + np.linspace(0.0, 1.0, n)[:, None] * (b - a)[None, :]
        return bool(np.all(self._traversable(pts)[0]))

    def plan(self, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """Return waypoints [start, ..., goal] (WORLD). Raises PlanningError when infeasible."""
        start, goal = np.asarray(start, dtype=np.float64), np.asarray(goal, dtype=np.float64)
        if self.segment_clear(start, goal):
            return np.stack([start, goal])
        if not self._traversable(goal[None, :])[0][0]:
            raise PlanningError("goal lies in space the active policy forbids")
        res = self.config.resolution_m
        lo = np.minimum(start, goal) - self.config.bounds_margin_m
        hi = np.maximum(start, goal) + self.config.bounds_margin_m
        origin = start

        def world(c: tuple[int, int, int]) -> np.ndarray:
            return origin + res * np.asarray(c, dtype=np.float64)

        gc = np.rint((goal - origin) / res).astype(int)
        goal_cell = (int(gc[0]), int(gc[1]), int(gc[2]))
        s0 = (0, 0, 0)
        h = lambda c: float(np.linalg.norm(world(c) - goal))  # noqa: E731
        open_heap: list[tuple[float, int, tuple[int, int, int]]] = [(h(s0), 0, s0)]
        g: dict[tuple[int, int, int], float] = {s0: 0.0}
        parent: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        counter = itertools.count(1)
        closed: set[tuple[int, int, int]] = set()
        while open_heap:
            _, _, cur = heapq.heappop(open_heap)
            if cur in closed:
                continue
            closed.add(cur)
            if len(closed) > self.config.max_expansions:
                raise PlanningError("A* expansion budget exhausted")
            if cur == goal_cell or np.linalg.norm(world(cur) - goal) < res:
                return self._finish(cur, parent, world, start, goal)
            cand = [(cur[0] + d[0], cur[1] + d[1], cur[2] + d[2]) for d in _NEIGHBOURS]
            cand = [c for c in cand if c not in closed]
            if not cand:
                continue
            pts = np.stack([world(c) for c in cand])
            inside = np.all((pts >= lo) & (pts <= hi), axis=1)
            free, mult = self._traversable(pts)
            here = world(cur)
            # the edge must be clear too, not only its end node (no diagonal corner cutting)
            fr = np.array([0.25, 0.5, 0.75])
            edge = here[None, None, :] + fr[None, :, None] * (pts - here)[:, None, :]
            edge_free = self._traversable(edge.reshape(-1, 3))[0].reshape(len(pts), len(fr)).all(axis=1)
            free = free & edge_free
            for c, p, ok, m in zip(cand, pts, inside & free, mult, strict=True):
                if not ok:
                    continue
                step = float(np.linalg.norm(p - here)) * float(m)
                if self._cost_hook is not None:
                    step += max(0.0, float(self._cost_hook(here, p)))
                ng = g[cur] + step
                if ng < g.get(c, math.inf):
                    g[c] = ng
                    parent[c] = cur
                    heapq.heappush(open_heap, (ng + h(c), next(counter), c))
        raise PlanningError("no path under the active unknown-space policy")

    def _finish(
        self,
        cell: tuple[int, int, int],
        parent: dict[tuple[int, int, int], tuple[int, int, int]],
        world: Callable[[tuple[int, int, int]], np.ndarray],
        start: np.ndarray,
        goal: np.ndarray,
    ) -> np.ndarray:
        chain = [cell]
        while chain[-1] in parent:
            chain.append(parent[chain[-1]])
        pts = [world(c) for c in reversed(chain)]
        pts[0] = start
        pts.append(goal)
        return self.shortcut(np.stack(pts))

    def shortcut(self, pts: np.ndarray) -> np.ndarray:
        """Greedy line-of-sight smoothing that preserves clearance under the same policy."""
        out, i = [pts[0]], 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self.segment_clear(pts[i], pts[j]):
                j -= 1
            out.append(pts[j])
            i = j
        return np.stack(out)
