"""Deterministic belief lifecycle engine (ch8). Transitions are validated before any write.

implementation_status: EXPERIMENTAL_CANDIDATE (policy thresholds are configuration)
"""

from __future__ import annotations

from conrad.core.config import PmblConfig
from conrad.schemas.belief import ALLOWED_LIFECYCLE_TRANSITIONS, Lifecycle


class LifecycleTransitionError(ValueError):
    """A lifecycle transition outside ``ALLOWED_LIFECYCLE_TRANSITIONS`` was requested."""


TERMINAL_STATES = frozenset(s for s, nxt in ALLOWED_LIFECYCLE_TRANSITIONS.items() if not nxt)


def check_transition(old: Lifecycle, new: Lifecycle) -> None:
    if new is not old and new not in ALLOWED_LIFECYCLE_TRANSITIONS[old]:
        raise LifecycleTransitionError(f"illegal lifecycle transition {old.value} -> {new.value}")


def is_terminal(state: Lifecycle) -> bool:
    return state in TERMINAL_STATES


class LifecycleEngine:
    def __init__(self, cfg: PmblConfig) -> None:
        self.cfg = cfg

    def initial(self, independent_observations: int) -> Lifecycle:
        """Unmatched evidence creates a CANDIDATE (phantom control) unless the configured confirmation
        threshold is already met (threshold 1 = the 'no candidate stage' ablation)."""
        if independent_observations >= self.cfg.confirm_independent_observations:
            return Lifecycle.CONFIRMED
        return Lifecycle.CANDIDATE

    def on_direct_evidence(self, current: Lifecycle, independent_observations: int) -> Lifecycle:
        """One legal step per committed revision."""
        if is_terminal(current):
            raise LifecycleTransitionError(f"{current.value} belief cannot consume evidence")
        if current is Lifecycle.CANDIDATE:
            if independent_observations >= self.cfg.confirm_independent_observations:
                return Lifecycle.CONFIRMED
            return Lifecycle.CANDIDATE
        if current in (Lifecycle.CONFIRMED, Lifecycle.REACTIVATED):
            return Lifecycle.ACTIVE
        if current is Lifecycle.DORMANT:
            return Lifecycle.REACTIVATED
        return current

    def on_silence(self, current: Lifecycle, idle_s: float) -> Lifecycle:
        """Policy for beliefs that received no evidence for ``idle_s`` physical seconds."""
        if current is Lifecycle.CANDIDATE and idle_s >= self.cfg.reject_after_s:
            return Lifecycle.REJECTED
        if (
            current in (Lifecycle.CONFIRMED, Lifecycle.ACTIVE, Lifecycle.REACTIVATED)
            and idle_s >= self.cfg.dormant_after_s
        ):
            return Lifecycle.DORMANT
        if (
            current is Lifecycle.DORMANT
            and self.cfg.retire_after_s is not None
            and idle_s >= self.cfg.retire_after_s
        ):
            return Lifecycle.RETIRED
        return current
