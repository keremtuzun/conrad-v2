"""Model 2 child interface (ch2). BELIEF PLANE: no truth types may appear here.

implementation_status: FROZEN_CONTRACT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from conrad.schemas.belief import Availability, BeliefMessage, BeliefQuery
from conrad.schemas.observation import Evidence
from conrad.schemas.timebase import TimeStamp
from conrad.schemas.world import Domain


class Model2Child(ABC):
    domain: Domain
    model_version: str

    @abstractmethod
    def initialize(self, context: dict[str, Any]) -> None:
        """``context`` holds permitted prior/mission context only (e.g. an asset registry)."""

    @abstractmethod
    def ingest(self, evidence: Sequence[Evidence]) -> None: ...

    @abstractmethod
    def update_beliefs(self, now: TimeStamp) -> list[BeliefMessage]:
        """Apply pending evidence; return messages for beliefs that changed materially."""

    @abstractmethod
    def predict(self, delta_t_s: float, now: TimeStamp) -> list[BeliefMessage]:
        """Temporal prediction over PHYSICAL elapsed seconds. Results are PREDICTED, never OBSERVED."""

    @abstractmethod
    def receive_context(self, messages: Sequence[BeliefMessage]) -> None:
        """Cross-domain messages from the bus. Consumed only as CROSS_DOMAIN_CONTEXT."""

    @abstractmethod
    def query(self, query: BeliefQuery) -> list[BeliefMessage]: ...

    @abstractmethod
    def export_beliefs(self) -> list[BeliefMessage]: ...

    @abstractmethod
    def reset_working_memory(self) -> None:
        """Clears working memory only. Persistent beliefs and the evidence archive survive (CC-09)."""

    @abstractmethod
    def availability(self) -> Availability: ...
