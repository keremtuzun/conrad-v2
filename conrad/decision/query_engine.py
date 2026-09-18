"""Belief Query Engine: decision-centric retrieval instead of dumping the world (ch16, ch17).

For each mission requirement the engine issues narrow, typed ``BeliefQuery`` objects against a
belief source (the Belief Bus) and merges the replies into ONE coherent snapshot whose revision map
lets the consumer detect mixed-time input.

implementation_status: FROZEN_CONTRACT (interface) / deterministic
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol
from uuid import UUID

from conrad.decision.context import MissionRequirement
from conrad.schemas.belief import Availability, BeliefMessage, BeliefQuery, BeliefSnapshot
from conrad.schemas.ids import IdFactory
from conrad.schemas.world import Domain


class BeliefSource(Protocol):
    def query(self, query: BeliefQuery, now_ns: int, max_age_s: float | None = None) -> BeliefSnapshot: ...


# Tool routing table (ch17 'Tool/model routing'): which child answers which requirement domain.
ROUTING_TOOLS: dict[Domain, str] = {
    Domain.TECHNICAL: "query_model2t",
    Domain.ECOLOGICAL: "query_model2e",
    Domain.SPATIAL: "query_model2s",
}


class BeliefQueryEngine:
    def __init__(self, source: BeliefSource, id_factory: IdFactory, max_results_per_query: int = 32) -> None:
        self._source = source
        self._ids = id_factory
        self._max_results = max_results_per_query
        self.issued: list[tuple[str, BeliefQuery]] = []

    def queries_for(self, requirement: MissionRequirement) -> list[tuple[str, BeliefQuery]]:
        """Primary query on the owning domain plus region-scoped context queries on other domains."""
        out: list[tuple[str, BeliefQuery]] = [
            (
                ROUTING_TOOLS[requirement.domain],
                BeliefQuery(
                    domain=requirement.domain,
                    belief_ids=requirement.target_belief_ids,
                    entity_ids=requirement.target_entity_ids,
                    region=None
                    if (requirement.target_belief_ids or requirement.target_entity_ids)
                    else requirement.region,
                    requested_fields=requirement.properties,
                    include_provenance=True,
                    max_results=self._max_results,
                ),
            )
        ]
        if requirement.region is not None:
            for domain in requirement.context_domains:
                if domain is requirement.domain:
                    continue
                out.append(
                    (
                        ROUTING_TOOLS[domain],
                        BeliefQuery(
                            domain=domain,
                            region=requirement.region,
                            include_provenance=True,
                            max_results=self._max_results,
                        ),
                    )
                )
        return out

    def retrieve(self, requirements: Sequence[MissionRequirement], now_ns: int) -> BeliefSnapshot:
        merged: dict[UUID, BeliefMessage] = {}
        availability: dict[str, Availability] = {}
        revisions: dict[str, int] = {}
        stale: set[str] = set()
        sequences: list[int] = []
        tools: list[str] = []
        for requirement in requirements:
            for tool, query in self.queries_for(requirement):
                self.issued.append((tool, query))
                tools.append(tool)
                reply = self._source.query(query, now_ns)
                availability.update(reply.domain_availability)
                stale.update(str(s) for s in reply.provenance.get("stale_belief_ids", []))
                if "bus_sequence" in reply.provenance:
                    sequences.append(int(reply.provenance["bus_sequence"]))
                for message in reply.messages:
                    known = merged.get(message.belief_id)
                    if known is None or message.revision > known.revision:
                        merged[message.belief_id] = message
                        revisions[str(message.belief_id)] = message.revision
        messages = sorted(merged.values(), key=lambda m: (m.domain.value, m.belief_id.int))
        return BeliefSnapshot(
            snapshot_id=self._ids.new(),
            created_time_ns=now_ns,
            messages=tuple(messages),
            domain_availability=availability,
            provenance={
                "revisions": revisions,
                "stale_belief_ids": sorted(stale),
                "coherent": len(set(sequences)) <= 1,
                "tools": tools,
            },
        )
