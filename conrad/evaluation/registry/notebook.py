"""Append-only research notebook (ch27 IP notebook, ch35 Priority 7).

Negative and inconclusive results are preserved. This is an engineering provenance record, not
a patentability or novelty conclusion.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from pydantic import Field

from conrad.schemas.base import ARCHITECTURE_ID, STACK_ID, ConradModel, digest_of
from conrad.settings import REPO_ROOT

NOTEBOOK_PATH = REPO_ROOT / "research_notebook" / "entries.jsonl"


class EntryResult(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    INCONCLUSIVE = "INCONCLUSIVE"
    NOT_RUN = "NOT_RUN"


class NotebookEntry(ConradModel):
    time_ns: int = Field(ge=0)
    contributor: str = Field(min_length=1)
    architecture_id: str = ARCHITECTURE_ID
    stack_id: str = STACK_ID
    hypothesis_id: str | None = None
    mechanism: str | None = None
    source_sections: tuple[str, ...] = ()
    scenario_data_versions: tuple[str, ...] = ()
    method: str = Field(min_length=1)
    assumptions: tuple[str, ...] = ()
    implementation_commit: str | None = None
    result: EntryResult
    result_summary: str = Field(min_length=1)
    raw_artifacts: tuple[str, ...] = ()
    limitation: str = Field(min_length=1)
    decision: str = Field(min_length=1)
    next_action: str = Field(min_length=1)
    external_disclosures: tuple[str, ...] = ()


class ResearchNotebook:
    def __init__(self, path: str | Path = NOTEBOOK_PATH) -> None:
        self.path = Path(path)

    def entries(self) -> list[NotebookEntry]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("digest") != digest_of(row.get("entry")):
                    raise ValueError("research notebook entry was edited after it was written")
                out.append(NotebookEntry.model_validate(row["entry"]))
        return out

    def append(self, entry: NotebookEntry) -> None:
        existing = self.entries()
        if existing and entry.time_ns < existing[-1].time_ns:
            raise ValueError("notebook entries are chronological; backdating is not allowed")
        body = entry.model_dump(mode="json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"entry": body, "digest": digest_of(body)}, sort_keys=True) + "\n")
