# Research notebook

Append-only engineering and research provenance record (spec ch27 "IP notebook", ch35 Priority 7).

- Entries live in `entries.jsonl`, one JSON object per line: `{"entry": {...}, "digest": sha256}`.
- Write entries only through `conrad.evaluation.registry.ResearchNotebook.append(NotebookEntry(...))`.
  Each entry records time, contributor, architecture/stack ID, hypothesis, source sections,
  scenario/data versions, method, assumptions, implementation commit, result, raw-artifact
  locations, limitation, decision and next action.
- Never edit or delete a line. A correction is a new entry that points to the earlier one.
  Reading the notebook fails if a stored entry no longer matches its digest.
- Record negative and inconclusive results as well as positive ones.
- This is not a patentability or novelty conclusion. Get qualified legal advice before making
  any external IP claim, and keep claims separate from measured results.

The experiment registry (`conrad.evaluation.registry.ExperimentRegistry`) is a separate
hash-chained JSONL file that holds per-experiment code/config/data/seed/checkpoint/metric records.
