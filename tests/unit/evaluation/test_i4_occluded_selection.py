from __future__ import annotations

from conrad.evaluation.decision_experiments.active_mcbr_i4_occluded import (
    I4_COMPARATORS,
    PRIMARY,
    select_validation_winner,
)
from conrad.evaluation.decision_experiments.active_mcbr_reeval import MISSION_METRICS


def _worlds(values: dict[str, float], n: int = 12) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for seed in range(n):
        out[str(seed)] = {}
        for arm, primary in values.items():
            out[str(seed)][arm] = dict.fromkeys(MISSION_METRICS, 0.0)
            out[str(seed)][arm][PRIMARY] = primary
    return out


def test_selection_picks_highest_mean_arm_that_beats_every_required_comparator() -> None:
    values = {**dict.fromkeys(I4_COMPARATORS, 0.0), "LOW": 1.0, "WINNER": 2.0}
    result = select_validation_winner(_worlds(values), list(values))

    assert result["eligible"] == ["LOW", "WINNER"]
    assert result["winner"] == "WINNER"
    assert all(
        result["pairwise"]["WINNER"][baseline][PRIMARY]["ci95"][0] > 0.0
        for baseline in I4_COMPARATORS
    )


def test_selection_freezes_nothing_when_no_arm_has_positive_required_cis() -> None:
    values = {**dict.fromkeys(I4_COMPARATORS, 0.0), "PRODUCTION": 0.0}
    result = select_validation_winner(_worlds(values), list(values))

    assert result["eligible"] == []
    assert result["winner"] is None
    assert result["decision"].startswith("no arm beat")
