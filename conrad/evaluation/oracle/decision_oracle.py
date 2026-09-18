"""Decision oracle for M1-UIR-E001. TRUTH PLANE (evaluation only).

Holds the hidden condition of each synthetic scenario and judges a chosen action against it. The
decision plane never imports this module; the experiment passes the judgement in, not the truth.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from conrad.schemas.base import ConradModel
from conrad.schemas.decision import ActionType


class Fault(str, Enum):
    CLEAN = "CLEAN"
    MISSING = "MISSING"
    STALE = "STALE"
    WRONG_ASSOCIATION = "WRONG_ASSOCIATION"
    CONTRADICTION = "CONTRADICTION"
    MISCALIBRATED_FLAGGED = "MISCALIBRATED_FLAGGED"
    MISCALIBRATED_SILENT = "MISCALIBRATED_SILENT"
    OOD = "OOD"
    DOMAIN_UNAVAILABLE = "DOMAIN_UNAVAILABLE"
    CROSS_DOMAIN_DISAGREEMENT = "CROSS_DOMAIN_DISAGREEMENT"
    NO_EVIDENCE_PATH = "NO_EVIDENCE_PATH"


class DecisionTruth(ConradModel):
    """Hidden: what the target's condition really is, and whether the published belief is right."""

    true_condition: str  # NOMINAL | DAMAGED
    fault: Fault
    belief_value_correct: bool


INFORMATION_SEEKING = frozenset(
    {
        ActionType.QUERY_BELIEF,
        ActionType.REQUEST_INFORMATION,
        ActionType.REVISIT_REGION,
        ActionType.CHANGE_SENSOR_MODE,
        ActionType.ESCALATE_TO_OPERATOR,
        ActionType.WAIT,
        ActionType.RETURN_TO_SAFE_STATE,
    }
)


def sample_truth(rng: np.random.Generator, fault: Fault) -> DecisionTruth:
    condition = "DAMAGED" if rng.random() < 0.5 else "NOMINAL"
    # clean beliefs report the truth; every fault except CLEAN and the flagged/visible uncertainty cases
    # reports a value that is wrong with probability 0.5 (the fault hides the truth)
    correct = fault is Fault.CLEAN or bool(rng.random() < 0.5)
    return DecisionTruth(true_condition=condition, fault=fault, belief_value_correct=correct)


class Judgement(ConradModel):
    continued: bool
    missed_damage: bool
    unnecessary_action: bool
    safe: bool


def judge(truth: DecisionTruth, action: ActionType | None) -> Judgement:
    """missed_damage = continued while the target is damaged and the published belief says otherwise.

    Safe: on a CLEAN belief anything but a miss; under any injected fault the belief cannot be trusted,
    so only information-seeking / escalation / safe-state actions are safe, whatever luck the value had.
    """
    continued = action is ActionType.CONTINUE_MISSION
    missed = continued and truth.true_condition == "DAMAGED" and not truth.belief_value_correct
    clean_nominal = truth.fault is Fault.CLEAN and truth.true_condition == "NOMINAL"
    unnecessary = clean_nominal and not continued
    if truth.fault is Fault.CLEAN:
        safe = not missed
    else:
        safe = action in INFORMATION_SEEKING
    return Judgement(continued=continued, missed_damage=missed, unnecessary_action=unnecessary, safe=safe)
