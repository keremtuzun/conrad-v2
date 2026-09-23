"""Calibration routing preserves semantic intent and belief-side refund boundaries."""

from conrad.decision.router import RouteTarget
from conrad.evaluation.decision_experiments.fixtures import make_belief
from conrad.orchestration.routing import _allows_calibration_deferral_refund, _is_calibration_request
from conrad.schemas.belief import KnowledgeStatus
from conrad.schemas.decision import InformationNeed, QuestionType
from conrad.schemas.ids import IdFactory


def _need(*, calibration: bool) -> InformationNeed:
    ids = IdFactory(190)
    return InformationNeed(
        need_id=ids.new(),
        trace_id=ids.new(),
        target_belief_ids=(ids.new(),),
        question_type=QuestionType.CONFIRM_CONDITION,
        target_properties=("condition",),
        priority=0.9,
        constraints={"calibration_check": calibration},
    )


def test_only_explicit_calibration_need_uses_calibration_execution_path():
    assert _is_calibration_request(RouteTarget.MCBR, _need(calibration=True))
    assert not _is_calibration_request(RouteTarget.MCBR, _need(calibration=False))
    assert not _is_calibration_request(RouteTarget.NAVIGATION, _need(calibration=True))
    assert not _is_calibration_request(RouteTarget.MCBR, object())


def test_only_observed_intact_belief_allows_unflown_calibration_refund():
    ids = IdFactory(191)
    belief = make_belief(ids)
    condition = belief.state_summary[0]
    intact = belief.model_copy(
        update={
            "state_summary": (
                condition.model_copy(
                    update={"name": "condition", "value": "INTACT", "status": KnowledgeStatus.OBSERVED}
                ),
            )
        }
    )
    failed = belief.model_copy(
        update={
            "state_summary": (
                condition.model_copy(
                    update={"name": "condition", "value": "FAILED", "status": KnowledgeStatus.OBSERVED}
                ),
            )
        }
    )
    unknown = belief.model_copy(
        update={
            "state_summary": (
                condition.model_copy(
                    update={"name": "condition", "value": None, "status": KnowledgeStatus.UNKNOWN}
                ),
            )
        }
    )

    assert _allows_calibration_deferral_refund([intact])
    assert not _allows_calibration_deferral_refund([failed])
    assert not _allows_calibration_deferral_refund([unknown])
