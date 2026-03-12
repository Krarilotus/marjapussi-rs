from ml.bidding_teacher import build_bidding_teacher_targets
from ml.neurosymbolic_state import CanonicalState
from ml.tests.test_decision_state import sample_record


def test_bidding_teacher_prefers_recommended_raise_when_legal():
    record = sample_record()
    state = CanonicalState.from_record(record)
    teacher = build_bidding_teacher_targets(record, state)
    assert tuple(teacher.aux_targets.shape) == (10,)
    assert teacher.teacher_policy.argmax().item() == 0


def test_bidding_teacher_falls_back_to_stop_when_raise_not_legal():
    record = sample_record()
    record["obs"]["legal_actions"][0]["bid_value"] = 130
    state = CanonicalState.from_record(record)
    teacher = build_bidding_teacher_targets(record, state)
    assert teacher.teacher_policy.argmax().item() == 1
