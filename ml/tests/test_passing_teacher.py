from ml.neurosymbolic_state import CanonicalState
from ml.passing_teacher import build_passing_teacher_targets
from ml.tests.test_decision_state import sample_passing_record


def test_passing_teacher_prefers_blanking_action():
    record = sample_passing_record()
    state = CanonicalState.from_record(record)
    teacher = build_passing_teacher_targets(record, state)
    assert tuple(teacher.aux_targets.shape) == (8,)
    assert abs(float(teacher.teacher_policy.sum().item()) - 1.0) < 1e-6
    assert int(teacher.teacher_policy.argmax().item()) == 0


def test_passing_teacher_falls_back_to_uniform_for_non_pass_actions():
    record = sample_passing_record()
    record["obs"]["legal_actions"] = [
        {"action_token": 42, "bid_value": None, "card_idx": None, "suit_idx": None},
        {"action_token": 42, "bid_value": None, "card_idx": None, "suit_idx": None},
    ]
    state = CanonicalState.from_record(record)
    teacher = build_passing_teacher_targets(record, state)
    assert teacher.teacher_policy.tolist() == [0.5, 0.5]
