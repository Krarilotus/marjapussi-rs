from ml.four_model_schedule import (
    BeliefMetricSnapshot,
    BeliefStabilityGate,
    JointScheduleConfig,
    belief_is_stable_enough,
    compute_joint_schedule,
)


def good_metrics() -> BeliefMetricSnapshot:
    return BeliefMetricSnapshot(
        card_owner_acc=0.82,
        constraint_consistency=1.0,
        void_suit_acc=0.91,
        half_pair_acc=0.88,
        calibration_score=0.72,
    )


def test_belief_gate_requires_duration_and_metrics():
    gate = BeliefStabilityGate(min_steps=100, min_epochs=2)
    assert not belief_is_stable_enough(good_metrics(), steps_seen=99, epochs_seen=2, gate=gate)
    assert not belief_is_stable_enough(good_metrics(), steps_seen=100, epochs_seen=1, gate=gate)
    assert belief_is_stable_enough(good_metrics(), steps_seen=100, epochs_seen=2, gate=gate)


def test_joint_schedule_switches_from_belief_to_decision_heavy():
    cfg = JointScheduleConfig(
        gate=BeliefStabilityGate(min_steps=10, min_epochs=1),
        belief_updates_early=5,
        decision_updates_early=1,
        belief_updates_late=1,
        decision_updates_late=5,
    )
    early = compute_joint_schedule(
        good_metrics(),
        steps_seen=0,
        epochs_seen=0,
        cfg=cfg,
    )
    assert early.phase_name == "belief_heavy"
    assert early.belief_updates == 5
    assert early.decision_updates == 1

    late = compute_joint_schedule(
        good_metrics(),
        steps_seen=10,
        epochs_seen=1,
        cfg=cfg,
    )
    assert late.phase_name == "decision_heavy"
    assert late.belief_updates == 1
    assert late.decision_updates == 5
