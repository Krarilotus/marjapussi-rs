from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BeliefMetricSnapshot:
    card_owner_acc: float
    constraint_consistency: float
    void_suit_acc: float
    half_pair_acc: float
    calibration_score: float


@dataclass(frozen=True)
class BeliefStabilityGate:
    min_steps: int = 1_000
    min_epochs: int = 1
    card_owner_acc: float = 0.70
    constraint_consistency: float = 0.999
    void_suit_acc: float = 0.85
    half_pair_acc: float = 0.80
    calibration_score: float = 0.60


@dataclass(frozen=True)
class JointScheduleConfig:
    gate: BeliefStabilityGate = BeliefStabilityGate()
    belief_updates_early: int = 4
    decision_updates_early: int = 1
    belief_updates_late: int = 1
    decision_updates_late: int = 4


@dataclass(frozen=True)
class JointScheduleSnapshot:
    stable: bool
    belief_updates: int
    decision_updates: int
    phase_name: str


def belief_is_stable_enough(
    metrics: BeliefMetricSnapshot,
    *,
    steps_seen: int,
    epochs_seen: int,
    gate: BeliefStabilityGate,
) -> bool:
    if steps_seen < gate.min_steps or epochs_seen < gate.min_epochs:
        return False
    return (
        metrics.card_owner_acc >= gate.card_owner_acc
        and metrics.constraint_consistency >= gate.constraint_consistency
        and metrics.void_suit_acc >= gate.void_suit_acc
        and metrics.half_pair_acc >= gate.half_pair_acc
        and metrics.calibration_score >= gate.calibration_score
    )


def compute_joint_schedule(
    metrics: BeliefMetricSnapshot,
    *,
    steps_seen: int,
    epochs_seen: int,
    cfg: JointScheduleConfig = JointScheduleConfig(),
) -> JointScheduleSnapshot:
    stable = belief_is_stable_enough(
        metrics,
        steps_seen=steps_seen,
        epochs_seen=epochs_seen,
        gate=cfg.gate,
    )
    if stable:
        return JointScheduleSnapshot(
            stable=True,
            belief_updates=cfg.belief_updates_late,
            decision_updates=cfg.decision_updates_late,
            phase_name="decision_heavy",
        )
    return JointScheduleSnapshot(
        stable=False,
        belief_updates=cfg.belief_updates_early,
        decision_updates=cfg.decision_updates_early,
        phase_name="belief_heavy",
    )
