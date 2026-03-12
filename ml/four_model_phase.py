from __future__ import annotations


PHASE_TO_TASK = {
    "Bidding": "bidding",
    "Raising": "bidding",
    "PassingForth": "passing",
    "PassingBack": "passing",
    "StartTrick": "playing",
    "Trick": "playing",
    "AnsweringPair": "playing",
    "AnsweringHalf": "playing",
}


def task_from_phase_name(phase_name: str) -> str:
    task = PHASE_TO_TASK.get(phase_name)
    if task is not None:
        return task
    if phase_name.startswith("AnsweringPair"):
        return "playing"
    if phase_name.startswith("AnsweringHalf"):
        return "playing"
    return "playing"


def start_trick_for_generation_target(target: str) -> int | None:
    if target == "full":
        return None
    if target == "bidding":
        return -1
    if target == "passing":
        return 0
    raise ValueError(f"unsupported generation target '{target}'")


def phase_matches_generation_target(phase_name: str, target: str) -> bool:
    if target == "full":
        return True
    if target == "bidding":
        return task_from_phase_name(phase_name) == "bidding"
    if target == "passing":
        return task_from_phase_name(phase_name) == "passing"
    raise ValueError(f"unsupported generation target '{target}'")
