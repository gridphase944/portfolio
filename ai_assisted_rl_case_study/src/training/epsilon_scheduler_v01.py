"""Epsilon scheduler helpers v0.1."""

from __future__ import annotations


def compute_epsilon_v01(
    global_env_step: int,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.1,
    epsilon_decay_steps: int = 20000,
    epsilon_schedule: str = "linear",
) -> float:
    """Compute epsilon for a constant or linear schedule."""

    step = int(global_env_step)
    decay_steps = int(epsilon_decay_steps)
    start = float(epsilon_start)
    end = float(epsilon_end)
    schedule = str(epsilon_schedule)

    if schedule not in {"constant", "linear"}:
        raise ValueError("epsilon_schedule must be 'constant' or 'linear'")
    if not 0.0 <= start <= 1.0 or not 0.0 <= end <= 1.0:
        raise ValueError("epsilon_start and epsilon_end must be in [0, 1]")
    if schedule == "constant":
        return start

    if decay_steps <= 0:
        return end
    if step <= 0:
        return start
    if step >= decay_steps:
        return end

    progress = float(step) / float(decay_steps)
    return start + (end - start) * progress
