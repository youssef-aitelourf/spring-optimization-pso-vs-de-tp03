from __future__ import annotations

from dataclasses import dataclass

import numpy as np


BOUNDS = np.array(
    [
        [0.05, 2.00],
        [0.25, 1.30],
        [2.00, 15.00],
    ],
    dtype=float,
)


def objective(x: np.ndarray) -> float:
    x1, x2, x3 = x
    return float((x1**2) * x2 * (2.0 + x3))


def constraints(x: np.ndarray) -> np.ndarray:
    x1, x2, x3 = x
    g1 = 1.0 - (x2**3 * x3) / (71785.0 * x1**4)
    g2 = (4.0 * x2**2 - x1 * x2) / (12566.0 * (x2 * x1**3 - x1**4)) + 1.0 / (5108.0 * x1**2) - 1.0
    g3 = 1.0 - (140.45 * x1) / (x2**2 * x3)
    g4 = (x1 + x2) / 1.5 - 1.0
    return np.array([g1, g2, g3, g4], dtype=float)


def bounds_violation(x: np.ndarray) -> float:
    lower = BOUNDS[:, 0]
    upper = BOUNDS[:, 1]
    below = np.maximum(0.0, lower - x)
    above = np.maximum(0.0, x - upper)
    return float(np.sum(below + above))


def constraints_violation(x: np.ndarray) -> float:
    return float(np.sum(np.maximum(0.0, constraints(x))))


def total_violation(x: np.ndarray) -> float:
    return bounds_violation(x) + constraints_violation(x)


def is_feasible(x: np.ndarray, tol: float = 1e-12) -> bool:
    return bool(total_violation(x) <= tol)


def random_population(rng: np.random.Generator, size: int) -> np.ndarray:
    low = BOUNDS[:, 0]
    high = BOUNDS[:, 1]
    return rng.uniform(low=low, high=high, size=(size, BOUNDS.shape[0]))


def project_clip(x: np.ndarray) -> np.ndarray:
    return np.clip(x, BOUNDS[:, 0], BOUNDS[:, 1])


def reflect_with_velocity(position: np.ndarray, velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low = BOUNDS[:, 0]
    high = BOUNDS[:, 1]
    pos = np.array(position, dtype=float, copy=True)
    vel = np.array(velocity, dtype=float, copy=True)

    for i in range(pos.shape[0]):
        if pos[i] < low[i]:
            over = low[i] - pos[i]
            pos[i] = low[i] + over
            vel[i] *= -1.0
        elif pos[i] > high[i]:
            over = pos[i] - high[i]
            pos[i] = high[i] - over
            vel[i] *= -1.0

        pos[i] = min(max(pos[i], low[i]), high[i])

    return pos, vel


@dataclass
class Evaluation:
    x: np.ndarray
    objective: float
    violation: float
    penalty: float
    penalized_cost: float
    feasible: bool


class PenalizedEvaluator:
    def __init__(self, penalty_coeff: float = 1e6) -> None:
        self.penalty_coeff = float(penalty_coeff)
        self.eval_count = 0

    def evaluate(self, x: np.ndarray) -> Evaluation:
        self.eval_count += 1
        x_arr = np.array(x, dtype=float)
        f = objective(x_arr)
        violation = total_violation(x_arr)
        penalty = self.penalty_coeff * violation
        return Evaluation(
            x=x_arr,
            objective=f,
            violation=violation,
            penalty=penalty,
            penalized_cost=f + penalty,
            feasible=bool(violation <= 1e-12),
        )
