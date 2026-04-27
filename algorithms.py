from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from problem import Evaluation, PenalizedEvaluator, project_clip, random_population, reflect_with_velocity


@dataclass
class IterationLog:
    iteration: int
    eval_count: int
    best_cost: float
    best_objective: float
    best_violation: float
    best_feasible: bool
    current_cost: float
    diversity: float


@dataclass
class AlgorithmResult:
    name: str
    best_eval: Evaluation
    history: list[IterationLog]
    stop_reason: str


def _mean_pairwise_distance(points: np.ndarray) -> float:
    if points.shape[0] <= 1:
        return 0.0
    diffs = points[:, None, :] - points[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    n = points.shape[0]
    triu = np.triu_indices(n, k=1)
    return float(np.mean(dists[triu]))


def _stagnation_update(best_cost: float, prev_best_cost: float, epsilon: float, no_improve: int) -> int:
    improvement = prev_best_cost - best_cost
    if improvement > epsilon:
        return 0
    return no_improve + 1


def run_pso(
    rng: np.random.Generator,
    evaluator: PenalizedEvaluator,
    swarm_size: int,
    max_iter: int,
    stagnation_patience: int,
    stagnation_epsilon: float,
    inertia: float = 0.72,
    c1: float = 1.49,
    c2: float = 1.49,
    neighborhood: Literal["global", "ring"] = "ring",
) -> AlgorithmResult:
    dim = 3
    positions = random_population(rng, swarm_size)
    span = (project_clip(np.array([2.0, 1.3, 15.0])) - project_clip(np.array([0.05, 0.25, 2.0])))
    velocities = rng.uniform(-0.15 * span, 0.15 * span, size=(swarm_size, dim))

    particle_eval = [evaluator.evaluate(positions[i]) for i in range(swarm_size)]
    pbest_positions = positions.copy()
    pbest_eval = list(particle_eval)
    gbest = min(pbest_eval, key=lambda e: e.penalized_cost)
    gbest_position = np.array(gbest.x, copy=True)

    history: list[IterationLog] = []
    no_improve = 0
    prev_best = gbest.penalized_cost
    stop_reason = "max_iter"

    for iteration in range(1, max_iter + 1):
        for i in range(swarm_size):
            if neighborhood == "ring":
                left = (i - 1) % swarm_size
                right = (i + 1) % swarm_size
                neighbor_ids = [left, i, right]
                lbest_eval = min((pbest_eval[idx] for idx in neighbor_ids), key=lambda e: e.penalized_cost)
                social_target = lbest_eval.x
            else:
                social_target = gbest_position

            r1 = rng.random(dim)
            r2 = rng.random(dim)
            velocities[i] = (
                inertia * velocities[i]
                + c1 * r1 * (pbest_positions[i] - positions[i])
                + c2 * r2 * (social_target - positions[i])
            )
            positions[i] = positions[i] + velocities[i]
            positions[i], velocities[i] = reflect_with_velocity(positions[i], velocities[i])

            cand = evaluator.evaluate(positions[i])
            particle_eval[i] = cand
            if cand.penalized_cost < pbest_eval[i].penalized_cost:
                pbest_eval[i] = cand
                pbest_positions[i] = cand.x

        iteration_best = min(pbest_eval, key=lambda e: e.penalized_cost)
        if iteration_best.penalized_cost < gbest.penalized_cost:
            gbest = iteration_best
            gbest_position = np.array(gbest.x, copy=True)

        current_cost = float(np.median([pe.penalized_cost for pe in particle_eval]))
        history.append(
            IterationLog(
                iteration=iteration,
                eval_count=evaluator.eval_count,
                best_cost=gbest.penalized_cost,
                best_objective=gbest.objective,
                best_violation=gbest.violation,
                best_feasible=gbest.feasible,
                current_cost=current_cost,
                diversity=_mean_pairwise_distance(positions),
            )
        )

        no_improve = _stagnation_update(gbest.penalized_cost, prev_best, stagnation_epsilon, no_improve)
        prev_best = gbest.penalized_cost
        if no_improve >= stagnation_patience:
            stop_reason = "stagnation"
            break

    return AlgorithmResult(name=f"PSO_{neighborhood}", best_eval=gbest, history=history, stop_reason=stop_reason)


def run_de_rand_1_bin(
    rng: np.random.Generator,
    evaluator: PenalizedEvaluator,
    population_size: int,
    max_iter: int,
    stagnation_patience: int,
    stagnation_epsilon: float,
    f_weight: float = 0.7,
    crossover_rate: float = 0.9,
    bounds_strategy: Literal["clip"] = "clip",
) -> AlgorithmResult:
    population = random_population(rng, population_size)
    pop_eval = [evaluator.evaluate(ind) for ind in population]
    best = min(pop_eval, key=lambda e: e.penalized_cost)
    history: list[IterationLog] = []
    no_improve = 0
    prev_best = best.penalized_cost
    stop_reason = "max_iter"

    dim = population.shape[1]
    for iteration in range(1, max_iter + 1):
        for i in range(population_size):
            choices = [idx for idx in range(population_size) if idx != i]
            r1, r2, r3 = rng.choice(choices, size=3, replace=False)
            donor = population[r1] + f_weight * (population[r2] - population[r3])
            if bounds_strategy == "clip":
                donor = project_clip(donor)

            j_rand = int(rng.integers(0, dim))
            trial = population[i].copy()
            for j in range(dim):
                if rng.random() <= crossover_rate or j == j_rand:
                    trial[j] = donor[j]

            if bounds_strategy == "clip":
                trial = project_clip(trial)

            trial_eval = evaluator.evaluate(trial)
            if trial_eval.penalized_cost <= pop_eval[i].penalized_cost:
                population[i] = trial
                pop_eval[i] = trial_eval

        iter_best = min(pop_eval, key=lambda e: e.penalized_cost)
        if iter_best.penalized_cost < best.penalized_cost:
            best = iter_best

        current_cost = float(np.median([pe.penalized_cost for pe in pop_eval]))
        history.append(
            IterationLog(
                iteration=iteration,
                eval_count=evaluator.eval_count,
                best_cost=best.penalized_cost,
                best_objective=best.objective,
                best_violation=best.violation,
                best_feasible=best.feasible,
                current_cost=current_cost,
                diversity=_mean_pairwise_distance(population),
            )
        )

        no_improve = _stagnation_update(best.penalized_cost, prev_best, stagnation_epsilon, no_improve)
        prev_best = best.penalized_cost
        if no_improve >= stagnation_patience:
            stop_reason = "stagnation"
            break

    return AlgorithmResult(name="DE_rand_1_bin", best_eval=best, history=history, stop_reason=stop_reason)
