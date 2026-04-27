from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from algorithms import run_de_rand_1_bin, run_pso
from problem import PenalizedEvaluator


@dataclass
class ExperimentConfig:
    runs: int = 50
    max_iter: int = 800
    stagnation_patience: int = 120
    stagnation_epsilon: float = 1e-8
    seed: int = 2026
    penalty_coeff: float = 1e6
    swarm_size: int = 40
    de_population_size: int = 40
    pso_inertia: float = 0.72
    pso_c1: float = 1.49
    pso_c2: float = 1.49
    pso_neighborhood: str = "ring"
    de_f: float = 0.7
    de_cr: float = 0.9


def run_monte_carlo(config: ExperimentConfig) -> pd.DataFrame:
    rows: list[dict] = []
    algo_final_rows: list[dict] = []

    for run_id in range(config.runs):
        run_seed = config.seed + run_id

        pso_rng = np.random.Generator(np.random.PCG64(run_seed))
        pso_eval = PenalizedEvaluator(penalty_coeff=config.penalty_coeff)
        pso_result = run_pso(
            rng=pso_rng,
            evaluator=pso_eval,
            swarm_size=config.swarm_size,
            max_iter=config.max_iter,
            stagnation_patience=config.stagnation_patience,
            stagnation_epsilon=config.stagnation_epsilon,
            inertia=config.pso_inertia,
            c1=config.pso_c1,
            c2=config.pso_c2,
            neighborhood=config.pso_neighborhood,
        )

        de_rng = np.random.Generator(np.random.PCG64(run_seed))
        de_eval = PenalizedEvaluator(penalty_coeff=config.penalty_coeff)
        de_result = run_de_rand_1_bin(
            rng=de_rng,
            evaluator=de_eval,
            population_size=config.de_population_size,
            max_iter=config.max_iter,
            stagnation_patience=config.stagnation_patience,
            stagnation_epsilon=config.stagnation_epsilon,
            f_weight=config.de_f,
            crossover_rate=config.de_cr,
            bounds_strategy="clip",
        )

        for result in (pso_result, de_result):
            for rec in result.history:
                rows.append(
                    {
                        "run": run_id,
                        "seed": run_seed,
                        "algo": result.name,
                        "iteration": rec.iteration,
                        "eval_count": rec.eval_count,
                        "best_cost": rec.best_cost,
                        "best_objective": rec.best_objective,
                        "best_violation": rec.best_violation,
                        "best_feasible": rec.best_feasible,
                        "current_cost": rec.current_cost,
                        "diversity": rec.diversity,
                        "stop_reason": result.stop_reason,
                    }
                )

            algo_final_rows.append(
                {
                    "run": run_id,
                    "seed": run_seed,
                    "algo": result.name,
                    "final_best_cost": result.best_eval.penalized_cost,
                    "final_objective": result.best_eval.objective,
                    "final_violation": result.best_eval.violation,
                    "final_feasible": result.best_eval.feasible,
                    "stop_reason": result.stop_reason,
                    "total_evals": result.history[-1].eval_count if result.history else 0,
                    "n_iterations": len(result.history),
                }
            )

    history_df = pd.DataFrame(rows)
    final_df = pd.DataFrame(algo_final_rows)
    return history_df.merge(final_df, on=["run", "seed", "algo", "stop_reason"], how="left")


def save_experiment_outputs(df: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    history_csv = output_dir / "runs_history.csv"
    history_pkl = output_dir / "runs_history.pkl"
    df.to_csv(history_csv, index=False)
    df.to_pickle(history_pkl)
    paths["history_csv"] = history_csv
    paths["history_pkl"] = history_pkl
    return paths
