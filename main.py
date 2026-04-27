from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analysis import (
    budget_comparability_table,
    convergence_profile,
    final_statistics_table,
    normalize_to_budget,
    plot_best_run_convergence,
    plot_boxplot_final,
    plot_convergence,
    plot_diversity_evolution,
    plot_feasibility_rate,
    plot_violin_final,
)
from experiments import ExperimentConfig, run_monte_carlo, save_experiment_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TP03 - 8INF852 - PSO vs DE on constrained spring design")
    parser.add_argument("--runs", type=int, default=50, help="Monte-Carlo runs per algorithm")
    parser.add_argument("--max-iter", type=int, default=800, help="Maximum iterations")
    parser.add_argument("--stagnation-patience", type=int, default=120, help="Stop after N non-improving iterations")
    parser.add_argument("--stagnation-epsilon", type=float, default=1e-8, help="Minimum significant improvement")
    parser.add_argument("--seed", type=int, default=2026, help="Base random seed")
    parser.add_argument("--penalty-coeff", type=float, default=1e6, help="Penalty coefficient")
    parser.add_argument("--swarm-size", type=int, default=40, help="PSO swarm size")
    parser.add_argument("--de-population-size", type=int, default=40, help="DE population size")
    parser.add_argument("--pso-neighborhood", choices=["ring", "global"], default="ring", help="PSO topology")
    parser.add_argument("--pso-inertia", type=float, default=0.72, help="PSO inertia weight")
    parser.add_argument("--pso-c1", type=float, default=1.49, help="PSO cognitive coefficient")
    parser.add_argument("--pso-c2", type=float, default=1.49, help="PSO social coefficient")
    parser.add_argument("--de-f", type=float, default=0.7, help="DE differential weight")
    parser.add_argument("--de-cr", type=float, default=0.9, help="DE crossover rate")
    parser.add_argument("--n-checkpoints", type=int, default=100, help="Checkpoints for budget-normalized profile")
    parser.add_argument("--tp1-best-cost", type=float, default=None, help="Best TP1 reference penalized cost")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory")
    return parser.parse_args()


def _compare_with_tp1(stats_df: pd.DataFrame, tp1_best_cost: float | None) -> pd.DataFrame:
    if tp1_best_cost is None:
        return pd.DataFrame(columns=["algo", "tp1_best_cost", "delta_to_tp1"])

    table = stats_df[["algo", "best"]].copy()
    table["tp1_best_cost"] = float(tp1_best_cost)
    table["delta_to_tp1"] = table["best"] - float(tp1_best_cost)
    return table.rename(columns={"best": "tp03_best_cost"})


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig(
        runs=args.runs,
        max_iter=args.max_iter,
        stagnation_patience=args.stagnation_patience,
        stagnation_epsilon=args.stagnation_epsilon,
        seed=args.seed,
        penalty_coeff=args.penalty_coeff,
        swarm_size=args.swarm_size,
        de_population_size=args.de_population_size,
        pso_inertia=args.pso_inertia,
        pso_c1=args.pso_c1,
        pso_c2=args.pso_c2,
        pso_neighborhood=args.pso_neighborhood,
        de_f=args.de_f,
        de_cr=args.de_cr,
    )

    print("[1/5] Running Monte-Carlo experiments...")
    history_df = run_monte_carlo(cfg)

    print("[2/5] Saving raw outputs...")
    output_paths = save_experiment_outputs(history_df, args.output_dir)

    print("[3/5] Computing statistics...")
    stats_df = final_statistics_table(history_df)
    budget_df = budget_comparability_table(history_df)
    norm_df = normalize_to_budget(history_df, n_checkpoints=args.n_checkpoints)
    profile_df = convergence_profile(norm_df)
    tp1_cmp_df = _compare_with_tp1(stats_df, args.tp1_best_cost)

    stats_path = args.output_dir / "final_statistics.csv"
    budget_path = args.output_dir / "budget_comparability.csv"
    profile_path = args.output_dir / "convergence_profile.csv"
    tp1_cmp_path = args.output_dir / "tp1_comparison.csv"
    stats_df.to_csv(stats_path, index=False)
    budget_df.to_csv(budget_path, index=False)
    profile_df.to_csv(profile_path, index=False)
    tp1_cmp_df.to_csv(tp1_cmp_path, index=False)

    print("[4/5] Generating figures...")
    convergence_png = args.output_dir / "convergence_pso_vs_de.png"
    boxplot_png = args.output_dir / "boxplot_final_costs.png"
    feasible_png = args.output_dir / "feasibility_rate.png"
    violin_png = args.output_dir / "violin_final_costs.png"
    diversity_png = args.output_dir / "diversity_evolution.png"
    best_run_png = args.output_dir / "best_run_convergence.png"
    plot_convergence(profile_df, convergence_png)
    plot_boxplot_final(history_df, boxplot_png)
    plot_feasibility_rate(history_df, feasible_png)
    plot_violin_final(history_df, violin_png)
    plot_diversity_evolution(history_df, diversity_png, n_checkpoints=args.n_checkpoints)
    plot_best_run_convergence(history_df, best_run_png)

    print("[5/5] Done.")
    print(f"- History CSV: {output_paths['history_csv']}")
    print(f"- History PKL: {output_paths['history_pkl']}")
    print(f"- Final stats: {stats_path}")
    print(f"- Budget table: {budget_path}")
    print(f"- Convergence profile: {profile_path}")
    print(f"- TP1 comparison: {tp1_cmp_path}")
    print(f"- Figure convergence: {convergence_png}")
    print(f"- Figure boxplot: {boxplot_png}")
    print(f"- Figure violin: {violin_png}")
    print(f"- Figure diversity: {diversity_png}")
    print(f"- Figure best run: {best_run_png}")
    print(f"- Figure feasibility: {feasible_png}")


if __name__ == "__main__":
    main()
