from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def final_statistics_table(df: pd.DataFrame) -> pd.DataFrame:
    final_rows = (
        df.sort_values(["algo", "run", "eval_count"]).groupby(["algo", "run"], as_index=False).tail(1)
    )
    stats = (
        final_rows.groupby("algo")
        .agg(
            n_runs=("run", "count"),
            best=("final_best_cost", "min"),
            median=("final_best_cost", "median"),
            q1=("final_best_cost", lambda x: float(np.quantile(x, 0.25))),
            q3=("final_best_cost", lambda x: float(np.quantile(x, 0.75))),
            std=("final_best_cost", "std"),
            worst=("final_best_cost", "max"),
            feasible_rate=("final_feasible", "mean"),
            median_final_objective=("final_objective", "median"),
        )
        .reset_index()
    )
    stats["iqr"] = stats["q3"] - stats["q1"]
    return stats[
        [
            "algo",
            "n_runs",
            "best",
            "median",
            "q1",
            "q3",
            "iqr",
            "std",
            "worst",
            "feasible_rate",
            "median_final_objective",
        ]
    ]


def budget_comparability_table(df: pd.DataFrame) -> pd.DataFrame:
    final_rows = (
        df.sort_values(["algo", "run", "eval_count"]).groupby(["algo", "run"], as_index=False).tail(1)
    )
    summary = (
        final_rows.groupby("algo")
        .agg(
            median_total_evals=("total_evals", "median"),
            mean_total_evals=("total_evals", "mean"),
            min_total_evals=("total_evals", "min"),
            max_total_evals=("total_evals", "max"),
        )
        .reset_index()
    )
    if summary.shape[0] == 2:
        e0 = float(summary.loc[0, "median_total_evals"])
        e1 = float(summary.loc[1, "median_total_evals"])
        ratio = abs(e0 - e1) / max(e0, e1) if max(e0, e1) > 0 else 0.0
        summary["median_eval_gap_ratio"] = ratio
    return summary


def normalize_to_budget(df: pd.DataFrame, n_checkpoints: int = 100) -> pd.DataFrame:
    max_eval = int(df["eval_count"].max())
    checkpoints = np.linspace(0, max_eval, n_checkpoints + 1, dtype=int)[1:]
    rows: list[dict] = []

    for (run, algo), group in df.groupby(["run", "algo"]):
        g = group.sort_values("eval_count")
        eval_arr = g["eval_count"].to_numpy()
        best_arr = g["best_cost"].to_numpy()

        for cp in checkpoints:
            mask = eval_arr <= cp
            if not mask.any():
                continue
            idx = int(np.where(mask)[0][-1])
            rows.append(
                {
                    "run": run,
                    "algo": algo,
                    "checkpoint_eval": int(cp),
                    "best_cost": float(best_arr[idx]),
                }
            )
    return pd.DataFrame(rows)


def convergence_profile(df_norm: pd.DataFrame) -> pd.DataFrame:
    return (
        df_norm.groupby(["algo", "checkpoint_eval"], as_index=False)
        .agg(
            median_cost=("best_cost", "median"),
            q1_cost=("best_cost", lambda x: float(np.quantile(x, 0.25))),
            q3_cost=("best_cost", lambda x: float(np.quantile(x, 0.75))),
            min_cost=("best_cost", "min"),
            max_cost=("best_cost", "max"),
        )
        .sort_values(["algo", "checkpoint_eval"])
    )


def plot_convergence(profile_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    for algo, data in profile_df.groupby("algo"):
        ax.plot(data["checkpoint_eval"], data["median_cost"], label=f"{algo} median")
        ax.fill_between(data["checkpoint_eval"], data["q1_cost"], data["q3_cost"], alpha=0.2, label=f"{algo} IQR")

    ax.set_yscale("log")
    ax.set_xlabel("Objective function evaluations")
    ax.set_ylabel("Best penalized cost")
    ax.set_title("PSO vs DE convergence profile (median + IQR)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_feasibility_rate(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_rows = (
        df.sort_values(["algo", "run", "eval_count"]).groupby(["algo", "run"], as_index=False).tail(1)
    )
    rate = final_rows.groupby("algo")["final_feasible"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(rate.index.tolist(), rate.values.tolist())
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Feasible final solutions ratio")
    ax.set_title("Feasibility rate by algorithm")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_boxplot_final(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_rows = (
        df.sort_values(["algo", "run", "eval_count"]).groupby(["algo", "run"], as_index=False).tail(1)
    )
    algos = sorted(final_rows["algo"].unique())
    series = [final_rows[final_rows["algo"] == algo]["final_best_cost"].to_numpy() for algo in algos]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(series, labels=algos)
    ax.set_yscale("log")
    ax.set_ylabel("Final penalized cost")
    ax.set_title("Final cost distribution by algorithm")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_violin_final(df: pd.DataFrame, output_path: Path) -> None:
    """Violin plot of final penalized cost by algorithm."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_rows = (
        df.sort_values(["algo", "run", "eval_count"]).groupby(["algo", "run"], as_index=False).tail(1)
    )
    algos = sorted(final_rows["algo"].unique())
    series = [final_rows[final_rows["algo"] == algo]["final_best_cost"].to_numpy() for algo in algos]

    fig, ax = plt.subplots(figsize=(8, 5))
    parts = ax.violinplot(series, positions=range(1, len(algos) + 1), showmedians=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_alpha(0.6)
    ax.set_xticks(range(1, len(algos) + 1))
    ax.set_xticklabels(algos)
    ax.set_yscale("log")
    ax.set_ylabel("Final penalized cost (log scale)")
    ax.set_title("Distribution finale du coût pénalisé par algorithme")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_diversity_evolution(df: pd.DataFrame, output_path: Path, n_checkpoints: int = 100) -> None:
    """Median diversity (mean pairwise distance) vs evaluation budget, with IQR band."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    max_eval = int(df["eval_count"].max())
    checkpoints = np.linspace(0, max_eval, n_checkpoints + 1, dtype=int)[1:]

    rows: list[dict] = []
    for (run, algo), group in df.groupby(["run", "algo"]):
        g = group.sort_values("eval_count")
        eval_arr = g["eval_count"].to_numpy()
        div_arr = g["diversity"].to_numpy()
        for cp in checkpoints:
            mask = eval_arr <= cp
            if not mask.any():
                continue
            idx = int(np.where(mask)[0][-1])
            rows.append({"run": run, "algo": algo, "checkpoint_eval": int(cp), "diversity": float(div_arr[idx])})

    norm_df = pd.DataFrame(rows)
    profile = (
        norm_df.groupby(["algo", "checkpoint_eval"], as_index=False)
        .agg(
            median_div=("diversity", "median"),
            q1_div=("diversity", lambda x: float(np.quantile(x, 0.25))),
            q3_div=("diversity", lambda x: float(np.quantile(x, 0.75))),
        )
        .sort_values(["algo", "checkpoint_eval"])
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    for algo, data in profile.groupby("algo"):
        ax.plot(data["checkpoint_eval"], data["median_div"], label=f"{algo} médiane")
        ax.fill_between(data["checkpoint_eval"], data["q1_div"], data["q3_div"], alpha=0.2, label=f"{algo} IQR")
    ax.set_xlabel("Appels à la fonction objective")
    ax.set_ylabel("Diversité (distance pairwise moyenne)")
    ax.set_title("Évolution de la diversité de la population — PSO vs DE")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_best_run_convergence(df: pd.DataFrame, output_path: Path) -> None:
    """Convergence curve of the single best run per algorithm."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_rows = (
        df.sort_values(["algo", "run", "eval_count"]).groupby(["algo", "run"], as_index=False).tail(1)
    )
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"DE_rand_1_bin": "tab:blue", "PSO_ring": "tab:orange"}
    for algo, group in final_rows.groupby("algo"):
        best_run_id = int(group.loc[group["final_best_cost"].idxmin(), "run"])
        run_history = df[(df["algo"] == algo) & (df["run"] == best_run_id)].sort_values("eval_count")
        color = colors.get(algo, None)
        ax.plot(run_history["eval_count"], run_history["best_cost"], label=f"{algo} (meilleur run #{best_run_id})", color=color)

    ax.set_yscale("log")
    ax.set_xlabel("Appels à la fonction objective")
    ax.set_ylabel("Meilleur coût pénalisé (log scale)")
    ax.set_title("Convergence du meilleur run individuel — PSO vs DE")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
