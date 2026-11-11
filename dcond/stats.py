from itertools import combinations

import pandas as pd
from scipy import stats

ALPHA = 0.05


def holm_bonferroni(p_values: list[float], alpha: float = ALPHA) -> list[tuple[float, bool]]:
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, i in enumerate(order):
        candidate = (n - rank) * p_values[i]
        running_max = max(running_max, candidate)
        adjusted[i] = min(running_max, 1.0)
    reject = [adjusted[i] < alpha for i in range(n)]
    return list(zip(adjusted, reject))


def seed_summary(df: pd.DataFrame, metrics: list[str], group_cols: list[str] = ("strategy", "budget")) -> pd.DataFrame:
    group_cols = list(group_cols)
    summary = df.groupby(group_cols)[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def paired_significance(
    df: pd.DataFrame, metrics: list[str], budget_col: str = "budget", strategy_col: str = "strategy",
    seed_col: str = "seed", alpha: float = ALPHA,
) -> pd.DataFrame:
    rows = []
    for budget in sorted(df[budget_col].unique()):
        strategies = sorted(df[df[budget_col] == budget][strategy_col].unique())
        for a, b in combinations(strategies, 2):
            da = df[(df[strategy_col] == a) & (df[budget_col] == budget)].sort_values(seed_col)
            db = df[(df[strategy_col] == b) & (df[budget_col] == budget)].sort_values(seed_col)
            if len(da) < 2 or len(da) != len(db):
                continue
            for metric in metrics:
                t, p = stats.ttest_rel(da[metric].values, db[metric].values)
                rows.append({
                    budget_col: budget, "strategy_a": a, "strategy_b": b, "metric": metric,
                    "mean_a": da[metric].mean(), "mean_b": db[metric].mean(),
                    "mean_diff_a_minus_b": da[metric].mean() - db[metric].mean(),
                    "n_seeds": len(da), "t_stat": t, "p_value": p,
                })
    sig_df = pd.DataFrame(rows)
    if sig_df.empty:
        return sig_df
    corrected = holm_bonferroni(sig_df["p_value"].tolist(), alpha=alpha)
    sig_df["p_value_holm"] = [c[0] for c in corrected]
    sig_df["significant_holm"] = [c[1] for c in corrected]
    return sig_df
