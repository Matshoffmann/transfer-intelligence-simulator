import numpy as np
import pandas as pd


def compute_position_depth(df: pd.DataFrame, position: str):
    return len(df[df["position"] == position])


def compute_relative_performance(df: pd.DataFrame, player_row: pd.Series):
    same_position = df[df["position"] == player_row["position"]]
    avg_performance = same_position["performance_index"].mean()
    return player_row["performance_index"] - avg_performance


def compute_tactical_fit(df: pd.DataFrame, player_row: pd.Series):
    depth = compute_position_depth(df, player_row["position"])
    relative_perf = compute_relative_performance(df, player_row)

    # Normalize depth effect
    depth_factor = np.clip(1 - depth / 8, 0, 1)

    # Normalize relative performance
    perf_factor = 1 / (1 + np.exp(-relative_perf))

    # Age factor (peak around 26-28)
    age_factor = 1 - abs(player_row["age"] - 27) / 15
    age_factor = np.clip(age_factor, 0, 1)

    fit_score = (
        0.4 * depth_factor +
        0.4 * perf_factor +
        0.2 * age_factor
    )

    return fit_score * 100