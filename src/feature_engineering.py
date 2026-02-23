import pandas as pd
import numpy as np


def compute_performance_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalize core metrics
    metrics = ["xg", "xa", "progressive_passes", "defensive_actions"]

    for m in metrics:
        df[f"{m}_z"] = (df[m] - df[m].mean()) / df[m].std()

    # Position-weighted scoring
    def weighted_score(row):
        if row["position"] == "ST":
            return 0.5 * row["xg_z"] + 0.2 * row["xa_z"] + 0.2 * row["progressive_passes_z"] + 0.1 * row["defensive_actions_z"]
        elif row["position"] in ["AM", "CM"]:
            return 0.25 * row["xg_z"] + 0.35 * row["xa_z"] + 0.25 * row["progressive_passes_z"] + 0.15 * row["defensive_actions_z"]
        elif row["position"] in ["DM", "CB"]:
            return 0.1 * row["xg_z"] + 0.1 * row["xa_z"] + 0.3 * row["progressive_passes_z"] + 0.5 * row["defensive_actions_z"]
        elif row["position"] == "FB":
            return 0.15 * row["xg_z"] + 0.2 * row["xa_z"] + 0.35 * row["progressive_passes_z"] + 0.3 * row["defensive_actions_z"]
        elif row["position"] == "GK":
            return row["defensive_actions_z"]
        else:
            return 0

    df["performance_index_raw"] = df.apply(weighted_score, axis=1)

    # Minutes reliability adjustment
    reliability_factor = np.clip(df["minutes"] / 2500, 0.3, 1.0)
    df["performance_index"] = df["performance_index_raw"] * reliability_factor

    return df