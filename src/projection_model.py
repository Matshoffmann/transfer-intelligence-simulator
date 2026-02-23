import numpy as np
import pandas as pd


def compute_baseline_for_position(df: pd.DataFrame, position: str) -> float:
    """Baseline = average performance_index of the current squad in that position."""
    sub = df[df["position"] == position]
    if len(sub) == 0:
        return df["performance_index"].mean()
    return float(sub["performance_index"].mean())


def compute_uncertainty_from_minutes(minutes: float) -> float:
    """
    Simple uncertainty proxy:
    - low minutes => high uncertainty
    - high minutes => low uncertainty
    Returns std-like value in performance_index units.
    """
    # minutes in [200..3200] in our synthetic data; map to [high..low]
    m = np.clip(minutes, 200, 3200)
    # high uncertainty ~ 0.35, low uncertainty ~ 0.10
    return float(0.35 - (m - 200) / (3200 - 200) * (0.35 - 0.10))


def project_uplift(df: pd.DataFrame, player_row: pd.Series) -> dict:
    """
    Project uplift vs position baseline using performance_index.
    Returns uplift + uncertainty band.
    """
    baseline = compute_baseline_for_position(df, player_row["position"])
    player_perf = float(player_row["performance_index"])
    uplift = player_perf - baseline

    sigma = compute_uncertainty_from_minutes(player_row["minutes"])
    lower = uplift - 1.0 * sigma
    upper = uplift + 1.0 * sigma

    return {
        "baseline": baseline,
        "player_perf": player_perf,
        "uplift": uplift,
        "uplift_lower": lower,
        "uplift_upper": upper,
        "uncertainty_sigma": sigma
    }