import numpy as np
import pandas as pd

def compute_baseline_for_position(df,position):
    sub = df[df["position"]==position]
    return float(sub["performance_index"].mean()) if len(sub)>0 else float(df["performance_index"].mean())

def compute_uncertainty_from_minutes(minutes):
    m = np.clip(minutes,200,3200)
    return float(0.35-(m-200)/(3200-200)*(0.35-0.10))

def project_uplift(df,player_row):
    baseline = compute_baseline_for_position(df,player_row["position"])
    player_perf = float(player_row["performance_index"])
    uplift = player_perf-baseline
    sigma = compute_uncertainty_from_minutes(player_row["minutes"])
    return {"baseline":baseline,"player_perf":player_perf,"uplift":uplift,
            "uplift_lower":uplift-sigma,"uplift_upper":uplift+sigma,"uncertainty_sigma":sigma}