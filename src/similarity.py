import numpy as np
import pandas as pd

def compute_tactical_fit(df,player_row):
    depth = len(df[df["position"]==player_row["position"]])
    same = df[df["position"]==player_row["position"]]
    rel_perf = player_row["performance_index"]-same["performance_index"].mean() if len(same)>0 else 0
    depth_factor = float(np.clip(1-depth/8,0,1))
    perf_factor = float(1/(1+np.exp(-rel_perf)))
    age_factor = float(np.clip(1-abs(player_row["age"]-27)/15,0,1))
    return float((0.4*depth_factor+0.4*perf_factor+0.2*age_factor)*100)