import pandas as pd
import numpy as np

def compute_performance_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for m in ["xg","xa","progressive_passes","defensive_actions"]:
        df[f"{m}_z"] = (df[m]-df[m].mean())/df[m].std()
    def ws(row):
        if row["position"]=="ST":
            return 0.5*row["xg_z"]+0.2*row["xa_z"]+0.2*row["progressive_passes_z"]+0.1*row["defensive_actions_z"]
        elif row["position"] in ["AM","CM"]:
            return 0.25*row["xg_z"]+0.35*row["xa_z"]+0.25*row["progressive_passes_z"]+0.15*row["defensive_actions_z"]
        elif row["position"] in ["DM","CB"]:
            return 0.1*row["xg_z"]+0.1*row["xa_z"]+0.3*row["progressive_passes_z"]+0.5*row["defensive_actions_z"]
        elif row["position"]=="FB":
            return 0.15*row["xg_z"]+0.2*row["xa_z"]+0.35*row["progressive_passes_z"]+0.3*row["defensive_actions_z"]
        elif row["position"]=="GK":
            return row["defensive_actions_z"]
        return 0
    df["performance_index_raw"] = df.apply(ws,axis=1)
    df["performance_index"] = df["performance_index_raw"]*np.clip(df["minutes"]/2500,0.3,1.0)
    return df