import numpy as np
import pandas as pd

def compute_financial_risk(player_row,transfer_budget_m,wage_capacity_m):
    transfer_fee = player_row["market_value_m"]*1.1
    wage_ratio = float(player_row["estimated_wage_m"])/max(wage_capacity_m,0.01)
    age_risk = float(np.clip(abs(player_row["age"]-27)/12,0,1))
    injury_risk = float(player_row["injury_risk"])
    budget_pressure = float(np.clip(transfer_fee/max(transfer_budget_m,0.01),0,1))
    risk_score = 0.3*wage_ratio+0.25*age_risk+0.25*injury_risk+0.2*budget_pressure
    return float(np.clip(risk_score*100,0,100))