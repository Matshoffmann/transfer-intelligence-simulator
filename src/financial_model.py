import numpy as np
import pandas as pd


def compute_amortization_cost(transfer_fee_m, contract_years=4):
    return transfer_fee_m / contract_years


def compute_wage_ratio(estimated_wage_m, wage_capacity_m):
    if wage_capacity_m <= 0:
        return 1.0
    return estimated_wage_m / wage_capacity_m


def compute_age_depreciation(age):
    peak = 27
    depreciation = abs(age - peak) / 12
    return np.clip(depreciation, 0, 1)


def compute_financial_risk(player_row: pd.Series,
                           transfer_budget_m,
                           wage_capacity_m):

    transfer_fee = player_row["market_value_m"] * 1.1
    amortization = compute_amortization_cost(transfer_fee)

    wage_ratio = compute_wage_ratio(
        player_row["estimated_wage_m"],
        wage_capacity_m
    )

    age_risk = compute_age_depreciation(player_row["age"])
    injury_risk = player_row["injury_risk"]

    budget_pressure = transfer_fee / transfer_budget_m
    budget_pressure = np.clip(budget_pressure, 0, 1)

    risk_score = (
        0.3 * wage_ratio +
        0.25 * age_risk +
        0.25 * injury_risk +
        0.2 * budget_pressure
    )

    return np.clip(risk_score * 100, 0, 100)