# FILE: src/shortlist.py

from __future__ import annotations

import numpy as np
import pandas as pd

from src.similarity import compute_tactical_fit
from src.financial_model import compute_financial_risk
from src.projection_model import project_uplift
from src.decision_engine import generate_decision

from src.ml_model import train_success_model, predict_success_proba


def compute_bundle(
    squad_df: pd.DataFrame,
    market_df: pd.DataFrame,
    player_row: pd.Series,
    budget_m: float,
    wage_capacity_m: float,
    model_pipe=None
) -> dict:
    """
    Compute a complete evaluation bundle for a market player vs the current squad.
    Optionally uses a trained ML model to estimate P(success) = P(uplift > 0).
    """
    fit = compute_tactical_fit(squad_df, player_row)
    risk = compute_financial_risk(player_row, budget_m, wage_capacity_m)
    decision = generate_decision(fit, risk)
    uplift = project_uplift(squad_df, player_row)

    mv = float(player_row["market_value_m"])
    value_per_euro = float(uplift["uplift"]) / mv if mv > 0 else 0.0

    # ML probability (if model provided)
    p_success = None
    if model_pipe is not None:
        p_success = float(predict_success_proba(model_pipe, pd.DataFrame([player_row]))[0])

    # Expected uplift (risk-neutral) if p_success exists, else plain uplift
    expected_uplift = float(uplift["uplift"]) * (p_success if p_success is not None else 1.0)

    # Risk-adjusted value (simple & transparent)
    rav = expected_uplift - 0.5 * (float(risk) / 100.0)

    return {
        "fit": float(fit),
        "risk": float(risk),
        "decision": decision,
        "uplift": uplift,
        "value_per_euro": float(value_per_euro),
        "p_success": p_success,
        "expected_uplift": float(expected_uplift),
        "risk_adjusted_value": float(rav),
    }


def shortlist_top_targets(
    squad_df: pd.DataFrame,
    market_df: pd.DataFrame,
    position: str,
    budget_m: float,
    wage_capacity_m: float,
    top_n: int = 5,
    min_minutes: int = 400,
    model_pipe=None
) -> pd.DataFrame:
    """
    Generate a shortlist of top targets for a given position under constraints.
    Returns a DataFrame with key KPIs (incl. ML probability if model_pipe is passed).
    """
    candidates = market_df[
        (market_df["position"] == position) &
        (market_df["minutes"] >= min_minutes)
    ].copy()

    rows = []
    for _, p in candidates.iterrows():
        b = compute_bundle(squad_df, market_df, p, budget_m, wage_capacity_m, model_pipe=model_pipe)

        rows.append({
            "name": p["name"],
            "age": int(p["age"]),
            "position": p["position"],
            "market_value_m": float(p["market_value_m"]),
            "estimated_wage_m": float(p["estimated_wage_m"]),
            "injury_risk": float(p["injury_risk"]),
            "fit_score": b["fit"],
            "financial_risk": b["risk"],
            "uplift": float(b["uplift"]["uplift"]),
            "uplift_low": float(b["uplift"]["uplift_lower"]),
            "uplift_high": float(b["uplift"]["uplift_upper"]),
            "p_success": (round(float(b["p_success"]), 3) if b["p_success"] is not None else None),
            "expected_uplift": round(float(b["expected_uplift"]), 3),
            "risk_adjusted_value": round(float(b["risk_adjusted_value"]), 3),
            "recommendation": b["decision"]["recommendation"],
            "decision_score": float(b["decision"]["decision_score"]),
        })

    out = pd.DataFrame(rows)

    # Feasibility filter (budget & wage)
    out = out[out["market_value_m"] <= budget_m].copy()
    out = out[out["estimated_wage_m"] <= wage_capacity_m].copy()

    # Rank primarily by risk-adjusted value (which uses expected uplift if ML prob exists)
    out = out.sort_values(
        by=["risk_adjusted_value", "fit_score", "decision_score"],
        ascending=False
    )

    return out.head(top_n).reset_index(drop=True)


def train_club_model_cached(squad_df: pd.DataFrame, market_df: pd.DataFrame, random_state: int = 42):
    """
    Convenience wrapper in case you want to call it from the app.
    Returns (model_pipe, metrics_dict).
    """
    return train_success_model(squad_df, market_df, random_state=random_state)