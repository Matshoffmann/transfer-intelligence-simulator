import numpy as np

def generate_decision(fit_score, financial_risk):
    fit_norm  = fit_score / 100
    risk_norm = financial_risk / 100
    decision_score = 0.6 * fit_norm + 0.4 * (1 - risk_norm)
    recommendation = "Sign" if decision_score > 0.7 else ("Watch" if decision_score > 0.5 else "Pass")
    risk_level = "Low" if financial_risk < 30 else ("Medium" if financial_risk < 60 else "High")

    reasons = []
    if fit_score > 70:
        reasons.append("Clear upgrade over current options at this position.")
    elif fit_score < 40:
        reasons.append("Limited added value given current squad depth.")

    if financial_risk > 60:
        reasons.append("Fee or wage exceeds comfortable thresholds for this budget.")
    elif financial_risk < 30:
        reasons.append("Fee and wage sit well within your defined limits.")

    if not reasons:
        reasons.append("Solid option with a manageable risk-reward trade-off.")

    return {
        "decision_score":   round(decision_score * 100, 2),
        "recommendation":   recommendation,
        "risk_level":       risk_level,
        "reasons":          reasons,
    }
