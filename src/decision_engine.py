import numpy as np


def generate_decision(fit_score, financial_risk):

    # Normalize inputs
    fit_norm = fit_score / 100
    risk_norm = financial_risk / 100

    # Composite decision score
    decision_score = 0.6 * fit_norm + 0.4 * (1 - risk_norm)

    if decision_score > 0.7:
        recommendation = "Proceed"
    elif decision_score > 0.5:
        recommendation = "Monitor"
    else:
        recommendation = "Avoid"

    # Risk Level
    if financial_risk < 30:
        risk_level = "Low"
    elif financial_risk < 60:
        risk_level = "Medium"
    else:
        risk_level = "High"

    # Generate reasons
    reasons = []

    if fit_score > 70:
        reasons.append("Strong tactical upgrade relative to squad.")
    elif fit_score < 40:
        reasons.append("Limited tactical improvement over current options.")

    if financial_risk > 60:
        reasons.append("High financial exposure under current constraints.")
    elif financial_risk < 30:
        reasons.append("Financially sustainable within current structure.")

    if len(reasons) == 0:
        reasons.append("Balanced trade-off between performance and risk.")

    return {
        "decision_score": np.round(decision_score * 100, 2),
        "recommendation": recommendation,
        "risk_level": risk_level,
        "reasons": reasons
    }