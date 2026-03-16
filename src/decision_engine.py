import numpy as np

def generate_decision(fit_score,financial_risk):
    fit_norm = fit_score/100; risk_norm = financial_risk/100
    decision_score = 0.6*fit_norm+0.4*(1-risk_norm)
    recommendation = "Proceed" if decision_score>0.7 else ("Monitor" if decision_score>0.5 else "Avoid")
    risk_level = "Low" if financial_risk<30 else ("Medium" if financial_risk<60 else "High")
    reasons = []
    if fit_score>70: reasons.append("Strong tactical upgrade relative to squad.")
    elif fit_score<40: reasons.append("Limited tactical improvement over current options.")
    if financial_risk>60: reasons.append("High financial exposure under current constraints.")
    elif financial_risk<30: reasons.append("Financially sustainable within current structure.")
    if not reasons: reasons.append("Balanced trade-off between performance and risk.")
    return {"decision_score":round(decision_score*100,2),"recommendation":recommendation,
            "risk_level":risk_level,"reasons":reasons}