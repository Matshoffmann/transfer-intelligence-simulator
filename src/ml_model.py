# FILE: src/ml_model.py

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

from src.projection_model import project_uplift


NUMERIC_FEATURES = [
    "age", "minutes", "xg", "xa", "progressive_passes",
    "defensive_actions", "injury_risk", "market_value_m", "estimated_wage_m"
]
CATEGORICAL_FEATURES = ["position"]


def build_training_table(squad_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a club-context training table:
    - For each market player, compute uplift vs selected club squad baseline
    - Define success label: uplift > 0
    """
    rows = []
    for _, p in market_df.iterrows():
        upl = project_uplift(squad_df, p)
        uplift_val = float(upl["uplift"])
        rows.append({
            **{col: p[col] for col in (NUMERIC_FEATURES + CATEGORICAL_FEATURES)},
            "uplift": uplift_val,
            "success": int(uplift_val > 0)
        })

    df_train = pd.DataFrame(rows)
    df_train = df_train.replace([np.inf, -np.inf], np.nan).dropna()
    return df_train


def train_success_model(squad_df: pd.DataFrame, market_df: pd.DataFrame, random_state: int = 42):
    """
    Train a Logistic Regression model predicting P(success) = P(uplift > 0)
    in the selected club context.
    Returns:
      model_pipeline, metrics_dict
    """
    df_train = build_training_table(squad_df, market_df)

    X = df_train[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df_train["success"].astype(int)

    if y.nunique() < 2:
        return None, {
            "n": int(len(df_train)),
            "auc": None,
            "accuracy": None,
            "note": "Training labels degenerate (all same). Try different club/seed."
        }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="drop"
    )

    clf = LogisticRegression(
        max_iter=2000,
        solver="lbfgs"
    )

    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("clf", clf)
    ])

    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    auc = float(roc_auc_score(y_test, proba))
    acc = float(accuracy_score(y_test, preds))

    metrics = {
        "n": int(len(df_train)),
        "auc": round(auc, 3),
        "accuracy": round(acc, 3),
        "note": "LogReg trained on club-context uplift label (uplift > 0)."
    }

    return pipe, metrics


def predict_success_proba(model_pipe, players_df: pd.DataFrame) -> np.ndarray:
    """
    Predict P(uplift > 0) for a set of players. Returns array of probabilities.
    If model_pipe is None (degenerate labels), returns 0.5 as neutral.
    """
    if model_pipe is None:
        return np.full(shape=(len(players_df),), fill_value=0.5, dtype=float)

    X = players_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    proba = model_pipe.predict_proba(X)[:, 1]
    return proba.astype(float)


def get_feature_importance(model_pipe, top_k: int = 15) -> pd.DataFrame:
    """
    Global explainability:
    Returns top features by absolute coefficient magnitude (LogReg).
    """
    if model_pipe is None:
        return pd.DataFrame(columns=["feature", "coefficient", "abs_coefficient"])

    prep = model_pipe.named_steps["prep"]
    clf = model_pipe.named_steps["clf"]

    # Works on sklearn >= 1.0
    feature_names = prep.get_feature_names_out()
    coefs = clf.coef_.ravel()

    df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs),
    }).sort_values("abs_coefficient", ascending=False)

    return df.head(top_k).reset_index(drop=True)


def explain_prediction(model_pipe, player_row: pd.Series, top_k: int = 8) -> pd.DataFrame:
    """
    Local explainability:
    Returns top positive and negative feature contributions for a single player.
    Contribution = transformed_feature_value * coefficient
    (in the linear log-odds space of logistic regression)
    """
    if model_pipe is None:
        return pd.DataFrame(columns=["feature", "value", "coefficient", "contribution"])

    prep = model_pipe.named_steps["prep"]
    clf = model_pipe.named_steps["clf"]

    x_raw = pd.DataFrame([player_row[NUMERIC_FEATURES + CATEGORICAL_FEATURES]])
    x_trans = prep.transform(x_raw)  # may be sparse
    x_vec = np.asarray(x_trans.todense()).ravel() if hasattr(x_trans, "todense") else np.asarray(x_trans).ravel()

    feature_names = prep.get_feature_names_out()
    coefs = clf.coef_.ravel()

    contrib = x_vec * coefs

    df = pd.DataFrame({
        "feature": feature_names,
        "value": x_vec,
        "coefficient": coefs,
        "contribution": contrib
    })

    # Pick strongest drivers (both directions)
    df_pos = df.sort_values("contribution", ascending=False).head(top_k)
    df_neg = df.sort_values("contribution", ascending=True).head(top_k)

    out = pd.concat([df_pos, df_neg], ignore_index=True)
    out["contribution"] = out["contribution"].round(4)
    out["coefficient"] = out["coefficient"].round(4)
    out["value"] = out["value"].round(4)

    # Add label for direction
    out["direction"] = np.where(out["contribution"] >= 0, "↑ increases P(success)", "↓ decreases P(success)")
    return out