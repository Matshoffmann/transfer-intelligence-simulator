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

NUMERIC_FEATURES = ["age","minutes","xg","xa","progressive_passes","defensive_actions","injury_risk","market_value_m","estimated_wage_m"]
CATEGORICAL_FEATURES = ["position"]

def build_training_table(squad_df,market_df):
    rows = []
    for _,p in market_df.iterrows():
        upl = project_uplift(squad_df,p)
        uplift_val = float(upl["uplift"])
        rows.append({**{col:p[col] for col in NUMERIC_FEATURES+CATEGORICAL_FEATURES},"uplift":uplift_val,"success":int(uplift_val>0)})
    df_train = pd.DataFrame(rows).replace([np.inf,-np.inf],np.nan).dropna()
    return df_train

def train_success_model(squad_df,market_df,random_state=42):
    df_train = build_training_table(squad_df,market_df)
    X = df_train[NUMERIC_FEATURES+CATEGORICAL_FEATURES]; y = df_train["success"].astype(int)
    if y.nunique()<2:
        return None,{"n":int(len(df_train)),"auc":None,"accuracy":None,"note":"Degenerate labels."}
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=random_state,stratify=y)
    preprocessor = ColumnTransformer([("num",StandardScaler(),NUMERIC_FEATURES),("cat",OneHotEncoder(handle_unknown="ignore"),CATEGORICAL_FEATURES)],remainder="drop")
    pipe = Pipeline([("prep",preprocessor),("clf",LogisticRegression(max_iter=2000,solver="lbfgs"))])
    pipe.fit(X_train,y_train)
    proba = pipe.predict_proba(X_test)[:,1]
    metrics = {"n":int(len(df_train)),"auc":round(float(roc_auc_score(y_test,proba)),3),
               "accuracy":round(float(accuracy_score(y_test,(proba>=0.5).astype(int))),3),
               "note":"LogReg trained on club-context uplift label."}
    return pipe,metrics

def predict_success_proba(model_pipe,players_df):
    if model_pipe is None: return np.full(len(players_df),0.5)
    return model_pipe.predict_proba(players_df[NUMERIC_FEATURES+CATEGORICAL_FEATURES])[:,1].astype(float)

def get_feature_importance(model_pipe,top_k=15):
    if model_pipe is None: return pd.DataFrame(columns=["feature","coefficient","abs_coefficient"])
    prep = model_pipe.named_steps["prep"]; clf = model_pipe.named_steps["clf"]
    feature_names = prep.get_feature_names_out(); coefs = clf.coef_.ravel()
    df = pd.DataFrame({"feature":feature_names,"coefficient":coefs,"abs_coefficient":np.abs(coefs)})
    return df.sort_values("abs_coefficient",ascending=False).head(top_k).reset_index(drop=True)

def explain_prediction(model_pipe,player_row,top_k=8):
    if model_pipe is None: return pd.DataFrame(columns=["feature","value","coefficient","contribution","direction"])
    prep = model_pipe.named_steps["prep"]; clf = model_pipe.named_steps["clf"]
    x_raw = pd.DataFrame([player_row[NUMERIC_FEATURES+CATEGORICAL_FEATURES]])
    x_trans = prep.transform(x_raw)
    x_vec = np.asarray(x_trans.todense()).ravel() if hasattr(x_trans,"todense") else np.asarray(x_trans).ravel()
    feature_names = prep.get_feature_names_out(); coefs = clf.coef_.ravel(); contrib = x_vec*coefs
    df = pd.DataFrame({"feature":feature_names,"value":x_vec,"coefficient":coefs,"contribution":contrib})
    out = pd.concat([df.sort_values("contribution",ascending=False).head(top_k),
                     df.sort_values("contribution",ascending=True).head(top_k)],ignore_index=True)
    out["contribution"] = out["contribution"].round(4); out["coefficient"] = out["coefficient"].round(4)
    out["value"] = out["value"].round(4)
    out["direction"] = np.where(out["contribution"]>=0,"increases P(success)","decreases P(success)")
    return out