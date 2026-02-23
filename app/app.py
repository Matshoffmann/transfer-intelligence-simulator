# FILE: app/app.py

import sys
from pathlib import Path

# Add project root to path (so `src` imports work when running `streamlit run app/app.py`)
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.data_loader import load_dataset
from src.feature_engineering import compute_performance_index
from src.shortlist import compute_bundle, shortlist_top_targets
from src.shortlist import train_club_model_cached  # wrapper around train_success_model
from src.ml_model import get_feature_importance, explain_prediction


st.set_page_config(
    page_title="Transfer Intelligence Simulator",
    layout="wide"
)

# ---------- Styling ----------
st.markdown("""
    <style>
        .main-title {
            font-size: 32px;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            font-size: 18px;
            color: #666;
            margin-bottom: 0.25rem;
        }
        .context-line {
            font-size: 14px;
            color: #9aa0a6;
            margin-bottom: 1.25rem;
        }
        .small-note {
            color: #888;
            font-size: 13px;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- Load Market Data ----------
market_df = load_dataset()
market_df = compute_performance_index(market_df)

# ---------- Club setup (Bundesliga) ----------
BUNDESLIGA_CLUBS = [
    "FC Bayern München", "Borussia Dortmund", "RB Leipzig", "Bayer 04 Leverkusen",
    "Eintracht Frankfurt", "SC Freiburg", "VfB Stuttgart", "TSG Hoffenheim",
    "1. FSV Mainz 05", "Borussia Mönchengladbach", "VfL Wolfsburg", "Werder Bremen",
    "FC Augsburg", "1. FC Köln", "Union Berlin", "VfL Bochum",
    "1. FC Heidenheim", "SV Darmstadt 98"
]
POSITION_BUCKETS = ["GK", "CB", "FB", "DM", "CM", "AM", "ST"]


def generate_squad_for_club(club_name: str, df_market: pd.DataFrame, squad_size: int = 23) -> pd.DataFrame:
    seed = abs(hash(club_name)) % (2**32 - 1)
    rng = np.random.default_rng(seed)

    composition = {"GK": 2, "CB": 4, "FB": 4, "DM": 3, "CM": 4, "AM": 3, "ST": 3}

    chosen_idx = []
    for pos, n in composition.items():
        pool = df_market[df_market["position"] == pos]
        if len(pool) == 0:
            continue
        pick = rng.choice(pool.index.to_numpy(), size=min(n, len(pool)), replace=False)
        chosen_idx.extend(pick.tolist())

    squad = df_market.loc[chosen_idx].copy()

    if len(squad) < squad_size:
        remaining = df_market.drop(index=chosen_idx)
        extra = rng.choice(
            remaining.index.to_numpy(),
            size=min(squad_size - len(squad), len(remaining)),
            replace=False
        )
        squad = pd.concat([squad, remaining.loc[extra]], ignore_index=True)

    squad["club"] = club_name
    return squad.reset_index(drop=True)


# ---------- Header ----------
st.markdown('<div class="main-title">Transfer Intelligence Simulator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Bundesliga Edition – Decision Support for Budget-Constrained Clubs</div>', unsafe_allow_html=True)

# ---------- Sidebar ----------
st.sidebar.header("Scenario Setup")

club = st.sidebar.selectbox("Bundesliga Club", BUNDESLIGA_CLUBS)
mode = st.sidebar.radio("Mode", ["Single Player", "A/B Comparison"], index=0)

transfer_budget = st.sidebar.slider("Transfer Budget (M€)", 5, 80, 30)
wage_capacity = st.sidebar.slider("Remaining Wage Capacity (M€/year)", 1, 30, 8)

squad_df = generate_squad_for_club(club, market_df)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Club Context:** {club}")
st.sidebar.caption("Fit and baselines are computed against the selected club's squad context.")
st.sidebar.markdown(f"<div class='small-note'>Squad size: {len(squad_df)}</div>", unsafe_allow_html=True)

# ---------- Train ML model (club-context) ----------
@st.cache_resource
def _train_model_for_club(club_name: str):
    squad_local = generate_squad_for_club(club_name, market_df)
    model, metrics = train_club_model_cached(squad_local, market_df, random_state=42)
    return model, metrics

model_pipe, model_metrics = _train_model_for_club(club)

# ---------- Context line under header ----------
st.markdown(
    f"<div class='context-line'>Active club context: <b>{club}</b> · "
    f"ML success model trained on: uplift &gt; 0 vs this squad baseline.</div>",
    unsafe_allow_html=True
)

# Player selection
if mode == "Single Player":
    player_name = st.sidebar.selectbox("Select Target Player (Market)", market_df["name"].tolist())
    player = market_df[market_df["name"] == player_name].iloc[0]
else:
    player_a_name = st.sidebar.selectbox("Select Player A (Market)", market_df["name"].tolist(), index=0)
    player_b_name = st.sidebar.selectbox("Select Player B (Market)", market_df["name"].tolist(), index=1)
    if player_a_name == player_b_name:
        st.sidebar.warning("Select two different players for comparison.")

# ---------- Model Card + Explainability ----------
with st.expander("Model Card & Explainability", expanded=False):
    st.write("#### What the model predicts")
    st.write(
        "The ML layer predicts **P(Positive Uplift)**, i.e. the probability that a candidate's projected uplift "
        "vs the selected club's squad baseline is **greater than zero**."
    )

    st.write("#### Model type & training")
    st.write(
        "- Model: Logistic Regression (interpretable linear model)\n"
        "- Features: numeric player stats + position one-hot\n"
        "- Label: `success = 1` if projected uplift > 0 else 0\n"
        "- Training context: club-specific (baseline depends on the selected squad)"
    )

    st.write("#### Prototype quality metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("Training N", model_metrics.get("n", "—"))
    m2.metric("AUC", model_metrics.get("auc", "—"))
    m3.metric("Accuracy", model_metrics.get("accuracy", "—"))
    if model_metrics.get("note"):
        st.caption(model_metrics["note"])

    st.write("#### Global drivers (feature importance)")
    imp_df = get_feature_importance(model_pipe, top_k=15)
    if len(imp_df) == 0:
        st.info("Feature importance not available (model not trained).")
    else:
        # Plotly horizontal bar of absolute coefficients
        fig_imp = go.Figure()
        fig_imp.add_trace(go.Bar(
            x=imp_df["abs_coefficient"],
            y=imp_df["feature"],
            orientation="h",
            name="|coefficient|"
        ))
        fig_imp.update_layout(
            title="Top global drivers (absolute logistic coefficients)",
            xaxis_title="Absolute coefficient magnitude",
            yaxis_title="Feature",
            template="plotly_dark",
            height=520
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    st.write("#### Limitations (prototype)")
    st.write(
        "- Training data is synthetic and calibrated (not real event-level match data).\n"
        "- The label is based on the prototype's uplift function, not observed outcomes.\n"
        "- Coefficients reflect correlations in the synthetic world, not causal effects.\n"
        "- Intended purpose: decision prototyping, not production scouting."
    )


# ---------- Tabs ----------
tab_single, tab_financial, tab_compare = st.tabs(
    ["Single Transfer Simulation", "Financial Analysis", "Comparison Mode"]
)

# ---------- SINGLE TAB ----------
with tab_single:
    if mode != "Single Player":
        st.info("Switch to 'Single Player' mode in the sidebar to run a single-player simulation.")
    else:
        bundle = compute_bundle(
            squad_df, market_df, player,
            transfer_budget, wage_capacity,
            model_pipe=model_pipe
        )

        st.write("### Executive Decision Overview")

        c1, c2, c3 = st.columns(3)
        c1.metric("Decision", bundle["decision"]["recommendation"])
        c2.metric("Decision Score", bundle["decision"]["decision_score"])
        c3.metric("Financial Risk Level", bundle["decision"]["risk_level"])

        st.write("#### Key Rationale")
        for r in bundle["decision"]["reasons"]:
            st.write(f"- {r}")

        st.divider()
        st.write("### Tactical, Risk & ML")

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Fit Score", f"{bundle['fit']:.2f}")
        k2.metric("Projected Uplift", f"{bundle['uplift']['uplift']:.3f}")
        k3.metric("Uncertainty (±)", f"{bundle['uplift']['uncertainty_sigma']:.3f}")
        k4.metric("P(Positive Uplift)", f"{(bundle['p_success'] if bundle['p_success'] is not None else 0.5):.3f}")
        k5.metric("Expected Uplift", f"{bundle['expected_uplift']:.3f}")

        st.write("#### Local explanation (why this probability?)")
        expl = explain_prediction(model_pipe, player, top_k=6)
        if len(expl) == 0:
            st.info("Local explanation not available (model not trained).")
        else:
            st.dataframe(
                expl[["feature", "contribution", "direction"]],
                use_container_width=True,
                hide_index=True
            )
            st.caption("Contributions are in the model's log-odds space (linear). Positive values increase P(success).")

        st.write("#### Projected Impact vs Squad Baseline")

        b1, b2, b3 = st.columns(3)
        b1.metric("Squad Baseline", round(bundle["uplift"]["baseline"], 3))
        b2.metric("Target Player", round(bundle["uplift"]["player_perf"], 3))
        b3.metric("Uplift Range", f"{bundle['uplift']['uplift_lower']:.3f} to {bundle['uplift']['uplift_upper']:.3f}")

        fig = go.Figure()
        fig.add_trace(go.Bar(x=["Squad Baseline"], y=[bundle["uplift"]["baseline"]], name="Baseline"))
        fig.add_trace(go.Bar(x=["Target Player"], y=[bundle["uplift"]["player_perf"]], name="Player"))
        fig.add_trace(go.Scatter(
            x=["Target Player", "Target Player"],
            y=[bundle["uplift"]["uplift_lower"] + bundle["uplift"]["baseline"],
               bundle["uplift"]["uplift_upper"] + bundle["uplift"]["baseline"]],
            mode="lines",
            line=dict(width=6),
            name="Uncertainty Range"
        ))
        fig.update_layout(
            title="Projected Performance Impact (vs Club Squad Baseline)",
            yaxis_title="Performance Index",
            template="plotly_dark",
            showlegend=True,
            height=420
        )
        st.plotly_chart(fig, use_container_width=True)

        st.write("### Squad Snapshot")
        st.caption("Current squad overview (synthetic, club-specific).")
        squad_summary = (
            squad_df.groupby("position")
            .agg(players=("name", "count"), avg_perf=("performance_index", "mean"))
            .reset_index()
        )
        st.dataframe(squad_summary, use_container_width=True)


# ---------- FINANCIAL TAB ----------
with tab_financial:
    if mode != "Single Player":
        st.info("Switch to 'Single Player' mode in the sidebar to view financial details for a single player.")
    else:
        bundle = compute_bundle(
            squad_df, market_df, player,
            transfer_budget, wage_capacity,
            model_pipe=model_pipe
        )

        st.write("### Financial Risk")
        st.metric("Risk Score", round(float(bundle["risk"]), 2))

        st.write("### Financial Inputs")
        st.write(f"Market Value: {round(float(player['market_value_m']), 2)} M€")
        st.write(f"Estimated Wage: {round(float(player['estimated_wage_m']), 2)} M€")
        st.write(f"Injury Risk: {round(float(player['injury_risk']), 2)}")

        st.write("### Efficiency")
        st.metric("Uplift per Market Value", f"{bundle['value_per_euro']:.4f}")
        st.metric("Expected Uplift (ML-adjusted)", f"{bundle['expected_uplift']:.3f}")


# ---------- COMPARISON TAB ----------
with tab_compare:
    st.write("### Comparison Mode")
    st.caption("Compare two targets under the same budget and wage constraints for the selected club context.")

    if mode != "A/B Comparison":
        st.info("Switch to 'A/B Comparison' mode in the sidebar to compare two players.")
    else:
        player_a = market_df[market_df["name"] == player_a_name].iloc[0]
        player_b = market_df[market_df["name"] == player_b_name].iloc[0]

        a = compute_bundle(squad_df, market_df, player_a, transfer_budget, wage_capacity, model_pipe=model_pipe)
        b = compute_bundle(squad_df, market_df, player_b, transfer_budget, wage_capacity, model_pipe=model_pipe)

        st.write("#### Executive Summary")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("A: Recommendation", a["decision"]["recommendation"])
        k2.metric("B: Recommendation", b["decision"]["recommendation"])
        k3.metric("A: Expected Uplift", f"{a['expected_uplift']:.3f}")
        k4.metric("B: Expected Uplift", f"{b['expected_uplift']:.3f}")

        st.divider()

        left, right = st.columns(2)

        with left:
            st.write(f"### Player A: {player_a_name}")
            st.metric("Fit Score", f"{a['fit']:.2f}")
            st.metric("Financial Risk", f"{a['risk']:.2f}")
            st.metric("P(Positive Uplift)", f"{(a['p_success'] if a['p_success'] is not None else 0.5):.3f}")
            st.metric("Expected Uplift", f"{a['expected_uplift']:.3f}")
            st.write("Key Rationale")
            for r in a["decision"]["reasons"]:
                st.write(f"- {r}")

            st.write("Local explanation (A)")
            expl_a = explain_prediction(model_pipe, player_a, top_k=4)
            if len(expl_a) > 0:
                st.dataframe(expl_a[["feature", "contribution", "direction"]], use_container_width=True, hide_index=True)

        with right:
            st.write(f"### Player B: {player_b_name}")
            st.metric("Fit Score", f"{b['fit']:.2f}")
            st.metric("Financial Risk", f"{b['risk']:.2f}")
            st.metric("P(Positive Uplift)", f"{(b['p_success'] if b['p_success'] is not None else 0.5):.3f}")
            st.metric("Expected Uplift", f"{b['expected_uplift']:.3f}")
            st.write("Key Rationale")
            for r in b["decision"]["reasons"]:
                st.write(f"- {r}")

            st.write("Local explanation (B)")
            expl_b = explain_prediction(model_pipe, player_b, top_k=4)
            if len(expl_b) > 0:
                st.dataframe(expl_b[["feature", "contribution", "direction"]], use_container_width=True, hide_index=True)

        st.divider()

        winner = "Tie"
        if a["risk_adjusted_value"] > b["risk_adjusted_value"]:
            winner = f"Player A ({player_a_name})"
        elif b["risk_adjusted_value"] > a["risk_adjusted_value"]:
            winner = f"Player B ({player_b_name})"

        st.write("#### Recommendation Under Constraints")
        st.write(
            f"Using a risk-adjusted value metric (expected uplift adjusted by financial risk), "
            f"the preferred option is: **{winner}**."
        )

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=["Player A", "Player B"],
            y=[a["expected_uplift"], b["expected_uplift"]],
            name="Expected Uplift"
        ))
        fig2.add_trace(go.Bar(
            x=["Player A", "Player B"],
            y=[-a["risk"] / 100, -b["risk"] / 100],
            name="Risk (scaled, negative)"
        ))
        fig2.update_layout(
            title="Expected Uplift vs Financial Risk (Scaled)",
            yaxis_title="Value Units",
            template="plotly_dark",
            barmode="group",
            height=420
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.divider()

        st.write("### Shortlist Generator")
        st.caption("Top targets for a specific position under the same constraints (club context).")

        pos = st.selectbox("Position", POSITION_BUCKETS, index=POSITION_BUCKETS.index(player_a["position"]))
        top_n = st.slider("Top N", 3, 10, 5)

        shortlist = shortlist_top_targets(
            squad_df=squad_df,
            market_df=market_df,
            position=pos,
            budget_m=transfer_budget,
            wage_capacity_m=wage_capacity,
            top_n=top_n,
            min_minutes=400,
            model_pipe=model_pipe
        )

        st.dataframe(shortlist, use_container_width=True)

        st.download_button(
            label="Download shortlist as CSV",
            data=shortlist.to_csv(index=False).encode("utf-8"),
            file_name=f"shortlist_{club}_{pos}.csv",
            mime="text/csv"
        )