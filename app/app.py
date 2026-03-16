# FILE: app/app.py  —  Assignment 2 (Transfer Intelligence Simulator v2)

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.data_loader import load_dataset
from src.feature_engineering import compute_performance_index
from src.shortlist import compute_bundle, shortlist_top_targets, train_club_model_cached
from src.ml_model import get_feature_importance, explain_prediction

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Transfer Intelligence Simulator",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS  — professional sports-analytics look
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Base & fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Background ── */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stRadio label {
        color: #8b949e !important;
        font-size: 12px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ── Hero header ── */
    .hero-container {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #1a2332 100%);
        border: 1px solid #21d07640;
        border-radius: 12px;
        padding: 28px 32px 20px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, #21d07615 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-title {
        font-size: 26px;
        font-weight: 700;
        color: #e6edf3;
        margin: 0 0 4px 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 14px;
        color: #8b949e;
        margin: 0 0 16px 0;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #21d07615;
        border: 1px solid #21d07640;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 12px;
        color: #21d076;
        font-weight: 500;
    }

    /* ── KPI cards ── */
    .kpi-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px 20px;
        position: relative;
    }
    .kpi-label {
        font-size: 11px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 6px;
        font-weight: 500;
    }
    .kpi-value {
        font-size: 26px;
        font-weight: 700;
        color: #e6edf3;
        line-height: 1;
    }
    .kpi-value.green  { color: #21d076; }
    .kpi-value.yellow { color: #f0c000; }
    .kpi-value.red    { color: #f85149; }
    .kpi-sub {
        font-size: 12px;
        color: #8b949e;
        margin-top: 4px;
    }

    /* ── Section headers ── */
    .section-header {
        font-size: 13px;
        font-weight: 600;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 24px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #21262d;
    }

    /* ── Decision badge ── */
    .decision-proceed {
        display: inline-block;
        background: #1f4a2e;
        border: 1px solid #2ea04380;
        color: #2ea043;
        border-radius: 6px;
        padding: 6px 14px;
        font-weight: 600;
        font-size: 14px;
    }
    .decision-monitor {
        display: inline-block;
        background: #3d2f00;
        border: 1px solid #f0c00080;
        color: #f0c000;
        border-radius: 6px;
        padding: 6px 14px;
        font-weight: 600;
        font-size: 14px;
    }
    .decision-avoid {
        display: inline-block;
        background: #3d0f0f;
        border: 1px solid #f8514980;
        color: #f85149;
        border-radius: 6px;
        padding: 6px 14px;
        font-weight: 600;
        font-size: 14px;
    }

    /* ── Agent chat ── */
    .agent-user-msg {
        background: #1c2840;
        border: 1px solid #264882;
        border-radius: 10px 10px 4px 10px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #cdd9e5;
        font-size: 14px;
        max-width: 80%;
        margin-left: auto;
    }
    .agent-ai-msg {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px 10px 10px 4px;
        padding: 14px 18px;
        margin: 8px 0;
        color: #e6edf3;
        font-size: 14px;
        max-width: 92%;
    }
    .agent-tool-call {
        background: #0d1117;
        border: 1px solid #21d07630;
        border-radius: 6px;
        padding: 8px 12px;
        margin: 6px 0;
        font-size: 12px;
        color: #21d076;
        font-family: 'Courier New', monospace;
    }

    /* ── Tables ── */
    .stDataFrame {
        border: 1px solid #30363d !important;
        border-radius: 8px;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #161b22;
        border-bottom: 1px solid #30363d;
        gap: 0;
    }
    .stTabs [data-baseweb="tab"] {
        color: #8b949e;
        font-size: 13px;
        font-weight: 500;
        padding: 10px 20px;
        border-bottom: 2px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        color: #e6edf3 !important;
        border-bottom: 2px solid #21d076 !important;
        background: transparent !important;
    }

    /* ── Expanders ── */
    .streamlit-expanderHeader {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        color: #8b949e !important;
        font-size: 13px;
    }

    /* ── Metrics override ── */
    [data-testid="metric-container"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px 16px;
    }
    [data-testid="metric-container"] label {
        color: #8b949e !important;
        font-size: 12px;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #e6edf3 !important;
        font-size: 22px;
        font-weight: 700;
    }

    /* ── Scrollable chat area ── */
    .chat-scroll {
        max-height: 520px;
        overflow-y: auto;
        padding-right: 8px;
    }

    /* ── Divider ── */
    hr {
        border-color: #21262d !important;
        margin: 20px 0 !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: #21d076;
        color: #0d1117;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-size: 13px;
        padding: 8px 18px;
        transition: opacity 0.2s;
    }
    .stButton > button:hover {
        opacity: 0.85;
        background: #21d076;
        color: #0d1117;
    }

    /* ── Input fields ── */
    .stTextInput > div > div > input {
        background: #21262d !important;
        border: 1px solid #30363d !important;
        color: #e6edf3 !important;
        border-radius: 8px !important;
        font-size: 14px;
    }
    .stTextInput > div > div > input:focus {
        border-color: #21d076 !important;
        box-shadow: 0 0 0 2px #21d07620 !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Data & constants
# ─────────────────────────────────────────────
BUNDESLIGA_CLUBS = [
    "FC Bayern München", "Borussia Dortmund", "RB Leipzig", "Bayer 04 Leverkusen",
    "Eintracht Frankfurt", "SC Freiburg", "VfB Stuttgart", "TSG Hoffenheim",
    "1. FSV Mainz 05", "Borussia Mönchengladbach", "VfL Wolfsburg", "Werder Bremen",
    "FC Augsburg", "1. FC Köln", "Union Berlin", "VfL Bochum",
    "1. FC Heidenheim", "SV Darmstadt 98"
]
POSITION_BUCKETS = ["GK", "CB", "FB", "DM", "CM", "AM", "ST"]

DECISION_COLOR = {"Proceed": "#21d076", "Monitor": "#f0c000", "Avoid": "#f85149"}
PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#8b949e", family="Inter"),
        xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
        yaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
        colorway=["#21d076", "#388bfd", "#f0c000", "#f85149", "#bc8cff"],
        margin=dict(t=40, l=16, r=16, b=40),
    )
)

market_df = load_dataset()
market_df = compute_performance_index(market_df)


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
        extra = rng.choice(remaining.index.to_numpy(),
                           size=min(squad_size - len(squad), len(remaining)), replace=False)
        squad = pd.concat([squad, remaining.loc[extra]], ignore_index=True)
    squad["club"] = club_name
    return squad.reset_index(drop=True)


def decision_badge(rec: str) -> str:
    cls = f"decision-{rec.lower()}"
    emoji = {"Proceed": "✅", "Monitor": "⚠️", "Avoid": "❌"}.get(rec, "")
    return f'<span class="{cls}">{emoji} {rec}</span>'


def radar_chart(player_row: pd.Series, squad_df: pd.DataFrame) -> go.Figure:
    pos = player_row["position"]
    baseline_sub = squad_df[squad_df["position"] == pos]

    categories = ["xG", "xA", "Prog. Passes", "Def. Actions", "Minutes\n(norm)"]
    player_vals = [
        float(player_row["xg"]),
        float(player_row["xa"]),
        float(player_row["progressive_passes"]),
        float(player_row["defensive_actions"]),
        float(player_row["minutes"]) / 3200,
    ]

    if len(baseline_sub) > 0:
        baseline_vals = [
            float(baseline_sub["xg"].mean()),
            float(baseline_sub["xa"].mean()),
            float(baseline_sub["progressive_passes"].mean()),
            float(baseline_sub["defensive_actions"].mean()),
            float(baseline_sub["minutes"].mean()) / 3200,
        ]
    else:
        baseline_vals = [0.2] * 5

    # Normalize to [0,1] scale using market max
    maxvals = [
        float(market_df["xg"].quantile(0.95)),
        float(market_df["xa"].quantile(0.95)),
        float(market_df["progressive_passes"].quantile(0.95)),
        float(market_df["defensive_actions"].quantile(0.95)),
        1.0,
    ]
    p_norm = [min(v / m, 1.0) for v, m in zip(player_vals, maxvals)]
    b_norm = [min(v / m, 1.0) for v, m in zip(baseline_vals, maxvals)]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=b_norm + [b_norm[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(56,139,253,0.15)",
        line=dict(color="#388bfd", width=2),
        name=f"Squad Baseline ({pos})"
    ))
    fig.add_trace(go.Scatterpolar(
        r=p_norm + [p_norm[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(33,208,118,0.15)",
        line=dict(color="#21d076", width=2.5),
        name=player_row["name"]
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#161b22",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#30363d", tickfont=dict(color="#8b949e")),
            angularaxis=dict(gridcolor="#30363d", tickfont=dict(color="#8b949e")),
        ),
        paper_bgcolor="#0d1117",
        showlegend=True,
        legend=dict(font=dict(color="#8b949e"), bgcolor="#161b22", bordercolor="#30363d"),
        height=360,
        margin=dict(t=20, l=20, r=20, b=20),
    )
    return fig


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚽ Scenario Setup")
    st.markdown("---")

    club = st.selectbox("Club", BUNDESLIGA_CLUBS)
    mode = st.radio("Mode", ["Single Player", "A/B Comparison"], index=0)

    st.markdown("---")
    st.markdown("**Budget & Wages**")
    transfer_budget = st.slider("Transfer Budget (M€)", 5, 80, 30)
    wage_capacity   = st.slider("Wage Capacity (M€/yr)", 1, 30, 8)

    st.markdown("---")
    squad_df = generate_squad_for_club(club, market_df)
    st.markdown(f"**Active Context:** {club}")
    st.caption(f"Squad: {len(squad_df)} players · Baselines computed per position")

    if mode == "Single Player":
        st.markdown("---")
        player_name = st.selectbox("Target Player", market_df["name"].tolist())
        player = market_df[market_df["name"] == player_name].iloc[0]
    else:
        st.markdown("---")
        player_a_name = st.selectbox("Player A", market_df["name"].tolist(), index=0)
        player_b_name = st.selectbox("Player B", market_df["name"].tolist(), index=1)
        if player_a_name == player_b_name:
            st.warning("Select two different players.")


# ─────────────────────────────────────────────
# ML model (cached per club)
# ─────────────────────────────────────────────
@st.cache_resource
def _train_model(club_name: str):
    sq = generate_squad_for_club(club_name, market_df)
    return train_club_model_cached(sq, market_df, random_state=42)

model_pipe, model_metrics = _train_model(club)


# ─────────────────────────────────────────────
# Hero header
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="hero-container">
    <div class="hero-title">Transfer Intelligence Simulator</div>
    <div class="hero-subtitle">Bundesliga Edition — Decision Support for Budget-Constrained Recruitment</div>
    <span class="hero-badge">⚡ Active: {club}</span>
    &nbsp;
    <span class="hero-badge" style="background:#388bfd15;border-color:#388bfd40;color:#388bfd;">
        🤖 LLM Agent Ready
    </span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab_single, tab_compare, tab_shortlist, tab_agent, tab_model = st.tabs([
    "🔍 Single Player",
    "⚖️ A/B Comparison",
    "📋 Shortlist",
    "🤖 Scouting Agent",
    "📊 Model Card"
])


# ══════════════════════════════════════════════
# TAB 1 — Single Player
# ══════════════════════════════════════════════
with tab_single:
    if mode != "Single Player":
        st.info("Switch to 'Single Player' mode in the sidebar.")
    else:
        bundle = compute_bundle(squad_df, market_df, player,
                                transfer_budget, wage_capacity, model_pipe=model_pipe)
        rec = bundle["decision"]["recommendation"]

        # ── Decision banner ──
        col_rec, col_score, col_risk = st.columns([2, 1, 1])
        with col_rec:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Recommendation</div>
                {decision_badge(rec)}
                <div class="kpi-sub" style="margin-top:8px;">
                    {" · ".join(bundle["decision"]["reasons"])}
                </div>
            </div>""", unsafe_allow_html=True)
        with col_score:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Decision Score</div>
                <div class="kpi-value">{bundle["decision"]["decision_score"]}</div>
                <div class="kpi-sub">out of 100</div>
            </div>""", unsafe_allow_html=True)
        with col_risk:
            risk_cls = "green" if bundle["risk"] < 35 else ("yellow" if bundle["risk"] < 65 else "red")
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Financial Risk</div>
                <div class="kpi-value {risk_cls}">{bundle["decision"]["risk_level"]}</div>
                <div class="kpi-sub">Score: {round(bundle['risk'], 1)}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header">Performance & ML</div>', unsafe_allow_html=True)

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Fit Score",          f"{bundle['fit']:.1f}")
        k2.metric("Projected Uplift",   f"{bundle['uplift']['uplift']:+.3f}")
        k3.metric("Uncertainty ±",      f"{bundle['uplift']['uncertainty_sigma']:.3f}")
        k4.metric("P(Positive Uplift)", f"{(bundle['p_success'] or 0.5):.3f}")
        k5.metric("Expected Uplift",    f"{bundle['expected_uplift']:+.3f}")

        st.markdown('<div class="section-header">Radar Profile</div>', unsafe_allow_html=True)

        col_radar, col_expl = st.columns([1, 1])
        with col_radar:
            st.plotly_chart(radar_chart(player, squad_df), use_container_width=True)

        with col_expl:
            st.markdown("**Local Explanation — Why this probability?**")
            expl = explain_prediction(model_pipe, player, top_k=5)
            if len(expl) > 0:
                fig_expl = go.Figure(go.Bar(
                    x=expl["contribution"],
                    y=expl["feature"],
                    orientation="h",
                    marker_color=["#21d076" if c > 0 else "#f85149" for c in expl["contribution"]],
                ))
                fig_expl.update_layout(
                    **PLOTLY_TEMPLATE,
                    height=300,
                    xaxis_title="Log-odds contribution",
                )
                st.plotly_chart(fig_expl, use_container_width=True)
                st.caption("Green = increases P(success) · Red = decreases")

        st.markdown('<div class="section-header">Projected Impact vs Squad Baseline</div>',
                    unsafe_allow_html=True)

        b1, b2, b3 = st.columns(3)
        b1.metric("Squad Baseline",  f"{bundle['uplift']['baseline']:.3f}")
        b2.metric("Player Perf.",    f"{bundle['uplift']['player_perf']:.3f}")
        b3.metric("Uplift Range",    f"{bundle['uplift']['uplift_lower']:+.3f} → {bundle['uplift']['uplift_upper']:+.3f}")

        fig_uplift = go.Figure()
        fig_uplift.add_trace(go.Bar(
            x=["Squad Baseline", player_name],
            y=[bundle["uplift"]["baseline"], bundle["uplift"]["player_perf"]],
            marker_color=["#388bfd", "#21d076"],
            name="Performance Index"
        ))
        # Uncertainty band as error bar on player
        fig_uplift.add_trace(go.Scatter(
            x=[player_name, player_name],
            y=[bundle["uplift"]["player_perf"] - bundle["uplift"]["uncertainty_sigma"],
               bundle["uplift"]["player_perf"] + bundle["uplift"]["uncertainty_sigma"]],
            mode="lines",
            line=dict(width=6, color="#f0c000"),
            name="Uncertainty Band"
        ))
        fig_uplift.update_layout(**PLOTLY_TEMPLATE, height=320,
                                  yaxis_title="Performance Index")
        st.plotly_chart(fig_uplift, use_container_width=True)

        with st.expander("Squad Snapshot"):
            summary = (squad_df.groupby("position")
                       .agg(players=("name","count"), avg_perf=("performance_index","mean"))
                       .reset_index())
            st.dataframe(summary, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# TAB 2 — A/B Comparison
# ══════════════════════════════════════════════
with tab_compare:
    if mode != "A/B Comparison":
        st.info("Switch to 'A/B Comparison' mode in the sidebar.")
    else:
        pa = market_df[market_df["name"] == player_a_name].iloc[0]
        pb = market_df[market_df["name"] == player_b_name].iloc[0]
        a  = compute_bundle(squad_df, market_df, pa, transfer_budget, wage_capacity, model_pipe=model_pipe)
        b  = compute_bundle(squad_df, market_df, pb, transfer_budget, wage_capacity, model_pipe=model_pipe)

        winner = "Tie"
        if   a["risk_adjusted_value"] > b["risk_adjusted_value"]: winner = f"Player A — {player_a_name}"
        elif b["risk_adjusted_value"] > a["risk_adjusted_value"]: winner = f"Player B — {player_b_name}"

        st.markdown(f"""
        <div class="kpi-card" style="margin-bottom:20px;">
            <div class="kpi-label">Risk-Adjusted Recommendation</div>
            <div class="kpi-value" style="font-size:20px; margin-top:4px;">🏆 {winner}</div>
            <div class="kpi-sub">Based on expected uplift adjusted for financial risk</div>
        </div>""", unsafe_allow_html=True)

        left, right = st.columns(2)

        for col, name, bun, prow in [
            (left,  player_a_name, a, pa),
            (right, player_b_name, b, pb)
        ]:
            with col:
                st.markdown(f"### {name}")
                st.markdown(decision_badge(bun["decision"]["recommendation"]), unsafe_allow_html=True)
                st.markdown("")

                m1, m2 = st.columns(2)
                m1.metric("Fit Score",         f"{bun['fit']:.1f}")
                m2.metric("Financial Risk",     f"{bun['risk']:.1f}")
                m1.metric("P(Success)",         f"{(bun['p_success'] or 0.5):.3f}")
                m2.metric("Expected Uplift",    f"{bun['expected_uplift']:+.3f}")

                st.plotly_chart(radar_chart(prow, squad_df), use_container_width=True)

                with st.expander("Rationale"):
                    for r in bun["decision"]["reasons"]:
                        st.write(f"• {r}")

        # ── Side-by-side bar chart ──
        st.markdown('<div class="section-header">Head-to-Head Metrics</div>', unsafe_allow_html=True)

        metrics_keys  = ["fit", "risk", "p_success", "expected_uplift"]
        metrics_labels = ["Fit Score (÷100)", "Financial Risk (÷100)", "P(Success)", "Expected Uplift"]
        a_vals = [a["fit"]/100, a["risk"]/100, (a["p_success"] or 0.5), a["expected_uplift"]]
        b_vals = [b["fit"]/100, b["risk"]/100, (b["p_success"] or 0.5), b["expected_uplift"]]

        fig_ab = go.Figure()
        fig_ab.add_trace(go.Bar(name=player_a_name, x=metrics_labels, y=a_vals, marker_color="#388bfd"))
        fig_ab.add_trace(go.Bar(name=player_b_name, x=metrics_labels, y=b_vals, marker_color="#21d076"))
        fig_ab.update_layout(**PLOTLY_TEMPLATE, barmode="group", height=340,
                              yaxis_title="Normalised Value")
        st.plotly_chart(fig_ab, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 3 — Shortlist
# ══════════════════════════════════════════════
with tab_shortlist:
    st.markdown('<div class="section-header">Shortlist Generator</div>', unsafe_allow_html=True)

    sc1, sc2, sc3 = st.columns([1, 1, 2])
    with sc1:
        pos_sel = st.selectbox("Position", POSITION_BUCKETS)
    with sc2:
        top_n = st.slider("Top N", 3, 12, 5)
    with sc3:
        st.markdown("&nbsp;")
        run_btn = st.button("▶ Generate Shortlist")

    if run_btn or "shortlist_df" in st.session_state:
        if run_btn:
            with st.spinner("Computing shortlist..."):
                sl = shortlist_top_targets(
                    squad_df=squad_df, market_df=market_df,
                    position=pos_sel, budget_m=transfer_budget,
                    wage_capacity_m=wage_capacity, top_n=top_n,
                    min_minutes=400, model_pipe=model_pipe
                )
            st.session_state["shortlist_df"] = sl
            st.session_state["shortlist_pos"] = pos_sel

        sl = st.session_state.get("shortlist_df", pd.DataFrame())

        if len(sl) == 0:
            st.warning("No feasible candidates found under current budget/wage constraints. Try relaxing filters.")
        else:
            # ── Bubble chart: uplift vs risk, sized by market value ──
            fig_sl = px.scatter(
                sl,
                x="financial_risk", y="expected_uplift",
                size="market_value_m", color="recommendation",
                text="name",
                color_discrete_map=DECISION_COLOR,
                size_max=40,
                labels={"financial_risk": "Financial Risk", "expected_uplift": "Expected Uplift"},
                title=f"{pos_sel} candidates — Risk vs Uplift (bubble = market value)"
            )
            fig_sl.update_traces(textposition="top center",
                                 textfont=dict(color="#8b949e", size=11))
            fig_sl.update_layout(
                **PLOTLY_TEMPLATE,
                height=400,
                legend=dict(bgcolor="#161b22", bordercolor="#30363d")
            )
            st.plotly_chart(fig_sl, use_container_width=True)

            # ── Table ──
            display_cols = ["name", "age", "market_value_m", "estimated_wage_m",
                            "fit_score", "expected_uplift", "p_success",
                            "financial_risk", "recommendation"]
            st.dataframe(sl[display_cols].style.format({
                "market_value_m":    "{:.1f}M",
                "estimated_wage_m":  "{:.1f}M",
                "fit_score":         "{:.1f}",
                "expected_uplift":   "{:+.3f}",
                "p_success":         "{:.3f}",
                "financial_risk":    "{:.1f}",
            }), use_container_width=True, hide_index=True)

            st.download_button(
                "⬇️ Download Shortlist CSV",
                data=sl.to_csv(index=False).encode("utf-8"),
                file_name=f"shortlist_{club.replace(' ','_')}_{pos_sel}.csv",
                mime="text/csv"
            )


# ══════════════════════════════════════════════
# TAB 4 — Scouting Agent  (LLM + Tool Use)
# ══════════════════════════════════════════════
with tab_agent:
    from src.scouting_agent import run_scouting_agent

    st.markdown("""
    <div class="kpi-card" style="margin-bottom:20px;">
        <div class="kpi-label">AI Scouting Agent</div>
        <div style="color:#e6edf3; font-size:15px; font-weight:500; margin:6px 0 4px 0;">
            Ask anything about transfers — in natural language
        </div>
        <div class="kpi-sub">
            The agent uses <strong>Cohere Command R+</strong> with tool use.
            It autonomously calls the simulation engine, shortlist generator, and
            comparison tools — then synthesises a structured scouting brief.
        </div>
    </div>""", unsafe_allow_html=True)

    # Example prompts
    st.markdown('<div class="section-header">Example Queries</div>', unsafe_allow_html=True)
    examples = [
        f"Which striker under 20M€ fits {club} best right now?",
        f"Compare the top 2 CM candidates for {club} and explain the risk trade-off.",
        f"Give me a scouting brief for the best value AM under 15M€ for {club}.",
        "Who should I prioritize: a cheap young DM or an experienced CB?",
    ]
    eq_cols = st.columns(2)
    for i, ex in enumerate(examples):
        with eq_cols[i % 2]:
            if st.button(ex, key=f"ex_{i}"):
                st.session_state["agent_input"] = ex

    st.markdown("---")

    # Chat history
    if "agent_history" not in st.session_state:
        st.session_state["agent_history"] = []

    # Display history
    if st.session_state["agent_history"]:
        st.markdown('<div class="section-header">Conversation</div>', unsafe_allow_html=True)
        for msg in st.session_state["agent_history"]:
            if msg["role"] == "user":
                st.markdown(f'<div class="agent-user-msg">💬 {msg["content"]}</div>',
                            unsafe_allow_html=True)
            elif msg["role"] == "tool_call":
                st.markdown(f'<div class="agent-tool-call">🔧 Tool: {msg["content"]}</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="agent-ai-msg">🤖 {msg["content"]}</div>',
                            unsafe_allow_html=True)

    # Input
    user_query = st.text_input(
        "Ask the agent",
        value=st.session_state.pop("agent_input", ""),
        placeholder="e.g. Which CB under 25M€ would improve Bayern's squad the most?",
        label_visibility="collapsed",
    )

    col_send, col_clear = st.columns([1, 5])
    with col_send:
        send = st.button("Send ➤")
    with col_clear:
        if st.button("Clear Chat"):
            st.session_state["agent_history"] = []
            st.rerun()

    if send and user_query.strip():
        st.session_state["agent_history"].append({"role": "user", "content": user_query})

        with st.spinner("Agent is thinking — calling tools…"):
            try:
                result = run_scouting_agent(
                    query=user_query,
                    club=club,
                    squad_df=squad_df,
                    market_df=market_df,
                    model_pipe=model_pipe,
                    transfer_budget=transfer_budget,
                    wage_capacity=wage_capacity,
                    history=st.session_state["agent_history"][:-1]
                )
                # Log tool calls
                for tc in result.get("tool_calls", []):
                    st.session_state["agent_history"].append(
                        {"role": "tool_call",
                         "content": f'{tc["tool"]}({", ".join(f"{k}={v}" for k, v in tc["args"].items())})'}
                    )
                st.session_state["agent_history"].append(
                    {"role": "assistant", "content": result["answer"]}
                )
            except Exception as e:
                st.session_state["agent_history"].append(
                    {"role": "assistant",
                     "content": f"⚠️ Agent error: {str(e)}\n\nMake sure your COHERE_API_KEY is set in `.env`."}
                )
        st.rerun()


# ══════════════════════════════════════════════
# TAB 5 — Model Card
# ══════════════════════════════════════════════
with tab_model:
    st.markdown('<div class="section-header">Model Card</div>', unsafe_allow_html=True)

    st.markdown("""
    **What the model predicts**

    The ML layer estimates **P(Positive Uplift)** — the probability that a candidate's
    projected performance index exceeds the selected club's position-specific baseline.
    The label is *club-context specific*: the same player may have different probabilities
    for different clubs.

    **Model type:** Logistic Regression (interpretable by design).
    Features: age, minutes, xG, xA, progressive passes, defensive actions, injury risk,
    market value, wage, position (one-hot).
    """)

    m1, m2, m3 = st.columns(3)
    m1.metric("Training N",  model_metrics.get("n", "—"))
    m2.metric("AUC",         model_metrics.get("auc", "—"))
    m3.metric("Accuracy",    model_metrics.get("accuracy", "—"))
    if model_metrics.get("note"):
        st.caption(model_metrics["note"])

    st.markdown('<div class="section-header">Global Feature Importance</div>', unsafe_allow_html=True)

    imp_df = get_feature_importance(model_pipe, top_k=15)
    if len(imp_df):
        fig_imp = go.Figure(go.Bar(
            x=imp_df["abs_coefficient"],
            y=imp_df["feature"],
            orientation="h",
            marker_color="#388bfd",
        ))
        fig_imp.update_layout(**PLOTLY_TEMPLATE, height=480,
                               xaxis_title="|Coefficient|",
                               yaxis_title="Feature")
        st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown('<div class="section-header">Limitations & Scope</div>', unsafe_allow_html=True)
    st.markdown("""
    - Trained on **synthetic** data; not validated against real transfer outcomes.
    - Uplift label derived from the projection function, not observed post-transfer data.
    - Coefficients reflect correlations within the synthetic world — not causal claims.
    - Intended for **decision prototyping**, not production scouting systems.
    """)
