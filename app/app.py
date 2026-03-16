# FILE: app/app.py  —  Transfer Intelligence Simulator v2

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

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Transfer Intelligence Simulator",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Apple-inspired CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    -webkit-font-smoothing: antialiased;
}

/* Background */
.stApp { background: #f5f5f7; color: #1d1d1f; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #d2d2d7;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stRadio label {
    color: #86868b !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] .stMarkdown {
    color: #1d1d1f;
    font-size: 13px;
}

/* Hero */
.hero {
    background: #1d1d1f;
    border-radius: 18px;
    padding: 36px 40px 28px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(0,113,227,0.18) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    font-size: 12px;
    font-weight: 600;
    color: #0071e3;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
}
.hero-title {
    font-size: 32px;
    font-weight: 700;
    color: #f5f5f7;
    letter-spacing: -0.5px;
    line-height: 1.1;
    margin-bottom: 6px;
}
.hero-sub {
    font-size: 15px;
    color: #86868b;
    margin-bottom: 20px;
}
.hero-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 12px;
    color: #f5f5f7;
    font-weight: 500;
    margin-right: 8px;
}
.hero-pill.blue {
    background: rgba(0,113,227,0.2);
    border-color: rgba(0,113,227,0.35);
    color: #60a5fa;
}

/* Cards */
.card {
    background: #ffffff;
    border: 1px solid #d2d2d7;
    border-radius: 14px;
    padding: 20px 24px;
}
.card-label {
    font-size: 11px;
    font-weight: 600;
    color: #86868b;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 6px;
}
.card-value {
    font-size: 28px;
    font-weight: 700;
    color: #1d1d1f;
    line-height: 1;
}
.card-value.green  { color: #30d158; }
.card-value.yellow { color: #ff9f0a; }
.card-value.red    { color: #ff3b30; }
.card-sub {
    font-size: 12px;
    color: #86868b;
    margin-top: 5px;
}

/* Decision chips */
.chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    border-radius: 8px;
    padding: 6px 14px;
    font-size: 13px;
    font-weight: 600;
}
.chip.proceed { background: #d1fae5; color: #065f46; border: 1px solid #6ee7b7; }
.chip.monitor { background: #fef3c7; color: #92400e; border: 1px solid #fcd34d; }
.chip.avoid   { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }

/* Section headers */
.sh {
    font-size: 11px;
    font-weight: 600;
    color: #86868b;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 28px 0 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid #d2d2d7;
}

/* Agent chat */
.msg-user {
    background: #0071e3;
    border-radius: 14px 14px 4px 14px;
    padding: 12px 16px;
    margin: 8px 0 8px auto;
    max-width: 78%;
    color: #ffffff;
    font-size: 14px;
    line-height: 1.5;
}
.msg-ai {
    background: #ffffff;
    border: 1px solid #d2d2d7;
    border-radius: 14px 14px 14px 4px;
    padding: 14px 18px;
    margin: 8px 0;
    max-width: 90%;
    color: #1d1d1f;
    font-size: 14px;
    line-height: 1.6;
}
.msg-tool {
    background: #f5f5f7;
    border: 1px solid #d2d2d7;
    border-radius: 8px;
    padding: 7px 12px;
    margin: 4px 0;
    font-size: 12px;
    color: #0071e3;
    font-family: 'SF Mono', 'Fira Code', monospace;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #ffffff;
    border-bottom: 1px solid #d2d2d7;
    border-radius: 12px 12px 0 0;
    gap: 0;
    padding: 0 8px;
}
.stTabs [data-baseweb="tab"] {
    color: #86868b;
    font-size: 13px;
    font-weight: 500;
    padding: 12px 18px;
    border-bottom: 2px solid transparent;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    color: #1d1d1f !important;
    border-bottom: 2px solid #0071e3 !important;
    background: transparent !important;
}

/* Metrics */
[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #d2d2d7;
    border-radius: 12px;
    padding: 14px 18px;
}
[data-testid="metric-container"] label { color: #86868b !important; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #1d1d1f !important; font-weight: 700; }

/* Buttons */
.stButton > button {
    background: #0071e3;
    color: #ffffff;
    border: none;
    border-radius: 980px;
    font-size: 13px;
    font-weight: 500;
    padding: 8px 20px;
    transition: all 0.2s;
}
.stButton > button:hover { background: #0077ed; color: #ffffff; }

/* Text input */
.stTextInput > div > div > input {
    background: #ffffff !important;
    border: 1px solid #d2d2d7 !important;
    border-radius: 10px !important;
    color: #1d1d1f !important;
    font-size: 14px;
    padding: 10px 14px;
}
.stTextInput > div > div > input:focus {
    border-color: #0071e3 !important;
    box-shadow: 0 0 0 3px rgba(0,113,227,0.15) !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: #ffffff !important;
    border: 1px solid #d2d2d7 !important;
    border-radius: 10px !important;
    color: #1d1d1f !important;
    font-size: 13px;
    font-weight: 500;
}

/* Dataframe */
.stDataFrame { border-radius: 10px; overflow: hidden; border: 1px solid #d2d2d7; }

hr { border-color: #d2d2d7 !important; margin: 20px 0 !important; }

/* Download button */
.stDownloadButton > button {
    background: #f5f5f7;
    color: #0071e3;
    border: 1px solid #d2d2d7;
    border-radius: 980px;
    font-size: 13px;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
BUNDESLIGA_CLUBS = [
    "FC Bayern München","Borussia Dortmund","RB Leipzig","Bayer 04 Leverkusen",
    "Eintracht Frankfurt","SC Freiburg","VfB Stuttgart","TSG Hoffenheim",
    "1. FSV Mainz 05","Borussia Mönchengladbach","VfL Wolfsburg","Werder Bremen",
    "FC Augsburg","1. FC Köln","Union Berlin","VfL Bochum","1. FC Heidenheim","SV Darmstadt 98",
]
POSITIONS = ["GK","CB","FB","DM","CM","AM","ST"]
DECISION_COLORS = {"Proceed":"#30d158","Monitor":"#ff9f0a","Avoid":"#ff3b30"}

# ── Plotly layout helper (fixes the **dict bug) ───────────────────────────────
def pl(height=340, barmode=None, xaxis_title=None, yaxis_title=None,
       showlegend=True, extra=None):
    """Return a dict of kwargs for fig.update_layout() — never nested."""
    kw = dict(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="Inter, -apple-system, sans-serif", color="#86868b", size=12),
        height=height,
        margin=dict(t=32, l=8, r=8, b=40),
        showlegend=showlegend,
        colorway=["#0071e3","#30d158","#ff9f0a","#ff3b30","#ac39ff"],
        xaxis=dict(gridcolor="#f0f0f5", linecolor="#d2d2d7",
                   title=xaxis_title or "", tickfont=dict(color="#86868b")),
        yaxis=dict(gridcolor="#f0f0f5", linecolor="#d2d2d7",
                   title=yaxis_title or "", tickfont=dict(color="#86868b")),
    )
    if barmode: kw["barmode"] = barmode
    if extra:   kw.update(extra)
    return kw


# ── Data ─────────────────────────────────────────────────────────────────────
market_df = load_dataset()
market_df = compute_performance_index(market_df)

def generate_squad(club, df_market, squad_size=23):
    seed = abs(hash(club)) % (2**32-1)
    rng  = np.random.default_rng(seed)
    comp = {"GK":2,"CB":4,"FB":4,"DM":3,"CM":4,"AM":3,"ST":3}
    idx  = []
    for pos, n in comp.items():
        pool = df_market[df_market["position"]==pos]
        if len(pool)==0: continue
        idx.extend(rng.choice(pool.index.to_numpy(), size=min(n,len(pool)), replace=False).tolist())
    squad = df_market.loc[idx].copy()
    if len(squad)<squad_size:
        rem   = df_market.drop(index=idx)
        extra = rng.choice(rem.index.to_numpy(), size=min(squad_size-len(squad),len(rem)), replace=False)
        squad = pd.concat([squad, rem.loc[extra]], ignore_index=True)
    squad["club"] = club
    return squad.reset_index(drop=True)

def chip(rec):
    cls = rec.lower()
    icon = {"Proceed":"●","Monitor":"●","Avoid":"●"}.get(rec,"●")
    return f'<span class="chip {cls}">{icon} {rec}</span>'

def radar(player_row, squad_df):
    pos  = player_row["position"]
    base = squad_df[squad_df["position"]==pos]
    cats = ["xG","xA","Progressive\nPasses","Defensive\nActions","Minutes"]
    pv   = [float(player_row["xg"]), float(player_row["xa"]),
            float(player_row["progressive_passes"]), float(player_row["defensive_actions"]),
            float(player_row["minutes"])/3200]
    bv   = ([float(base["xg"].mean()), float(base["xa"].mean()),
              float(base["progressive_passes"].mean()), float(base["defensive_actions"].mean()),
              float(base["minutes"].mean())/3200] if len(base)>0 else [0.2]*5)
    mx   = [float(market_df["xg"].quantile(0.95)), float(market_df["xa"].quantile(0.95)),
            float(market_df["progressive_passes"].quantile(0.95)),
            float(market_df["defensive_actions"].quantile(0.95)), 1.0]
    pn   = [min(v/m,1.0) for v,m in zip(pv,mx)]
    bn   = [min(v/m,1.0) for v,m in zip(bv,mx)]
    fig  = go.Figure()
    fig.add_trace(go.Scatterpolar(r=bn+[bn[0]], theta=cats+[cats[0]], fill="toself",
        fillcolor="rgba(0,113,227,0.1)", line=dict(color="#0071e3",width=2),
        name=f"Squad Baseline ({pos})"))
    fig.add_trace(go.Scatterpolar(r=pn+[pn[0]], theta=cats+[cats[0]], fill="toself",
        fillcolor="rgba(48,209,88,0.1)", line=dict(color="#30d158",width=2.5),
        name=player_row["name"]))
    fig.update_layout(
        polar=dict(bgcolor="#ffffff",
            radialaxis=dict(visible=True,range=[0,1],gridcolor="#d2d2d7",
                tickfont=dict(color="#86868b",size=10)),
            angularaxis=dict(gridcolor="#d2d2d7",tickfont=dict(color="#86868b",size=11))),
        paper_bgcolor="#ffffff", showlegend=True,
        legend=dict(font=dict(color="#86868b",size=11),bgcolor="#ffffff",bordercolor="#d2d2d7"),
        height=340, margin=dict(t=16,l=16,r=16,b=16))
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Scenario Setup")
    st.markdown("---")
    club = st.selectbox("Club", BUNDESLIGA_CLUBS)
    mode = st.radio("Mode", ["Single Player","A/B Comparison"])
    st.markdown("---")
    transfer_budget = st.slider("Transfer Budget (M€)", 5, 80, 30)
    wage_capacity   = st.slider("Wage Capacity (M€/yr)", 1, 30, 8)
    squad_df = generate_squad(club, market_df)
    st.markdown("---")
    st.markdown(f"**{club}**")
    st.caption(f"{len(squad_df)} players · baselines per position")
    if mode=="Single Player":
        st.markdown("---")
        player_name = st.selectbox("Target Player", market_df["name"].tolist())
        player = market_df[market_df["name"]==player_name].iloc[0]
    else:
        st.markdown("---")
        pA_name = st.selectbox("Player A", market_df["name"].tolist(), index=0)
        pB_name = st.selectbox("Player B", market_df["name"].tolist(), index=1)

# ── ML model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def _train(club_name):
    sq = generate_squad(club_name, market_df)
    return train_club_model_cached(sq, market_df, random_state=42)

model_pipe, model_metrics = _train(club)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
    <div class="hero-eyebrow">Bundesliga Edition</div>
    <div class="hero-title">Transfer Intelligence Simulator</div>
    <div class="hero-sub">Decision support for budget-constrained recruitment</div>
    <span class="hero-pill">{club}</span>
    <span class="hero-pill blue">AI Scouting Agent</span>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5 = st.tabs([
    "Single Player", "A/B Comparison", "Shortlist", "Scouting Agent", "Model Card"
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Player
# ═════════════════════════════════════════════════════════════════════════════
with t1:
    if mode != "Single Player":
        st.info("Switch to 'Single Player' mode in the sidebar.")
    else:
        b = compute_bundle(squad_df, market_df, player, transfer_budget, wage_capacity, model_pipe=model_pipe)
        rec  = b["decision"]["recommendation"]
        risk_cls = "green" if b["risk"]<35 else ("yellow" if b["risk"]<65 else "red")

        # Decision row
        c1, c2, c3 = st.columns([3,1,1])
        with c1:
            st.markdown(f"""
            <div class="card">
                <div class="card-label">Recommendation</div>
                {chip(rec)}
                <div class="card-sub" style="margin-top:10px;">
                    {" &nbsp;·&nbsp; ".join(b["decision"]["reasons"])}
                </div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="card">
                <div class="card-label">Decision Score</div>
                <div class="card-value">{b["decision"]["decision_score"]}</div>
                <div class="card-sub">out of 100</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="card">
                <div class="card-label">Financial Risk</div>
                <div class="card-value {risk_cls}">{b["decision"]["risk_level"]}</div>
                <div class="card-sub">Score: {round(b['risk'],1)}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sh">Performance & ML Layer</div>', unsafe_allow_html=True)
        k1,k2,k3,k4,k5 = st.columns(5)
        k1.metric("Fit Score",          f"{b['fit']:.1f}")
        k2.metric("Projected Uplift",   f"{b['uplift']['uplift']:+.3f}")
        k3.metric("Uncertainty",        f"± {b['uplift']['uncertainty_sigma']:.3f}")
        k4.metric("P(Positive Uplift)", f"{(b['p_success'] or 0.5):.3f}")
        k5.metric("Expected Uplift",    f"{b['expected_uplift']:+.3f}")

        st.markdown('<div class="sh">Radar Profile vs Squad Baseline</div>', unsafe_allow_html=True)
        col_r, col_e = st.columns([1,1])
        with col_r:
            st.plotly_chart(radar(player, squad_df), use_container_width=True)
        with col_e:
            st.markdown("**Feature Contributions to P(Success)**")
            expl = explain_prediction(model_pipe, player, top_k=5)
            if len(expl):
                fig_e = go.Figure(go.Bar(
                    x=expl["contribution"], y=expl["feature"], orientation="h",
                    marker_color=["#30d158" if c>0 else "#ff3b30" for c in expl["contribution"]],
                    marker_line_width=0,
                ))
                fig_e.update_layout(**pl(height=300, xaxis_title="Log-odds contribution"))
                st.plotly_chart(fig_e, use_container_width=True)
                st.caption("Green increases · Red decreases P(success)")

        st.markdown('<div class="sh">Projected Impact vs Squad Baseline</div>', unsafe_allow_html=True)
        b1,b2,b3 = st.columns(3)
        b1.metric("Squad Baseline", f"{b['uplift']['baseline']:.3f}")
        b2.metric("Player Index",   f"{b['uplift']['player_perf']:.3f}")
        b3.metric("Uplift Range",   f"{b['uplift']['uplift_lower']:+.3f} → {b['uplift']['uplift_upper']:+.3f}")

        fig_u = go.Figure()
        fig_u.add_trace(go.Bar(x=["Squad Baseline", player_name],
            y=[b["uplift"]["baseline"], b["uplift"]["player_perf"]],
            marker_color=["#0071e3","#30d158"], marker_line_width=0))
        fig_u.add_trace(go.Scatter(
            x=[player_name, player_name],
            y=[b["uplift"]["player_perf"]-b["uplift"]["uncertainty_sigma"],
               b["uplift"]["player_perf"]+b["uplift"]["uncertainty_sigma"]],
            mode="lines", line=dict(width=5, color="#ff9f0a"), name="Uncertainty Band"))
        fig_u.update_layout(**pl(height=300, yaxis_title="Performance Index"))
        st.plotly_chart(fig_u, use_container_width=True)

        with st.expander("Squad Snapshot"):
            s = squad_df.groupby("position").agg(
                players=("name","count"), avg_perf=("performance_index","mean")).reset_index()
            st.dataframe(s, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — A/B Comparison
# ═════════════════════════════════════════════════════════════════════════════
with t2:
    if mode != "A/B Comparison":
        st.info("Switch to 'A/B Comparison' mode in the sidebar.")
    else:
        pa = market_df[market_df["name"]==pA_name].iloc[0]
        pb = market_df[market_df["name"]==pB_name].iloc[0]
        ba = compute_bundle(squad_df, market_df, pa, transfer_budget, wage_capacity, model_pipe=model_pipe)
        bb = compute_bundle(squad_df, market_df, pb, transfer_budget, wage_capacity, model_pipe=model_pipe)

        if   ba["risk_adjusted_value"] > bb["risk_adjusted_value"]: winner = f"Player A — {pA_name}"
        elif bb["risk_adjusted_value"] > ba["risk_adjusted_value"]: winner = f"Player B — {pB_name}"
        else: winner = "Tie"

        st.markdown(f"""
        <div class="card" style="margin-bottom:20px;">
            <div class="card-label">Risk-Adjusted Verdict</div>
            <div style="font-size:20px;font-weight:700;color:#1d1d1f;margin-top:6px;">{winner}</div>
            <div class="card-sub">Based on expected uplift minus financial risk penalty</div>
        </div>""", unsafe_allow_html=True)

        left, right = st.columns(2)
        for col, name, bun, prow in [(left, pA_name, ba, pa),(right, pB_name, bb, pb)]:
            with col:
                st.markdown(f"**{name}**")
                st.markdown(chip(bun["decision"]["recommendation"]), unsafe_allow_html=True)
                st.markdown("")
                m1,m2 = st.columns(2)
                m1.metric("Fit Score",      f"{bun['fit']:.1f}")
                m2.metric("Financial Risk", f"{bun['risk']:.1f}")
                m1.metric("P(Success)",     f"{(bun['p_success'] or 0.5):.3f}")
                m2.metric("Exp. Uplift",    f"{bun['expected_uplift']:+.3f}")
                st.plotly_chart(radar(prow, squad_df), use_container_width=True)
                with st.expander("Rationale"):
                    for r in bun["decision"]["reasons"]: st.write(f"• {r}")

        st.markdown('<div class="sh">Head-to-Head Metrics</div>', unsafe_allow_html=True)
        labels = ["Fit (÷100)","Risk (÷100)","P(Success)","Expected Uplift"]
        av = [ba["fit"]/100, ba["risk"]/100, ba["p_success"] or 0.5, ba["expected_uplift"]]
        bv = [bb["fit"]/100, bb["risk"]/100, bb["p_success"] or 0.5, bb["expected_uplift"]]
        fig_ab = go.Figure()
        fig_ab.add_trace(go.Bar(name=pA_name, x=labels, y=av, marker_color="#0071e3", marker_line_width=0))
        fig_ab.add_trace(go.Bar(name=pB_name, x=labels, y=bv, marker_color="#30d158", marker_line_width=0))
        fig_ab.update_layout(**pl(height=320, barmode="group", yaxis_title="Normalised Value"))
        st.plotly_chart(fig_ab, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Shortlist
# ═════════════════════════════════════════════════════════════════════════════
with t3:
    st.markdown('<div class="sh">Shortlist Generator</div>', unsafe_allow_html=True)
    sc1,sc2,sc3 = st.columns([1,1,2])
    with sc1: pos_sel = st.selectbox("Position", POSITIONS)
    with sc2: top_n   = st.slider("Top N", 3, 12, 5)
    with sc3:
        st.markdown("&nbsp;")
        run_btn = st.button("Generate Shortlist")

    if run_btn:
        with st.spinner("Computing shortlist..."):
            sl = shortlist_top_targets(
                squad_df=squad_df, market_df=market_df, position=pos_sel,
                budget_m=transfer_budget, wage_capacity_m=wage_capacity,
                top_n=top_n, min_minutes=400, model_pipe=model_pipe)
        st.session_state["shortlist_df"]  = sl
        st.session_state["shortlist_pos"] = pos_sel

    sl = st.session_state.get("shortlist_df", pd.DataFrame())
    if len(sl)==0:
        if run_btn:
            st.warning("No feasible candidates under current budget/wage constraints.")
    else:
        fig_sl = px.scatter(
            sl, x="financial_risk", y="expected_uplift",
            size="market_value_m", color="recommendation", text="name",
            color_discrete_map=DECISION_COLORS, size_max=38,
            labels={"financial_risk":"Financial Risk","expected_uplift":"Expected Uplift"})
        fig_sl.update_traces(textposition="top center",
            textfont=dict(color="#86868b", size=11))
        fig_sl.update_layout(**pl(height=380,
            extra=dict(legend=dict(bgcolor="#ffffff", bordercolor="#d2d2d7"))))
        st.plotly_chart(fig_sl, use_container_width=True)

        disp = ["name","age","market_value_m","estimated_wage_m",
                "fit_score","expected_uplift","p_success","financial_risk","recommendation"]
        st.dataframe(sl[disp].style.format({
            "market_value_m":   "{:.1f}M",
            "estimated_wage_m": "{:.1f}M",
            "fit_score":        "{:.1f}",
            "expected_uplift":  "{:+.3f}",
            "p_success":        "{:.3f}",
            "financial_risk":   "{:.1f}",
        }), use_container_width=True, hide_index=True)

        st.download_button(
            "Download CSV",
            data=sl.to_csv(index=False).encode("utf-8"),
            file_name=f"shortlist_{club.replace(' ','_')}_{pos_sel}.csv",
            mime="text/csv")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — Scouting Agent
# ═════════════════════════════════════════════════════════════════════════════
with t4:
    from src.scouting_agent import run_scouting_agent

    st.markdown(f"""
    <div class="card" style="margin-bottom:20px;">
        <div class="card-label">AI Scouting Agent</div>
        <div style="font-size:16px;font-weight:600;color:#1d1d1f;margin:8px 0 4px;">
            Ask transfer questions in natural language
        </div>
        <div class="card-sub">
            Powered by Cohere Command R+ with tool use. The agent calls the simulation engine,
            shortlist generator, and comparison tools autonomously — then delivers a grounded scouting brief.
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sh">Example Queries</div>', unsafe_allow_html=True)
    examples = [
        f"Which striker under 20M fits {club} best?",
        f"Compare the top 2 CM candidates for {club}.",
        f"What is the weakest position in {club}'s squad?",
        "Which position should I prioritise this transfer window?",
    ]
    eq1, eq2 = st.columns(2)
    for i, ex in enumerate(examples):
        with (eq1 if i%2==0 else eq2):
            if st.button(ex, key=f"ex_{i}"):
                st.session_state["agent_input"] = ex

    st.markdown("---")

    if "agent_history" not in st.session_state:
        st.session_state["agent_history"] = []

    for msg in st.session_state["agent_history"]:
        if msg["role"] == "user":
            st.markdown(f'<div class="msg-user">{msg["content"]}</div>', unsafe_allow_html=True)
        elif msg["role"] == "tool_call":
            pass  # hidden from user
        elif msg["role"] == "assistant":
            st.markdown('<div class="msg-ai">', unsafe_allow_html=True)
            st.markdown(msg["content"])
            st.markdown('</div>', unsafe_allow_html=True)

    user_q = st.text_input(
        "Query", value=st.session_state.pop("agent_input",""),
        placeholder="e.g. Which CB under 25M would improve our squad most?",
        label_visibility="collapsed")

    col_s, col_c = st.columns([1,6])
    with col_s:
        send = st.button("Send")
    with col_c:
        if st.button("Clear"):
            st.session_state["agent_history"] = []
            st.rerun()

    if send and user_q.strip():
        st.session_state["agent_history"].append({"role":"user","content":user_q})
        with st.spinner("Analysing..."):
            try:
                res = run_scouting_agent(
                    query=user_q, club=club, squad_df=squad_df,
                    market_df=market_df, model_pipe=model_pipe,
                    transfer_budget=transfer_budget, wage_capacity=wage_capacity,
                    history=st.session_state["agent_history"][:-1])
                for tc in res.get("tool_calls",[]):
                    st.session_state["agent_history"].append(
                        {"role":"tool_call",
                         "content":f'{tc["tool"]}({", ".join(f"{k}={v}" for k,v in tc["args"].items())})'})
                st.session_state["agent_history"].append({"role":"assistant","content":res["answer"]})
            except Exception as e:
                st.session_state["agent_history"].append(
                    {"role":"assistant","content":f"Error: {str(e)}"})
        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — Model Card
# ═════════════════════════════════════════════════════════════════════════════
with t5:
    st.markdown('<div class="sh">What the Model Predicts</div>', unsafe_allow_html=True)
    st.markdown("""
The ML layer estimates **P(Positive Uplift)** — the probability that a candidate's projected
performance index exceeds the selected club's position-specific baseline.
The label is club-context specific: the same player may receive different probabilities for different clubs.

**Model:** Logistic Regression (interpretable). Features: age, minutes, xG, xA, progressive passes,
defensive actions, injury risk, market value, wage, position (one-hot encoded).
""")

    m1,m2,m3 = st.columns(3)
    m1.metric("Training N",  model_metrics.get("n","—"))
    m2.metric("AUC",         model_metrics.get("auc","—"))
    m3.metric("Accuracy",    model_metrics.get("accuracy","—"))
    if model_metrics.get("note"): st.caption(model_metrics["note"])

    st.markdown('<div class="sh">Global Feature Importance</div>', unsafe_allow_html=True)
    imp = get_feature_importance(model_pipe, top_k=15)
    if len(imp):
        fig_i = go.Figure(go.Bar(
            x=imp["abs_coefficient"], y=imp["feature"], orientation="h",
            marker_color="#0071e3", marker_line_width=0))
        fig_i.update_layout(**pl(height=460, xaxis_title="|Coefficient|"))
        st.plotly_chart(fig_i, use_container_width=True)

    st.markdown('<div class="sh">Limitations</div>', unsafe_allow_html=True)
    st.markdown("""
- Trained on **synthetic** data; not validated against real transfer outcomes.
- Uplift label derived from the projection function, not observed post-transfer data.
- Coefficients reflect correlations within the synthetic world — not causal claims.
- Intended for **decision prototyping**, not production scouting systems.
""")