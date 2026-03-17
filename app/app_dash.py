# FILE: app/app_dash.py — Transfer Intelligence Simulator (Dash)
# Assignment 2 · Prototyping Products with Data & AI · ESADE

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Prevent Flask from auto-loading the .env (has UTF-16 encoding on Windows).
# We load it manually below with encoding fallback (same pattern as scouting_agent.py).
os.environ["FLASK_SKIP_DOTENV"] = "1"
try:
    from dotenv import load_dotenv
    _env = ROOT_DIR / ".env"
    try:
        load_dotenv(_env, encoding="utf-8")
    except Exception:
        load_dotenv(_env, encoding="utf-16")
except ImportError:
    pass

import dash
from dash import html, dcc, Input, Output, State, callback, ctx, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json

from src.data_loader import load_dataset
from src.feature_engineering import compute_performance_index
from src.shortlist import compute_bundle, shortlist_top_targets, train_club_model_cached
from src.ml_model import get_feature_importance, explain_prediction
from src.brief_generator import generate_player_brief
from src.scouting_agent import run_scouting_agent

# ── App init ──────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="Transfer Intelligence · Bundesliga",
)
server = app.server  # for deployment

# ── Constants ─────────────────────────────────────────────────────────────────
CLUBS = [
    "FC Bayern München", "Borussia Dortmund", "RB Leipzig", "Bayer 04 Leverkusen",
    "Eintracht Frankfurt", "SC Freiburg", "VfB Stuttgart", "TSG Hoffenheim",
    "1. FSV Mainz 05", "Borussia Mönchengladbach", "VfL Wolfsburg", "Werder Bremen",
    "FC Augsburg", "1. FC Köln", "Union Berlin", "VfL Bochum",
    "1. FC Heidenheim", "SV Darmstadt 98",
]
POSITIONS = ["GK", "CB", "FB", "DM", "CM", "AM", "ST"]
POS_COLORS = {
    "GK": "#f59e0b", "CB": "#6366f1", "FB": "#0ea5e9",
    "DM": "#22c55e", "CM": "#3b82f6", "AM": "#f43f5e", "ST": "#ef4444",
}
DEC_COLORS = {"Sign": "#16a34a", "Watch": "#d97706", "Pass": "#dc2626"}

# ── Club logos (football-data.org public CDN — no auth required) ──────────────
_CREST_BASE = "https://crests.football-data.org/{}.png"
CLUB_LOGOS = {
    "FC Bayern München":          _CREST_BASE.format(5),
    "Borussia Dortmund":          _CREST_BASE.format(4),
    "RB Leipzig":                 _CREST_BASE.format(721),
    "Bayer 04 Leverkusen":        _CREST_BASE.format(3),
    "Eintracht Frankfurt":        _CREST_BASE.format(19),
    "SC Freiburg":                _CREST_BASE.format(17),
    "VfB Stuttgart":              _CREST_BASE.format(10),
    "TSG Hoffenheim":             _CREST_BASE.format(720),
    "1. FSV Mainz 05":            _CREST_BASE.format(15),
    "Borussia Mönchengladbach":   _CREST_BASE.format(18),
    "VfL Wolfsburg":              _CREST_BASE.format(11),
    "Werder Bremen":              _CREST_BASE.format(12),
    "FC Augsburg":                _CREST_BASE.format(16),
    "1. FC Köln":                 _CREST_BASE.format(1),
    "Union Berlin":               _CREST_BASE.format(28),
    "VfL Bochum":                 _CREST_BASE.format(36),
    "1. FC Heidenheim":           _CREST_BASE.format(387),
    "SV Darmstadt 98":            _CREST_BASE.format(94),
}


def _club_logo(club, size=32):
    url = CLUB_LOGOS.get(club)
    if not url:
        return html.Span()
    return html.Img(src=url, style={
        "width": f"{size}px", "height": f"{size}px",
        "objectFit": "contain", "flexShrink": "0",
    })

# ── Data + model cache ────────────────────────────────────────────────────────
market_df = load_dataset()
market_df = compute_performance_index(market_df)
_cache: dict = {}   # club → (squad_df, model_pipe, metrics)


def get_context(club: str, budget: float, wage: float):
    """Return (squad_df, model_pipe, metrics). Model cached per club."""
    if club not in _cache:
        sq = _make_squad(club)
        pipe, metrics = train_club_model_cached(sq, market_df, random_state=42)
        _cache[club] = (sq, pipe, metrics)
    sq, pipe, metrics = _cache[club]
    return sq, pipe, metrics


def _make_squad(club: str) -> pd.DataFrame:
    seed = abs(hash(club)) % (2 ** 32 - 1)
    rng  = np.random.default_rng(seed)
    comp = {"GK": 2, "CB": 4, "FB": 4, "DM": 3, "CM": 4, "AM": 3, "ST": 3}
    idx  = []
    for pos, n in comp.items():
        pool = market_df[market_df["position"] == pos]
        if len(pool) == 0:
            continue
        idx.extend(rng.choice(pool.index.to_numpy(),
                               size=min(n, len(pool)), replace=False).tolist())
    squad = market_df.loc[idx].copy()
    if len(squad) < 23:
        rem   = market_df.drop(index=idx)
        extra = rng.choice(rem.index.to_numpy(),
                            size=min(23 - len(squad), len(rem)), replace=False)
        squad = pd.concat([squad, rem.loc[extra]], ignore_index=True)
    squad["club"] = club
    return squad.reset_index(drop=True)


# ── Feature name mapping ──────────────────────────────────────────────────────
_FEAT_NAMES = {
    "num__xg":                  "Expected Goals (xG)",
    "num__xa":                  "Expected Assists (xA)",
    "num__progressive_passes":  "Progressive Passes",
    "num__defensive_actions":   "Defensive Actions",
    "num__age":                 "Age",
    "num__minutes":             "Minutes Played",
    "num__market_value_m":      "Market Value",
    "num__estimated_wage_m":    "Wage",
    "num__injury_risk":         "Injury Risk",
    "num__performance_index":   "Performance Index",
    "cat__position_GK":         "Position: GK",
    "cat__position_CB":         "Position: CB",
    "cat__position_FB":         "Position: FB",
    "cat__position_DM":         "Position: DM",
    "cat__position_CM":         "Position: CM",
    "cat__position_AM":         "Position: AM",
    "cat__position_ST":         "Position: ST",
}

def _fmt_feat(name: str) -> str:
    return _FEAT_NAMES.get(name, name.replace("num__", "").replace("cat__", "").replace("_", " ").title())


# ── Chart helpers ─────────────────────────────────────────────────────────────
_BG   = "#161b22"   # --bg-1 (panel background)
_BG2  = "#1c2128"   # --bg-2
_BG3  = "#21262d"   # --bg-3
_GRID = "rgba(240,246,252,0.07)"
_LINE = "rgba(240,246,252,0.10)"
_TEXT = "#8b949e"

CHART = dict(
    paper_bgcolor=_BG, plot_bgcolor=_BG,
    font=dict(family="DM Sans, system-ui, sans-serif", color=_TEXT, size=12),
    margin=dict(t=24, l=0, r=0, b=40),
    colorway=["#388bfd", "#3fb950", "#d29922", "#f85149", "#a371f7"],
)

def _radar_fig(player_row, squad_df):
    pos  = player_row["position"]
    base = squad_df[squad_df["position"] == pos]
    cats = ["xG", "xA", "Prog Passes", "Def Actions", "Minutes"]
    pv = [float(player_row["xg"]), float(player_row["xa"]),
          float(player_row["progressive_passes"]),
          float(player_row["defensive_actions"]),
          float(player_row["minutes"]) / 3200]
    bv = ([float(base["xg"].mean()), float(base["xa"].mean()),
           float(base["progressive_passes"].mean()),
           float(base["defensive_actions"].mean()),
           float(base["minutes"].mean()) / 3200]
          if len(base) > 0 else [0.2] * 5)
    mx = [float(market_df["xg"].quantile(0.95)),
          float(market_df["xa"].quantile(0.95)),
          float(market_df["progressive_passes"].quantile(0.95)),
          float(market_df["defensive_actions"].quantile(0.95)), 1.0]
    pn = [min(v / m, 1.0) for v, m in zip(pv, mx)]
    bn = [min(v / m, 1.0) for v, m in zip(bv, mx)]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=bn + [bn[0]], theta=cats + [cats[0]], fill="toself",
        fillcolor="rgba(59,130,246,0.08)", line=dict(color="#3b82f6", width=2),
        name=f"Squad ({pos})"))
    fig.add_trace(go.Scatterpolar(r=pn + [pn[0]], theta=cats + [cats[0]], fill="toself",
        fillcolor="rgba(34,197,94,0.1)", line=dict(color="#22c55e", width=2.5),
        name=str(player_row["name"]).replace("_", " ")))
    fig.update_layout(
        polar=dict(bgcolor=_BG2,
            radialaxis=dict(visible=True, range=[0, 1],
                            gridcolor=_GRID, tickfont=dict(color=_TEXT, size=10)),
            angularaxis=dict(gridcolor=_GRID, tickfont=dict(color=_TEXT, size=11))),
        paper_bgcolor=_BG, showlegend=True,
        legend=dict(font=dict(color=_TEXT, size=11), bgcolor=_BG2,
                    bordercolor=_LINE),
        height=320, margin=dict(t=16, l=16, r=16, b=16))
    return fig


def _feature_fig(expl):
    if not len(expl):
        return go.Figure()
    colors = ["#22c55e" if c > 0 else "#ef4444" for c in expl["contribution"]]
    labels = [_fmt_feat(f) for f in expl["feature"]]
    fig = go.Figure(go.Bar(
        x=expl["contribution"], y=labels, orientation="h",
        marker_color=colors, marker_line_width=0))
    fig.update_layout(**{**CHART, "height": 320,
                          "xaxis": {"title": "Log-odds contribution",
                                    "gridcolor": _GRID,
                                    "linecolor": _LINE,
                                    "tickfont": dict(color=_TEXT)},
                          "yaxis": {"gridcolor": _GRID,
                                    "linecolor": _LINE,
                                    "tickfont": dict(color=_TEXT)},
                          "showlegend": False})
    return fig


def _impact_fig(baseline, player_perf, sigma, pname):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Squad Baseline", pname], y=[baseline, player_perf],
        marker_color=["#3b82f6", "#22c55e"], marker_line_width=0, name="Performance"))
    fig.add_trace(go.Scatter(
        x=[pname, pname],
        y=[player_perf - sigma, player_perf + sigma],
        mode="lines", line=dict(width=5, color="#f59e0b"), name="Uncertainty"))
    fig.update_layout(**{**CHART, "height": 260,
                          "yaxis": {"title": "Performance Index",
                                    "gridcolor": _GRID, "linecolor": _LINE,
                                    "tickfont": dict(color=_TEXT)},
                          "xaxis": {"gridcolor": _GRID, "linecolor": _LINE,
                                    "tickfont": dict(color=_TEXT)},
                          "showlegend": True})
    return fig


# ── UI component helpers ──────────────────────────────────────────────────────
def _chip(rec: str):
    cls  = {"Sign": "chip-proceed", "Watch": "chip-monitor", "Pass": "chip-avoid"}
    icon = {"Sign": "✓", "Watch": "~", "Pass": "✕"}
    return html.Span(f'{icon.get(rec,"·")} {rec}', className=f"chip {cls.get(rec,'')}")


_SILHOUETTE_SVG = html.Span("👤", style={"fontSize": "22px", "opacity": "0.75"})


def _player_header(row):
    name  = str(row["name"]).replace("_", " ")
    color = POS_COLORS.get(str(row["position"]), "#3b82f6")
    return html.Div(className="player-card", children=[
        html.Div(className="player-avatar",
                 style={"background": f"linear-gradient(135deg, {color}99, {color})"},
                 children=[_SILHOUETTE_SVG]),
        html.Div([
            html.Div(name, className="player-name"),
            html.Div(className="player-tags", children=[
                html.Div(className="player-stat-item", children=[
                    html.Div("Position", className="player-stat-label"),
                    html.Span(str(row["position"]), className="tag tag-pos"),
                ]),
                html.Div(className="player-stat-item", children=[
                    html.Div("Age", className="player-stat-label"),
                    html.Span(str(int(row["age"])), className="tag tag-age"),
                ]),
                html.Div(className="player-stat-item", children=[
                    html.Div("Market Value", className="player-stat-label"),
                    html.Span(f"EUR {float(row['market_value_m']):.1f}M", className="tag tag-val"),
                ]),
                html.Div(className="player-stat-item", children=[
                    html.Div("Salary", className="player-stat-label"),
                    html.Span(f"EUR {float(row['estimated_wage_m']):.1f}M/yr", className="tag"),
                ]),
                html.Div(className="player-stat-item", children=[
                    html.Div("Minutes", className="player-stat-label"),
                    html.Span(f"{int(row['minutes']):,}", className="tag tag-min"),
                ]),
            ]),
        ]),
    ])


def _verdict_card(rec, score, risk_lbl, risk_val, reasons):
    risk_cls = ("green" if risk_val < 35 else ("orange" if risk_val < 65 else "red"))
    return html.Div(className="verdict-card", children=[
        html.Div(className="d-flex justify-content-between align-items-center", children=[
            html.Div("Recommendation", className="card-label"),
            _chip(rec),
        ]),
        html.Div(className="verdict-scores", children=[
            html.Div(className="verdict-score-item", children=[
                html.Div("Overall Score", className="v-label"),
                html.Div(f"{score:.0f}", className="v-value"),
            ]),
            html.Div(className="verdict-score-item", children=[
                html.Div("Financial Risk", className="v-label"),
                html.Div(risk_lbl, className=f"v-value card-value {risk_cls}"),
                html.Div(f"{risk_val:.1f} / 100", className="card-sub"),
            ]),
        ]),
        html.Div(" · ".join(reasons), className="verdict-reasons"),
    ])


def _score_bar(label, value, color="#3b82f6", invert=False):
    if invert:
        color = "#ef4444" if value > 65 else ("#f59e0b" if value > 35 else "#22c55e")
    pct = min(value, 100)
    return html.Div(className="score-bar-wrap", children=[
        html.Div(className="score-bar-header", children=[
            html.Span(label, className="score-bar-label"),
            html.Span(f"{value:.1f}", className="score-bar-value"),
        ]),
        html.Div(className="score-bar-track", children=[
            html.Div(className="score-bar-fill",
                     style={"width": f"{pct:.1f}%", "background": color}),
        ]),
    ])


def _kpi(label, value):
    return html.Div(className="kpi-card", children=[
        html.Div(label, className="kpi-label"),
        html.Div(value, className="kpi-value"),
    ])


def _brief_section(icon, icon_cls, title, body):
    return html.Div(className="brief-section", children=[
        html.Div(className="d-flex align-items-center", children=[
            html.Span(icon, className=f"brief-icon {icon_cls}"),
            html.Span(title, className="brief-section-label"),
        ]),
        html.Div(body, className="brief-body"),
    ])


def _chart_note(text):
    return html.Div(text, style={
        "fontSize": "12px", "color": "#8b949e", "marginTop": "6px",
        "lineHeight": "1.55", "paddingLeft": "2px"
    })


def _empty_state(icon, title, subtitle):
    return html.Div(className="empty-state", children=[
        html.Div(icon, className="empty-state-icon"),
        html.Div(title, className="empty-state-title"),
        html.Div(subtitle, className="empty-state-sub"),
    ])


# ── Sidebar ───────────────────────────────────────────────────────────────────
sidebar = html.Div(id="sidebar", children=[
    html.Div("Bundesliga · 2025/26", className="sidebar-eyebrow"),

    html.Div(className="sidebar-section", children=[
        html.Label("Club", className="sidebar-label"),
        dcc.Dropdown(
            id="sb-club", options=[{"label": c, "value": c} for c in CLUBS],
            value=CLUBS[0], clearable=False, className="sidebar-select",
        ),
        html.Div(
            "Select your club. All analyses, squad comparisons, and AI briefs will be tailored to this club's squad profile.",
            className="sidebar-hint"
        ),
    ]),

    html.Hr(className="sidebar-divider"),

    html.Div(className="sidebar-section", children=[
        html.Label("Transfer Budget (M€)", className="sidebar-label"),
        dcc.Slider(id="sb-budget", min=5, max=80, step=1, value=30,
                   marks={5: "5", 40: "40", 80: "80"},
                   tooltip={"placement": "bottom", "always_visible": True}),
        html.Div(
            "Maximum transfer fee you are willing to pay. Players above this threshold are flagged as financially infeasible.",
            className="sidebar-hint"
        ),
    ]),

    html.Div(className="sidebar-section", children=[
        html.Label("Wage Capacity (M€/yr)", className="sidebar-label"),
        dcc.Slider(id="sb-wage", min=1, max=30, step=1, value=8,
                   marks={1: "1", 15: "15", 30: "30"},
                   tooltip={"placement": "bottom", "always_visible": True}),
        html.Div(
            "Maximum annual salary you can offer. Candidates exceeding this cap receive a higher financial risk score.",
            className="sidebar-hint"
        ),
    ]),

    html.Hr(className="sidebar-divider"),

    # Club card — updated by callback
    html.Div(id="sb-club-card"),
])


def _build_club_card(club, budget, wage, squad_df):
    avg_age = squad_df["age"].mean()
    avg_val = squad_df["market_value_m"].mean()
    return html.Div(className="club-card", children=[
        html.Div(className="club-card-header", children=[
            _club_logo(club, size=36),
            html.Div([
                html.Div(club, className="club-card-name"),
                html.Div(f"{len(squad_df)} players · {len(squad_df['position'].unique())} positions",
                         className="club-card-meta"),
            ]),
        ]),
        html.Div(className="club-stat-row", children=[
            html.Div(className="club-stat", children=[
                html.Div("Budget", className="club-stat-label"),
                html.Div(f"€{budget}M", className="club-stat-value"),
            ]),
            html.Div(className="club-stat", children=[
                html.Div("Wage Cap", className="club-stat-label"),
                html.Div(f"€{wage}M", className="club-stat-value"),
            ]),
        ]),
        html.Div(className="club-stat-row mt-2", children=[
            html.Div(className="club-stat", children=[
                html.Div("Avg Age", className="club-stat-label"),
                html.Div(f"{avg_age:.1f}", className="club-stat-value"),
            ]),
            html.Div(className="club-stat", children=[
                html.Div("Avg Value", className="club-stat-label"),
                html.Div(f"€{avg_val:.1f}M", className="club-stat-value"),
            ]),
        ]),
    ])


# ── Player options helper ─────────────────────────────────────────────────────
PLAYER_OPTIONS = [{"label": n.replace("_", " "), "value": n}
                  for n in market_df["name"].tolist()]


# ── Tab layouts ───────────────────────────────────────────────────────────────
def _tab1():
    return html.Div([
        dbc.Row(dbc.Col(
            dcc.Dropdown(id="t1-player", options=PLAYER_OPTIONS,
                         value=PLAYER_OPTIONS[0]["value"], clearable=False,
                         placeholder="Search for a player…"),
            width=6), className="mb-4"),
        dcc.Loading(html.Div(id="t1-content"), type="circle", color="#2563eb"),
    ])


def _tab2():
    return html.Div([
        html.Div("Head-to-Head Comparison", className="section-title"),
        html.Div("Compare two candidates directly. Same squad context, same constraints — see who comes out ahead.",
                 className="section-sub"),
        dbc.Row([
            dbc.Col(dcc.Dropdown(id="t2-pa", options=PLAYER_OPTIONS,
                                 value=PLAYER_OPTIONS[0]["value"], clearable=False,
                                 placeholder="First candidate…"), width=6),
            dbc.Col(dcc.Dropdown(id="t2-pb", options=PLAYER_OPTIONS,
                                 value=PLAYER_OPTIONS[1]["value"], clearable=False,
                                 placeholder="Second candidate…"), width=6),
        ], className="mb-4"),
        dcc.Loading(html.Div(id="t2-content"), type="circle", color="#2563eb"),
    ])


def _tab3():
    return html.Div([
        html.Div("Market Scan", className="section-title"),
        html.Div("Find the best available candidates within your budget and wage limits. Ranked by risk-adjusted value.",
                 className="section-sub"),
        dbc.Row([
            dbc.Col(dcc.Dropdown(id="t3-pos",
                                 options=[{"label": p, "value": p} for p in POSITIONS],
                                 value="ST", clearable=False), width=3),
            dbc.Col([
                html.Label("Number of candidates", className="card-label"),
                dcc.Slider(id="t3-topn", min=3, max=12, step=1, value=5,
                           marks={3: "3", 7: "7", 12: "12"},
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], width=4),
            dbc.Col(html.Div(
                dbc.Button("Find Candidates  →", id="t3-btn", color="primary"),
                className="d-flex align-items-end h-100 pb-1"), width=3),
        ], className="mb-4"),
        dcc.Loading(html.Div(id="t3-content"), type="circle", color="#2563eb"),
    ])


def _tab4():
    return html.Div([
        html.Div("Scouting Report", className="section-title"),
        html.Div(
            "Get a written summary ready to share with your board. "
            "The AI pulls together all scores and squad data into a clear, concise brief.",
            className="section-sub"),
        dbc.Row([
            dbc.Col(dcc.Dropdown(id="t4-player", options=PLAYER_OPTIONS,
                                 value=PLAYER_OPTIONS[0]["value"], clearable=False), width=6),
            dbc.Col(html.Div(
                dbc.Button("Generate Report  →", id="t4-btn", color="primary"),
                className="d-flex align-items-end h-100 pb-1"), width=4),
        ], className="mb-3"),
        html.Div(id="t4-preview"),
        dcc.Loading(html.Div(id="t4-content"), type="circle", color="#2563eb"),
    ])


def _tab5():
    return html.Div([
        html.Div(className="agent-header", children=[
            html.Div("AI Scouting Assistant", className="agent-eyebrow"),
            html.Div("Ask the Scout", className="agent-title"),
            html.Div(id="agent-sub", className="agent-sub"),
        ]),
        html.Div("Try asking…", className="sh"),
        dbc.Row([
            dbc.Col([
                dbc.Button(e, id={"type": "ex-btn", "index": i},
                           className="btn-example", n_clicks=0)
                for i, e in enumerate(["Which striker under €20M fits us best?",
                                       "Where is our squad weakest right now?"])
            ], width=6),
            dbc.Col([
                dbc.Button(e, id={"type": "ex-btn", "index": i + 2},
                           className="btn-example", n_clicks=0)
                for i, e in enumerate(["Compare the top 2 central midfield options.",
                                       "Which position should we target this window?"])
            ], width=6),
        ], className="mb-3"),
        html.Hr(),
        dcc.Loading(
            html.Div(id="agent-chat", className="msg-wrap",
                     style={"minHeight": "80px"}),
            type="circle", color="#2563eb"),
        dbc.Row([
            dbc.Col(dbc.Input(id="agent-input", placeholder="e.g. Which CB under €25M would improve our defensive line?",
                              className="agent-input"), width=9),
            dbc.Col(dbc.Button("Send", id="agent-send", color="primary"), width=1),
            dbc.Col(dbc.Button("Clear", id="agent-clear",
                               className="btn-ghost"), width=2),
        ], className="mt-3 align-items-center"),
    ])


def _tab6():
    return html.Div([
        html.Div("How the Scores Work", className="section-title"),
        html.Div("A plain-language explanation of what drives each recommendation — and where to treat the numbers with caution.",
                 className="section-sub"),
        dcc.Loading(html.Div(id="t6-content"), type="circle", color="#2563eb"),
    ])


# ── Hero ──────────────────────────────────────────────────────────────────────
hero = html.Div(className="hero", children=[
    html.Div("Bundesliga · Recruitment Intelligence", className="hero-eyebrow"),
    html.Div("Transfer Intelligence Simulator", className="hero-title"),
    html.Div("Know who to sign, who to watch, and who to pass on — "
             "before you spend a single euro. Built for Sporting Directors "
             "who need fast, grounded decisions under budget pressure.", className="hero-sub"),
    html.Div(id="hero-badges", className="hero-badges"),
])


# ── App layout ────────────────────────────────────────────────────────────────
app.layout = html.Div([
    dcc.Store(id="agent-history", data=[]),
    dcc.Store(id="shortlist-store", data=None),

    dbc.Row(className="g-0 root-row", children=[
        # Sidebar
        dbc.Col(sidebar, width=3, style={"maxWidth": "280px"}),

        # Main
        dbc.Col(html.Div(id="main-content", children=[
            hero,
            dbc.Tabs(id="main-tabs", active_tab="t1", className="custom-tabs", children=[
                dbc.Tab(_tab1(), tab_id="t1", label="Player Analysis"),
                dbc.Tab(_tab2(), tab_id="t2", label="Head-to-Head"),
                dbc.Tab(_tab3(), tab_id="t3", label="Market Scan"),
                dbc.Tab(_tab4(), tab_id="t4", label="Scouting Report"),
                dbc.Tab(_tab5(), tab_id="t5", label="Ask the Scout"),
                dbc.Tab(_tab6(), tab_id="t6", label="How It Works"),
            ]),
        ]), className="tab-content-area-wrap"),
    ]),
])


# ═══════════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Sidebar club card ─────────────────────────────────────────────────────────
@app.callback(
    Output("sb-club-card", "children"),
    Output("hero-badges", "children"),
    Input("sb-club", "value"),
    Input("sb-budget", "value"),
    Input("sb-wage", "value"),
)
def update_sidebar(club, budget, wage):
    sq, _, _ = get_context(club, budget, wage)
    card = _build_club_card(club, budget, wage, sq)
    badges = [
        html.Span(className="hero-badge primary", children=[
            _club_logo(club, size=16),
            html.Span(club, style={"marginLeft": "6px"}),
        ]),
        html.Span(f"€{budget}M Transfer Budget", className="hero-badge"),
        html.Span("Success Prediction", className="hero-badge"),
        html.Span("AI Scouting Assistant", className="hero-badge"),
        html.Span("277 Market Players", className="hero-badge"),
    ]
    return card, badges


# ── Agent subtitle ────────────────────────────────────────────────────────────
@app.callback(
    Output("agent-sub", "children"),
    Input("sb-club", "value"),
    Input("sb-budget", "value"),
    Input("sb-wage", "value"),
)
def update_agent_sub(club, budget, wage):
    return (f"Your recruitment assistant for {club}. "
            f"Budget: €{budget}M · Wage cap: €{wage}M/yr. "
            "Ask in plain language — it runs the analysis and comes back with a clear answer.")


# ── TAB 1 — Player Intelligence ───────────────────────────────────────────────
@app.callback(
    Output("t1-content", "children"),
    Input("t1-player", "value"),
    Input("sb-club", "value"),
    Input("sb-budget", "value"),
    Input("sb-wage", "value"),
)
def render_player_tab(player_name, club, budget, wage):
    sq, pipe, metrics = get_context(club, budget, wage)
    row = market_df[market_df["name"] == player_name].iloc[0]
    b   = compute_bundle(sq, market_df, row, budget, wage, model_pipe=pipe)
    rec = b["decision"]["recommendation"]
    fit = b["fit"]
    risk = b["risk"]
    p_s = b["p_success"] or 0.5
    eu  = b["expected_uplift"]

    expl   = explain_prediction(pipe, row, top_k=6)
    pname  = str(player_name).replace("_", " ")

    return html.Div([
        _player_header(row),
        dbc.Row([
            dbc.Col(_verdict_card(
                rec, b["decision"]["decision_score"],
                b["decision"]["risk_level"], risk,
                b["decision"]["reasons"]), width=7, style={"display": "flex", "flexDirection": "column"}),
            dbc.Col(html.Div(className="card-base h-100", children=[
                html.Div("Scores at a Glance", className="card-label mb-3"),
                _score_bar("Squad Fit", fit, "#3b82f6"),
                _score_bar("Financial Risk", risk, invert=True),
                _score_bar("Success Probability", p_s * 100, "#22c55e"),
                _score_bar("Uplift Potential",
                           max(min((eu + 1) / 2 * 100, 100), 0), "#a855f7"),
            ]), width=5, style={"display": "flex", "flexDirection": "column"}),
        ], className="align-items-stretch"),
        html.Div("Performance Summary", className="sh"),
        dbc.Row([
            dbc.Col(_kpi("Squad Fit", f"{fit:.0f} / 100"), width=True),
            dbc.Col(_kpi("Performance Uplift", f"{b['uplift']['uplift']:+.2f}"), width=True),
            dbc.Col(_kpi("Confidence Range", f"±{b['uplift']['uncertainty_sigma']:.2f}"), width=True),
            dbc.Col(_kpi("Success Likelihood", f"{p_s:.0%}"), width=True),
            dbc.Col(_kpi("Risk-Adj. Uplift", f"{eu:+.2f}"), width=True),
        ], className="g-2 mb-3"),
        dbc.Row([
            dbc.Col([
                html.Div("Radar Profile vs Squad Baseline", className="sh"),
                dcc.Graph(figure=_radar_fig(row, sq), config={"displayModeBar": False}),
                _chart_note("The radar compares the player's key stats against the current squad average at the same position. A larger green area means the player outperforms the squad baseline. Blue shows the squad average. All values are normalised to the top 5% of the market."),
            ], width=6),
            dbc.Col([
                html.Div("Feature Contributions to P(Success)", className="sh"),
                dcc.Graph(figure=_feature_fig(expl), config={"displayModeBar": False}),
                _chart_note("Each bar shows how strongly a player attribute pushes the ML success probability up (green) or down (red). The longer the bar, the more influential that attribute is for this specific player."),
            ], width=6),
        ]),
        html.Div("Expected Squad Impact", className="sh"),
        dbc.Row([
            dbc.Col(_kpi("Current Squad Level", f"{b['uplift']['baseline']:.2f}"), width=4),
            dbc.Col(_kpi("Player Level",         f"{b['uplift']['player_perf']:.2f}"), width=4),
            dbc.Col(_kpi("Projected Range",
                         f"{b['uplift']['uplift_lower']:+.2f} to {b['uplift']['uplift_upper']:+.2f}"),
                    width=4),
        ], className="g-2 mb-3"),
        dcc.Graph(figure=_impact_fig(
            b["uplift"]["baseline"], b["uplift"]["player_perf"],
            b["uplift"]["uncertainty_sigma"], pname),
            config={"displayModeBar": False}),
        _chart_note("Blue bar: current squad average performance at this position. Green bar: player's projected performance index. A higher green bar means the player would improve the squad. The orange line shows the uncertainty range — a wider range means fewer minutes played and lower confidence in the projection."),
        dbc.Accordion([
            dbc.AccordionItem(
                html.Div(children=[
                    html.P("Current squad composition by position — depth and average performance levels.",
                           style={"fontSize": "13px", "color": "#8b949e"}),
                    html.Div(_squad_table(sq)),
                ], style={"padding": "4px"}),
                title="Current Squad Overview",
            )
        ], start_collapsed=True, className="mt-2"),
    ])


def _squad_table(sq):
    snap = sq.groupby("position").agg(
        players=("name", "count"),
        avg_perf=("performance_index", "mean"),
        avg_age=("age", "mean"),
        avg_val=("market_value_m", "mean"),
    ).reset_index()
    return dbc.Table.from_dataframe(
        snap.rename(columns={"position": "Pos", "players": "Players",
                              "avg_perf": "Avg Perf",
                              "avg_age": "Avg Age", "avg_val": "Avg Value (M€)"}
                    ).assign(**{
                        "Avg Perf": snap["avg_perf"].map("{:.3f}".format),
                        "Avg Age": snap["avg_age"].map("{:.1f}".format),
                        "Avg Value (M€)": snap["avg_val"].map("€{:.1f}M".format),
                    }),
        striped=True, hover=True, size="sm",
        className="mb-0", style={"fontSize": "13px"})


# ── TAB 2 — Transfer Comparison ───────────────────────────────────────────────
@app.callback(
    Output("t2-content", "children"),
    Input("t2-pa", "value"),
    Input("t2-pb", "value"),
    Input("sb-club", "value"),
    Input("sb-budget", "value"),
    Input("sb-wage", "value"),
)
def render_comparison(pA, pB, club, budget, wage):
    if pA == pB:
        return dbc.Alert("Select two different players to enable comparison.",
                         color="warning", className="rounded-3")
    sq, pipe, _ = get_context(club, budget, wage)
    ra = market_df[market_df["name"] == pA].iloc[0]
    rb = market_df[market_df["name"] == pB].iloc[0]
    ba = compute_bundle(sq, market_df, ra, budget, wage, model_pipe=pipe)
    bb = compute_bundle(sq, market_df, rb, budget, wage, model_pipe=pipe)

    if   ba["risk_adjusted_value"] > bb["risk_adjusted_value"]: winner, wcolor = pA.replace("_"," "), "#3b82f6"
    elif bb["risk_adjusted_value"] > ba["risk_adjusted_value"]: winner, wcolor = pB.replace("_"," "), "#22c55e"
    else: winner, wcolor = "Tie", "#8b949e"

    delta = abs(ba["risk_adjusted_value"] - bb["risk_adjusted_value"])
    winner_card = html.Div(className="winner-card",
                           style={"borderLeftColor": wcolor}, children=[
        html.Div("Better Option", className="card-label"),
        html.Div(winner, className="winner-name"),
        html.Div(f"Scored higher on performance upside vs. financial risk — gap: {delta:.2f}",
                 className="winner-sub"),
    ])

    def _player_col(name, b, row, letter, color):
        chip_icon = {"Sign": "✓", "Watch": "~", "Pass": "✕"}
        rec = b["decision"]["recommendation"]
        rec_cls = {"Sign": "chip-proceed", "Watch": "chip-monitor", "Pass": "chip-avoid"}
        return html.Div([
            html.Div(className="card-sm mb-3", style={"borderLeft": f"3px solid {color}"}, children=[
                html.Div(f"Player {letter}", style={"fontSize": "10px", "fontWeight": "700",
                         "color": color, "textTransform": "uppercase",
                         "letterSpacing": "0.8px", "marginBottom": "4px"}),
                html.Div(name.replace("_", " "),
                         style={"fontSize": "16px", "fontWeight": "700", "marginBottom": "6px"}),
                html.Div(className="player-tags", children=[
                    html.Span(str(row["position"]), className="tag tag-pos"),
                    html.Span(f"Age {int(row['age'])}", className="tag tag-age"),
                    html.Span(f"€{float(row['market_value_m']):.1f}M", className="tag tag-val"),
                ]),
            ]),
            html.Span(f'{chip_icon.get(rec,"·")} {rec}',
                      className=f"chip {rec_cls.get(rec,'')} mb-3 d-inline-block"),
            dbc.Row([
                dbc.Col(_kpi("Fit Score",   f"{b['fit']:.1f}"), width=6),
                dbc.Col(_kpi("Fin. Risk",   f"{b['risk']:.1f}"), width=6),
            ], className="g-2 mb-2"),
            dbc.Row([
                dbc.Col(_kpi("P(Success)", f"{(b['p_success'] or 0.5):.3f}"), width=6),
                dbc.Col(_kpi("Exp. Uplift",f"{b['expected_uplift']:+.3f}"), width=6),
            ], className="g-2 mb-3"),
            dcc.Graph(figure=_radar_fig(row, sq), config={"displayModeBar": False}),
            _chart_note("Green = this player. Blue = squad average at same position."),
            dbc.Accordion([
                dbc.AccordionItem(
                    html.Ul([html.Li(r, style={"fontSize": "13px"})
                             for r in b["decision"]["reasons"]]),
                    title="Scouting Rationale",
                )
            ], start_collapsed=True),
        ])

    labels = ["Tactical Fit (÷100)", "Financial Risk (÷100)", "P(Success)", "Exp. Uplift"]
    av = [ba["fit"]/100, ba["risk"]/100, ba["p_success"] or 0.5, ba["expected_uplift"]]
    bv = [bb["fit"]/100, bb["risk"]/100, bb["p_success"] or 0.5, bb["expected_uplift"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(name=f"A · {pA.replace('_',' ')}", x=labels, y=av,
                         marker_color="#3b82f6", marker_line_width=0))
    fig.add_trace(go.Bar(name=f"B · {pB.replace('_',' ')}", x=labels, y=bv,
                         marker_color="#22c55e", marker_line_width=0))
    fig.update_layout(**{**CHART, "height": 300, "barmode": "group",
                          "yaxis": {"title": "Normalised Value", "gridcolor": _GRID,
                                    "linecolor": _LINE, "tickfont": dict(color=_TEXT)},
                          "xaxis": {"gridcolor": _GRID, "linecolor": _LINE,
                                    "tickfont": dict(color=_TEXT)}})

    return html.Div([
        winner_card,
        dbc.Row([
            dbc.Col(_player_col(pA, ba, ra, "A", "#3b82f6"), width=6),
            dbc.Col(_player_col(pB, bb, rb, "B", "#22c55e"), width=6),
        ]),
        html.Div("Head-to-Head Analytics", className="sh"),
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
        _chart_note("Side-by-side comparison across four metrics (all normalised to 0-1 scale). Higher is better for Tactical Fit and P(Success). For Financial Risk, lower is better. Expected Uplift can be negative if the player is projected to underperform the squad baseline at their position."),
    ])


# ── TAB 3 — Market Shortlist ──────────────────────────────────────────────────
@app.callback(
    Output("t3-content", "children"),
    Output("shortlist-store", "data"),
    Input("t3-btn", "n_clicks"),
    State("t3-pos", "value"),
    State("t3-topn", "value"),
    State("sb-club", "value"),
    State("sb-budget", "value"),
    State("sb-wage", "value"),
    prevent_initial_call=True,
)
def generate_shortlist(n, pos, top_n, club, budget, wage):
    sq, pipe, _ = get_context(club, budget, wage)
    sl = shortlist_top_targets(sq, market_df, pos, budget, wage,
                               top_n=top_n, min_minutes=400, model_pipe=pipe)
    if len(sl) == 0:
        content = dbc.Alert(
            "No feasible candidates found under current budget and wage constraints. "
            "Try increasing the budget or wage capacity.", color="warning",
            className="rounded-3 mt-2")
        return content, None

    fig = px.scatter(
        sl, x="financial_risk", y="expected_uplift",
        size="market_value_m", color="recommendation",
        text="name",
        color_discrete_map=DEC_COLORS, size_max=36,
        labels={"financial_risk": "Financial Risk →",
                "expected_uplift": "Expected Uplift →"})
    fig.update_traces(textposition="top center",
                      textfont=dict(color=_TEXT, size=10))
    fig.update_layout(**{**CHART, "height": 360,
                          "xaxis": {"title": "Financial Risk →", "gridcolor": _GRID,
                                    "linecolor": _LINE, "tickfont": dict(color=_TEXT)},
                          "yaxis": {"title": "Expected Uplift →", "gridcolor": _GRID,
                                    "linecolor": _LINE, "tickfont": dict(color=_TEXT)}})

    disp = ["name", "age", "market_value_m", "estimated_wage_m",
            "fit_score", "expected_uplift", "p_success", "financial_risk", "recommendation"]
    tbl_df = sl[disp].copy()
    tbl_df["market_value_m"]   = tbl_df["market_value_m"].map("€{:.1f}M".format)
    tbl_df["estimated_wage_m"] = tbl_df["estimated_wage_m"].map("€{:.1f}M".format)
    tbl_df["fit_score"]        = tbl_df["fit_score"].map("{:.1f}".format)
    tbl_df["expected_uplift"]  = tbl_df["expected_uplift"].map("{:+.3f}".format)
    tbl_df["p_success"]        = tbl_df["p_success"].map("{:.3f}".format)
    tbl_df["financial_risk"]   = tbl_df["financial_risk"].map("{:.1f}".format)
    tbl_df = tbl_df.rename(columns={
        "name": "Player", "age": "Age",
        "market_value_m": "Market Value", "estimated_wage_m": "Wage / yr",
        "fit_score": "Squad Fit", "expected_uplift": "Uplift",
        "p_success": "Success %", "financial_risk": "Fin. Risk",
        "recommendation": "Verdict",
    })

    content = html.Div([
        html.Div("Opportunity Map", className="sh"),
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
        _chart_note("Each bubble is a candidate within your budget. Top-left is the sweet spot — high upside, low financial risk. Bubble size reflects market value. Green = Sign, Orange = Watch, Red = Pass."),
        html.Div("Ranked Candidates", className="sh"),
        dbc.Table.from_dataframe(tbl_df, striped=True, hover=True, size="sm",
                                 style={"fontSize": "13px"}),
        html.Div(className="mt-3", children=[
            dbc.Button("Export as CSV", id="t3-download-btn",
                       className="btn-ghost", size="sm"),
            dcc.Download(id="t3-download"),
        ]),
    ])
    return content, sl.to_dict("records")


@app.callback(
    Output("t3-download", "data"),
    Input("t3-download-btn", "n_clicks"),
    State("shortlist-store", "data"),
    State("sb-club", "value"),
    State("t3-pos", "value"),
    prevent_initial_call=True,
)
def download_shortlist(n, data, club, pos):
    if not data:
        raise PreventUpdate
    df = pd.DataFrame(data)
    fname = f"shortlist_{club.replace(' ','_')}_{pos}.csv"
    return dcc.send_data_frame(df.to_csv, fname, index=False)


# ── TAB 4 — AI Transfer Brief ─────────────────────────────────────────────────
@app.callback(
    Output("t4-preview", "children"),
    Input("t4-player", "value"),
    Input("sb-club", "value"),
    Input("sb-budget", "value"),
    Input("sb-wage", "value"),
)
def update_brief_preview(player_name, club, budget, wage):
    sq, pipe, _ = get_context(club, budget, wage)
    row = market_df[market_df["name"] == player_name].iloc[0]
    b   = compute_bundle(sq, market_df, row, budget, wage, model_pipe=pipe)
    rec = b["decision"]["recommendation"]
    return dbc.Row([
        dbc.Col(_kpi("Squad Fit",          f"{b['fit']:.0f} / 100"), width=3),
        dbc.Col(_kpi("Financial Risk",     f"{b['risk']:.0f} / 100"), width=3),
        dbc.Col(_kpi("Success Likelihood", f"{(b['p_success'] or 0.5):.0%}"), width=3),
        dbc.Col(_kpi("Recommendation",     rec), width=3),
    ], className="g-2 mb-3")


@app.callback(
    Output("t4-content", "children"),
    Input("t4-btn", "n_clicks"),
    State("t4-player", "value"),
    State("sb-club", "value"),
    State("sb-budget", "value"),
    State("sb-wage", "value"),
    prevent_initial_call=True,
)
def generate_brief(n, player_name, club, budget, wage):
    sq, pipe, _ = get_context(club, budget, wage)
    row    = market_df[market_df["name"] == player_name].iloc[0]
    bundle = compute_bundle(sq, market_df, row, budget, wage, model_pipe=pipe)
    brief, warn = generate_player_brief(row, bundle, sq, club, budget, wage)

    if brief is None:
        return dbc.Alert(f"Brief generation failed: {warn}", color="danger",
                         className="rounded-3")

    rec  = bundle["decision"]["recommendation"]
    conf = brief.get("confidence", "Medium")
    conf_cls = {"High": "conf-high", "Medium": "conf-medium", "Low": "conf-low"}.get(conf, "conf-medium")

    chip_cls = {"Sign": "chip-proceed", "Watch": "chip-monitor", "Pass": "chip-avoid"}
    chip_icon= {"Sign": "✓", "Watch": "~", "Pass": "✕"}

    return html.Div([
        html.Div(className="d-flex align-items-center gap-2 mb-1", children=[
            _club_logo(club, size=28),
            html.Div("Scouting Report: " + player_name.replace("_", " ") + " — " + club,
                     className="sh", style={"margin": "0"}),
        ]),
        html.Div(className="d-flex align-items-center gap-3 mb-3", children=[
            html.Span(f'{chip_icon.get(rec,"·")} {rec}',
                      className=f"chip {chip_cls.get(rec,'')}"),
            html.Span(f"◆ {conf} Confidence", className=f"conf-badge {conf_cls}"),
            html.Span("AI-generated · Cohere Command R+",
                      style={"fontSize": "11px", "color": "#8b949e", "marginLeft": "8px"}),
        ]),
        _brief_section("—", "brief-icon-blue",   "Summary",                       brief.get("executive_summary", "")),
        _brief_section("—", "brief-icon-green",  "On the Pitch",                  brief.get("performance_verdict", "")),
        _brief_section("—", "brief-icon-orange", "Financial Picture",              brief.get("financial_assessment", "")),
        _brief_section("—", "brief-icon-red",    "Risks to Consider",             brief.get("risk_factors", "")),
        _brief_section("—", "brief-icon-blue",   "What We Recommend",             brief.get("recommendation", "")),
        html.Div("Based on synthetic data. For demonstration purposes only.",
                 style={"fontSize": "11px", "color": "#8b949e", "marginTop": "8px"}),
        dbc.Alert(warn, color="warning", className="rounded-3 mt-2") if warn else html.Div(),
    ])


# ── TAB 5 — Scouting Agent ────────────────────────────────────────────────────
@app.callback(
    Output("agent-history", "data"),
    Output("agent-chat", "children"),
    Output("agent-input", "value"),
    Input("agent-send", "n_clicks"),
    Input({"type": "ex-btn", "index": dash.ALL}, "n_clicks"),
    State("agent-input", "value"),
    State("agent-history", "data"),
    State("sb-club", "value"),
    State("sb-budget", "value"),
    State("sb-wage", "value"),
    prevent_initial_call=True,
)
def handle_agent(send_clicks, ex_clicks, user_q, history, club, budget, wage):
    triggered = ctx.triggered_id

    # Example button clicked
    if isinstance(triggered, dict) and triggered.get("type") == "ex-btn":
        idx = triggered["index"]
        examples = [
            f"Which striker under €20M fits {club} best?",
            f"What is the weakest position in {club}'s squad right now?",
            f"Compare the top 2 central midfield candidates for {club}.",
            "Which position should I prioritise this transfer window?",
        ]
        user_q = examples[idx]

    if not user_q or not str(user_q).strip():
        raise PreventUpdate

    sq, pipe, _ = get_context(club, budget, wage)
    history = history or []
    history.append({"role": "user", "content": user_q})

    try:
        res = run_scouting_agent(
            query=user_q, club=club, squad_df=sq,
            market_df=market_df, model_pipe=pipe,
            transfer_budget=budget, wage_capacity=wage,
            history=[h for h in history[:-1] if h["role"] in ("user", "assistant")])
        for tc in res.get("tool_calls", []):
            arg_str = ", ".join(f"{k}={v}" for k, v in tc["args"].items())
            history.append({"role": "tool_call",
                            "content": f'{tc["tool"]}({arg_str})'})
        history.append({"role": "assistant", "content": res["answer"]})
    except Exception as e:
        history.append({"role": "assistant", "content": f"⚠️ Agent error: {e}"})

    return history, _render_chat(history), ""


@app.callback(
    Output("agent-history", "data", allow_duplicate=True),
    Output("agent-chat", "children", allow_duplicate=True),
    Input("agent-clear", "n_clicks"),
    prevent_initial_call=True,
)
def clear_chat(n):
    return [], [_empty_state("🤖", "Agent ready",
                             "Ask any transfer question. The agent calls simulation "
                             "and shortlist tools autonomously.")]


def _render_chat(history):
    if not history:
        return [_empty_state("🤖", "Agent ready",
                             "Ask any transfer question. The agent calls simulation "
                             "and shortlist tools autonomously.")]
    msgs = []
    for m in history:
        if m["role"] == "user":
            msgs.append(html.Div(m["content"], className="msg-user"))
        elif m["role"] == "tool_call":
            msgs.append(html.Div(f"⚙ {m['content']}", className="msg-tool"))
        elif m["role"] == "assistant":
            msgs.append(html.Div(
                dcc.Markdown(m["content"], style={"margin": 0}),
                className="msg-ai"))
    return msgs


# ── TAB 6 — Model Card ────────────────────────────────────────────────────────
@app.callback(
    Output("t6-content", "children"),
    Input("sb-club", "value"),
    Input("sb-budget", "value"),
    Input("sb-wage", "value"),
)
def render_model_card(club, budget, wage):
    _, pipe, metrics = get_context(club, budget, wage)
    imp = get_feature_importance(pipe, top_k=15)

    feat_labels = [_fmt_feat(f) for f in imp["feature"]]
    fig = go.Figure(go.Bar(x=imp["abs_coefficient"], y=feat_labels,
                           orientation="h", marker_color="#3b82f6",
                           marker_line_width=0))
    fig.update_layout(**{**CHART, "height": 440,
                          "xaxis": {"title": "|Coefficient|", "gridcolor": _GRID,
                                    "linecolor": _LINE, "tickfont": dict(color=_TEXT)},
                          "yaxis": {"gridcolor": _GRID, "linecolor": _LINE,
                                    "tickfont": dict(color=_TEXT)},
                          "showlegend": False})

    auc = metrics.get("auc", "N/A")
    acc = metrics.get("accuracy", "N/A")
    n   = metrics.get("n", "N/A")

    return html.Div([
        html.Div("What the Model Predicts", className="sh"),
        dcc.Markdown("""
The ML layer estimates **P(Positive Uplift)**: the probability that a candidate's projected
performance index exceeds the selected club's position-specific baseline.

**Model:** Logistic Regression (interpretable by design).
**Features:** age, minutes, xG, xA, progressive passes, defensive actions, injury risk,
market value, wage, position (one-hot encoded).
        """),
        dbc.Row([
            dbc.Col(_kpi("Training N",  str(n)),   width=4),
            dbc.Col(_kpi("AUC",         str(auc)), width=4),
            dbc.Col(_kpi("Accuracy",    str(acc)), width=4),
        ], className="g-2 mb-3"),
        metrics.get("note") and html.Div(metrics["note"],
                                          style={"fontSize": "12px", "color": "#8b949e"}),
        html.Div("Global Feature Importance", className="sh"),
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
        _chart_note("These bars show which player attributes have the strongest influence on the model's transfer success prediction. Longer bar = more important. This is a global view across all players — it shows what the model considers most relevant when deciding whether a transfer will generate positive performance uplift for this club."),
        html.Div("Limitations & Scope", className="sh"),
        dcc.Markdown("""
- Trained on **synthetic** data; not validated against real transfer outcomes.
- Uplift label derived from the projection function, not observed post-transfer data.
- Coefficients reflect correlations in the synthetic world, not causal claims.
- No contract or agent-fee modelling.
- Intended for **decision-support prototyping**, not production scouting systems.
        """),
    ])


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    debug = os.environ.get("RENDER") is None  # debug off on Render, on locally
    app.run(host="0.0.0.0", port=port, debug=debug, dev_tools_ui=False)
