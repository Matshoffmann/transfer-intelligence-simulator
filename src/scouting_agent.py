# FILE: src/scouting_agent.py
from __future__ import annotations
import os, json
import pandas as pd
import numpy as np

try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

from src.shortlist import compute_bundle, shortlist_top_targets

# Cohere V2 / OpenAI-compatible tool schema
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "simulate_player_transfer",
            "description": "Runs a full transfer simulation for one player vs the club squad. Returns fit score, projected uplift, financial risk, ML success probability, and recommendation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "player_name": {"type": "string", "description": "Name of the player to simulate."}
                },
                "required": ["player_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_shortlist_for_position",
            "description": "Returns a ranked shortlist of the best transfer targets for a position under budget and wage constraints.",
            "parameters": {
                "type": "object",
                "properties": {
                    "position":    {"type": "string",  "description": "Position code: GK, CB, FB, DM, CM, AM, or ST."},
                    "top_n":       {"type": "integer", "description": "Number of candidates (default 3)."},
                    "max_value_m": {"type": "number",  "description": "Optional max market value filter in M euros."}
                },
                "required": ["position"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_two_players",
            "description": "Head-to-head comparison of two players with a risk-adjusted verdict.",
            "parameters": {
                "type": "object",
                "properties": {
                    "player_a": {"type": "string", "description": "First player name."},
                    "player_b": {"type": "string", "description": "Second player name."}
                },
                "required": ["player_a", "player_b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_squad_overview",
            "description": "Returns squad depth and average performance per position. Use to find weaknesses.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
]

def _find_player(name, market_df):
    m = market_df[market_df["name"].str.lower() == name.strip().lower()]
    if len(m) == 0:
        m = market_df[market_df["name"].str.lower().str.contains(name.strip().lower(), na=False)]
    return None if len(m) == 0 else m.iloc[0]

def _tool_simulate(player_name, squad_df, market_df, model_pipe, budget, wage):
    p = _find_player(player_name, market_df)
    if p is None:
        return {"error": f"Player '{player_name}' not found."}
    b = compute_bundle(squad_df, market_df, p, budget, wage, model_pipe=model_pipe)
    return {
        "player_name": p["name"], "position": p["position"], "age": int(p["age"]),
        "market_value_m": round(float(p["market_value_m"]), 2),
        "fit_score": round(float(b["fit"]), 2),
        "uplift": round(float(b["uplift"]["uplift"]), 4),
        "financial_risk": round(float(b["risk"]), 2),
        "p_success": round(float(b["p_success"] or 0.5), 3),
        "expected_uplift": round(float(b["expected_uplift"]), 4),
        "risk_adjusted_value": round(float(b["risk_adjusted_value"]), 4),
        "recommendation": b["decision"]["recommendation"],
        "risk_level": b["decision"]["risk_level"],
        "reasons": b["decision"]["reasons"],
    }

def _tool_shortlist(position, squad_df, market_df, model_pipe, budget, wage, top_n=3, max_value_m=None):
    pos = position.upper().strip()
    if pos not in ["GK","CB","FB","DM","CM","AM","ST"]:
        return {"error": f"Unknown position '{pos}'."}
    eff = min(budget, max_value_m) if max_value_m else budget
    sl = shortlist_top_targets(squad_df, market_df, pos, eff, wage, top_n=top_n, min_minutes=300, model_pipe=model_pipe)
    if len(sl) == 0:
        return {"error": f"No feasible {pos} candidates under current constraints."}
    return {"position": pos, "candidates": sl[["name","age","market_value_m","fit_score",
        "expected_uplift","p_success","financial_risk","recommendation"]].to_dict(orient="records")}

def _tool_compare(pa, pb, squad_df, market_df, model_pipe, budget, wage):
    ra = _tool_simulate(pa, squad_df, market_df, model_pipe, budget, wage)
    rb = _tool_simulate(pb, squad_df, market_df, model_pipe, budget, wage)
    if "error" in ra or "error" in rb:
        return {"error": ra.get("error") or rb.get("error")}
    verdict = (ra["player_name"] if ra["risk_adjusted_value"] > rb["risk_adjusted_value"]
               else (rb["player_name"] if rb["risk_adjusted_value"] > ra["risk_adjusted_value"] else "tie"))
    return {"player_a": ra, "player_b": rb, "verdict": verdict,
            "delta_expected_uplift": round(ra["expected_uplift"] - rb["expected_uplift"], 4)}

def _tool_squad(squad_df):
    s = squad_df.groupby("position").agg(
        count=("name","count"), avg_perf=("performance_index","mean"), avg_age=("age","mean")
    ).reset_index().sort_values("avg_perf")
    return {"squad_size": int(len(squad_df)),
            "position_summary": s.round(3).to_dict(orient="records"),
            "weakest_position": s.iloc[0]["position"],
            "strongest_position": s.iloc[-1]["position"]}

def _dispatch(tool_name, args, squad_df, market_df, model_pipe, budget, wage):
    try:
        if tool_name == "simulate_player_transfer":
            r = _tool_simulate(args.get("player_name",""), squad_df, market_df, model_pipe, budget, wage)
        elif tool_name == "get_shortlist_for_position":
            r = _tool_shortlist(args.get("position","ST"), squad_df, market_df, model_pipe, budget, wage,
                                int(args.get("top_n", 3)), args.get("max_value_m"))
        elif tool_name == "compare_two_players":
            r = _tool_compare(args.get("player_a",""), args.get("player_b",""),
                              squad_df, market_df, model_pipe, budget, wage)
        elif tool_name == "get_squad_overview":
            r = _tool_squad(squad_df)
        else:
            r = {"error": f"Unknown tool: {tool_name}"}
    except Exception as e:
        r = {"error": str(e)}
    return json.dumps(r, default=str)

def run_scouting_agent(query, club, squad_df, market_df, model_pipe,
                       transfer_budget, wage_capacity, history=None):
    if not COHERE_AVAILABLE:
        raise ImportError("cohere not installed. Run: pip install cohere")
    api_key = os.environ.get("COHERE_API_KEY", "")
    if not api_key:
        try:
            from dotenv import load_dotenv
            from pathlib import Path
            env_path = Path(__file__).resolve().parent.parent / ".env"
            try:
                load_dotenv(dotenv_path=env_path, encoding="utf-8")
            except Exception:
                load_dotenv(dotenv_path=env_path, encoding="utf-16")
            api_key = os.environ.get("COHERE_API_KEY", "")
        except ImportError:
            pass
    if not api_key:
        raise ValueError("COHERE_API_KEY not set. Add it to .env or Streamlit Secrets.")

    co = cohere.ClientV2(api_key=api_key)
    system_prompt = (
        f"You are a senior football transfer analyst for {club}. "
        f"Budget: {transfer_budget}M euros | Wage capacity: {wage_capacity}M euros/year. "
        "Always use tools to ground your answers in data. "
        "Respond as a concise professional scouting brief (150-250 words). "
        "Lead with the recommendation, cite exact numbers. "
        "End with one concrete next step for the Sporting Director."
    )
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        for h in history:
            if h["role"] in ("user", "assistant"):
                messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": query})

    tool_calls_log = []
    response = None

    for _ in range(4):
        response = co.chat(model="command-r-plus-08-2024", messages=messages, tools=TOOLS)
        if not response.message.tool_calls:
            break
        messages.append({
            "role": "assistant",
            "tool_calls": [{"id": tc.id, "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in response.message.tool_calls],
            "tool_plan": getattr(response.message, "tool_plan", ""),
            "content": "",
        })
        for tc in response.message.tool_calls:
            try: args = json.loads(tc.function.arguments)
            except: args = {}
            tool_calls_log.append({"tool": tc.function.name, "args": args})
            result_str = _dispatch(tc.function.name, args, squad_df, market_df, model_pipe, transfer_budget, wage_capacity)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_str})

    final = ""
    try:
        for block in response.message.content:
            if hasattr(block, "text"): final += block.text
    except: final = str(response.message.content)
    if not final.strip():
        final = "The agent completed analysis but produced no text. Please try rephrasing."
    return {"answer": final, "tool_calls": tool_calls_log}