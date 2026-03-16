# FILE: src/scouting_agent.py
#
# Non-trivial LLM integration for Assignment 2:
# ─────────────────────────────────────────────
# Architecture: Cohere Command R+ with Tool Use (function calling)
#
# The agent autonomously decides which tools to call, calls them, receives
# structured Python outputs, and synthesises a grounded scouting brief.
#
# Tool loop:
#   1. User asks a natural-language transfer question.
#   2. Cohere returns a tool-use plan (which functions + arguments).
#   3. We execute the tools in Python (calling the real simulation engine).
#   4. Tool results are fed back to Cohere as structured context.
#   5. Cohere produces a final, grounded, cited answer.
#
# This is non-trivial because:
#   - The LLM output is NOT directly shown to the user.
#   - The LLM drives an agentic loop that calls the ML/simulation pipeline.
#   - Structured tool results (JSON) are post-processed before the final answer.
#   - The agent can chain multiple tool calls in one turn.

from __future__ import annotations

import os
import json
import pandas as pd
import numpy as np

try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

from src.shortlist import compute_bundle, shortlist_top_targets
from src.feature_engineering import compute_performance_index

# ─────────────────────────────────────────────
# Tool definitions (schema for Cohere)
# ─────────────────────────────────────────────

TOOLS = [
    {
        "name": "simulate_player_transfer",
        "description": (
            "Runs a full transfer simulation for a single player against the selected club's squad. "
            "Returns fit score, projected uplift, financial risk, ML success probability, "
            "expected uplift, and a decision recommendation (Proceed / Monitor / Avoid). "
            "Use this when the user asks about a specific player."
        ),
        "parameter_definitions": {
            "player_name": {
                "description": "The name of the player to simulate (must match the market dataset).",
                "type": "str",
                "required": True,
            }
        },
    },
    {
        "name": "get_shortlist_for_position",
        "description": (
            "Generates a ranked shortlist of the best transfer targets for a given position "
            "under the current club's budget and wage constraints. Returns the top N candidates "
            "with their fit score, expected uplift, financial risk, and recommendation. "
            "Use this when the user asks for recommendations without naming a specific player."
        ),
        "parameter_definitions": {
            "position": {
                "description": "Position code: one of GK, CB, FB, DM, CM, AM, ST.",
                "type": "str",
                "required": True,
            },
            "top_n": {
                "description": "Number of top candidates to return (default: 3).",
                "type": "int",
                "required": False,
            },
            "max_value_m": {
                "description": "Optional maximum market value filter in M€.",
                "type": "float",
                "required": False,
            },
        },
    },
    {
        "name": "compare_two_players",
        "description": (
            "Runs a head-to-head comparison of exactly two players under the club's current context. "
            "Returns both simulation bundles plus a risk-adjusted verdict. "
            "Use this when the user explicitly asks to compare two players."
        ),
        "parameter_definitions": {
            "player_a": {
                "description": "Name of the first player.",
                "type": "str",
                "required": True,
            },
            "player_b": {
                "description": "Name of the second player.",
                "type": "str",
                "required": True,
            },
        },
    },
    {
        "name": "get_squad_overview",
        "description": (
            "Returns a summary of the current squad's average performance index "
            "and squad depth per position. Use this when the user asks about squad "
            "weaknesses or positions that need strengthening."
        ),
        "parameter_definitions": {},
    },
]


# ─────────────────────────────────────────────
# Tool implementations (Python side)
# ─────────────────────────────────────────────

def _tool_simulate_player(
    player_name: str,
    squad_df: pd.DataFrame,
    market_df: pd.DataFrame,
    model_pipe,
    transfer_budget: float,
    wage_capacity: float,
) -> dict:
    matches = market_df[market_df["name"].str.lower() == player_name.strip().lower()]
    if len(matches) == 0:
        # fuzzy fallback: partial match
        matches = market_df[market_df["name"].str.lower().str.contains(
            player_name.strip().lower(), na=False)]
    if len(matches) == 0:
        return {"error": f"Player '{player_name}' not found in the market dataset."}

    player = matches.iloc[0]
    bundle = compute_bundle(squad_df, market_df, player,
                            transfer_budget, wage_capacity, model_pipe=model_pipe)
    return {
        "player_name":      player["name"],
        "position":         player["position"],
        "age":              int(player["age"]),
        "market_value_m":   round(float(player["market_value_m"]), 2),
        "estimated_wage_m": round(float(player["estimated_wage_m"]), 2),
        "fit_score":        round(float(bundle["fit"]), 2),
        "uplift":           round(float(bundle["uplift"]["uplift"]), 4),
        "uncertainty_sigma":round(float(bundle["uplift"]["uncertainty_sigma"]), 4),
        "financial_risk":   round(float(bundle["risk"]), 2),
        "p_success":        round(float(bundle["p_success"] or 0.5), 3),
        "expected_uplift":  round(float(bundle["expected_uplift"]), 4),
        "risk_adjusted_value": round(float(bundle["risk_adjusted_value"]), 4),
        "recommendation":   bundle["decision"]["recommendation"],
        "risk_level":       bundle["decision"]["risk_level"],
        "reasons":          bundle["decision"]["reasons"],
    }


def _tool_shortlist(
    position: str,
    squad_df: pd.DataFrame,
    market_df: pd.DataFrame,
    model_pipe,
    transfer_budget: float,
    wage_capacity: float,
    top_n: int = 3,
    max_value_m: float | None = None,
) -> dict:
    position = position.upper().strip()
    valid = ["GK", "CB", "FB", "DM", "CM", "AM", "ST"]
    if position not in valid:
        return {"error": f"Unknown position '{position}'. Must be one of {valid}"}

    effective_budget = min(transfer_budget, max_value_m) if max_value_m else transfer_budget

    sl = shortlist_top_targets(
        squad_df=squad_df, market_df=market_df,
        position=position, budget_m=effective_budget,
        wage_capacity_m=wage_capacity, top_n=top_n,
        min_minutes=300, model_pipe=model_pipe
    )

    if len(sl) == 0:
        return {"error": f"No feasible {position} candidates under current constraints."}

    return {
        "position":     position,
        "budget_used_m": round(effective_budget, 1),
        "candidates": sl[[
            "name", "age", "market_value_m", "fit_score",
            "expected_uplift", "p_success", "financial_risk", "recommendation"
        ]].to_dict(orient="records")
    }


def _tool_compare(
    player_a: str,
    player_b: str,
    squad_df: pd.DataFrame,
    market_df: pd.DataFrame,
    model_pipe,
    transfer_budget: float,
    wage_capacity: float,
) -> dict:
    ra = _tool_simulate_player(player_a, squad_df, market_df, model_pipe, transfer_budget, wage_capacity)
    rb = _tool_simulate_player(player_b, squad_df, market_df, model_pipe, transfer_budget, wage_capacity)

    if "error" in ra or "error" in rb:
        return {"error": ra.get("error") or rb.get("error")}

    verdict = "tie"
    if ra["risk_adjusted_value"] > rb["risk_adjusted_value"]:
        verdict = ra["player_name"]
    elif rb["risk_adjusted_value"] > ra["risk_adjusted_value"]:
        verdict = rb["player_name"]

    return {
        "player_a": ra,
        "player_b": rb,
        "risk_adjusted_verdict": verdict,
        "delta_expected_uplift": round(ra["expected_uplift"] - rb["expected_uplift"], 4),
        "delta_risk":            round(ra["financial_risk"]  - rb["financial_risk"],  2),
    }


def _tool_squad_overview(squad_df: pd.DataFrame) -> dict:
    summary = (
        squad_df.groupby("position")
        .agg(
            count=("name", "count"),
            avg_perf=("performance_index", "mean"),
            avg_age=("age", "mean"),
        )
        .reset_index()
        .sort_values("avg_perf")
    )
    return {
        "squad_size": int(len(squad_df)),
        "position_summary": summary.round(3).to_dict(orient="records"),
        "weakest_position": summary.iloc[0]["position"],
        "strongest_position": summary.iloc[-1]["position"],
    }


# ─────────────────────────────────────────────
# Tool dispatcher
# ─────────────────────────────────────────────

def _dispatch_tool(
    tool_name: str,
    tool_args: dict,
    squad_df: pd.DataFrame,
    market_df: pd.DataFrame,
    model_pipe,
    transfer_budget: float,
    wage_capacity: float,
) -> str:
    """Execute a tool and return its result as a JSON string."""
    try:
        if tool_name == "simulate_player_transfer":
            result = _tool_simulate_player(
                tool_args.get("player_name", ""),
                squad_df, market_df, model_pipe,
                transfer_budget, wage_capacity
            )
        elif tool_name == "get_shortlist_for_position":
            result = _tool_shortlist(
                tool_args.get("position", "ST"),
                squad_df, market_df, model_pipe,
                transfer_budget, wage_capacity,
                top_n=int(tool_args.get("top_n", 3)),
                max_value_m=tool_args.get("max_value_m")
            )
        elif tool_name == "compare_two_players":
            result = _tool_compare(
                tool_args.get("player_a", ""),
                tool_args.get("player_b", ""),
                squad_df, market_df, model_pipe,
                transfer_budget, wage_capacity
            )
        elif tool_name == "get_squad_overview":
            result = _tool_squad_overview(squad_df)
        else:
            result = {"error": f"Unknown tool: {tool_name}"}
    except Exception as e:
        result = {"error": f"Tool execution error: {str(e)}"}

    return json.dumps(result, default=str)


# ─────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────

def _build_system_prompt(club: str, transfer_budget: float, wage_capacity: float) -> str:
    return f"""You are a senior football transfer analyst AI assistant for {club}.

Your role is to provide concise, data-driven scouting briefs to the Sporting Director.
You have access to tools that query a live transfer simulation engine.

Current context:
- Club: {club}
- Transfer Budget: {transfer_budget}M€
- Remaining Wage Capacity: {wage_capacity}M€/year

Guidelines:
1. ALWAYS use tools to ground your answer in data — never speculate without tool results.
2. Structure your answers as professional scouting briefs: lead with the recommendation,
   then explain the key numbers (fit, uplift, risk, P(success)), then caveats.
3. Be specific about trade-offs (e.g. high uplift but high risk).
4. Keep answers concise but substantive — 150-300 words is ideal.
5. If comparing players, always mention the risk-adjusted verdict.
6. Reference actual numbers from the tool outputs (fit score, expected uplift, etc.)
7. End with one concrete next step for the Sporting Director.
"""


# ─────────────────────────────────────────────
# Main agent entry point
# ─────────────────────────────────────────────

def run_scouting_agent(
    query: str,
    club: str,
    squad_df: pd.DataFrame,
    market_df: pd.DataFrame,
    model_pipe,
    transfer_budget: float,
    wage_capacity: float,
    history: list[dict] | None = None,
) -> dict:
    """
    Run the Cohere scouting agent with tool use.

    Returns:
        {
            "answer":      str,           # final LLM-synthesised answer
            "tool_calls":  list[dict],    # tools that were called + args
        }
    """
    if not COHERE_AVAILABLE:
        raise ImportError("cohere package not installed. Run: pip install cohere")

    api_key = os.environ.get("COHERE_API_KEY", "")
    if not api_key:
        # Try loading from .env manually
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.environ.get("COHERE_API_KEY", "")
        except ImportError:
            pass

    if not api_key:
        raise ValueError(
            "COHERE_API_KEY not found. "
            "Set it in your .env file or as an environment variable."
        )

    co = cohere.ClientV2(api_key=api_key)

    system_prompt = _build_system_prompt(club, transfer_budget, wage_capacity)

    # Build message list
    messages = [{"role": "system", "content": system_prompt}]

    # Add previous turns (user + assistant only, not tool calls)
    if history:
        for h in history:
            if h["role"] in ("user", "assistant"):
                messages.append({"role": h["role"], "content": h["content"]})

    messages.append({"role": "user", "content": query})

    tool_calls_log = []
    MAX_TOOL_ROUNDS = 4

    for round_i in range(MAX_TOOL_ROUNDS):
        response = co.chat(
            model="command-r-plus-08-2024",
            messages=messages,
            tools=TOOLS,
        )

        # Check finish reason
        finish_reason = getattr(response, "finish_reason", None)

        # If no tool calls → final answer
        if not response.message.tool_calls:
            break

        # ── Tool use round ──
        # 1. Append assistant message with tool_calls
        messages.append({
            "role": "assistant",
            "tool_calls": [
                {
                    "id":       tc.id,
                    "type":     "function",
                    "function": {
                        "name":      tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in response.message.tool_calls
            ],
            "tool_plan": getattr(response.message, "tool_plan", ""),
            "content":   "",
        })

        # 2. Execute tools + collect results
        tool_results = []
        for tc in response.message.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                args = {}

            tool_calls_log.append({"tool": tc.function.name, "args": args})

            result_str = _dispatch_tool(
                tc.function.name, args,
                squad_df, market_df, model_pipe,
                transfer_budget, wage_capacity
            )

            tool_results.append({
                "role":        "tool",
                "tool_call_id": tc.id,
                "content":     result_str,
            })

        messages.extend(tool_results)

    # Extract final text answer
    final_answer = ""
    try:
        for block in response.message.content:
            if hasattr(block, "text"):
                final_answer += block.text
    except Exception:
        final_answer = str(response.message.content)

    if not final_answer.strip():
        final_answer = "The agent completed tool calls but produced no text response. Please try rephrasing."

    return {
        "answer":     final_answer,
        "tool_calls": tool_calls_log,
    }
