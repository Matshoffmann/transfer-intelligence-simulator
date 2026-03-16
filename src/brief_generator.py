"""
src/brief_generator.py — AI Transfer Brief Generator

Second LLM feature for Assignment 2.

Non-straightforward aspects (per assignment requirements):
  1. DATA ASSEMBLY: Aggregates outputs from 5 independent model layers
     (performance index, projection model, financial risk engine, ML success model,
     decision engine) into a structured analytical context object.
  2. CONTEXTUAL ENRICHMENT: Computes squad-level peer statistics (depth, avg age,
     avg performance index for the candidate's position) and injects them into the prompt
     alongside budget utilisation percentages and financial feasibility flags.
  3. SOPHISTICATED PROMPT ENGINEERING: Uses a role-specific system prompt (senior
     recruitment analyst persona), quantitative grounding, hard output-format constraints
     (JSON with 6 named sections), and domain-specific language requirements.
  4. STRUCTURED OUTPUT PARSING: LLM response is expected as JSON.  A two-pass
     extraction strategy (direct parse → regex block extraction) handles cases where the
     model wraps JSON in prose or markdown fences.
  5. GRACEFUL DEGRADATION: If JSON parsing fails entirely, key quantitative facts are
     assembled into a structured fallback response so the UI always shows something useful.
"""

from __future__ import annotations
import os
import re
import json

try:
    import cohere
    _COHERE_AVAILABLE = True
except ImportError:
    _COHERE_AVAILABLE = False


# ── Context assembly ─────────────────────────────────────────────────────────

def _build_context(player_row, bundle, squad_df, club, budget_m, wage_cap_m):
    """Aggregate all model layer outputs into a single analytical context dict."""
    fit        = float(bundle["fit"])
    risk       = float(bundle["risk"])
    risk_lvl   = bundle["decision"]["risk_level"]
    rec        = bundle["decision"]["recommendation"]
    uplift     = float(bundle["uplift"]["uplift"])
    u_lower    = float(bundle["uplift"]["uplift_lower"])
    u_upper    = float(bundle["uplift"]["uplift_upper"])
    uncertainty= float(bundle["uplift"]["uncertainty_sigma"])
    p_success  = float(bundle.get("p_success") or 0.5)
    exp_uplift = float(bundle["expected_uplift"])
    reasons    = bundle["decision"]["reasons"]

    # Squad peer analysis
    pos_peers      = squad_df[squad_df["position"] == player_row["position"]]
    peer_count     = int(len(pos_peers))
    avg_peer_age   = float(pos_peers["age"].mean())   if peer_count > 0 else 26.0
    avg_peer_perf  = float(pos_peers["performance_index"].mean()) if peer_count > 0 else 0.0

    # Financial context
    market_val  = float(player_row["market_value_m"])
    est_fee     = market_val * 1.1
    wage        = float(player_row["estimated_wage_m"])
    budget_pct  = (est_fee / budget_m * 100) if budget_m > 0 else 999.0
    wage_pct    = (wage / wage_cap_m * 100)   if wage_cap_m > 0 else 999.0
    affordable  = est_fee <= budget_m and wage <= wage_cap_m
    feasibility = (
        f"FEASIBLE — fee is {budget_pct:.0f}% of budget, wage is {wage_pct:.0f}% of capacity"
        if affordable else
        f"EXCEEDS CONSTRAINTS — fee is {budget_pct:.0f}% of budget / wage is {wage_pct:.0f}% of capacity"
    )

    return dict(
        name         = str(player_row["name"]).replace("_", " "),
        pos          = str(player_row["position"]),
        age          = int(player_row["age"]),
        minutes      = int(player_row["minutes"]),
        xg           = float(player_row["xg"]),
        xa           = float(player_row["xa"]),
        prog_passes  = float(player_row["progressive_passes"]),
        def_actions  = float(player_row["defensive_actions"]),
        injury_risk  = float(player_row["injury_risk"]),
        market_val   = market_val,
        est_fee      = est_fee,
        wage         = wage,
        budget_pct   = budget_pct,
        wage_pct     = wage_pct,
        affordable   = affordable,
        feasibility  = feasibility,
        fit          = fit,
        risk         = risk,
        risk_lvl     = risk_lvl,
        rec          = rec,
        uplift       = uplift,
        u_lower      = u_lower,
        u_upper      = u_upper,
        uncertainty  = uncertainty,
        p_success    = p_success,
        exp_uplift   = exp_uplift,
        reasons      = reasons,
        peer_count   = peer_count,
        avg_peer_age = avg_peer_age,
        avg_peer_perf= avg_peer_perf,
        club         = club,
        budget       = budget_m,
        wage_cap     = wage_cap_m,
    )


# ── Prompt construction ───────────────────────────────────────────────────────

def _build_prompts(ctx):
    """Build system and user prompts. Returns (system_str, user_str)."""
    system = (
        "You are a senior football recruitment analyst at a top European club. "
        "Your scouting briefs are read by the sporting director and chairman. "
        "Style: precise, authoritative, concise — cite exact numbers from the data provided, "
        "use football industry terminology, never use generic filler sentences. "
        "Produce ONLY a valid JSON object — no markdown code fences, no prose outside the JSON."
    )

    reasons_str = "; ".join(ctx["reasons"])
    uncertainty_note = (
        "high data uncertainty (few minutes)" if ctx["uncertainty"] > 0.25 else
        "moderate uncertainty" if ctx["uncertainty"] > 0.15 else
        "low uncertainty (high-minute player)"
    )

    user = f"""Write a professional transfer scouting brief for {ctx['name']} \
({ctx['pos']}, age {ctx['age']}) — target club {ctx['club']}.

TRANSFER CONSTRAINTS:
• Budget: €{ctx['budget']}M | Estimated fee: €{ctx['est_fee']:.1f}M | {ctx['feasibility']}
• Wage capacity: €{ctx['wage_cap']}M/yr | Player wage: €{ctx['wage']:.1f}M/yr

ANALYTICS PIPELINE OUTPUTS:
• Tactical Fit Score: {ctx['fit']:.1f}/100
• Financial Risk Score: {ctx['risk']:.1f}/100 ({ctx['risk_lvl']})
• Projected Uplift vs Squad: {ctx['uplift']:+.3f} (range {ctx['u_lower']:+.3f} → {ctx['u_upper']:+.3f}; {uncertainty_note})
• ML Success Probability P(Positive Uplift): {ctx['p_success']:.1%}
• Risk-Adjusted Expected Uplift: {ctx['exp_uplift']:+.3f}
• System Decision: {ctx['rec']}
• Key Drivers: {reasons_str}

PLAYER PROFILE:
• Minutes: {ctx['minutes']:,} | xG: {ctx['xg']:.2f} | xA: {ctx['xa']:.2f}
• Progressive Passes: {ctx['prog_passes']:.1f} | Defensive Actions: {ctx['def_actions']:.1f}
• Injury Risk Index: {ctx['injury_risk']:.0%} | Market Value: €{ctx['market_val']:.1f}M

SQUAD CONTEXT ({ctx['pos']} role at {ctx['club']}):
• Current {ctx['pos']} depth: {ctx['peer_count']} player(s)
• Average age at position: {ctx['avg_peer_age']:.1f} | Average performance index: {ctx['avg_peer_perf']:.3f}

Respond with ONLY this JSON (no code fences):
{{
  "executive_summary": "2-3 sentences for the sporting director — lead with the recommendation \
and cite the fit score and key risk",
  "performance_verdict": "2-3 sentences on the player's performance profile and specific \
tactical fit for {ctx['club']} — reference xG, xA or progressive passes as appropriate",
  "financial_assessment": "2-3 sentences on fee feasibility, wage sustainability, and \
value-for-money — use concrete numbers",
  "risk_factors": "2-3 sentences naming the top 2 risk factors with specific figures",
  "recommendation": "1-2 sentences with a direct, actionable recommendation and a \
concrete next step for the Sporting Director",
  "confidence": "High or Medium or Low — based on minutes played and uncertainty band width"
}}"""

    return system, user


# ── Response parsing ──────────────────────────────────────────────────────────

def _parse_response(raw: str, ctx: dict):
    """Extract JSON from LLM response; return (brief_dict, warning_str|None)."""
    # Strip markdown code fences if present
    clean = re.sub(r"```(?:json)?", "", raw).strip()

    # Attempt 1: direct JSON parse
    try:
        return json.loads(clean), None
    except json.JSONDecodeError:
        pass

    # Attempt 2: extract the first {...} block
    m = re.search(r"\{[\s\S]*\}", clean)
    if m:
        try:
            return json.loads(m.group()), None
        except json.JSONDecodeError:
            pass

    # Fallback: structure raw text
    fallback = {
        "executive_summary": (raw[:400] + "…") if len(raw) > 400 else raw,
        "performance_verdict": (
            f"Tactical Fit Score: {ctx['fit']:.1f}/100. "
            f"Projected uplift: {ctx['uplift']:+.3f} (range {ctx['u_lower']:+.3f} → {ctx['u_upper']:+.3f})."
        ),
        "financial_assessment": (
            f"Estimated fee: €{ctx['est_fee']:.1f}M ({ctx['budget_pct']:.0f}% of budget). "
            f"Wage: €{ctx['wage']:.1f}M/yr ({ctx['wage_pct']:.0f}% of capacity). {ctx['feasibility']}."
        ),
        "risk_factors": "; ".join(ctx["reasons"]),
        "recommendation": ctx["rec"],
        "confidence": "Low",
    }
    return fallback, "JSON parse failed — degraded brief shown"


# ── Public API ────────────────────────────────────────────────────────────────

def generate_player_brief(player_row, bundle, squad_df, club, budget_m, wage_cap_m):
    """
    Generate a structured scouting brief via Cohere Command R+.

    Returns:
        (brief_dict, warning_str | None)
        brief_dict has keys: executive_summary, performance_verdict,
          financial_assessment, risk_factors, recommendation, confidence
        warning_str is None on clean success, a string on degraded output.
    """
    if not _COHERE_AVAILABLE:
        return None, "cohere package not installed. Run: pip install cohere"

    # Load API key (same pattern as scouting_agent.py)
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
        return None, "COHERE_API_KEY not set. Add it to your .env file or Streamlit Secrets."

    ctx = _build_context(player_row, bundle, squad_df, club, budget_m, wage_cap_m)
    system_prompt, user_prompt = _build_prompts(ctx)

    try:
        co = cohere.ClientV2(api_key=api_key)
        response = co.chat(
            model="command-r-plus-08-2024",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.25,
        )
        # Extract text from content blocks (same pattern as scouting_agent.py)
        raw = ""
        try:
            for block in response.message.content:
                if hasattr(block, "text"):
                    raw += block.text
        except Exception:
            raw = str(response.message.content)

        brief, warning = _parse_response(raw, ctx)
        return brief, warning

    except Exception as e:
        return None, f"Cohere API error: {e}"
