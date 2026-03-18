# Transfer Intelligence Simulator — Bundesliga Edition
Decision Support Prototype for Budget-Constrained Recruitment · ESADE Assignment 2

**Live Demo:**
https://transfer-intelligence-simulator.onrender.com/

> Note: The app is hosted on Render's free tier. On first load after inactivity, the server spins up in approximately 50–60 seconds. Subsequent interactions are immediate.

**GitHub Repository:**
https://github.com/Matshoffmann/transfer-intelligence-simulator

---

## 1. Overview

The Transfer Intelligence Simulator is a Plotly Dash prototype that supports transfer decision-making for Bundesliga clubs under financial constraints. The app models a realistic recruitment workflow, incorporating five analytical model layers and two non-straightforward LLM integrations powered by Cohere Command R+.

The intended user is a Sporting Director or Head of Recruitment who needs a structured, transparent, and reproducible way to move from a large candidate pool to a defensible shortlist and transfer decision.

The dataset contains 277 real Bundesliga 2025/26 players (Manuel Neuer, Harry Kane, Jamal Musiala, etc.) with synthetically generated performance statistics calibrated to realistic distributions.

---

## 2. How to Run Locally

```bash
# 1. Create environment
micromamba create -n tis_bundesliga python=3.11 -y
micromamba activate tis_bundesliga

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Cohere API key
# Create a .env file in the project root:
# COHERE_API_KEY=your_key_here

# 4. Run the app
python app/app_dash.py
# Open http://127.0.0.1:8050
```

---

## 3. Features

### Tab 1: Player Analysis
Evaluate any market candidate against a selected Bundesliga club's squad:
- Tactical fit score (position + performance profile match)
- Projected uplift vs squad baseline with uncertainty band
- Financial risk score (budget, wage, injury risk)
- ML success probability P(Positive Uplift)
- Sign / Watch / Pass recommendation with rationale
- Color-coded score bars (green = good, yellow = moderate, red = risk)
- Radar chart vs squad average (Goals, Assists, Progressive Passes, Defensive Actions, Minutes Played)
- Feature contribution chart (log-odds, explainable ML)

### Tab 2: Head-to-Head
Side-by-side A/B evaluation of two candidates under identical constraints:
- Risk-adjusted verdict with winner card
- Side-by-side color-coded score bars
- Comparative analytics bar chart

### Tab 3: Market Scan
Scan all market players filtered by position, budget, and wage cap:
- Risk-adjusted ranking
- Opportunity scatter (Financial Risk vs Expected Uplift)
- Exportable results table

### Tab 4: Scouting Report (LLM Feature 1)
Generates a structured executive scouting brief via Cohere Command R+.

Non-straightforward aspects:
1. All 5 model layers run and their outputs are assembled into a structured context object
2. Squad peer statistics (depth, average age, average performance index) are computed and injected into the prompt
3. A role-specific system prompt (senior recruitment analyst persona) with hard JSON output format constraints is constructed
4. Two-pass JSON extraction handles cases where the model wraps output in markdown or prose
5. Graceful degradation provides a fallback structured response if parsing fails

### Tab 5: Ask the Scout (LLM Feature 2)
A multi-turn conversational agent backed by Cohere Command R+ with tool calling.

Non-straightforward aspects:
1. Multi-call architecture: the agent runs a full reasoning loop (think, call tool, observe result, continue)
2. Two custom tools are available: `simulate_transfer` runs the full 5-layer pipeline for any player/club combination; `generate_shortlist` runs the position-filtered, budget-constrained shortlist engine
3. The agent decides autonomously which tool to call and in what order based on the user query
4. Tool results are injected back into the conversation as observations; the agent synthesises them into a final analytical response
5. Full tool call history is rendered in the UI for transparency

### Tab 6: How It Works
Interpretable ML layer documentation:
- Global feature importance (logistic regression coefficients)
- Training metrics (N, AUC, Accuracy)
- Model card with limitations and scope

---

## 4. Analytics Pipeline

```
data_loader          Load Bundesliga player dataset (277 real 2025/26 players)
feature_engineering  Compute performance_index from core stat proxies
projection_model     Compute uplift vs club-specific positional baseline
financial_model      Compute financial risk score (budget, wage, injury)
decision_engine      Convert fit + risk into Sign / Watch / Pass recommendation
ml_model             Train logistic regression per club: P(Positive Uplift)
brief_generator      Assemble context and generate LLM scouting brief
scouting_agent       Multi-turn agent with tool-calling over simulation engine
```

---

## 5. Data

The dataset uses real Bundesliga 2025/26 player names and positions (277 players across all 18 clubs), with synthetically generated performance statistics. This approach ensures no external API dependency and consistent demo results, while making the prototype feel grounded in the real league context. The architecture allows straightforward replacement with real event-level data (e.g., FBref, StatsBomb).

---

## 6. Project Structure

```
transfer_intelligence_simulator/
├── app/
│   ├── app_dash.py          Main Dash application (entry point)
│   └── assets/
│       └── style.css        Full dark theme design system (~650 lines)
├── src/
│   ├── data_loader.py       Dataset generation with real player names
│   ├── feature_engineering.py
│   ├── projection_model.py
│   ├── financial_model.py
│   ├── decision_engine.py   Sign / Watch / Pass decision logic
│   ├── shortlist.py
│   ├── ml_model.py
│   ├── brief_generator.py   LLM Feature 1: Scouting Report
│   └── scouting_agent.py    LLM Feature 2: Tool-calling Agent
├── data/
│   └── processed/
│       └── bundesliga_players.csv
├── requirements.txt
└── README.md
```

---

## 7. Limitations

- Performance statistics are synthetic; not validated against real match data
- Uplift label derived from projection logic, not observed post-transfer outcomes
- No contract length, agent fees, or sell-on clauses modelled
- Intended for decision-support prototyping, not production scouting systems
- Hosted on Render free tier: cold-start latency of ~60 seconds after inactivity
