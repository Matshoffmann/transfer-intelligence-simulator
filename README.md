# Transfer Intelligence Simulator — Bundesliga Edition
Decision Support Prototype for Budget-Constrained Recruitment · ESADE Assignment 2

**GitHub Repository:**
https://github.com/Matshoffmann/transfer-intelligence-simulator

---

## 1. Overview

The Transfer Intelligence Simulator is a Plotly Dash prototype that supports transfer decision-making for Bundesliga clubs under financial constraints. The app models a realistic recruitment workflow, incorporating five analytical model layers and two non-straightforward LLM integrations powered by Cohere Command R+.

The intended user is a Sporting Director or Head of Recruitment who needs a structured, transparent, and reproducible way to move from a large candidate pool to a defensible shortlist and transfer decision.

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

### Tab 1: Player Intelligence
Evaluate any market candidate against a selected Bundesliga club's squad:
- Tactical fit score (position + performance profile match)
- Projected uplift vs squad baseline with uncertainty band
- Financial risk score (budget, wage, injury risk)
- ML success probability P(Positive Uplift)
- Proceed / Monitor / Avoid recommendation with rationale
- Radar chart vs squad average
- Feature contribution chart (log-odds, explainable ML)

### Tab 2: Transfer Comparison
Head-to-head A/B evaluation of two candidates under identical constraints:
- Risk-adjusted verdict with winner card
- Side-by-side score bars
- Comparative analytics bar chart

### Tab 3: Market Shortlist Generator
Scan all market players filtered by position, budget, and wage cap:
- Risk-adjusted ranking
- Opportunity scatter (Financial Risk vs Expected Uplift)
- Exportable results table

### Tab 4: AI Transfer Brief (LLM Feature 1)
Generates a structured executive scouting brief via Cohere Command R+.

Non-straightforward aspects:
1. All 5 model layers run and their outputs are assembled into a structured context object
2. Squad peer statistics (depth, average age, average performance index) are computed and injected into the prompt
3. A role-specific system prompt (senior recruitment analyst persona) with hard JSON output format constraints is constructed
4. Two-pass JSON extraction handles cases where the model wraps output in markdown or prose
5. Graceful degradation provides a fallback structured response if parsing fails

### Tab 5: Transfer Intelligence Agent (LLM Feature 2)
A multi-turn conversational agent backed by Cohere Command R+ with tool calling.

Non-straightforward aspects:
1. Multi-call architecture: the agent runs a full reasoning loop (think, call tool, observe result, continue)
2. Two custom tools are available: `simulate_transfer` runs the full 5-layer pipeline for any player/club combination; `generate_shortlist` runs the position-filtered, budget-constrained shortlist engine
3. The agent decides autonomously which tool to call and in what order based on the user query
4. Tool results are injected back into the conversation as observations; the agent synthesises them into a final analytical response
5. Full tool call history is rendered in the UI for transparency

### Tab 6: Model Card
Interpretable ML layer documentation:
- Global feature importance (logistic regression coefficients)
- Training metrics (N, AUC, Accuracy)
- Limitations and scope

---

## 4. Analytics Pipeline

```
data_loader          Load synthetic Bundesliga market dataset
feature_engineering  Compute performance_index from stat proxies
projection_model     Compute uplift vs club-specific positional baseline
financial_model      Compute financial risk score (budget, wage, injury)
decision_engine      Convert fit + risk into recommendation + rationale
ml_model             Train logistic regression per club: P(Positive Uplift)
brief_generator      Assemble context and generate LLM scouting brief
scouting_agent       Multi-turn agent with tool-calling over simulation engine
```

---

## 5. Data

The market dataset is synthetic by design to ensure no external API dependency and consistent demo results. It is calibrated to resemble plausible football distributions: position-specific stat ranges, age-market value curves, wage proportional to market value, and minutes as a reliability proxy.

---

## 6. Project Structure

```
transfer_intelligence_simulator/
├── app/
│   ├── app_dash.py          Main Dash application (entry point)
│   └── assets/
│       └── style.css        Full dark theme design system
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── projection_model.py
│   ├── financial_model.py
│   ├── decision_engine.py
│   ├── shortlist.py
│   ├── ml_model.py
│   ├── brief_generator.py   LLM Feature 1: AI Transfer Brief
│   └── scouting_agent.py    LLM Feature 2: Tool-calling Agent
├── data/
│   └── processed/
│       └── bundesliga_players.csv
├── requirements.txt
└── README.md
```

---

## 7. Limitations

- Trained on synthetic data; not validated against real transfer outcomes
- Uplift label derived from projection logic, not observed post-transfer data
- No contract length, agent fees, or sell-on clauses modelled
- Intended for decision-support prototyping, not production scouting systems
