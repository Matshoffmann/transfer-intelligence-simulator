# Transfer Intelligence Simulator (Bundesliga Edition)
Decision Support Prototype for Budget-Constrained Recruitment

**Live App:**  
https://transfer-intelligence-simulator-gszqxgguhhieqzifhrfox5.streamlit.app  

**GitHub Repository:**  
https://github.com/Matshoffmann/transfer-intelligence-simulator  

## 1) Overview

The Transfer Intelligence Simulator is a Streamlit-based product prototype that supports transfer decision-making for Bundesliga clubs under financial constraints. Instead of simply displaying metrics, the app models a realistic decision workflow:

- club-specific squad context (baseline differs by club and by position)
- tactical fit and projected performance uplift (with uncertainty)
- financial feasibility and risk
- A/B candidate comparison
- position-based shortlist generation with export
- an interpretable ML layer that estimates **P(Positive Uplift)**

The intended user is a Sporting Director / Head of Recruitment who needs a structured, transparent, and reproducible way to move from “many candidates” to a defensible shortlist and decision.

## 2) Key Product Idea (Why this is not “just a dashboard”)

Transfer decisions are multi-constraint decisions:

- A player can be “good” but not improve the club’s current baseline at that position.
- A player can improve performance but break wage structure or exceed budget.
- A player’s projected impact can be highly uncertain when minutes played are low.

This prototype formalizes that logic into a decision-support flow:

1) Set context (club + constraints)  
2) Evaluate candidate(s) vs that context  
3) Produce a recommendation and rationale  
4) Shortlist feasible targets under the same constraints  


## 3) Core Features

### 3.1 Club Context (Bundesliga)

- Select a Bundesliga club in the sidebar
- A reproducible club squad is generated (synthetic, calibrated) so:
  - squad baselines are club-specific
  - tactical fit and uplift are computed against *this* squad

### 3.2 Single Transfer Simulation

For one target player:

- Tactical fit score
- Projected uplift vs squad baseline
- Uncertainty band (minutes as reliability proxy)
- Financial risk score (budget, wage capacity, injury risk)
- ML success probability
- Decision recommendation (Proceed / Monitor / Avoid) + rationale

### 3.3 A/B Comparison

Compare two candidates under identical constraints:

- Side-by-side metrics (fit, risk, P(success), expected uplift)
- Comparison chart (expected uplift vs scaled risk)
- Risk-adjusted recommendation

### 3.4 Shortlist Generator (+ Export)

- Choose position (GK / CB / FB / DM / CM / AM / ST)
- Choose Top-N
- Feasibility filtering (budget and wage capacity)
- Risk-adjusted ranking
- Export shortlist to CSV via `st.download_button`

## 4) Data and Assumptions

### 4.1 Synthetic dataset (reproducible prototype)

The market dataset is synthetic by design to ensure:

- no external API downtime
- stable demo and consistent results

The synthetic data is calibrated to resemble plausible football distributions:

- position-specific stat distributions
- age–market value curve (prime age effect)
- wage roughly linked to market value
- injury risk correlated with age
- minutes used as a reliability proxy

## 5) Modeling Approach (Pipeline)

High-level pipeline:

1) `data_loader`  
   Generates / loads a synthetic market dataset  

2) `feature_engineering`  
   Computes a `performance_index` from core stat proxies  

3) `projection_model`  
   Computes:
   - baseline performance by position within the selected club squad  
   - candidate performance  
   - uplift = candidate − baseline  
   - uncertainty band (minutes-based reliability proxy)  

4) `financial_model`  
   Computes a financial risk score using:
   - market value vs transfer budget  
   - wage vs wage capacity  
   - injury risk  

5) `decision_engine`  
   Converts fit + risk into:
   - decision score  
   - recommendation (Proceed / Monitor / Avoid)  
   - human-readable rationale  

6) ML layer (`ml_model`)  
   Trains an interpretable logistic regression per club context to estimate:
   - **P(Positive Uplift)** = probability that uplift > 0 vs the club’s baseline  

7) Streamlit UI (`app/app.py`)  
   Presents the end-to-end workflow with simulation, comparison and shortlisting  

## 6) ML Layer + Explainability

### 6.1 What the ML model predicts

The model predicts **P(Positive Uplift)**:

- Label definition: `success = 1` if projected uplift > 0, else 0  
- Training is club-context specific because the baseline depends on the selected squad  

### 6.2 Model Type

- Logistic Regression (interpretable by design)
- Numeric features:
  - age
  - minutes
  - xG
  - xA
  - progressive passes
  - defensive actions
  - injury risk
  - market value
  - wage
- Categorical:
  - position (one-hot encoded)

### 6.3 Model Quality (Prototype)

The app displays:

- training size (N)
- AUC
- Accuracy

These metrics are prototype-level metrics on a holdout split within the synthetic dataset.

### 6.4 Explainability

The app includes a dedicated **Model Card & Explainability** section:

- Global importance: absolute logistic coefficients
- Local explanation: feature-level contribution to predicted probability (log-odds space)

This supports transparency and responsible AI prototyping.

## 7) Limitations (Prototype Scope)

- Synthetic training data; not validated on real event-level match data
- “Uplift” derived from projection logic, not observed post-transfer outcomes
- No contract length, agent fees, or sell-on clauses
- No tactical system metadata beyond position features

## 8) Future Extensions (Production Path)

- Replace synthetic market with real snapshot data (e.g., FBref export)
- Validate uplift against real match outcomes
- Separate offline model training from runtime inference
- Add budget sensitivity stress tests
- Extend to multi-season financial simulations

## 9) Project Structure
transfer-intelligence-simulator/
│
├── app/
│ ├── app.py
│ └── init.py
│
├── src/
│ ├── data_loader.py
│ ├── feature_engineering.py
│ ├── projection_model.py
│ ├── similarity.py
│ ├── financial_model.py
│ ├── decision_engine.py
│ ├── shortlist.py
│ ├── ml_model.py
│ └── init.py
│
├── data/
│ ├── raw/
│ └── processed/
│
├── models/
├── requirements.txt
└── README.md

## 10) Run Locally

### 10.1 Create / Activate Environment (micromamba example)

```bash
micromamba create -n tis_bundesliga python=3.11 -y
micromamba activate tis_bundesliga
pip install -r requirements.txt
From the repository root:

streamlit run app/app.py