# FILE: src/data_loader.py

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/processed/bundesliga_players.csv")

# Synthetic but realistic-looking German(-ish) name pool
FIRST_NAMES = [
    "Jonas", "Lukas", "Felix", "Noah", "Max", "Leon", "Tim", "Paul", "Nico", "Julian",
    "Marco", "David", "Simon", "Tobias", "Emil", "Moritz", "Kevin", "Philipp", "Daniel", "Robin",
    "Jan", "Florian", "Sebastian", "Adrian", "Fabian", "Matteo", "Lucas", "Elias", "Milan", "Aaron"
]

LAST_NAMES = [
    "Schneider", "Fischer", "Weber", "Meyer", "Wagner", "Becker", "Hoffmann", "Schulz", "Koch", "Bauer",
    "Richter", "Klein", "Wolf", "Neumann", "Schwarz", "Zimmermann", "Krüger", "Hartmann", "Lange", "Schmitt",
    "Krause", "Werner", "Schuster", "Vogel", "Fuchs", "Peters", "Arnold", "Bergmann", "Keller", "Frank"
]


def random_name(i: int) -> str:
    """
    Deterministic name mapping: stable across runs for the same i.
    """
    fn = FIRST_NAMES[i % len(FIRST_NAMES)]
    ln = LAST_NAMES[(i * 7) % len(LAST_NAMES)]
    return f"{fn} {ln}"


def generate_bundesliga_dataset(n_players: int = 250, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic Bundesliga-like market dataset (semi-realistic).
    - Positions are sampled with plausible frequencies.
    - Stats distributions differ by position.
    - Market value peaks around prime age and is linked to performance proxies.
    - Wages correlate with market value.
    - Injury risk increases slightly with age + noise.
    """
    rng = np.random.default_rng(random_state)

    positions = ["GK", "CB", "FB", "DM", "CM", "AM", "ST"]
    pos_probs = [0.08, 0.18, 0.14, 0.14, 0.18, 0.14, 0.14]

    rows = []

    for i in range(n_players):
        position = rng.choice(positions, p=pos_probs)
        age = int(rng.integers(18, 34))
        minutes = int(rng.integers(200, 3200))

        # --- Performance metrics (position-dependent) ---
        # xG: higher for attackers; xA: higher for AM/CM; progression: mid/wing-backs; defense: CB/DM
        if position in ["ST", "AM"]:
            xg = float(rng.gamma(2.2, 0.20))
        else:
            xg = float(rng.gamma(1.1, 0.10))

        if position in ["AM", "CM", "FB"]:
            xa = float(rng.gamma(2.0, 0.14))
        else:
            xa = float(rng.gamma(1.0, 0.05))

        if position in ["CM", "AM", "FB", "DM"]:
            progressive_passes = float(rng.normal(3.2, 1.0))
        else:
            progressive_passes = float(rng.normal(2.4, 0.9))

        if position in ["CB", "DM", "FB"]:
            defensive_actions = float(rng.normal(4.4, 1.3))
        elif position == "GK":
            defensive_actions = float(rng.normal(3.0, 1.0))  # proxy (since we don't model saves)
        else:
            defensive_actions = float(rng.normal(3.2, 1.2))

        # Clip to avoid weird negatives from normals
        progressive_passes = float(np.clip(progressive_passes, 0.0, 8.0))
        defensive_actions = float(np.clip(defensive_actions, 0.0, 8.0))

        # --- Market value model (semi-realistic heuristic) ---
        # Prime-age peak around 26-28; slight linkage to "activity" (minutes) and attacking contribution.
        prime_factor = 1 - abs(age - 27) / 18  # ~[0..1]
        prime_factor = float(np.clip(prime_factor, 0.2, 1.0))

        activity_factor = float(np.clip(minutes / 2800, 0.2, 1.0))

        contribution = 0.6 * xg + 0.4 * xa + 0.15 * progressive_passes
        contribution = float(np.clip(contribution, 0.0, 3.0))

        base_value = float(rng.uniform(1.0, 35.0))
        market_value_m = base_value * prime_factor * (0.65 + 0.35 * activity_factor) * (0.8 + 0.25 * contribution)

        # Keep values within a plausible Bundesliga range
        market_value_m = float(np.clip(market_value_m, 0.5, 70.0))

        # --- Wage estimation (linked to market value) ---
        # Approx: wage_m is in M€/year (not monthly)
        wage_multiplier = float(rng.uniform(0.05, 0.12))
        estimated_wage_m = float(np.clip(market_value_m * wage_multiplier, 0.2, 15.0))

        # --- Injury risk (simple proxy) ---
        # increases with age, plus noise; clipped [0..1]
        injury_risk = float(np.clip((age - 27) / 10 + rng.normal(0, 0.18), 0, 1))

        rows.append([
            f"BL{i:04d}",             # player_id
            random_name(i),           # name
            age,
            position,
            minutes,
            xg,
            xa,
            progressive_passes,
            defensive_actions,
            market_value_m,
            estimated_wage_m,
            injury_risk
        ])

    df = pd.DataFrame(rows, columns=[
        "player_id",
        "name",
        "age",
        "position",
        "minutes",
        "xg",
        "xa",
        "progressive_passes",
        "defensive_actions",
        "market_value_m",
        "estimated_wage_m",
        "injury_risk"
    ])

    return df


def save_dataset(n_players: int = 250, random_state: int = 42) -> None:
    df = generate_bundesliga_dataset(n_players=n_players, random_state=random_state)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Bundesliga dataset generated: {DATA_PATH}")


def load_dataset(n_players: int = 250, random_state: int = 42) -> pd.DataFrame:
    """
    Loads the processed dataset; if missing, generates it.
    If you change the generator and want new names/values, delete:
      data/processed/bundesliga_players.csv
    """
    if not DATA_PATH.exists():
        save_dataset(n_players=n_players, random_state=random_state)
    return pd.read_csv(DATA_PATH)