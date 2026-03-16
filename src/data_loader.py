from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data/processed/bundesliga_players.csv"

FIRST_NAMES = ["Jonas","Lukas","Felix","Noah","Max","Leon","Tim","Paul","Nico","Julian",
    "Marco","David","Simon","Tobias","Emil","Moritz","Kevin","Philipp","Daniel","Robin",
    "Jan","Florian","Sebastian","Adrian","Fabian","Matteo","Lucas","Elias","Milan","Aaron"]
LAST_NAMES = ["Schneider","Fischer","Weber","Meyer","Wagner","Becker","Hoffmann","Schulz","Koch","Bauer",
    "Richter","Klein","Wolf","Neumann","Schwarz","Zimmermann","Krüger","Hartmann","Lange","Schmitt",
    "Krause","Werner","Schuster","Vogel","Fuchs","Peters","Arnold","Bergmann","Keller","Frank"]

def random_name(i):
    return f"{FIRST_NAMES[i%len(FIRST_NAMES)]} {LAST_NAMES[(i*7)%len(LAST_NAMES)]}"

def generate_bundesliga_dataset(n_players=250, random_state=42):
    rng = np.random.default_rng(random_state)
    positions = ["GK","CB","FB","DM","CM","AM","ST"]
    pos_probs = [0.08,0.18,0.14,0.14,0.18,0.14,0.14]
    rows = []
    for i in range(n_players):
        position = rng.choice(positions, p=pos_probs)
        age = int(rng.integers(18, 34))
        minutes = int(rng.integers(200, 3200))
        if position in ["ST","AM"]: xg = float(rng.gamma(2.2,0.20))
        else: xg = float(rng.gamma(1.1,0.10))
        if position in ["AM","CM","FB"]: xa = float(rng.gamma(2.0,0.14))
        else: xa = float(rng.gamma(1.0,0.05))
        if position in ["CM","AM","FB","DM"]: pp = float(rng.normal(3.2,1.0))
        else: pp = float(rng.normal(2.4,0.9))
        if position in ["CB","DM","FB"]: da = float(rng.normal(4.4,1.3))
        elif position=="GK": da = float(rng.normal(3.0,1.0))
        else: da = float(rng.normal(3.2,1.2))
        pp = float(np.clip(pp,0,8)); da = float(np.clip(da,0,8))
        prime_factor = float(np.clip(1-abs(age-27)/18, 0.2, 1.0))
        activity_factor = float(np.clip(minutes/2800, 0.2, 1.0))
        contribution = float(np.clip(0.6*xg+0.4*xa+0.15*pp, 0, 3))
        base_value = float(rng.uniform(1.0,35.0))
        mv = float(np.clip(base_value*prime_factor*(0.65+0.35*activity_factor)*(0.8+0.25*contribution),0.5,70))
        wage = float(np.clip(mv*float(rng.uniform(0.05,0.12)),0.2,15))
        injury = float(np.clip((age-27)/10+rng.normal(0,0.18),0,1))
        rows.append([f"BL{i:04d}",random_name(i),age,position,minutes,xg,xa,pp,da,mv,wage,injury])
    return pd.DataFrame(rows,columns=["player_id","name","age","position","minutes","xg","xa",
        "progressive_passes","defensive_actions","market_value_m","estimated_wage_m","injury_risk"])

def load_dataset(n_players=250, random_state=42):
    if not DATA_PATH.exists():
        df = generate_bundesliga_dataset(n_players,random_state)
        DATA_PATH.parent.mkdir(parents=True,exist_ok=True)
        df.to_csv(DATA_PATH,index=False)
    return pd.read_csv(DATA_PATH)