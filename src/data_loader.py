from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data/processed/bundesliga_players.csv"

# ── Real Bundesliga 2025/26 player pool ───────────────────────────────────────
# Format: (name, age, position)
REAL_PLAYERS = [
    # FC Bayern München
    ("Manuel Neuer",       39, "GK"), ("Sven Ulreich",        36, "GK"),
    ("Dayot Upamecano",    26, "CB"), ("Min-jae Kim",         28, "CB"), ("Matthijs de Ligt",   25, "CB"),
    ("Alphonso Davies",    24, "FB"), ("Konrad Laimer",        27, "FB"), ("Josip Stanisic",     24, "FB"),
    ("Joao Palhinha",      29, "DM"), ("Joshua Kimmich",      30, "DM"),
    ("Leon Goretzka",      30, "CM"),
    ("Jamal Musiala",      22, "AM"), ("Michael Olise",       23, "AM"), ("Leroy Sane",         29, "AM"),
    ("Thomas Müller",      35, "AM"), ("Kingsley Coman",      29, "AM"),
    ("Harry Kane",         31, "ST"),

    # Borussia Dortmund
    ("Gregor Kobel",       27, "GK"), ("Alexander Meyer",     32, "GK"),
    ("Nico Schlotterbeck", 25, "CB"), ("Waldemar Anton",      28, "CB"), ("Niklas Süle",        29, "CB"),
    ("Ramy Bensebaini",    29, "FB"), ("Julian Ryerson",      27, "FB"), ("Yan Couto",          22, "FB"),
    ("Marcel Sabitzer",    31, "DM"), ("Emre Can",            31, "DM"),
    ("Pascal Gross",       33, "CM"), ("Felix Nmecha",        24, "CM"),
    ("Julian Brandt",      29, "AM"), ("Jamie Gittens",       20, "AM"),
    ("Serhou Guirassy",    28, "ST"), ("Karim Adeyemi",       23, "ST"),

    # RB Leipzig
    ("Peter Gulacsi",      35, "GK"), ("Janis Blaswich",      33, "GK"),
    ("Willi Orban",        32, "CB"), ("Mohamed Simakan",     24, "CB"), ("Castello Lukeba",    22, "CB"),
    ("Benjamin Henrichs",  28, "FB"), ("David Raum",          26, "FB"),
    ("Nicolas Seiwald",    23, "DM"), ("Kevin Kampl",         34, "DM"),
    ("Xaver Schlager",     27, "CM"), ("Christoph Baumgartner",25,"CM"),
    ("Xavi Simons",        22, "AM"), ("Amadou Haidara",      26, "AM"), ("Antonio Nusa",       20, "AM"),
    ("Lois Openda",        24, "ST"), ("Benjamin Sesko",      21, "ST"),

    # Bayer 04 Leverkusen
    ("Lukas Hradecky",     35, "GK"), ("Matej Kovar",         23, "GK"),
    ("Odilon Kossounou",   24, "CB"), ("Piero Hincapie",      23, "CB"), ("Edmond Tapsoba",     25, "CB"),
    ("Alejandro Grimaldo", 29, "FB"), ("Jeremie Frimpong",    24, "FB"),
    ("Granit Xhaka",       32, "DM"), ("Robert Andrich",      30, "DM"),
    ("Florian Wirtz",      22, "AM"), ("Exequiel Palacios",   26, "CM"),
    ("Jonas Hofmann",      32, "AM"), ("Amine Adli",          24, "AM"), ("Adam Hlozek",        22, "AM"),
    ("Victor Boniface",    24, "ST"), ("Patrik Schick",       29, "ST"),

    # Eintracht Frankfurt
    ("Kevin Trapp",        34, "GK"), ("Kaua Santos",         21, "GK"),
    ("Tuta",               26, "CB"), ("Robin Koch",          28, "CB"), ("Aurele Amenda",      21, "CB"),
    ("Philipp Max",        31, "FB"), ("Niels Nkounkou",      24, "FB"),
    ("Ellyes Skhiri",      29, "DM"), ("Hugo Larsson",        20, "DM"),
    ("Mario Götze",        33, "CM"), ("Junior Dina Ebimbe",  23, "CM"),
    ("Ansgar Knauff",      23, "AM"), ("Jessic Ngankam",      23, "AM"),
    ("Hugo Ekitike",       22, "ST"), ("Nathaniel Brown",     20, "ST"),

    # SC Freiburg
    ("Noah Atubolu",       22, "GK"), ("Florian Müller",      27, "GK"),
    ("Philipp Lienhart",   28, "CB"), ("Matthias Ginter",     31, "CB"), ("Kiliann Sildillia",  22, "CB"),
    ("Christian Günter",   31, "FB"), ("Lukas Kübler",        32, "FB"),
    ("Nicolas Höfler",     32, "DM"), ("Maximilian Eggestein",26, "DM"),
    ("Yannik Keitel",      23, "CM"), ("Merlin Röhl",         22, "CM"),
    ("Ritsu Doan",         26, "AM"), ("Vincenzo Grifo",      31, "AM"), ("Junior Adamu",       23, "AM"),
    ("Lucas Höler",        30, "ST"), ("Eren Dinkci",         22, "ST"),

    # VfB Stuttgart
    ("Alexander Nübel",    28, "GK"), ("Fabian Bredlow",      30, "GK"),
    ("Konstantinos Mavropanos",27,"CB"),("Jeff Chabot",        24, "CB"), ("Anthony Rouault",   23, "CB"),
    ("Maximilian Mittelstädt", 27,"FB"),("Josha Vagnoman",    24, "FB"),
    ("Atakan Karazor",     27, "DM"), ("Angelo Stiller",      23, "DM"),
    ("Enzo Millot",        23, "CM"), ("Chris Führich",       26, "CM"),
    ("Ermedin Demirovic",  26, "AM"), ("Jamie Leweling",      24, "AM"),
    ("Wouter Weghorst",    32, "ST"), ("Nick Woltemade",      23, "ST"),

    # TSG Hoffenheim
    ("Oliver Baumann",     34, "GK"), ("Luca Philipp",        24, "GK"),
    ("Kevin Vogt",         33, "CB"), ("Stanley Nsoki",        26, "CB"), ("David Jurasek",      24, "CB"),
    ("Pavel Kaderabek",    32, "FB"), ("John Anthony Brooks", 31, "FB"),
    ("Dennis Geiger",      26, "DM"), ("Tom Bischof",         21, "DM"),
    ("Grischa Prömel",     31, "CM"), ("Florian Grillitsch",  29, "CM"),
    ("Andrej Kramaric",    33, "AM"), ("Jacob Bruun Larsen",  26, "AM"), ("Robert Skov",        28, "AM"),
    ("Mergim Berisha",     25, "ST"), ("Sargis Adamyan",      31, "ST"),

    # 1. FSV Mainz 05
    ("Robin Zentner",      29, "GK"), ("Finn Dahmen",         26, "GK"),
    ("Andreas Hanche-Olsen",28,"CB"), ("Stefan Bell",         33, "CB"), ("Maxim Leitsch",      26, "CB"),
    ("Aaron",              28, "FB"), ("Silvan Widmer",       31, "FB"),
    ("Leandro Barreiro",   24, "DM"), ("Dominik Kohr",        31, "DM"),
    ("Kaishu Sano",        23, "CM"), ("Aymen Barkok",        26, "CM"),
    ("Jonathan Burkardt",  24, "AM"), ("Karim Onisiwo",       31, "AM"),
    ("Ludovic Ajorque",    31, "ST"), ("Paul Nebel",          21, "ST"),

    # Borussia Mönchengladbach
    ("Jonas Omlin",        30, "GK"), ("Jan Olschowsky",      23, "GK"),
    ("Nico Elvedi",        28, "CB"), ("Marvin Friedrich",    28, "CB"), ("Ko Itakura",         27, "CB"),
    ("Luca Netz",          22, "FB"), ("Joe Scally",          22, "FB"),
    ("Florian Neuhaus",    27, "DM"), ("Julian Weigl",        29, "DM"),
    ("Rocco Reitz",        22, "CM"), ("Christoph Kramer",    33, "CM"),
    ("Franck Honorat",     27, "AM"), ("Alassane Plea",       31, "AM"),
    ("Tim Kleindienst",    28, "ST"), ("Robin Hack",          25, "ST"),

    # VfL Wolfsburg
    ("Kamil Grabara",      25, "GK"), ("Niklas Klinger",      24, "GK"),
    ("Maxence Lacroix",    24, "CB"), ("Sebastiaan Bornauw",  25, "CB"), ("Danilho Doekhi",     26, "CB"),
    ("Ridle Baku",         26, "FB"), ("Joakim Maehle",       27, "FB"),
    ("Maximilian Arnold",  30, "DM"), ("Mattias Svanberg",    25, "DM"),
    ("Yannick Gerhardt",   29, "CM"), ("Aster Vranckx",       22, "CM"),
    ("Patrick Wimmer",     23, "AM"), ("Jakub Kaminski",      22, "AM"),
    ("Jonas Wind",         25, "ST"), ("Lukas Nmecha",        25, "ST"),

    # Werder Bremen
    ("Michael Zetterer",   29, "GK"), ("Jiri Pavlenka",       32, "GK"),
    ("Amos Pieper",        27, "CB"), ("Niklas Stark",        29, "CB"), ("Marco Friedl",       26, "CB"),
    ("Mitchell Weiser",    30, "FB"), ("Anthony Jung",        30, "FB"),
    ("Ilia Gruev",         24, "DM"), ("Christian Gross",     36, "DM"),
    ("Leonardo Bittencourt",30,"CM"), ("Romano Schmid",       24, "CM"),
    ("Marvin Ducksch",     30, "AM"), ("Jens Stage",          27, "AM"),
    ("Dawid Kownacki",     27, "ST"), ("Oliver Burke",        27, "ST"),

    # FC Augsburg
    ("Tomas Koubek",       32, "GK"), ("Finn Dahmen",         26, "GK"),
    ("Jeffrey Gouweleeuw", 32, "CB"), ("Maximilian Bauer",    27, "CB"), ("Luca Kilian",        25, "CB"),
    ("Robert Gumny",       26, "FB"), ("Mads Pedersen",       28, "FB"),
    ("Arne Maier",         25, "DM"), ("Elvis Rexhbecaj",     26, "DM"),
    ("Fredrik Jensen",     27, "CM"), ("Alexis Claude-Maurice",26,"CM"),
    ("Mergim Berisha",     25, "AM"), ("Phillip Tietz",       26, "AM"),
    ("Sven Michel",        33, "ST"), ("Ermedin Demirovic",   26, "ST"),

    # 1. FC Köln
    ("Marvin Schwäbe",     29, "GK"), ("Jonas Urbig",         21, "GK"),
    ("Timo Hübers",        28, "CB"), ("Luca Kilian",         25, "CB"), ("Jeff Chabot",        24, "CB"),
    ("Benno Schmitz",      29, "FB"), ("Jannes Horn",         27, "FB"),
    ("Dejan Ljubicic",     26, "DM"), ("Eric Martel",         22, "DM"),
    ("Florian Kainz",      31, "CM"), ("Ondrej Duda",         29, "CM"),
    ("Linton Maina",       24, "AM"), ("Steffen Tigges",      26, "AM"),
    ("Davie Selke",        29, "ST"), ("Mark Uth",            32, "ST"),

    # Union Berlin
    ("Frederik Rönnow",    31, "GK"), ("Lennart Grill",       25, "GK"),
    ("Robin Knoche",       33, "CB"), ("Diogo Leite",         25, "CB"), ("Timo Baumgartl",     27, "CB"),
    ("Josip Juranovic",    29, "FB"), ("Tom Rothe",           20, "FB"),
    ("Rani Khedira",       29, "DM"), ("Andras Schäfer",      25, "DM"),
    ("Brenden Aaronson",   24, "CM"), ("Janik Haberer",       30, "CM"),
    ("Sheraldo Becker",    29, "AM"), ("Kevin Volland",       32, "AM"),
    ("Kevin Behrens",      33, "ST"), ("Jordan Pefok",        29, "ST"),

    # VfL Bochum
    ("Manuel Riemann",     35, "GK"), ("Patrick Drewes",      33, "GK"),
    ("Ivan Ordets",        29, "CB"), ("Erhan Masovic",       25, "CB"), ("Armel Bella-Kotchap",23,"CB"),
    ("Danilo Soares",      34, "FB"), ("Cristian Gamboa",     34, "FB"),
    ("Patrick Osterhage",  25, "DM"), ("Maximilian Wittek",   30, "DM"),
    ("Kevin Stöger",       31, "CM"), ("Philipp Förster",     28, "CM"),
    ("Christopher Antwi-Adjei",31,"AM"),("Simon Zoller",      33, "AM"),
    ("Takuma Asano",       30, "ST"), ("Philipp Hofmann",     31, "ST"),

    # 1. FC Heidenheim
    ("Kevin Müller",       30, "GK"), ("Morten Behrens",      23, "GK"),
    ("Patrick Mainka",     29, "CB"), ("Benedikt Gimber",     27, "CB"), ("Jan Schöppner",      25, "CB"),
    ("Jan-Niklas Beste",   25, "FB"), ("Jonas Föhrenbach",    27, "FB"),
    ("Tobias Raschl",      24, "DM"), ("Norman Theuerkauf",   34, "DM"),
    ("Lennard Maloney",    23, "CM"), ("Denis Thomalla",      28, "CM"),
    ("Marvin Pieringer",   25, "AM"), ("Niklas Dorsch",       27, "AM"),
    ("Stefan Schimmer",    29, "ST"), ("Christian Kühlwetter",30, "ST"),

    # SV Darmstadt 98
    ("Marcel Schuhen",     31, "GK"), ("Florian Stritzel",    28, "GK"),
    ("Clemens Riedel",     25, "CB"), ("Thomas Isherwood",    23, "CB"), ("Fabian Holland",     32, "CB"),
    ("Emir Karic",         26, "FB"), ("Klaus Gjasula",       33, "FB"),
    ("Fabian Schnellhardt",33, "DM"), ("Tobias Kempe",        32, "DM"),
    ("Marvin Mehlem",      28, "CM"), ("Oscar Vilhelmsson",   21, "CM"),
    ("Braydon Manu",       27, "AM"), ("Matthias Bader",      28, "AM"),
    ("Luca Pfeiffer",      28, "ST"), ("Mathias Honsak",      28, "ST"),
]


def generate_bundesliga_dataset(random_state=42):
    rng = np.random.default_rng(random_state)
    rows = []
    seen_names = {}

    for i, (raw_name, age, position) in enumerate(REAL_PLAYERS):
        # Deduplicate names (same player at multiple clubs in list)
        name = raw_name
        if name in seen_names:
            seen_names[name] += 1
            name = f"{raw_name} ({seen_names[name]})"
        else:
            seen_names[name] = 1

        minutes = int(rng.integers(200, 3200))

        if position in ["ST", "AM"]:
            xg = float(rng.gamma(2.2, 0.20))
        else:
            xg = float(rng.gamma(1.1, 0.10))

        if position in ["AM", "CM", "FB"]:
            xa = float(rng.gamma(2.0, 0.14))
        else:
            xa = float(rng.gamma(1.0, 0.05))

        if position in ["CM", "AM", "FB", "DM"]:
            pp = float(rng.normal(3.2, 1.0))
        else:
            pp = float(rng.normal(2.4, 0.9))

        if position in ["CB", "DM", "FB"]:
            da = float(rng.normal(4.4, 1.3))
        elif position == "GK":
            da = float(rng.normal(3.0, 1.0))
        else:
            da = float(rng.normal(3.2, 1.2))

        pp = float(np.clip(pp, 0, 8))
        da = float(np.clip(da, 0, 8))

        prime_factor   = float(np.clip(1 - abs(age - 27) / 18, 0.2, 1.0))
        activity_factor= float(np.clip(minutes / 2800, 0.2, 1.0))
        contribution   = float(np.clip(0.6 * xg + 0.4 * xa + 0.15 * pp, 0, 3))
        base_value     = float(rng.uniform(1.0, 35.0))
        mv   = float(np.clip(base_value * prime_factor * (0.65 + 0.35 * activity_factor) * (0.8 + 0.25 * contribution), 0.5, 70))
        wage = float(np.clip(mv * float(rng.uniform(0.05, 0.12)), 0.2, 15))
        injury = float(np.clip((age - 27) / 10 + rng.normal(0, 0.18), 0, 1))

        rows.append([f"BL{i:04d}", name, age, position, minutes,
                     xg, xa, pp, da, mv, wage, injury])

    return pd.DataFrame(rows, columns=[
        "player_id", "name", "age", "position", "minutes",
        "xg", "xa", "progressive_passes", "defensive_actions",
        "market_value_m", "estimated_wage_m", "injury_risk",
    ])


def load_dataset():
    if not DATA_PATH.exists():
        df = generate_bundesliga_dataset()
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
    return pd.read_csv(DATA_PATH)
