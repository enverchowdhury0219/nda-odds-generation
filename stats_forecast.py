import pandas as pd

# If you're loading from the CSV you just wrote:
refined = pd.read_csv("lebron_core_games.csv")

def prepare_lebron_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    ts = df.copy()

    # 1) Keep only Regular Season games (optional but recommended for now)
    ts = ts[ts["season_type"] == "Regular Season"].copy()

    # 2) Parse GAME_DATE and sort chronologically
    ts["GAME_DATE"] = pd.to_datetime(ts["GAME_DATE"], errors="coerce")
    ts = ts.sort_values("GAME_DATE")

    # 3) Create HOME flag from MATCHUP
    # MATCHUP examples: "CLE @ SAC", "CLE vs. DEN"
    ts["HOME"] = ts["MATCHUP"].str.contains("vs").astype(int)

    # 4) Extract opponent abbreviation from MATCHUP
    # "CLE @ SAC"  -> "SAC"
    # "CLE vs. DEN" -> "DEN"
    def extract_opp(matchup: str) -> str:
        # Split by spaces and take the last token ("SAC", "DEN", etc.)
        return matchup.split(" ")[-1]

    ts["OPP_TEAM_ABBR"] = ts["MATCHUP"].apply(extract_opp)

    # 5) Set GAME_DATE as index (time series index)
    ts = ts.set_index("GAME_DATE")

    # 6) Ensure numeric columns are truly numeric
    numeric_cols = [
        "MIN", "PTS", "REB", "AST",
        "STL", "BLK", "TOV",
        "FGM", "FGA", "FG_PCT",
        "FG3M", "FG3A", "FG3_PCT",
        "FTM", "FTA", "FT_PCT",
        "PLUS_MINUS",
        "HOME",
    ]
    for col in numeric_cols:
        if col in ts.columns:
            ts[col] = pd.to_numeric(ts[col], errors="coerce")

    # 7) Optional: drop any rows with missing target PTS
    ts = ts.dropna(subset=["PTS"])

    # 8) Add a simple game counter (1..N), useful for plotting/debug
    ts["GAME_NUM"] = range(1, len(ts) + 1)

    return ts

# Usage:
lebron_ts = prepare_lebron_timeseries(refined)
print(lebron_ts)
# print(lebron_ts[["PTS", "MIN", "OPP_TEAM_ABBR", "HOME"]].head())
