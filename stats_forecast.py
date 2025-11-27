import pandas as pd

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

def add_rolling_form_features(ts: pd.DataFrame,
                              stats=None,
                              windows=(3, 5, 10)) -> pd.DataFrame:
    df = ts.copy()
    if stats is None:
        stats = ["PTS", "MIN", "FGA", "FGM", "FG3A", "FG3M", "AST", "REB"]

    for stat in stats:
        if stat not in df.columns:
            continue

        # use only PAST values of this stat
        shifted = df[stat].shift(1)

        for w in windows:
            roll = shifted.rolling(window=w, min_periods=1)
            df[f"{stat}_roll_mean_{w}"] = roll.mean()
            df[f"{stat}_roll_std_{w}"] = roll.std()

    return df

def add_schedule_features(ts: pd.DataFrame) -> pd.DataFrame:
    df = ts.copy()

    # Days since previous game
    df["DAYS_SINCE_PREV"] = df.index.to_series().diff().dt.days

    # First game has NaN: replace with median or a default like 3 days
    default_rest = df["DAYS_SINCE_PREV"].median()
    df["DAYS_SINCE_PREV"] = df["DAYS_SINCE_PREV"].fillna(default_rest)

    # Back-to-back flag
    df["IS_BACK_TO_BACK"] = (df["DAYS_SINCE_PREV"] == 1).astype(int)

    # Long rest flag (e.g., 4+ days)
    df["IS_LONG_REST"] = (df["DAYS_SINCE_PREV"] >= 4).astype(int)

    # Recent minutes load: rolling mean of MIN over last 5 games (non-leaky)
    if "MIN" in df.columns:
        df["MIN_roll_mean_5"] = df["MIN"].shift(1).rolling(5, min_periods=1).mean()

    return df

def add_opponent_history_features(ts: pd.DataFrame) -> pd.DataFrame:
    df = ts.copy()

    # Average PTS vs this opponent *before* current game
    df["OPP_MEAN_PTS_PRIOR"] = (
        df.groupby("OPP_TEAM_ABBR")["PTS"]
          .apply(lambda s: s.shift(1).expanding().mean())
    )

    # Number of prior games vs opponent
    df["OPP_NUM_PRIOR_MATCHUPS"] = (
        df.groupby("OPP_TEAM_ABBR")["PTS"]
          .apply(lambda s: s.shift(1).expanding().count())
    )

    # Fill initial NaNs (first encounter vs each team)
    overall_mean_pts = df["PTS"].mean()
    df["OPP_MEAN_PTS_PRIOR"] = df["OPP_MEAN_PTS_PRIOR"].fillna(overall_mean_pts)
    df["OPP_NUM_PRIOR_MATCHUPS"] = df["OPP_NUM_PRIOR_MATCHUPS"].fillna(0)

    return df

def build_lebron_feature_table(lebron_ts: pd.DataFrame) -> pd.DataFrame:
    df = lebron_ts.copy()

    # 1) Rolling form stats
    df = add_rolling_form_features(df)

    # 2) Schedule / fatigue
    df = add_schedule_features(df)

    # 3) Opponent history
    df = add_opponent_history_features(df)

    # 4) Optional: drop early games where some rolling windows are too short
    #    (or just keep them; many models can handle NaNs after imputation)
    # For now, let's just drop any rows with missing key features:
    feature_cols = [c for c in df.columns if c not in ["MATCHUP", "WL", "season_type"]]
    df = df.dropna(subset=["PTS_roll_mean_3", "MIN_roll_mean_5", "OPP_MEAN_PTS_PRIOR"])

    return df

# Usage:
lebron_features = build_lebron_feature_table(lebron_ts)
print(lebron_features.shape)
print(lebron_features[[
    "PTS",
    "HOME",
    "PTS_roll_mean_3",
    "PTS_roll_mean_5",
    "MIN_roll_mean_5",
    "DAYS_SINCE_PREV",
    "IS_BACK_TO_BACK",
    "OPP_TEAM_ABBR",
    "OPP_MEAN_PTS_PRIOR",
    "OPP_NUM_PRIOR_MATCHUPS"
]].head(15))
