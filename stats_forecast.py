import pandas as pd

refined = pd.read_csv("lebron_core_games.csv")

def prepare_lebron_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    ts = df.copy()

    ts = ts[ts["season_type"] == "Regular Season"].copy()

    ts["GAME_DATE"] = pd.to_datetime(ts["GAME_DATE"], errors="coerce")
    ts = ts.sort_values("GAME_DATE")

    ts["HOME"] = ts["MATCHUP"].str.contains("vs").astype(int)

    def extract_opp(matchup: str) -> str:
        return matchup.split(" ")[-1]

    ts["OPP_TEAM_ABBR"] = ts["MATCHUP"].apply(extract_opp)

    ts = ts.set_index("GAME_DATE")

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

    ts = ts.dropna(subset=["PTS"])

    ts["GAME_NUM"] = range(1, len(ts) + 1)

    return ts

# Usage:
lebron_ts = prepare_lebron_timeseries(refined)
print(lebron_ts)

def add_rolling_form_features(ts: pd.DataFrame,
                              stats=None,
                              windows=(3, 5, 10)) -> pd.DataFrame:
    df = ts.copy()
    if stats is None:
        stats = ["PTS", "MIN", "FGA", "FGM", "FG3A", "FG3M", "AST", "REB"]

    for stat in stats:
        if stat not in df.columns:
            continue
        shifted = df[stat].shift(1)

        for w in windows:
            roll = shifted.rolling(window=w, min_periods=1)
            df[f"{stat}_roll_mean_{w}"] = roll.mean()
            df[f"{stat}_roll_std_{w}"] = roll.std()

    return df

def add_schedule_features(ts: pd.DataFrame) -> pd.DataFrame:
    df = ts.copy()

    df["DAYS_SINCE_PREV"] = df.index.to_series().diff().dt.days

    default_rest = df["DAYS_SINCE_PREV"].median()
    df["DAYS_SINCE_PREV"] = df["DAYS_SINCE_PREV"].fillna(default_rest)

    df["IS_BACK_TO_BACK"] = (df["DAYS_SINCE_PREV"] == 1).astype(int)

    df["IS_LONG_REST"] = (df["DAYS_SINCE_PREV"] >= 4).astype(int)

    if "MIN" in df.columns:
        df["MIN_roll_mean_5"] = df["MIN"].shift(1).rolling(5, min_periods=1).mean()

    return df

def add_opponent_history_features(ts: pd.DataFrame) -> pd.DataFrame:
    df = ts.copy()
    g = df.groupby("OPP_TEAM_ABBR")["PTS"]
    df["OPP_MEAN_PTS_PRIOR"] = g.transform(
        lambda s: s.shift(1).expanding().mean()
    )
    df["OPP_NUM_PRIOR_MATCHUPS"] = g.transform(
        lambda s: s.shift(1).expanding().count()
    )
    overall_mean_pts = df["PTS"].mean()
    df["OPP_MEAN_PTS_PRIOR"] = df["OPP_MEAN_PTS_PRIOR"].fillna(overall_mean_pts)
    df["OPP_NUM_PRIOR_MATCHUPS"] = df["OPP_NUM_PRIOR_MATCHUPS"].fillna(0)

    return df


def build_lebron_feature_table(lebron_ts: pd.DataFrame) -> pd.DataFrame:
    df = lebron_ts.copy()
    df = add_rolling_form_features(df)
    df = add_schedule_features(df)
    df = add_opponent_history_features(df)
    feature_cols = [c for c in df.columns if c not in ["MATCHUP", "WL", "season_type"]]
    df = df.dropna(subset=["PTS_roll_mean_3", "MIN_roll_mean_5", "OPP_MEAN_PTS_PRIOR"])

    return df

lebron_ts = prepare_lebron_timeseries(refined)
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
