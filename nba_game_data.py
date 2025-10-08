# pip install nba_api pandas
import time
from datetime import datetime
import pandas as pd
from nba_api.stats.endpoints import commonplayerinfo, playergamelog
from nba_api.stats.library.parameters import SeasonTypeAllStar

def _season_strings(from_year: int, to_year: int) -> list[str]:
    """Build 'YYYY-YY' strings inclusive: e.g., 2003..2024 -> ['2003-04', ... '2024-25']"""
    seasons = []
    for y in range(from_year, to_year + 1):
        seasons.append(f"{y}-{str((y + 1) % 100).zfill(2)}")
    return seasons

def _safe_player_gamelog(player_id: int, season: str, season_type: str, pause=0.7, retries=3) -> pd.DataFrame:
    """Call PlayerGameLog with basic retry + polite pacing."""
    last_err = None
    for _ in range(retries):
        try:
            gl = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                season_type_all_star=season_type,  # "Regular Season" or "Playoffs"
                timeout=30
            ).get_data_frames()[0]
            if not gl.empty:
                gl["season"] = season
                gl["season_type"] = season_type
            time.sleep(pause)  # be nice to NBA Stats
            return gl
        except Exception as e:
            last_err = e
            time.sleep(1.5)
    raise last_err

def fetch_all_games_all_seasons(player_id: int) -> pd.DataFrame:
    """
    Returns a DataFrame of every game for this player across all seasons
    (regular + playoffs). Columns include: GAME_DATE, MATCHUP, WL, MIN, PTS, REB, AST, STL, BLK, TOV, etc.
    """
    # 1) Get active season range from CommonPlayerInfo
    info = commonplayerinfo.CommonPlayerInfo(player_id=player_id, timeout=30).get_data_frames()
    header = info[0]  # "CommonPlayerInfo"
    # Fallback if fields missing: infer from DOB/first season, but for NBA players these exist.
    from_year = int(header.at[0, "FROM_YEAR"])
    to_year = int(header.at[0, "TO_YEAR"])

    # In case the player is still active this season, bump to current start year if needed
    current_year = datetime.now().year
    # NBA season starts in Oct; crude adjustment to include the current season if applicable
    likely_season_start = current_year if datetime.now().month >= 7 else current_year - 1
    to_year = max(to_year, likely_season_start)

    seasons = _season_strings(from_year, to_year)

    # 2) Pull logs for each season & both season types
    frames = []
    for s in seasons:
        for st in (SeasonTypeAllStar.regular, SeasonTypeAllStar.playoffs):
            df = _safe_player_gamelog(player_id, s, st)
            if not df.empty:
                frames.append(df)

    if not frames:
        return pd.DataFrame()  # nothing found

    all_games = pd.concat(frames, ignore_index=True)

    # 3) Light cleaning/convenience
    # - standardize date to datetime
    # - sort chronologically
    # - ensure numeric cols are numeric
    num_cols = ["PTS","REB","AST","STL","BLK","TOV","FGM","FGA","FG3M","FG3A","FTM","FTA","PLUS_MINUS"]
    for c in num_cols:
        if c in all_games.columns:
            all_games[c] = pd.to_numeric(all_games[c], errors="coerce")
    if "GAME_DATE" in all_games.columns:
        all_games["GAME_DATE"] = pd.to_datetime(all_games["GAME_DATE"])
        all_games = all_games.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    return all_games

# -------------------------
# USAGE
# -------------------------
# You said you already have `player_id`. For LeBron it's typically 2544, but use your own resolver.
player_id = 2544  # replace with your resolved id

all_lbj_games = fetch_all_games_all_seasons(player_id)
print(all_lbj_games.head())
print(len(all_lbj_games), "total games")

# Optional: save for reuse
all_lbj_games.to_csv("lebron_all_games.csv", index=False)
# or parquet:
# all_lbj_games.to_parquet("lebron_all_games.parquet", index=False)
