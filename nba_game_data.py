from nba_api.stats.endpoints import commonplayerinfo, playergamelog
import pandas as pd
import time
from datetime import datetime

def season_strings(start, end):
    return [f"{y}-{str((y+1)%100).zfill(2)}" for y in range(start, end+1)]

def safe_gamelog(player_id, season, stype, max_retries=3, sleep=1.0):
    for i in range(max_retries):
        try:
            gl = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                season_type_all_star=stype,
                timeout=30
            ).get_data_frames()[0]
            if not gl.empty:
                # normalize column names for consistency
                if "Game_ID" in gl.columns and "GAME_ID" not in gl.columns:
                    gl = gl.rename(columns={"Game_ID": "GAME_ID"})
                if "Game_ID" in gl.columns and "GAME_ID" in gl.columns:
                    # unlikely, but avoid duplicate
                    gl = gl.drop(columns=["Game_ID"])
                gl["season"] = season
                gl["season_type"] = stype
                return gl
        except Exception as e:
            print(f"Error {stype} {season}: {e}")
        time.sleep(sleep * (i+1))
    return pd.DataFrame()

def get_all_games(player_id):
    info = commonplayerinfo.CommonPlayerInfo(player_id=player_id, timeout=30).get_data_frames()[0]
    start_year = int(info["FROM_YEAR"].iloc[0])
    end_year = int(info["TO_YEAR"].iloc[0])

    # include current season start if still active
    now = datetime.now()
    season_start_year = now.year if now.month >= 7 else now.year - 1
    end_year = max(end_year, season_start_year)

    logs = []
    for season in season_strings(start_year, end_year):
        for stype in ["Regular Season", "Playoffs"]:
            df = safe_gamelog(player_id, season, stype)
            if not df.empty:
                logs.append(df)
                print(f"✅ Pulled {len(df)} games for {season} {stype}")
            else:
                print(f"⚠️ Did not make/play {season} {stype}")
            time.sleep(1.0)

    if not logs:
        return pd.DataFrame()

    combined = pd.concat(logs, ignore_index=True)

    # ensure expected columns / dtypes
    if "GAME_DATE" in combined.columns:
        combined["GAME_DATE"] = pd.to_datetime(combined["GAME_DATE"], errors="coerce")

    # sort by whatever exists
    sort_keys = [c for c in ["GAME_DATE", "GAME_ID"] if c in combined.columns]
    if sort_keys:
        combined = combined.sort_values(sort_keys).reset_index(drop=True)

    return combined

# ---- usage ----
player_id = 2544  # your resolved id
df = get_all_games(player_id)
core_cols = [
    "GAME_DATE", "MATCHUP", "WL", "MIN", "PTS", "REB", "AST",
    "STL", "BLK", "TOV", "FGM", "FGA", "FG_PCT",
    "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT",
    "PLUS_MINUS", "season", "season_type"
]
refined = df[core_cols]
refined.to_csv("lebron_core_games.csv", index=False)

print(f"Total games pulled: {len(df)}")
print(df)

