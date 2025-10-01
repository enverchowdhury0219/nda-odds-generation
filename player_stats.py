# for plotting
import matplotlib.pyplot as plt
import pandas as pd

#nba api calls
from nba_api_functions import get_nba_id
from nba_api.stats.endpoints import playercareerstats

player_name = input("Enter NBA player name: ")

# use functions to get player stats
career = playercareerstats.PlayerCareerStats(get_nba_id(player_name)) 

nba_data_frame = career.get_data_frames()[0]

# what the data looks like:
# ['PLAYER_ID', 'LEAGUE_ID', 'Team_ID', 'GP', 'GS', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']

print(nba_data_frame)

def pick_season_row(g: pd.DataFrame) -> pd.Series:
    tot = g[g["TEAM_ABBREVIATION"] == "TOT"]
    return tot.iloc[0] if not tot.empty else g.iloc[-1]

df_plot = (
    nba_data_frame.sort_values(["SEASON_ID", "TEAM_ABBREVIATION"])
      .groupby("SEASON_ID", as_index=False)
      .apply(pick_season_row)
      .reset_index(drop=True)
)

# total points by season
plt.figure()
plt.plot(df_plot["SEASON_ID"], df_plot["PTS"], marker="o")
plt.title(f"{player_name} - Total Points by Season")
plt.xlabel("Season")
plt.ylabel("PTS (total)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()