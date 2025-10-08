from nba_api.live.nba.endpoints import scoreboard
import pandas as pd
import json


# Today's Score Board
games = scoreboard.ScoreBoard()
games_json = games.get_json()

#converts the json into a dataframe
def format_nba_scoreboard(nba_json):
    """
    Takes NBA API scoreboard JSON and formats it into a pandas DataFrame.
    """
    games = nba_json.get("scoreboard", {}).get("games", [])
    data = []
    
    for g in games:
        data.append({
            "GameID": g.get("gameId"),
            "Matchup": f"{g['awayTeam']['teamTricode']} @ {g['homeTeam']['teamTricode']}",
            "Status": g.get("gameStatusText"),
            "Home Team": f"{g['homeTeam']['teamCity']} {g['homeTeam']['teamName']}",
            "Home Score": g['homeTeam'].get("score", 0),
            "Away Team": f"{g['awayTeam']['teamCity']} {g['awayTeam']['teamName']}",
            "Away Score": g['awayTeam'].get("score", 0),
        })
    
    return pd.DataFrame(data)

nba_json = json.loads(games_json)

df = format_nba_scoreboard(nba_json)

print(games_json)