import pandas as pd

DATA_PATH = "data/NBA_Player_IDs.csv"
df = pd.read_csv(DATA_PATH)

def get_nba_id(player_name: str) -> str:
    """
    gets nba player id from csv using name
    """
    match = df[df['NBAName'].str.lower() == player_name.lower()]
    if not match.empty:
        return str(int(match.iloc[0]['NBAID']))
    else:
        return f"No NBAID found for {player_name}"

#testing
print(get_nba_id("Aaron Gordon"))   # should return 203932
