# connecting to an api in python
import requests
from nba_api.stats.endpoints import playercareerstats

# LeBron James, added a csv to get nba player ids for this nba api
career = playercareerstats.PlayerCareerStats(player_id='2544') 

# pandas data frames
nba_data_frame = career.get_data_frames()[0]

# json
nba_json = career.get_json()

# dictionary
nba_dict = career.get_dict()

print(nba_data_frame)