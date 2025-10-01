# connecting to an api in python
import requests
from nba_api_functions import get_nba_id
from nba_api.stats.endpoints import playercareerstats

# use functions to get name
career = playercareerstats.PlayerCareerStats(get_nba_id("LeBron James")) 

# pandas data frames
nba_data_frame = career.get_data_frames()[0]

print(nba_data_frame)