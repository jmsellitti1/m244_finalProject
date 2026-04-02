from pybaseball import statcast
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

all_pitches = statcast('2025-03-27', '2025-09-28')
all_pitches[:100].to_csv("data/2025_all_pitches.csv", index=False)

in_play = all_pitches[(all_pitches['description'] == 'hit_into_play') & (~all_pitches['estimated_ba_using_speedangle'].isna())]
data = in_play[['game_pk', 'home_team', 'batter', 'pitcher', 'launch_speed', 'launch_angle', 'estimated_ba_using_speedangle', 'events']]
data = data.rename(columns = {'estimated_ba_using_speedangle': 'xBA', 'home_team': 'stadium'})
data['hit'] = [1 if row['events'] in ['single', 'double', 'triple', 'home_run'] else 0 for index, row in data.iterrows()]
data = data.reset_index(drop=True)

data.to_csv("data/2025_hits.csv", index=False)