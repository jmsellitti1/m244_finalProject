from pybaseball import statcast, team_batting, batting_stats
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# player_stats = batting_stats(2025)
# player_stats.to_csv("data/2025_player_batting.csv", index=False)

all_pitches = statcast('2025-03-27', '2025-09-28')
all_pitches = all_pitches[all_pitches['events'].notna()]
all_pitches.to_parquet("data/2025_all_pitches.parquet", index=False)
all_pitches[:100].to_csv("data/2025_all_pitches_preview.csv", index=False)

in_play = all_pitches[(all_pitches['description'] == 'hit_into_play') & (~all_pitches['estimated_ba_using_speedangle'].isna())]
data = in_play[[
    'game_pk', 'home_team', 'away_team', 'inning_topbot', 'batter', 'pitcher', 'launch_speed', 'launch_angle', 'hit_distance_sc', 'hc_x', 'hc_y', 'effective_speed',
    'release_spin_rate', 'estimated_ba_using_speedangle', 'events'
    ]].copy()
data['team'] = data.apply(lambda row: row['home_team'] if row['inning_topbot'] == 'Bot' else row['away_team'], axis=1)
data = data.drop(columns=['home_team', 'away_team', 'inning_topbot'])
data = data.rename(columns = {'estimated_ba_using_speedangle': 'xBA',
                              'hit_distance_sc': 'hit_distance',
                              'hc_x': 'hit_location_x',
                              'hc_y': 'hit_location_y',
                              'effective_speed': 'pitch_speed'})
data['hit'] = [1 if row['events'] in ['single', 'double', 'triple', 'home_run'] else 0 for index, row in data.iterrows()]

# Rotate hit locations so that home plate is at 0,0 and balls in fair territory are in Q1
points = data[['hit_location_x', 'hit_location_y']].values
# Known points
home = np.array([126, 204])
rf_point = np.array([61, 136])

points_rel = points - home
rf_rel = rf_point - home
theta = np.arctan2(rf_rel[1], rf_rel[0])
cos_t = np.cos(-theta)
sin_t = np.sin(-theta)
R = np.array([[cos_t, -sin_t],[sin_t,  cos_t]])
points_rot = points_rel @ R.T
data['hit_location_x'] = points_rot[:, 0]
data['hit_location_y'] = points_rot[:, 1]

# Drop incomplete rows
data = data.dropna()

data.to_parquet("data/2025_hits.parquet", index=False)
data[:100].to_csv("data/2025_hits_preview.csv", index=False)
print("Data loading and preprocessing complete.")