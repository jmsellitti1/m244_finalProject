import pandas as pd
import matplotlib.pyplot as plt

all_pitches = pd.read_parquet("data/2025_all_pitches.parquet")
data = pd.read_parquet("data/2025_hits.parquet")
team_data = {}
for team in data['team'].unique():
    single_data = data[data['team'] == team]
    team_xba = single_data['xBA'].mean()
    team_strikeouts_count = len(all_pitches[
        ((all_pitches['inning_topbot'] == 'Top') & (all_pitches['away_team'] == team) |
         (all_pitches['inning_topbot'] == 'Bot') & (all_pitches['home_team'] == team)) &
        (all_pitches['events'] == 'strikeout')
    ])
    team_avg = len(single_data[single_data['hit'] == 1]) / (len(single_data) + team_strikeouts_count) # Not 100% accurate
    team_data[team] = [team_xba, team_avg]

team_df = pd.DataFrame.from_dict(team_data, orient='index', columns=['xBA', 'avg'])
team_df = team_df.sort_values(by='xBA', ascending=False)
plt.figure(figsize=(10, 6))
plt.scatter(team_df['avg'], team_df['xBA'])
plt.xlabel('Actual Batting Average')
plt.ylabel('xBA')
plt.title('Team xBA vs Actual Batting Average for 2025 Season')
plt.grid()
plt.savefig("data/team_xba_vs_avg.png")