import pandas as pd
import matplotlib.pyplot as plt

all_pitches = pd.read_parquet("data/2025_all_pitches.parquet")
data = pd.read_parquet("data/2025_hits.parquet")
team_stats = pd.read_csv("data/2025_team_stats.csv")

team_data = {}
for team in data['team'].unique():
    single_data = data[data['team'] == team]
    team_xba = single_data['xBA'].mean()
    team_avg = team_stats[team_stats['Team'] == team]['AVG'].iloc[0]
    team_data[team] = [team_xba, team_avg]

team_df = pd.DataFrame.from_dict(team_data, orient='index', columns=['xBA', 'avg'])
team_df = team_df.sort_values(by='xBA', ascending=False)
plt.figure(figsize=(10, 6))
plt.scatter(team_df['avg'], team_df['xBA'])
plt.xlabel('Actual Batting Average')
plt.ylabel('xBA')
plt.title('Team xBA vs Actual Batting Average for 2025 Season')
plt.grid()
plt.savefig("figures/team_xba_vs_avg.png")

team_df['Win%'] = team_stats.set_index('Team').loc[team_df.index]['Win%']
plt.figure(figsize=(10, 6))
plt.scatter(team_df['Win%'], team_df['xBA'])
plt.xlabel('Win Percentage')
plt.ylabel('xBA')
plt.title('Team xBA vs Win Percentage for 2025 Season')
plt.grid()
plt.savefig("figures/team_xba_vs_win%.png")