import pandas as pd
import matplotlib.pyplot as plt

all_pitches = pd.read_parquet("data/2025_all_pitches.parquet")
data = pd.read_parquet("data/2025_hits.parquet")
batter_data = {}
# Only use batters with more than 50 plate appearances
batter_list = data['batter'].value_counts()
batter_list = batter_list[batter_list > 50].index.tolist()
for batter in batter_list:
    single_data = data[data['batter'] == batter]
    batter_xba = single_data['xBA'].mean()
    batter_strikeouts_count = len(all_pitches[(all_pitches['batter'] == batter) & (all_pitches['events'] == 'strikeout')])
    batter_avg = len(single_data[single_data['hit'] == 1]) / (len(single_data) + batter_strikeouts_count) # Not 100% accurate
    batter_data[batter] = [batter_xba, batter_avg]
    
batter_df = pd.DataFrame.from_dict(batter_data, orient='index', columns=['xBA', 'avg'])
batter_df = batter_df.sort_values(by='xBA', ascending=False)
plt.figure(figsize=(10, 6))
plt.scatter(batter_df['avg'], batter_df['xBA'])
plt.xlabel('Actual Batting Average')
plt.ylabel('xBA')
plt.title('Player xBA vs Actual Batting Average for 2025 Season')
plt.grid()
plt.savefig("figures/player_xba_vs_avg.png")

player_id = 592450 # Aaron Judge
player_data = data[data['batter'] == player_id]
plt.figure(figsize=(8, 6))
player_data.boxplot(column='xBA', by='hit')
plt.title(f'xBA Distribution for Player ID: {player_id}')
plt.suptitle('')
plt.xlabel('Hit Outcome (0 = No Hit, 1 = Hit)')
plt.ylabel('xBA')
plt.grid()
plt.savefig(f"figures/player_xba_boxplot.png")