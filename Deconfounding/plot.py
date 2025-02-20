import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize a list to store all data
data = []

# Regular expressions to match lines with average rewards and window size
baseline_pattern = re.compile(r'Baseline PPO Agent - Episode (\d+), Average Reward \(last (\d+)\): ([\d\.\-]+)')
causal_pattern = re.compile(r'Causal-enhanced PPO Agent - Episode (\d+), Average Reward \(last (\d+)\): ([\d\.\-]+)')

# Read the log file with explicit encoding
with open('output.log', 'r', encoding='utf-8') as file:
    for line in file:
        # Check for Baseline PPO average reward
        baseline_match = baseline_pattern.search(line)
        if baseline_match:
            episode = int(baseline_match.group(1))
            window_size = int(baseline_match.group(2))
            reward = float(baseline_match.group(3))
            data.append({'Episode': episode, 'Reward': reward, 'Agent': 'Tabular Augmented'})
        
        # Check for Causal PPO average reward
        causal_match = causal_pattern.search(line)
        if causal_match:
            episode = int(causal_match.group(1))
            window_size = int(causal_match.group(2))
            reward = float(causal_match.group(3))
            data.append({'Episode': episode, 'Reward': reward, 'Agent': 'Causal PPO (DualVAE-HCD)'})

# Create a DataFrame
df = pd.DataFrame(data)

# Sort the DataFrame by Agent and Episode
df = df.sort_values(by=['Agent', 'Episode'])

# Set the window size for moving average
window_size = 50

# Compute the moving average for each agent
df['Moving_Avg'] = df.groupby('Agent')['Reward'].transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())

# Set up the plot aesthetics using seaborn
sns.set(style="darkgrid")
plt.figure(figsize=(12, 6))

# Plotting using seaborn's lineplot
sns.lineplot(data=df, x='Episode', y='Moving_Avg', hue='Agent', linewidth=2)

# Labels and titles
plt.xlabel('Episode', fontsize=14)
plt.ylabel(f'Average Reward ({window_size}-episode moving average)', fontsize=14)
plt.title('Comparison of Causal PPO and Tabular Augmented Average Rewards', fontsize=16)
plt.legend(title='Agent', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add grid
plt.grid(True, linestyle='--', linewidth=0.5)

# Adjust layout to prevent clipping of labels/titles
plt.tight_layout()

# Optionally save the figure
# plt.savefig('ppo_comparison_plot.png', dpi=300)

# Show the plot
plt.show()
