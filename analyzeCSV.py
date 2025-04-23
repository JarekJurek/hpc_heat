import pandas as pd
import matplotlib.pyplot as plt

print("Loading and analyzing building temperature data...")
df = pd.read_csv('building_summary_stats.csv')
print(f"Analyzed {len(df)} buildings\n")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
for ax in axes.flatten():
    ax.grid(True, linestyle='--', alpha=0.7)

# Mean temperature
axes[0, 0].hist(df['mean_temp'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Distribution of Mean Temperatures', fontsize=14)
axes[0, 0].set_xlabel('Mean Temperature (°C)', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)

# Standard deviation 
axes[0, 1].hist(df['std_temp'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
axes[0, 1].set_title('Distribution of Temperature Standard Deviations', fontsize=14)
axes[0, 1].set_xlabel('Standard Deviation (°C)', fontsize=12)
axes[0, 1].set_ylabel('Frequency', fontsize=12)

# Percentage above 18°C
axes[1, 0].hist(df['pct_above_18'], bins=30, alpha=0.7, color='salmon', edgecolor='black')
axes[1, 0].set_title('Percentage of Area Above 18°C', fontsize=14)
axes[1, 0].set_xlabel('Percentage (%)', fontsize=12)
axes[1, 0].set_ylabel('Frequency', fontsize=12)

# Percentage below 15°C 
axes[1, 1].hist(df['pct_below_15'], bins=30, alpha=0.7, color='mediumpurple', edgecolor='black')
axes[1, 1].set_title('Percentage of Area Below 15°C', fontsize=14)
axes[1, 1].set_xlabel('Percentage (%)', fontsize=12)
axes[1, 1].set_ylabel('Frequency', fontsize=12)

plt.tight_layout()
plt.savefig('Figures/temperature_distributions.png')
print("Plots saved to 'Figures/temperature_distributions.png'")


avg_mean_temp = df['mean_temp'].mean()
avg_std_temp = df['std_temp'].mean()
buildings_above_18_50pct = (df['pct_above_18'] >= 50).sum()
buildings_below_15_50pct = (df['pct_below_15'] >= 50).sum()
print("\n===== Temperature Analysis Results =====")
print(f"Average mean temperature: {avg_mean_temp:.2f}°C")
print(f"Average temperature standard deviation: {avg_std_temp:.2f}°C")
print(f"Buildings with at least 50% area above 18°C: {buildings_above_18_50pct}")
print(f"Buildings with at least 50% area below 15°C: {buildings_below_15_50pct}")