
import pandas as pd
import matplotlib.pyplot as plt
import os

# Option to enable subplot mode
subplot_mode = True  # Set to True for subplot, False for single plot
USE_FUTURE = True  # Set to True to use the future simulation data


# Read and combine historical and future data if USE_FUTURE is True
if USE_FUTURE:
    df_hist = pd.read_csv('presidential_margins.csv')
    df_future = pd.read_csv('presidential_margins_future.csv')
    df = df_future#pd.concat([df_hist[df_hist['year'] <= 2024], df_future], ignore_index=True)
else:
    df = pd.read_csv('presidential_margins.csv')

# Ensure output directory exists
output_dir = 'state_trends'
os.makedirs(output_dir, exist_ok=True)
# clear all files in the output directory
for file in os.listdir(output_dir):
    file_path = os.path.join(output_dir, file)
    if os.path.isfile(file_path):
        os.remove(file_path)

# Get unique states
states = df['abbr'].unique()

plt.style.use('dark_background')

for state in states:
    state_df = df[df['abbr'] == state]
    years = state_df['year']
    pres_margin = state_df['pres_margin']
    national_margin = state_df['national_margin']
    relative_margin = state_df['relative_margin']

    # Find where the future years start (first year > 2024)
    future_start = None
    if USE_FUTURE:
        years_sorted = sorted(years)
        for idx, y in enumerate(years_sorted):
            if y > 2024:
                future_start = y
                break

    if subplot_mode:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        # Left plot: line plot
        # Color pres_margin points by party
        pres_colors = ['blue' if m > 0 else 'red' for m in pres_margin]
        axes[0].plot(years, pres_margin, label='Presidential Margin', marker='o', linestyle='-')
        #axes[0].scatter(years, pres_margin, c=pres_colors, s=60, zorder=3, label='Pres Margin Points')
        axes[0].plot(years, national_margin, label='National Margin', marker='o', color='gold')
        axes[0].set_title(f'{state} Presidential vs National Margin')
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel('Margin')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(0, color='red', linestyle='--', linewidth=1)
        axes[0].set_xticks(years)
        axes[0].set_xticklabels(years, rotation=45)
        # Add yellow dashed vertical line between 2024 and first future year
        if USE_FUTURE and future_start is not None:
            axes[0].axvline(x=2026, color='yellow', linestyle='--', linewidth=3, label='Future Boundary')

        # Right plot: bar graph of relative margin
        # Color bars by pres_margin sign
        bar_colors = ['deepskyblue' if m > 0 else 'red' for m in pres_margin]
        axes[1].bar(years, relative_margin, color=bar_colors)
        axes[1].set_title(f'{state} Relative Margin (Bar)')
        axes[1].set_xlabel('Year')
        axes[1].set_ylabel('Relative Margin')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(0, color='red', linestyle='--', linewidth=1)
        axes[1].set_xticks(years)
        axes[1].set_xticklabels(years, rotation=45)
        if USE_FUTURE and future_start is not None:
            axes[1].axvline(x=2026, color='yellow', linestyle='--', linewidth=3, label='Future Boundary')

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f'{state}_trend_subplot.png'))
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        pres_colors = ['blue' if m > 0 else 'red' for m in pres_margin]
        ax.plot(years, pres_margin, label='Presidential Margin', color='gray', marker='o', linestyle='-')
        ax.scatter(years, pres_margin, c=pres_colors, s=60, zorder=3, label='Pres Margin Points')
        ax.plot(years, national_margin, label='National Margin', marker='o', color='gold')
        ax.set_title(f'{state} Presidential vs National Margin Over Time')
        ax.set_xlabel('Year')
        ax.set_ylabel('Margin')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='red', linestyle='--', linewidth=1)
        ax.set_xticks(years)
        ax.set_xticklabels(years, rotation=45)
        if USE_FUTURE and future_start is not None:
            ax.axvline(x=2026, color='yellow', linestyle='--', linewidth=3, label='Future Boundary')
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f'{state}_trend.png'))
        plt.close(fig)
