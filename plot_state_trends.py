import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FixedLocator, FuncFormatter

import utils

# Option to enable subplot mode
subplot_mode = True  # Set to True for subplot, False for single plot
USE_FUTURE = False  # Set to True to use the future simulation data

# Option to enable house margins
plot_house_margins = False  # Set to True to include house margins
house_on_same_plot = False  # Set to True to plot house and pres on the same plot


# Read and combine historical and future data if USE_FUTURE is True
if USE_FUTURE:
    df_hist = pd.read_csv('presidential_margins.csv')
    df_future = pd.read_csv('presidential_margins_future.csv')
    df = df_future#pd.concat([df_hist[df_hist['year'] <= 2024], df_future], ignore_index=True)
else:
    df = pd.read_csv('presidential_margins.csv')

# Read house margins if enabled
if plot_house_margins:
    house_df = pd.read_csv('house_margins.csv')
    house_df = house_df[house_df['year'] <= 2024]  # Only use years <= 2024

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

# Open files for writing outside the loop
left_file = open("trended_left_in_2024.txt", "w")
right_file = open("trended_right_in_2024.txt", "w")

for state in states:
    state_df = df[df['abbr'] == state]
    years = state_df['year']
    pres_margin = state_df['pres_margin']
    national_margin = state_df['national_margin']
    relative_margin = state_df['relative_margin']

    if plot_house_margins:
        house_state_df = house_df[house_df['abbr'] == state]
        house_years = house_state_df['year']
        house_margin = house_state_df['house_margin']
        house_national_margin = house_state_df['national_margin']
        house_relative_margin = house_state_df['relative_margin']

    # Find where the future years start (first year > 2024)
    future_start = None
    if USE_FUTURE:
        years_sorted = sorted(years)
        for idx, y in enumerate(years_sorted):
            if y > 2024:
                future_start = y
                break
    # Clear the files once before the loop
    if subplot_mode and not plot_house_margins:
        open("trended_left_in_2024.txt", "w").close()
        open("trended_right_in_2024.txt", "w").close()
    if subplot_mode:
        if plot_house_margins and house_on_same_plot:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            # Left plot: line plot
            # Color pres_margin points by party
            pres_colors = ['deepskyblue' if m > 0 else 'red' for m in pres_margin]
            house_colors = ['deepskyblue' if m > 0 else 'red' for m in house_margin]
            axes[0].plot(years, pres_margin, label='Presidential Margin', marker='o', linestyle='-')
            axes[0].plot(house_years, house_margin, label='House Margin', marker='o', linestyle='--')
            axes[0].plot(years, national_margin, label='Pres National Margin', marker='o', color='gold')
            axes[0].plot(house_years, house_national_margin, label='House National Margin', marker='o', color='silver')
            axes[0].set_title(f'{state} Margins')
            axes[0].set_xlabel('Year')
            axes[0].set_ylabel('Margin')
            y_vals = axes[0].get_yticks()
            axes[0].yaxis.set_major_locator(FixedLocator(y_vals))
            axes[0].yaxis.set_major_formatter(FuncFormatter(lambda y, _: utils.lean_str(y)))
            print(axes[0].get_lines())
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[0].axhline(0, color='red', linestyle='--', linewidth=1)
            axes[0].set_xticks(years)
            axes[0].set_xticklabels(years, rotation=45)
            # Add yellow dashed vertical line between 2024 and first future year
            if USE_FUTURE and future_start is not None:
                axes[0].axvline(x=2026, color='yellow', linestyle='--', linewidth=3, label='Future Boundary')

            # Right plot: bar graph of relative margin
            bar_width = 0.4
            x_indices = range(len(years))
            axes[1].bar([x - bar_width / 2 for x in x_indices], relative_margin, width=bar_width, label='Pres Relative Margin', color='deepskyblue')
            axes[1].bar([x + bar_width / 2 for x in x_indices], house_relative_margin, width=bar_width, label='House Relative Margin', color='orange')
            axes[1].set_title(f'{state} Relative Margins')
            axes[1].set_xlabel('Year')
            axes[1].set_ylabel('Relative Margin')
            y_vals = axes[1].get_yticks()
            axes[1].yaxis.set_major_locator(FixedLocator(y_vals))
            axes[1].yaxis.set_major_formatter(FuncFormatter(lambda y, _: utils.lean_str(y)))
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].axhline(0, color='red', linestyle='--', linewidth=1)
            axes[1].set_xticks(x_indices)
            axes[1].set_xticklabels(years, rotation=45)
            if USE_FUTURE and future_start is not None:
                axes[1].axvline(x=2026, color='yellow', linestyle='--', linewidth=3, label='Future Boundary')

            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, f'{state}_trend_subplot.png'))
            plt.close(fig)
        elif plot_house_margins:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            # Top row: Presidential margins
            pres_colors = ['deepskyblue' if m > 0 else 'red' for m in pres_margin]
            axes[0, 0].plot(years, pres_margin, label='Presidential Margin', marker='o', linestyle='-')
            axes[0, 0].plot(years, national_margin, label='National Margin', marker='o', color='gold')
            axes[0, 0].set_title(f'{state} Presidential Margins')
            axes[0, 0].set_xlabel('Year')
            axes[0, 0].set_ylabel('Margin')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=1)
            axes[0, 0].set_xticks(years)
            axes[0, 0].set_xticklabels(years, rotation=45)

            axes[0, 1].bar(years, relative_margin, color=pres_colors)
            axes[0, 1].set_title(f'{state} Presidential Relative Margins')
            axes[0, 1].set_xlabel('Year')
            axes[0, 1].set_ylabel('Relative Margin')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].axhline(0, color='red', linestyle='--', linewidth=1)
            axes[0, 1].set_xticks(years)
            axes[0, 1].set_xticklabels(years, rotation=45)

            # Bottom row: House margins
            house_colors = ['deepskyblue' if m > 0 else 'red' for m in house_margin]
            axes[1, 0].plot(house_years, house_margin, label='House Margin', marker='o', linestyle='--')
            axes[1, 0].plot(house_years, house_national_margin, label='House National Margin', marker='o', color='silver')
            axes[1, 0].set_title(f'{state} House Margins')
            axes[1, 0].set_xlabel('Year')
            axes[1, 0].set_ylabel('Margin')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=1)
            axes[1, 0].set_xticks(house_years)
            axes[1, 0].set_xticklabels(house_years, rotation=45)

            axes[1, 1].bar(house_years, house_relative_margin, color=house_colors)
            axes[1, 1].set_title(f'{state} House Relative Margins')
            axes[1, 1].set_xlabel('Year')
            axes[1, 1].set_ylabel('Relative Margin')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=1)
            axes[1, 1].set_xticks(house_years)
            axes[1, 1].set_xticklabels(house_years, rotation=45)

            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, f'{state}_trend_subplot.png'))
            plt.close(fig)
        elif subplot_mode and not plot_house_margins:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # Left plot: line plot for presidential margins
            pres_colors = ['deepskyblue' if m > 0 else 'red' for m in pres_margin]
            axes[0].plot(years, pres_margin, label='Presidential Margin', marker='o', linestyle='-', color='gray')
            axes[0].scatter(years, pres_margin, c=pres_colors, s=60, zorder=3, label='Pres Margin Points')
            axes[0].plot(years, national_margin, label='National Margin', marker='o', color='gold')
            axes[0].set_title(f'{state} Presidential Margins')
            axes[0].set_xlabel('Year')
            axes[0].set_ylabel('Margin')
            y_vals = axes[0].get_yticks()
            axes[0].set_yticklabels([utils.lean_str(y_val) for y_val in y_vals], color='white')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[0].axhline(0, color='red', linestyle='--', linewidth=1)
            axes[0].set_xticks(years)
            axes[0].set_xticklabels(years, rotation=45)

            # Right plot: bar graph for relative margins
            bar_width = 0.4
            x_indices = range(len(years))
            axes[1].bar(x_indices, relative_margin, width=bar_width, label='Pres Relative Margin', color=pres_colors)
            if relative_margin.iloc[-1] > relative_margin.iloc[-2]:
                #print(f"{state} trended left")
                left_file.write(f"{state} trended left ({utils.lean_str(relative_margin.iloc[-2])} -> {utils.lean_str(relative_margin.iloc[-1])},\tdifference: {utils.lean_str(relative_margin.iloc[-1] - relative_margin.iloc[-2])})\n")
            elif relative_margin.iloc[-1] < relative_margin.iloc[-2]:
                #print(f"{state} trended right")
                right_file.write(f"{state} trended right ({utils.lean_str(relative_margin.iloc[-2])} -> {utils.lean_str(relative_margin.iloc[-1])},\tdifference: {utils.lean_str(relative_margin.iloc[-1] - relative_margin.iloc[-2])})\n")
                
            axes[1].set_title(f'{state} Relative Margins')
            axes[1].set_xlabel('Year')
            axes[1].set_ylabel('Relative Margin')
            y_vals = axes[1].get_yticks()
            axes[1].set_yticklabels([utils.lean_str(y_val) if y_val != 0 else "NATIONAL MARGIN" for y_val in y_vals], color='white')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].axhline(0, color='red', linestyle='--', linewidth=1)
            axes[1].set_xticks(x_indices)
            axes[1].set_xticklabels(years, rotation=45)

            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, f'{state}_trend_subplot.png'))
            plt.close(fig)
        else:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
            pres_colors = ['deepskyblue' if m > 0 else 'red' for m in pres_margin]
            ax.plot(years, pres_margin, label='Presidential Margin', color='gray', marker='o', linestyle='-')
            ax.scatter(years, pres_margin, c=pres_colors, s=60, zorder=3, label='Pres Margin Points')
            ax.plot(years, national_margin, label='National Margin', marker='o', color='gold')
            ax.set_title(f'{state} Presidential vs National Margin Over Time')
            ax.set_xlabel('Year')
            ax.set_ylabel('Margin')
            y_vals = plt.gca().get_yticks()
            plt.gca().set_yticklabels([utils.lean_str(y_val) for y_val in y_vals], color='white')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color='purple', linestyle='--', linewidth=1)
            ax.set_xticks(years)
            ax.set_xticklabels(years, rotation=45)
            if USE_FUTURE and future_start is not None:
                ax.axvline(x=2026, color='yellow', linestyle='--', linewidth=3, label='Future Boundary')
            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, f'{state}_trend_subplot.png'))
            plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        pres_colors = ['deepskyblue' if m > 0 else 'red' for m in pres_margin]
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

# Close the files after the loop
left_file.close()
right_file.close()
