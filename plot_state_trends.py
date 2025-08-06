import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
from scipy.optimize import curve_fit

import utils

# Option to enable subplot mode
subplot_mode = True  # Set to True for subplot, False for single plot
USE_FUTURE = True  # Set to True to use the future simulation data

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
output_dir = 'state_trends/state_trends_future' if USE_FUTURE else 'state_trends'
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

# Define the sine function for fitting
def sine_function(x, A, B, C, D):
    return A * np.sin(B * x + C) + D

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
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Left plot: line plot for presidential margins
        pres_colors = ['deepskyblue' if m > 0 else 'red' for m in pres_margin]
        axes[0].plot(years, pres_margin, label='Presidential Margin', marker='o', linestyle='-', color='gray')
        axes[0].scatter(years, pres_margin, c=pres_colors, s=60, zorder=3, label='Pres Results')
        axes[0].plot(years, national_margin, label='National Margin', marker='o', color='gold')
        axes[0].set_title(f'{state} Presidential Margins')
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel('Margin')
        #axes[0].set_yticks(pres_margin)
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
        #axes[1].bar(x_indices, relative_margin, width=bar_width, label='Pres Relative Margin', color=pres_colors)

        # Add a line of best fit to the bar plot
        z = np.polyfit(x_indices, relative_margin, 1)
        p = np.poly1d(z)
        axes[1].plot(x_indices, p(x_indices), linestyle='--', color='yellow', label='Line of Best Fit')

        # Verify the data being used for sine fitting
        print(f"Fitting sine for {state} with relative_margin: {relative_margin.values}")
        print(f"Years: {years.values}")

        # Ensure data is sorted by year
        sorted_indices = np.argsort(years)
        relative_margin = relative_margin.iloc[sorted_indices]
        x_indices = np.arange(len(relative_margin))

        # Add a sine of best fit to the bar plot
        try:
            # Fit the sine function to the data
            params, _ = curve_fit(sine_function, x_indices, relative_margin, p0=[1, 1, 0, np.mean(relative_margin)])
            A, B, C, D = params

            # Generate the sine curve
            sine_fit = sine_function(np.array(x_indices), A, B, C, D)

            # Generate a denser set of x values for a smoother sine curve
            x_dense = np.linspace(min(x_indices.tolist()), max(x_indices.tolist()), 500)
            sine_fit_dense = sine_function(x_dense, A, B, C, D)

            # Plot the smoother sine of best fit
            axes[1].plot(x_dense, sine_fit_dense, linestyle='--', color='cyan', label='Sine of Best Fit')
        except Exception as e:
            print(f"Could not fit sine function for {state}: {e}")
            
        bars = axes[1].bar(x_indices, relative_margin, width=bar_width, label='Pres Relative Margin', color=pres_colors)
        axes[1].bar_label(bars, labels=[utils.lean_str(v) for v in relative_margin], padding=3, fontsize=8, color='white')

        if relative_margin.iloc[-1] > relative_margin.iloc[-2]:
            #print(f"{state} trended left")
            left_file.write(f"{state} trended left ({utils.lean_str(relative_margin.iloc[-2])} -> {utils.lean_str(relative_margin.iloc[-1])},\tdifference: {utils.lean_str(relative_margin.iloc[-1] - relative_margin.iloc[-2])})\n")
        elif relative_margin.iloc[-1] < relative_margin.iloc[-2]:
            #print(f"{state} trended right")
            right_file.write(f"{state} trended right ({utils.lean_str(relative_margin.iloc[-2])} -> {utils.lean_str(relative_margin.iloc[-1])},\tdifference: {utils.lean_str(relative_margin.iloc[-1] - relative_margin.iloc[-2])})\n")
            
        axes[1].set_title(f'{state} Relative Margins')
        axes[1].set_xlabel('Year')
        axes[1].set_ylabel('Relative Margin')
        #axes[1].set_yticks(relative_margin)
        y_vals = axes[1].get_yticks()
        axes[1].set_yticklabels([utils.lean_str(y_val) if y_val != 0 else "NAT. MARGIN" for y_val in y_vals], color='white')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(0, color='red', linestyle='--', linewidth=1)
        axes[1].set_xticks(x_indices)
        axes[1].set_xticklabels(years, rotation=45)

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
