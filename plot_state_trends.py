import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
from scipy.optimize import curve_fit

import utils

# Option to enable subplot mode
subplot_mode = True  # Set to True for subplot, False for single plot
USE_FUTURE = False  # Set to True to use the future simulation data

# Optional: only include data from this year onward (None means use all years)
start_year = None  # e.g. 2000 to only plot years >= 2000
# Optional end year (None means up to the latest available)
end_year = None  # e.g. 2024 to limit to years <= 2024
if end_year is None:
    end_year = 2024

# Sine fitting configuration: require a minimum number of points and a minimum R^2
sine_min_points = 3
sine_r2_threshold = 0.5

# Option: include delta subplot (bottom row merged). If True, figure is 2x2 with bottom row merged
include_deltas = True

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

# Ensure base output directory exists
base_output_dir = 'state_trends/state_trends_future' if USE_FUTURE else 'state_trends'

# If a year filter is provided, create a subdirectory to avoid clobbering the default outputs
if start_year is None and (end_year is None or end_year == 2024):
    output_dir = base_output_dir
else:
    if start_year is not None and end_year is not None:
        suffix = f"{start_year}_{end_year}"
    elif start_year is not None:
        suffix = f"{start_year}_plus"
    else:
        suffix = f"up_to_{end_year}"
    output_dir = os.path.join(base_output_dir, suffix)

os.makedirs(output_dir, exist_ok=True)

# Clear only files in the chosen output directory
for file in os.listdir(output_dir):
    file_path = os.path.join(output_dir, file)
    if os.path.isfile(file_path):
        os.remove(file_path)

# Get unique states
states = df['abbr'].unique()

plt.style.use('dark_background')

# Open trended files inside the output directory
left_file = open(os.path.join(output_dir, f"trended_left_in_{end_year}.txt"), "w")
right_file = open(os.path.join(output_dir, f"trended_right_in_{end_year}.txt"), "w")

# Define the sine function for fitting
def sine_function(x, A, B, C, D):
    return A * np.sin(B * x + C) + D


def create_figure_axes(include_deltas, figsize=(16, 6)):
    """Create figure and axes. If include_deltas is True, return three axes (ax_line, ax_bar, ax_delta).
    Otherwise return (ax_line, ax_bar, None) with a 1x2 layout."""
    if include_deltas:
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
        ax_line = fig.add_subplot(gs[0, 0])
        ax_bar = fig.add_subplot(gs[0, 1])
        ax_delta = fig.add_subplot(gs[1, :])
        return fig, (ax_line, ax_bar, ax_delta)
    else:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        return fig, (axes[0], axes[1], None)


def style_line_axis(ax, years, pres_margin, national_margin, state):
    pres_colors = ['deepskyblue' if m > 0 else 'red' for m in pres_margin]
    ax.plot(years, pres_margin, label='Presidential Margin', marker='o', linestyle='-', color='gray')
    ax.scatter(years, pres_margin, c=pres_colors, s=60, zorder=3, label='Pres Results')
    ax.plot(years, national_margin, label='National Margin', marker='o', color='gold')
    ax.set_title(f'{state} Presidential Margins')
    ax.set_xlabel('Year')
    ax.set_ylabel('Margin')
    y_vals = ax.get_yticks()
    ax.set_yticklabels([utils.lean_str(y_val) for y_val in y_vals], color='white')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='red', linestyle='--', linewidth=1)
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45)
    return pres_colors


def style_bar_axis(ax, x_indices, relative_margin, pres_colors, years):
    # Line of best fit
    try:
        z = np.polyfit(x_indices, relative_margin, 1)
        p = np.poly1d(z)
        ax.plot(x_indices, p(x_indices), linestyle='--', color='yellow', label='Line of Best Fit')
    except Exception:
        pass
    bars = ax.bar(x_indices, relative_margin, width=0.4, label='Pres Relative Margin', color=pres_colors)
    ax.bar_label(bars, labels=[utils.lean_str(v) for v in relative_margin], padding=3, fontsize=8, color='white')
    ax.set_title('Relative Margins')
    ax.set_xlabel('Year')
    ax.set_ylabel('Relative Margin')
    y_vals = ax.get_yticks()
    ax.set_yticklabels([utils.lean_str(y_val) if y_val != 0 else "NAT. MARGIN" for y_val in y_vals], color='white')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='red', linestyle='--', linewidth=1)
    ax.set_xticks(x_indices)
    ax.set_xticklabels(years, rotation=45)


def plot_delta_axis(ax, x_indices, deltas, years_for_delta):
    # x_indices and years_for_delta should align with deltas
    if len(deltas) == 0:
        ax.text(0.5, 0.5, 'No delta data', ha='center')
        return
    colors = ['deepskyblue' if d > 0 else 'red' for d in deltas]
    # Bars styled like the relative-margin bar plot
    bars = ax.bar(x_indices, deltas, width=0.4, label='Delta Relative Margin', color=colors)
    ax.bar_label(bars, labels=[utils.lean_str(v) for v in deltas], padding=3, fontsize=8, color='white')
    ax.set_title('Change in Relative Margin')
    ax.set_xlabel('Year')
    ax.set_ylabel('Delta Relative Margin')
    y_vals = ax.get_yticks()
    ax.set_yticklabels([utils.lean_str(y_val) if y_val != 0 else "NAT. MARGIN" for y_val in y_vals], color='white')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='red', linestyle='--', linewidth=1)
    ax.set_xticks(x_indices)
    ax.set_xticklabels(years_for_delta, rotation=45)

for state in states:
    state_df = df[df['abbr'] == state]
    # Apply start_year / end_year filters if provided
    if start_year is not None:
        state_df = state_df[state_df['year'] >= start_year]
    if end_year is not None:
        state_df = state_df[state_df['year'] <= end_year]

    # Skip states with no data in the requested range
    if state_df.empty:
        print(f"Skipping {state}: no data in requested year range")
        continue
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
    # Create figure & axes using the centralized helper; prefer subplot styling
    if subplot_mode:
        fig, (ax_line, ax_bar, ax_delta) = create_figure_axes(include_deltas)
    else:
        fig, (ax_line, ax_bar, ax_delta) = create_figure_axes(False, figsize=(10, 6))

    # Sort data by year for consistent plotting and fitting
    order = np.argsort(years.values)
    years_sorted = years.values[order]
    pres_sorted = pres_margin.values[order]
    nat_sorted = national_margin.values[order]
    rel_sorted = relative_margin.values[order]

    # Line plot (top-left)
    pres_colors = style_line_axis(ax_line, years_sorted, pres_sorted, nat_sorted, state)

    # Bar plot (top-right)
    x_indices = np.arange(len(rel_sorted))
    style_bar_axis(ax_bar, x_indices, rel_sorted, pres_colors, years_sorted)

    # Sine fitting: only if enough points and satisfactory R^2
    if len(x_indices) >= sine_min_points:
        try:
            params, _ = curve_fit(sine_function, x_indices, rel_sorted, p0=[1, 1, 0, np.mean(rel_sorted)])
            A, B, C, D = params
            sine_fit = sine_function(x_indices, A, B, C, D)
            ss_res = np.sum((rel_sorted - sine_fit) ** 2)
            ss_tot = np.sum((rel_sorted - np.mean(rel_sorted)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            print(f"Sine fit for {state}: R^2={r2:.3f} (n={len(x_indices)})")
            if r2 >= sine_r2_threshold:
                x_dense = np.linspace(x_indices.min(), x_indices.max(), 500)
                sine_fit_dense = sine_function(x_dense, A, B, C, D)
                ax_bar.plot(x_dense, sine_fit_dense, linestyle='--', color='cyan', label='Sine of Best Fit')
            else:
                print(f"Sine fit rejected for {state} because R^2 < {sine_r2_threshold}")
        except Exception as e:
            print(f"Could not fit sine function for {state}: {e}")
    else:
        print(f"Skipping sine fit for {state}: not enough points (need {sine_min_points}, have {len(x_indices)})")

    # Write trended files using sorted data (ensure we have at least 2 points)
    if not USE_FUTURE and len(rel_sorted) >= 2:
        diff = rel_sorted[-1] - rel_sorted[-2]
        if diff > 0:
            left_file.write(f"{state} trended left ({utils.lean_str(rel_sorted[-2])} -> {utils.lean_str(rel_sorted[-1])},\tdifference: {utils.lean_str(diff)})\n")
        elif diff < 0:
            right_file.write(f"{state} trended right ({utils.lean_str(rel_sorted[-2])} -> {utils.lean_str(rel_sorted[-1])},\tdifference: {utils.lean_str(diff)})\n")

    # Delta subplot (merged bottom row) if requested
    if include_deltas and ax_delta is not None:
        # compute year-to-year deltas and skip placeholder zeros
        deltas = np.diff(rel_sorted)
        years_for_delta = years_sorted[1:]
        mask = deltas != 0
    deltas_filtered = deltas[mask]
    years_filtered = years_for_delta[mask]
    x_idx_delta = np.arange(len(deltas_filtered))
    plot_delta_axis(ax_delta, x_idx_delta, deltas_filtered, years_filtered)

    # Finalize and save
    plt.tight_layout()
    filename = f'{state}_trend_subplot.png' if subplot_mode else f'{state}_trend.png'
    fig.savefig(os.path.join(output_dir, filename))
    plt.close(fig)

# Close the files after the loop
left_file.close()
right_file.close()
