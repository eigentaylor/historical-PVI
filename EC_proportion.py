import os
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils  # Import utils for lean_str


# Configuration
USE_FUTURE = False  # Set True to append future projections to history
OUTPUT_DIR_BASE = "EC_proportion"
# clear files in the output directory
if not os.path.exists(OUTPUT_DIR_BASE):
    os.makedirs(OUTPUT_DIR_BASE)
# Clear any existing files in the output directory
for filename in os.listdir(OUTPUT_DIR_BASE):
    file_path = os.path.join(OUTPUT_DIR_BASE, filename)
    if os.path.isfile(file_path):
        os.remove(file_path)

# Add a configuration option for scaling
SCALE_TO_TOTAL_EVS = True  # Set True to scale y-axis to (0, 538)
FUTURE_YEAR_MARK = 2026  # Vertical marker year when future data present

# Category thresholds (relative_margin)
# Order matters (from strong R to strong D)
CATEGORY_ORDER: List[str] = [
    "lockedR", "safeR", "leanR", "tiltR",
    "swing",
    "tiltD", "leanD", "safeD", "lockedD",
]

# Colors (dark mode friendly): R darkest -> lightest, swing purple, D lightest -> darkest
CATEGORY_COLORS = {
    "lockedR": "#8B0000",      # darkred
    "safeR":   "#B22222",      # firebrick
    "leanR":   "#CD5C5C",      # indianred
    "tiltR":   "#F08080",      # lightcoral
    "swing":   "#C3B1E1",      # light purple
    "tiltD":   "#87CEFA",      # lightskyblue
    "leanD":   "#6495ED",      # cornflowerblue
    "safeD":   "#4169E1",      # royalblue
    "lockedD": "#00008B",      # darkblue
}

LOCKED = 0.2
SAFE = 0.1
LEAN = 0.08
TILT = 0.05

# Ranges for each category
CATEGORY_THRESHOLDS = {
    "lockedR": -LOCKED,
    "safeR":   -SAFE,
    "leanR":   -LEAN,
    "tiltR":   -TILT,
    "swing":    TILT,
    "tiltD":    LEAN,
    "leanD":    SAFE,
    "safeD":    LOCKED,
    "lockedD":  float("inf"),
}

def get_category_ranges(thresholds: dict, order: List[str]) -> dict:
    """Generate human-readable ranges for each category using CATEGORY_THRESHOLDS."""
    ranges = {}
    # Build a list of thresholds in order
    thresh_vals = [thresholds[c] for c in order]
    for i, cat in enumerate(order):
        low = thresh_vals[i]
        if i == 0:
            # Lowest category: less than threshold
            ranges[cat] = f"{utils.lean_str(low)}+"
        elif i == len(order) - 1:
            # Highest category: greater than previous threshold
            ranges[cat] = f"{utils.lean_str(thresh_vals[i-1])}+"
        else:
            # Range between previous and current threshold
            if abs(low) >= abs(thresh_vals[i-1]):
                ranges[cat] = f"{utils.lean_str(thresh_vals[i-1])} to {utils.lean_str(low)}" 
            else:
                ranges[cat] = f"{utils.lean_str(low)} to {utils.lean_str(thresh_vals[i-1])}"
    return ranges

CATEGORY_RANGES = get_category_ranges(CATEGORY_THRESHOLDS, CATEGORY_ORDER)


def filter_year_range(df: pd.DataFrame, year_range=None) -> pd.DataFrame:
    """Filter df to inclusive [start, end] if year_range is provided; otherwise return df."""
    if not year_range:
        return df
    start, end = year_range
    return df[(df["year"] >= start) & (df["year"] <= end)].copy()


def make_range_suffix(year_range=None) -> str:
    """Return a filename suffix like _2000_2024 for a given range, else empty string."""
    if not year_range:
        return ""
    start, end = year_range
    return f"_{int(start)}_{int(end)}"


def load_data(use_future: bool) -> pd.DataFrame:
    """Load margins data. Optionally append future projections to history."""
    if use_future:
        df_hist = pd.read_csv("presidential_margins.csv")
        df_future = pd.read_csv("presidential_margins_future.csv")
        # Use history up to and including 2024, then append future (>2024 only)
        df = pd.concat([
            df_hist[df_hist["year"] <= 2024],
            df_future[df_future["year"] > 2024]
        ], ignore_index=True)
    else:
        df = pd.read_csv("presidential_margins.csv")
    # Ensure expected columns
    required = {"year", "abbr", "electoral_votes", "relative_margin"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in CSV: {missing}")
    return df


def categorize_relative_margin(x: float) -> str:
    """Map relative_margin to category per provided thresholds."""
    # Using strict < thresholds, otherwise falls into the next band toward center.
    for cat in CATEGORY_ORDER:
        if x < CATEGORY_THRESHOLDS[cat]:
            return cat
    return CATEGORY_ORDER[-1]


def build_proportion_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe indexed by year with columns per category of EV proportions and their deltas."""
    tmp = df.copy()
    # Convert relative_margin to float if needed
    tmp["relative_margin"] = pd.to_numeric(tmp["relative_margin"], errors="coerce")
    tmp = tmp.dropna(subset=["relative_margin", "electoral_votes"])  # drop any malformed rows

    # Categorize each state-year
    tmp["category"] = tmp["relative_margin"].apply(categorize_relative_margin)

    # Sum EVs by year and category
    by_year_cat = (
        tmp.groupby(["year", "category"], as_index=False)["electoral_votes"].sum()
    )
    pivot = by_year_cat.pivot(index="year", columns="category", values="electoral_votes").fillna(0)

    # Ensure all categories exist and ordered
    for c in CATEGORY_ORDER:
        if c not in pivot.columns:
            pivot[c] = 0
    pivot = pivot[CATEGORY_ORDER].sort_index()

    # Convert to proportions of that year's total EVs
    totals = pivot.sum(axis=1)
    proportions = pivot.div(totals, axis=0)

    # Calculate deltas (year-over-year differences)
    deltas = proportions.diff().fillna(0)  # Fill NaN for the first year with 0
    deltas.columns = [f"{col}_delta" for col in deltas.columns]

    # Combine proportions and deltas into a single DataFrame
    combined = pd.concat([proportions, deltas], axis=1)
    return combined


def plot_stacked_area(proportions: pd.DataFrame, output_dir: str, filename_suffix: str = "") -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Cast to numpy arrays for plotting
    years = proportions.index.astype(int).to_numpy()
    series = [proportions[c].to_numpy(dtype=float) for c in CATEGORY_ORDER]
    colors = [CATEGORY_COLORS[c] for c in CATEGORY_ORDER]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(14, 8))

    if SCALE_TO_TOTAL_EVS:
        # Scale proportions to total EVs (538)
        series = [s * 538 for s in series]
        y_label = "Total EVs"
        y_limit = 538
        dashed_line_value = 270
        dashed_line_label = "270 EVs"
        yticks = np.concatenate([np.arange(0, 270+1, 15), np.arange(283, 538+1, 15), [270, 0, 538]])
        ax.set_yticks(yticks)
        # Custom tick labels: if > 270, show as "538-x"
        ytick_labels = []
        for val in yticks:
            if val > 270:
                ytick_labels.append(f"{538-val}")
            else:
                ytick_labels.append(str(val))
        ax.set_yticklabels(ytick_labels)
    else:
        y_label = "Proportion of EVs"
        y_limit = 1
        dashed_line_value = 0.5
        dashed_line_label = "0.5"

    # Unpack the series so stackplot receives y1, y2, ...
    ax.stackplot(years, *series, labels=CATEGORY_ORDER, colors=colors, alpha=0.9)

    title = "Electoral College EV Proportions by Partisan Category of Relative Margins Over Time"
    if years.size:
        title += f" ({int(years.min())}-{int(years.max())})"
    ax.set_title(title)
    ax.set_xlabel("Election year")
    ax.set_ylabel(y_label)
    ax.set_ylim(0, y_limit)
    ax.set_xlim(int(years.min()), int(years.max()))
    # set x ticks to be every 4 years
    ax.set_xticks(np.arange(int(years.min()), int(years.max()) + 1, 4))
    ax.grid(True, alpha=0.25)

    # dash red line at the specified value for visual reference
    ax.axhline(dashed_line_value, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.text(int(years.min()), dashed_line_value, dashed_line_label, color='red', alpha=0.7, verticalalignment='bottom')

    # If data includes years past 2024, draw a big yellow dashed vertical line at 2026
    if years.max() > 2024:
        ax.axvline(FUTURE_YEAR_MARK, color='#FFD700', linestyle='--', linewidth=2.5, alpha=0.95)
        ax.text(FUTURE_YEAR_MARK + 0.2, y_limit * 0.98, str(FUTURE_YEAR_MARK), color='#FFD700', rotation=90,
                va='top', ha='left', fontsize=10)

    # Update legend to include ranges
    legend_labels = [f"{cat} ({CATEGORY_RANGES[cat]})" for cat in CATEGORY_ORDER]
    leg = ax.legend(legend_labels, loc="center left", bbox_to_anchor=(1.02, 0.5), framealpha=0.2)
    for text in leg.get_texts():
        text.set_color("white")

    fig.tight_layout()
    out_png = os.path.join(output_dir, f"EC_proportions{filename_suffix}.png")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def write_category_details(df: pd.DataFrame, output_dir: str, filename_suffix: str = "") -> None:
    """Write text files for each year listing states in each category."""
    os.makedirs(output_dir, exist_ok=True)

    # Ensure relative_margin is numeric
    df["relative_margin"] = pd.to_numeric(df["relative_margin"], errors="coerce")
    df = df.dropna(subset=["relative_margin", "electoral_votes"])  # drop malformed rows

    # Group by year
    years = df["year"].unique()
    for year in years:
        year_df = df[df["year"] == year]
        year_file = os.path.join(output_dir, f"{year}_categories{filename_suffix}.txt")

        with open(year_file, "w") as f:
            f.write(f"Year: {year}\n\n")
            for category in CATEGORY_ORDER:
                # calculate all states in this category
                # Assign category to each state using categorize_relative_margin
                year_df["category"] = year_df["relative_margin"].apply(categorize_relative_margin)
                cat_df = year_df[year_df["category"] == category]
                total_evs = cat_df["electoral_votes"].sum()

                f.write(f"Category: {category} ({CATEGORY_RANGES[category]})\n")
                f.write(f"Total EVs: {total_evs} ({total_evs / 538:.2%})\n")
                f.write("States:\n")

                # Sort states in this category by relative_margin (ascending)
                cat_df_sorted = cat_df.sort_values("relative_margin")
                for _, row in cat_df_sorted.iterrows():
                    state = row["abbr"]
                    evs = row["electoral_votes"]
                    margin = utils.lean_str(row["relative_margin"])
                    f.write(f"  {state}: {evs} EVs, {margin}\n")

                f.write("\n")


def run_for_range(df: pd.DataFrame, output_dir: str, year_range=None) -> None:
    """Build CSV, per-year category files, and plot for the given year_range into output_dir."""
    df_r = filter_year_range(df, year_range)
    if df_r.empty:
        return

    # Ensure output dir exists before writing CSV
    os.makedirs(output_dir, exist_ok=True)

    suffix = make_range_suffix(year_range)

    proportions = build_proportion_table(df_r)
    # Write category details for each year
    if year_range is None:
        # Write proportions (with deltas) to CSV for inspection
        proportions.to_csv(os.path.join(output_dir, f"EC_proportions{suffix}.csv"))
        write_category_details(df_r, output_dir, filename_suffix=suffix)
    else:
        # find the existing EC_proportions.csv and filter it to the year range
        existing_file = os.path.join(output_dir, f"EC_proportions.csv")
        if os.path.exists(existing_file):
            proportions = pd.read_csv(existing_file)
            proportions = proportions[proportions["year"].between(year_range[0], year_range[1])]
            proportions.to_csv(os.path.join(output_dir, f"EC_proportions{suffix}.csv"), index=False)

    # Plot
    plot_stacked_area(proportions, output_dir, filename_suffix=suffix)


def run(use_future: bool = USE_FUTURE, ranges=None) -> None:
    """Run outputs for provided ranges. ranges is a list like [None, (2000,2024)]."""
    df = load_data(use_future)
    # Choose output dir based on future/history to avoid mixing datasets
    output_dir = os.path.join(OUTPUT_DIR_BASE, 'future') if use_future else OUTPUT_DIR_BASE

    if ranges is None:
        ranges = [None, (2000, 2024)]

    for r in ranges:
        run_for_range(df, output_dir, year_range=r)


def main():
    # Default behavior replicates prior outputs and also 2000–2024
    #run(True)
    run(False, ranges=[None, (2000, 2024)])


if __name__ == "__main__":
    main()
