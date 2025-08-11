import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import utils

# Settings
plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor': '#0e1117',
    'axes.facecolor': '#0e1117',
    'savefig.facecolor': '#0e1117',
    'axes.edgecolor': '#c9d1d9',
    'text.color': '#c9d1d9',
    'axes.labelcolor': '#c9d1d9',
    'xtick.color': '#c9d1d9',
    'ytick.color': '#c9d1d9',
    'grid.color': '#30363d',
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.3,
    'font.size': 11,
})

ROOT = Path(__file__).parent
OUT_DIR = ROOT / 'margin_changes'
OUT_DIR.mkdir(exist_ok=True)

CSV_PATH = ROOT / 'presidential_margins.csv'
FUTURE_CSV_PATH = ROOT / 'presidential_margins_future.csv'

# Toggle to include future projections
USE_FUTURE = True
FUTURE_YEAR_MARK = 2026
if USE_FUTURE:
    OUT_DIR = OUT_DIR / 'future'
    os.makedirs(OUT_DIR, exist_ok=True)


def load_data(csv_path: Path, use_future: bool = False) -> pd.DataFrame:
    if use_future:
        df_hist = pd.read_csv(csv_path)
        # Filter hist to <= 2024
        df_hist = df_hist[df_hist['year'].astype(int) <= 2024]
        # Append future if available
        if FUTURE_CSV_PATH.exists():
            df_future = pd.read_csv(FUTURE_CSV_PATH)
            df = pd.concat([df_hist, df_future], ignore_index=True)
        else:
            df = df_hist
    else:
        df = pd.read_csv(csv_path)
    # Keep only required columns and ensure correct types
    df = df[['year', 'abbr', 'relative_margin']].copy()
    df['year'] = df['year'].astype(int)
    df.sort_values(['abbr', 'year'], inplace=True)
    return df


def compute_margin_changes(df: pd.DataFrame) -> pd.DataFrame:
    # Compute within-state 4-year change in relative_margin
    df['rel_margin_change'] = (
        df.groupby('abbr')['relative_margin'].diff()
    )
    # Ensure only 4-year gaps are used (guard against any data glitches)
    year_diff = df.groupby('abbr')['year'].diff()
    df.loc[year_diff != 4, 'rel_margin_change'] = np.nan
    return df.dropna(subset=['rel_margin_change']).copy()


def save_all_states_histogram(changes: pd.Series, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    # Auto bins around zero, but enforce symmetric range for nicer view
    min_c, max_c = float(changes.min()), float(changes.max())
    lim = max(abs(min_c), abs(max_c))
    lim = max(lim, 0.05)  # ensure some width
    bins = np.linspace(-lim, lim, 50).tolist()

    ax.hist(changes, bins=bins, color='#58a6ff', edgecolor='#161b22', alpha=0.9)
    ax.axvline(0, color='#f78166', linestyle='-', linewidth=1.5, alpha=0.9)

    ax.set_title('Distribution of 4-year Changes in Relative Margin (All States)')
    ax.set_xlabel('Change in relative_margin (this year - 4 years prior)')
    ax.set_ylabel('Count')
    
    # set x ticks to be intervals of 0.05 and to util.lean_str
    x_ticks = np.arange(-0.3, 0.35, 0.05)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([utils.lean_str(float(x)) for x in x_ticks])
    # display x ticks slanted
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Summary stats annotation
    s = changes.astype(float)
    # Use numpy for stats to avoid typing issues
    arr = s.to_numpy(dtype=float)
    mean = float(np.mean(arr))
    median = float(np.median(arr))
    var = float(np.var(arr, ddof=1)) if arr.size > 1 else float('nan')
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else float('nan')
    text = (
        f"n = {arr.size}\n"
        f"mean = {mean:.4f}\n"
        f"median = {median:.4f}\n"
        f"variance = {var:.6f}\n"
        f"std dev = {std:.4f}"
    )
    ax.text(0.99, 0.98, text, transform=ax.transAxes,
            ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.4', fc='#161b22', ec='#30363d', alpha=0.9))

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_state_barplot(state_df: pd.DataFrame, out_path: Path) -> None:
    # state_df has columns: year, abbr, relative_margin, rel_margin_change
    years = state_df['year'].to_numpy()
    changes = state_df['rel_margin_change'].to_numpy()

    colors = np.where(changes >= 0, '#58a6ff', '#f78166')  # blue for +, orange for -

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(years, changes, color=colors, edgecolor='#161b22')
    ax.axhline(0, color='#c9d1d9', linewidth=1)

    abbr = state_df['abbr'].iloc[0]
    ax.set_title(f'{abbr}: 4-year Changes in Relative Margin')
    ax.set_xlabel('Election year')
    ax.set_ylabel('Change in relative_margin')

    # Ensure all years are shown and pad x-limits for clarity
    ax.set_xticks(years)
    ax.set_xlim(int(years.min()) - 1, int(years.max()) + 1)

    # Show yellow dashed line at 2026 if future data is present
    if years.max() > 2024:
        ax.axvline(FUTURE_YEAR_MARK, color='#FFD700', linestyle='--', linewidth=2.5, alpha=0.95)
        ymax = ax.get_ylim()[1]
        ax.text(FUTURE_YEAR_MARK + 0.2, ymax * 0.95, str(FUTURE_YEAR_MARK), color='#FFD700', rotation=90,
                va='top', ha='left', fontsize=9)

    y_vals = ax.get_yticks()
    ax.set_yticklabels([utils.lean_str(float(y)) for y in y_vals])

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def compute_stats(changes: pd.Series) -> dict:
    s = changes.astype(float)
    arr = s.to_numpy(dtype=float)
    return {
        'count': int(arr.size),
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'variance': float(np.var(arr, ddof=1)) if arr.size > 1 else float('nan'),
        'std_dev': float(np.std(arr, ddof=1)) if arr.size > 1 else float('nan'),
    }


def main():
    df = load_data(CSV_PATH, USE_FUTURE)
    changes_df = compute_margin_changes(df)

    # All-states histogram
    all_changes = changes_df['rel_margin_change']
    save_all_states_histogram(all_changes, OUT_DIR / 'all_states_margin_change_hist.png')

    # Per-state bar plots and stats
    records = []
    for abbr, g in changes_df.groupby('abbr'):
        # Save bar plot
        out_plot = OUT_DIR / f'{abbr}_margin_changes.png'
        save_state_barplot(g, out_plot)

        # Stats for this state
        st = compute_stats(g['rel_margin_change'])
        st['abbr'] = abbr
        records.append(st)

    # All-states stats
    overall = compute_stats(all_changes)
    overall['abbr'] = 'ALL'
    records.append(overall)

    stats_df = pd.DataFrame.from_records(records)[['abbr', 'count', 'mean', 'median', 'variance', 'std_dev']]
    stats_df.sort_values(by=['abbr'], inplace=True)
    stats_df.to_csv(OUT_DIR / 'margin_change_stats.csv', index=False)

    # Also store raw changes for reference
    changes_df[['abbr', 'year', 'rel_margin_change']].to_csv(OUT_DIR / 'margin_changes_raw.csv', index=False)

    print('Saved:')
    print(' -', OUT_DIR / 'all_states_margin_change_hist.png')
    print(' -', OUT_DIR / 'margin_change_stats.csv')
    print(' -', OUT_DIR / 'margin_changes_raw.csv')
    print(' - Per-state plots in', OUT_DIR)


if __name__ == '__main__':
    main()
