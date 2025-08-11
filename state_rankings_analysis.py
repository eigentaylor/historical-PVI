import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Keep only the columns we need
    required_cols = {"year", "abbr", "relative_margin"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path.name}: {sorted(missing)}")

    # Drop rows where key fields are missing
    df = df.dropna(subset=["year", "abbr", "relative_margin"]).copy()
    # Ensure proper dtypes
    df["year"] = df["year"].astype(int)
    df["abbr"] = df["abbr"].astype(str)
    df["relative_margin"] = pd.to_numeric(df["relative_margin"], errors="coerce")
    df = df.dropna(subset=["relative_margin"])  # if any were coerced to NaN
    return df


def compute_yearly_ranks(df: pd.DataFrame) -> pd.DataFrame:
    # For each year, rank states by relative_margin (descending: most D-leaning rank 1)
    df = df.sort_values(["year", "relative_margin"], ascending=[True, False]).copy()
    df["rank"] = df.groupby("year")["relative_margin"].rank(method="dense", ascending=False).astype(int)
    return df


def plot_state_rank_over_time(ranks_df: pd.DataFrame, out_dir: Path) -> None:
    plt.style.use("dark_background")

    years_per_state = ranks_df.groupby("abbr")["year"].nunique()
    all_years = sorted(ranks_df["year"].unique())

    # Determine max number of ranked units in any year for consistent y-limits across plots
    states_per_year = ranks_df.groupby("year")["abbr"].nunique()
    max_rank = int(states_per_year.max())

    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare pivot for quicker plotting
    pivot = ranks_df.pivot_table(index="year", columns="abbr", values="rank", aggfunc="first").sort_index()

    for abbr in sorted(pivot.columns):
        y = pivot[abbr].dropna()
        if y.empty:
            continue
        x = y.index

        fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
        ax.plot(x, y, color="#33C3F0", marker="o", linewidth=2, markersize=4)

        ax.set_title(f"{abbr} state rank by relative margin", fontsize=14, pad=10)
        ax.set_xlabel("Year")
        ax.set_ylabel("Rank (1 = most D-leaning relative to nation)")

        # Place rank 1 at the top
        ax.set_ylim(max_rank + 1, 0)

        # Grid and bounds
        ax.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.set_xlim(min(all_years) - 1, max(all_years) + 1)

        # Ticks every 4 years if plausible
        if len(all_years) > 1:
            step = 4 if (max(all_years) - min(all_years)) >= 16 else max(1, (max(all_years) - min(all_years)) // 6)
            xticks = list(range(min(all_years), max(all_years) + 1, step))
            ax.set_xticks(xticks)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fname = out_dir / f"{abbr}_rank_over_time.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)


def main():
    repo_root = Path(__file__).resolve().parent
    csv_path = repo_root / "presidential_margins.csv"
    out_dir = repo_root / "state_rankings"

    df = load_data(csv_path)
    ranks_df = compute_yearly_ranks(df)
    plot_state_rank_over_time(ranks_df, out_dir)
    print(f"Saved per-state rank plots to: {out_dir}")


if __name__ == "__main__":
    main()
