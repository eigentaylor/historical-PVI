"""
Visualize energy components for historical deltas and sampled proposals.

Outputs go to energy_predict/visuals/ as dark-mode PNGs and CSVs.
"""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List, Tuple

from pathlib import Path
import sys

# ensure project root (one level up) is on sys.path so `import energy_predict` works
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import energy_predict as ep

VIS_DIR = os.path.join("energy_predict", "visuals")
CSV_PATH = ep.CSV_PATH
TAU = ep.TAU


def _ensure_dirs():
    os.makedirs(VIS_DIR, exist_ok=True)


def _dark_mode():
    mpl.rcParams.update({
        "figure.facecolor": "#0f1115",
        "axes.facecolor": "#0f1115",
        "savefig.facecolor": "#0f1115",
        "axes.edgecolor": "#e0e6f1",
        "axes.labelcolor": "#e0e6f1",
        "xtick.color": "#cbd5e1",
        "ytick.color": "#cbd5e1",
        "text.color": "#e0e6f1",
        "grid.color": "#334155",
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.5,
        "axes.titleweight": "semibold",
        "font.size": 11,
    })
    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except Exception:
        pass


def build_targets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[int], ep.Targets]:
    pivot_m, pivot_ev, ev_w, states, years = ep.load_and_align(CSV_PATH)
    targets = ep.build_delta_targets(pivot_m, ev_w, years, states, tau=ep.TAU, var_frac=ep.VAR_FRAC, K_min=ep.K_MIN)
    return pivot_m, pivot_ev, ev_w, states, years, targets


def compute_historical_components(pivot_m, ev_w, years, states, targets) -> pd.DataFrame:
    # 4-year delta pairs present in data
    y_pairs = [(y, y - 4) for y in years if (y - 4) in years]
    rows = []
    for y, y0 in y_pairs:
        d = (pivot_m.loc[y].reindex(states).to_numpy() -
             pivot_m.loc[y0].reindex(states).to_numpy())
        w = ev_w.loc[y].reindex(states).to_numpy()
        comps = ep.energy_components(d, targets, w)
        rows.append({"year": y, "prev": y0, **comps})
    return pd.DataFrame(rows)


def plot_hist_components(df: pd.DataFrame, title: str, fname: str) -> None:
    _dark_mode()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    parts = ["E_EV", "E_VAR", "E_SHOCK", "E_RESID"]
    axes = axes.ravel()
    colors = ["#60a5fa", "#f97316", "#22d3ee", "#a78bfa"]
    for ax, p, c in zip(axes, parts, colors):
        sns.histplot(data=df, x=p, bins=24, kde=True, ax=ax, color=c, alpha=0.8)
        ax.set_title(p)
        ax.set_xlabel("energy")
    fig.suptitle(title)
    out = os.path.join(VIS_DIR, fname)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_series(df: pd.DataFrame, fname: str) -> None:
    _dark_mode()
    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    for col, c in zip(["E_EV", "E_VAR", "E_SHOCK", "E_RESID", "E_TOTAL"],
                      ["#60a5fa", "#f97316", "#22d3ee", "#a78bfa", "#86efac"]):
        ax.plot(df["year"], df[col], marker="o", lw=1.6, label=col, color=c)
    ax.set_title("Energy components over time (4-year deltas)")
    ax.set_xlabel("year vs prior")
    ax.legend(ncol=3, frameon=False)
    out = os.path.join(VIS_DIR, fname)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_evd_shock(df: pd.DataFrame, fname: str) -> None:
    _dark_mode()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    sns.scatterplot(data=df, x="evd", y="E_EV", ax=ax[0], color="#60a5fa")
    ax[0].axvline(df["evd"].mean(), color="#475569", ls="--", lw=1)
    ax[0].set_title("EV-weighted delta vs EV energy")
    ax[0].set_xlabel("EV-weighted delta")

    sns.scatterplot(data=df, x="shock_frac", y="E_SHOCK", ax=ax[1], color="#22d3ee")
    ax[1].axvline(df["shock_frac"].mean(), color="#475569", ls="--", lw=1)
    ax[1].set_title(f"Shock fraction vs SHOCK energy (tau={TAU:.2f})")
    ax[1].set_xlabel("shock fraction (|d| >= tau)")

    out = os.path.join(VIS_DIR, fname)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def sample_and_plot_proposals(pivot_m, pivot_ev, ev_w, states, targets, year=2024) -> None:
    # Baseline and weights for a specific year (2024 default)
    m = pivot_m.loc[year].reindex(states).to_numpy()
    w = ev_w.loc[year].reindex(states).to_numpy()

    keep_d, keep_E = ep.sample_deltas(targets, w, N_draw=2000, N_keep=400, seed=ep.SEED)
    # components for kept proposals
    rows = []
    for i in range(keep_d.shape[0]):
        comps = ep.energy_components(keep_d[i], targets, w)
        rows.append({"rank": i, **comps})
    df = pd.DataFrame(rows).sort_values("E_TOTAL").reset_index(drop=True)
    df.to_csv(os.path.join(VIS_DIR, f"proposals_{year}_components.csv"), index=False)

    _dark_mode()
    fig, ax = plt.subplots(figsize=(11, 4.2), constrained_layout=True)
    ax.plot(df.index, df["E_TOTAL"], lw=1.4, color="#86efac")
    ax.set_title(f"Kept proposal energies (N={len(df)}) for baseline {year}")
    ax.set_xlabel("proposal rank (by energy)")
    ax.set_ylabel("E_TOTAL")
    fig.savefig(os.path.join(VIS_DIR, f"proposals_{year}_energies.png"), dpi=160)
    plt.close(fig)

    # heatmap of a random subset of proposal deltas
    sel = min(60, len(df))
    idx = np.linspace(0, len(df) - 1, sel, dtype=int)
    mat = keep_d[idx]
    _dark_mode()
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    sns.heatmap(mat, cmap="vlag", center=0, cbar_kws={"label": "delta"}, ax=ax)
    ax.set_title(f"Sample proposal deltas (subset) for {year}")
    ax.set_xlabel("state index")
    ax.set_ylabel("sample index")
    fig.savefig(os.path.join(VIS_DIR, f"proposals_{year}_deltas_heatmap.png"), dpi=160)
    plt.close(fig)


def main():
    _ensure_dirs()
    pivot_m, pivot_ev, ev_w, states, years, targets = build_targets()

    # Historical component series
    hist_df = compute_historical_components(pivot_m, ev_w, years, states, targets)
    hist_df.to_csv(os.path.join(VIS_DIR, "historical_energy_components.csv"), index=False)

    plot_hist_components(hist_df, "Historical energy components (4-year deltas)", "hist_energy_components.png")
    plot_series(hist_df, "hist_energy_series.png")
    plot_evd_shock(hist_df, "hist_scatter_evd_shock.png")

    # Proposals relative to 2024 baseline
    sample_and_plot_proposals(pivot_m, pivot_ev, ev_w, states, targets, year=2024)

    print(f"Saved visuals to: {VIS_DIR}")


if __name__ == "__main__":
    main()
