"""
PCA analysis utilities for presidential margins and Electoral College proportions.

Features
- Year range filter (e.g., 2000–2024)
- Datasets:
  * margins_all: relative margin for every state + national margin (52 columns)
  * margins_deltas: relative margin deltas for every state + national margin delta
  * ec: EC proportions (lockedR, safeR, …, lockedD)
  * ec_deltas: EC proportion deltas (…_delta columns)
- Dark-mode plots and human-oriented summaries
- Saves scores, loadings, and figures to an output folder

Usage examples
  # 2000–2024, relative margins for all states + national
  python pca_analysis.py --dataset margins_all --year-start 2000 --year-end 2024

  # EC proportion deltas for all available years
  python pca_analysis.py --dataset ec_deltas

  # Specify custom file paths
  python pca_analysis.py --dataset margins_deltas --margins-csv presidential_margins.csv \
                         --ec-csv EC_proportion/EC_proportions.csv
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import utils
import import_csv

# Use dark mode for all plots
plt.style.use("dark_background")
sns.set_context("talk")


@dataclass
class PCAData:
    X: np.ndarray
    years: List[int]
    feature_names: List[str]


def _find_default_paths(margins_csv: str | None, ec_csv: str | None) -> Tuple[str | None, str | None]:
    """Return existing paths or sensible defaults within this repo."""
    candidates_margins = [
        margins_csv,
        "presidential_margins.csv",
    ]
    candidates_ec = [
        ec_csv,
        os.path.join("EC_proportion", "EC_proportions.csv"),  # repo structure uses EC_proportion/
        os.path.join("EC_proportion", "future", "EC_proportions.csv"),
    ]
    margins_path = next((p for p in candidates_margins if p and os.path.exists(p)), None)
    ec_path = next((p for p in candidates_ec if p and os.path.exists(p)), None)
    return margins_path, ec_path


def load_margins(margins_csv: str, mode: str, year_start: int | None, year_end: int | None, include_nat: bool = True) -> PCAData:
    df = pd.read_csv(margins_csv)

    # Ensure correct types
    for col in ["year", "relative_margin", "relative_margin_delta", "national_margin", "national_margin_delta"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filter year range
    if year_start is not None:
        df = df[df["year"] >= year_start]
    if year_end is not None:
        df = df[df["year"] <= year_end]

    # Construct matrices
    if mode == "margins_all":
        # Pivot relative margins by state (columns are state abbrs). Add national margin as separate column.
        pivot = df.pivot_table(index="year", columns="abbr", values="relative_margin", aggfunc="first")
        # Add national margin column only if requested and present in source
        if include_nat and "national_margin" in df.columns:
            nat = df.drop_duplicates("year").set_index("year")["national_margin"]
            pivot["NAT"] = nat
        pivot = pivot.sort_index()
        # Drop any years with missing data (common for early years or partial datasets)
        pivot = pivot.dropna(how="any")
        years = pivot.index.astype(int).tolist()
        X = pivot.to_numpy(dtype=float)
        feature_names = pivot.columns.tolist()
        return PCAData(X=X, years=years, feature_names=feature_names)

    if mode == "margins_deltas":
        pivot = df.pivot_table(index="year", columns="abbr", values="relative_margin_delta", aggfunc="first")
        # Add national delta only if requested and present
        if include_nat and "national_margin_delta" in df.columns:
            nat = df.drop_duplicates("year").set_index("year")["national_margin_delta"]
            pivot["NAT"] = nat
        pivot = pivot.sort_index()
        # Drop first/any years lacking a previous-year delta
        pivot = pivot.dropna(how="any")
        years = pivot.index.astype(int).tolist()
        X = pivot.to_numpy(dtype=float)
        feature_names = pivot.columns.tolist()
        return PCAData(X=X, years=years, feature_names=feature_names)

    raise ValueError(f"Unknown margins mode: {mode}")


def load_ec(ec_csv: str, mode: str, year_start: int | None, year_end: int | None) -> PCAData:
    df = pd.read_csv(ec_csv)

    # Identify columns
    base_cols = [
        "lockedR", "safeR", "leanR", "tiltR", "swing", "tiltD", "leanD", "safeD", "lockedD"
    ]
    delta_cols = [f"{c}_delta" for c in base_cols]

    if mode == "ec":
        use_cols = ["year"] + base_cols
    elif mode == "ec_deltas":
        use_cols = ["year"] + delta_cols
    else:
        raise ValueError(f"Unknown ec mode: {mode}")

    missing = [c for c in use_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in {ec_csv}: {missing}")

    df = df[use_cols].copy()

    # Filter year range
    if year_start is not None:
        df = df[df["year"] >= year_start]
    if year_end is not None:
        df = df[df["year"] <= year_end]

    df = df.sort_values("year")
    years = df["year"].astype(int).tolist()
    X = df.drop(columns=["year"]).to_numpy(dtype=float)
    feature_names = df.columns.drop("year").tolist()
    return PCAData(X=X, years=years, feature_names=feature_names)


def run_pca(data: PCAData, n_components: int | None = None):
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data.X)

    # Fit PCA
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_scaled)
    loadings = pca.components_.T  # shape: features x components
    explained = pca.explained_variance_ratio_
    return scaler, pca, scores, loadings, explained


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _save_csvs(out_dir: str, years: List[int], feature_names: List[str], scores: np.ndarray, loadings: np.ndarray):
    # Scores (years x components)
    scores_df = pd.DataFrame(scores, index=years)
    scores_df.index.name = "year"
    scores_df.columns = [f"PC{i+1}" for i in range(scores.shape[1])]
    scores_df.to_csv(os.path.join(out_dir, "scores.csv"))

    # Loadings (features x components)
    loadings_df = pd.DataFrame(loadings, index=feature_names)
    loadings_df.index.name = "feature"
    loadings_df.columns = [f"PC{i+1}" for i in range(loadings.shape[1])]
    loadings_df.to_csv(os.path.join(out_dir, "loadings.csv"))


def _format_feature_labels(feature_names: List[str]) -> List[str]:
    mapping = {
        "lockedR": "Locked R", "safeR": "Safe R", "leanR": "Lean R", "tiltR": "Tilt R",
        "swing": "Swing", "tiltD": "Tilt D", "leanD": "Lean D", "safeD": "Safe D", "lockedD": "Locked D",
        "NAT": "National",
    }
    out = []
    for f in feature_names:
        label = mapping.get(f, f)
        # Prettify state/feature names like "CA" remain as-is
        out.append(label)
    return out


def plot_scree(explained: np.ndarray, out_dir: str):
    _ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#111")
    x = np.arange(1, len(explained) + 1)
    ax.bar(x, explained * 100, color="#56b4e9")
    ax.plot(x, np.cumsum(explained) * 100, color="#e69f00", marker="o", label="Cumulative")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title("PCA Scree Plot")
    # set y ticks to intervals of 10%
    ax.set_yticks(np.arange(0, 101, 10))
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "scree.png"), dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_scores_2d(scores: np.ndarray, years: List[int], out_dir: str):
    _ensure_dir(out_dir)
    if scores.shape[1] < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#111")
    # Color by year gradient
    years_arr = np.array(years)
    norm = (years_arr - years_arr.min()) / (years_arr.max() - years_arr.min() + 1e-9)
    colors = sns.color_palette("viridis", n_colors=len(years))

    for i, (x, y) in enumerate(scores[:, :2]):
        ax.scatter(x, y, color=colors[i], s=60)
        ax.text(x, y, str(years[i]), fontsize=9, color="#cccccc", ha="center", va="center")

    ax.axhline(0, color="#444", lw=1)
    ax.axvline(0, color="#444", lw=1)
    ax.set_xlabel("PC1 score")
    ax.set_ylabel("PC2 score")
    ax.set_title("Years in PC space (PC1 vs PC2)")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "scores_pc1_pc2.png"), dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_scores_timeseries(scores: np.ndarray, years: List[int], out_dir: str, k: int = 3):
    _ensure_dir(out_dir)
    k = min(k, scores.shape[1])
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#111")
    for i in range(k):
        ax.plot(years, scores[:, i], marker="o", label=f"PC{i+1}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Component score")
    ax.set_title("PC scores over time")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "scores_timeseries.png"), dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_loadings(loadings: np.ndarray, feature_names: List[str], out_dir: str, component: int = 1, top_n: int = 12):
    _ensure_dir(out_dir)
    comp_idx = component - 1
    if comp_idx < 0 or comp_idx >= loadings.shape[1]:
        return
    comp = loadings[:, comp_idx]
    labels = _format_feature_labels(feature_names)

    # Top positive/negative
    order = np.argsort(comp)
    neg_idx = order[:top_n]
    pos_idx = order[-top_n:][::-1]

    sel_idx = np.concatenate([neg_idx, pos_idx])
    sel_vals = comp[sel_idx]
    sel_labels = [labels[i] for i in sel_idx]

    colors = ["#d55e00" if v < 0 else "#009e73" for v in sel_vals]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#111")
    y = np.arange(len(sel_vals))
    ax.barh(y, sel_vals, color=colors)
    ax.set_yticks(y, sel_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Loading weight")
    ax.set_title(f"PC{component} top loadings (+/-)")
    ax.grid(True, axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"loadings_pc{component}.png"), dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_biplot(scores: np.ndarray, loadings: np.ndarray, feature_names: List[str], years: List[int], out_dir: str, component_pair: tuple[int, int] = (1, 2), top_n: int = 25):
    """PC biplot with year points and top feature vectors.
    component_pair is 1-based (e.g., (1,2)).
    """
    _ensure_dir(out_dir)
    c1, c2 = component_pair
    c1 -= 1
    c2 -= 1
    if scores.shape[1] <= max(c1, c2) or loadings.shape[1] <= max(c1, c2):
        return
    fig, ax = plt.subplots(figsize=(9, 7), facecolor="#111")

    # Scatter of scores
    colors = sns.color_palette("viridis", n_colors=len(years))
    for i, (x, y) in enumerate(scores[:, [c1, c2]]):
        ax.scatter(x, y, color=colors[i], s=60)
        ax.text(x, y, str(years[i]), fontsize=9, color="#cccccc", ha="center", va="center")

    # Feature vectors: choose top by magnitude across the two components
    vecs = loadings[:, [c1, c2]]
    mags = np.linalg.norm(vecs, axis=1)
    idx = np.argsort(mags)[-top_n:]
    labels = _format_feature_labels(feature_names)

    # Scale arrows to fit nicely
    # base scale based on scores spread
    sx = float(np.percentile(np.abs(scores[:, c1]), 95)) + 1e-9
    sy = float(np.percentile(np.abs(scores[:, c2]), 95)) + 1e-9
    scale = 0.8 * min(sx, sy)

    for i in idx:
        vx, vy = vecs[i]
        ax.arrow(0, 0, float(vx) * scale, float(vy) * scale, color="#56b4e9", head_width=0.03 * scale, length_includes_head=True, alpha=0.9)
        ax.text(float(vx) * scale * 1.05, float(vy) * scale * 1.05, labels[i], color="#aaaaaa", fontsize=8)

    ax.axhline(0, color="#444", lw=1)
    ax.axvline(0, color="#444", lw=1)
    ax.set_xlabel(f"PC{c1+1}")
    ax.set_ylabel(f"PC{c2+1}")
    ax.set_title(f"Biplot PC{c1+1} vs PC{c2+1}")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"biplot_pc{c1+1}_pc{c2+1}.png"), dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def save_interpretation_log(feature_names: List[str], loadings: np.ndarray, explained: np.ndarray, out_dir: str, top_n: int = 8, include_nat: bool = True, years: List[int] | None = None, scores: np.ndarray | None = None):
    labels = _format_feature_labels(feature_names)
    log_path = os.path.join(out_dir, "interpretation_log.txt")
    with open(log_path, "w") as f:
        f.write("PCA interpretation (top features per component):\n")
        f.write(f"National column (NAT) included: {'Yes' if include_nat else 'No'}\n")
        for i in range(min(5, loadings.shape[1])):
            comp = loadings[:, i]
            order = np.argsort(comp)
            neg = [(labels[idx], comp[idx]) for idx in order[:top_n]]
            pos = [(labels[idx], comp[idx]) for idx in order[-top_n:][::-1]]
            ev = explained[i] * 100
            f.write(f"\nPC{i+1}: {ev:.1f}% variance\n")
            f.write("  Strongly positive (move together):\n")
            for name, v in pos:
                f.write(f"    + {name:>15s}: {utils.lean_str(v)} ({v:+.3f})\n")
            f.write("  Strongly negative (opposite direction):\n")
            for name, v in neg:
                f.write(f"    - {name:>15s}: {utils.lean_str(v)} ({v:+.3f})\n")
        f.write("\nNotes:\n")
        f.write("- Features with large positive loadings rise/fall together along that PC; large negatives move inversely.\n")
        f.write("- For margins datasets: positive = more Democratic relative to nation; negative = more Republican relative to nation.\n")
        f.write("- For EC datasets: positive loadings increase the share in those buckets; negatives decrease it.\n")
        # Add scores by year if available
        if years is not None and scores is not None:
            f.write("\nPC scores by year:\n")
            # limit decimals to 3 per score
            for year, score_row in zip(years, scores):
                scores_str = ", ".join(f"PC{i+1}: {float(s):+.3f}" for i, s in enumerate(score_row))
                f.write(f"  {year}: {scores_str}\n")



def _total_variance(X_scaled: np.ndarray) -> float:
    # Use sample variance (ddof=1) to match sklearn PCA explained_variance
    return float(np.var(X_scaled, axis=0, ddof=1).sum())


def save_feature_rankings_csv(out_dir: str, feature_names: List[str], loadings: np.ndarray):
    """Create CSV where rows are ranks (1..n_features) and columns are grouped per PC:
    for each PC there are three columns: PCn_abbr, PCn_label, PCn_val.
    Rows are ordered from most negative (rank 1) to most positive (rank N).
    """
    n_features, n_comps = loadings.shape
    pretty_labels = _format_feature_labels(feature_names)

    data = {}
    for j in range(n_comps):
        comp = loadings[:, j]
        order = np.argsort(comp)  # most negative -> most positive
        abbrs = [feature_names[idx] for idx in order]
        labels = [pretty_labels[idx] for idx in order]
        vals = [float(comp[idx]) for idx in order]
        data[f"PC{j+1}_abbr"] = abbrs
        #data[f"PC{j+1}_label"] = labels
        data[f"PC{j+1}_val"] = [f"{v:+.3f}" for v in vals]

    df_rank = pd.DataFrame(data)
    df_rank.index = pd.RangeIndex(start=1, stop=n_features+1, name="rank")
    df_rank.to_csv(os.path.join(out_dir, "feature_rankings_per_pc.csv"), index=True)



def varimax(Phi: np.ndarray, gamma: float = 1.0, q: int = 20, tol: float = 1e-6) -> np.ndarray:
    """Orthogonal varimax rotation.
    Returns rotation matrix R such that loadings_rot = Phi @ R.
    """
    p, k = Phi.shape
    R = np.eye(k)
    d = 0.0
    for _ in range(q):
        d_old = d
        Lambda = Phi @ R
        u, s, vt = np.linalg.svd(
            Phi.T @ (Lambda**3 - (gamma / p) * Lambda @ np.diag(np.diag(Lambda.T @ Lambda)))
        )
        R = u @ vt
        d = float(s.sum())
        if d_old != 0 and d / d_old < 1 + tol:
            break
    return R


def apply_rotation(X_scaled: np.ndarray, loadings: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate loadings and derive rotated scores and explained variance ratio.
    Returns (scores_rot, loadings_rot, explained_ratio_rot).
    """
    method = method.lower()
    if method == "varimax":
        R = varimax(loadings)
        loadings_rot = loadings @ R
        scores_rot = X_scaled @ loadings_rot
        # explained variance ratio for rotated components
        comp_var = np.var(scores_rot, axis=0, ddof=1)
        total_var = _total_variance(X_scaled)
        explained_ratio_rot = comp_var / total_var
        return scores_rot, loadings_rot, explained_ratio_rot
    else:
        raise ValueError(f"Unknown rotation: {method}")


def main(year_start: int | None = None, year_end: int | None = None, 
         n_components: int | None = None,
         margins_csv: str | None = None, ec_csv: str | None = None, 
         out_dir: str | None = None,
         dataset: str = "margins_all", no_plot: bool = False,
         include_nat: bool = False,
         rotation: str = "none", biplot: bool = False, biplot_top: int = 25):
    parser = argparse.ArgumentParser(description="PCA on presidential margins and EC proportions (dark-mode plots)")
    parser.add_argument("--dataset", required=False, choices=[
        "margins_all", "margins_deltas", "ec", "ec_deltas"
    ], help="Which dataset to analyze")
    parser.add_argument("--year-start", type=int, default=None, help="Start year inclusive (e.g., 2000)")
    parser.add_argument("--year-end", type=int, default=None, help="End year inclusive (e.g., 2024)")
    parser.add_argument("--n-components", type=int, default=None, help="Number of PCA components (default: all)")
    parser.add_argument("--margins-csv", type=str, default=None, help="Path to presidential_margins.csv")
    parser.add_argument("--ec-csv", type=str, default=None, help="Path to EC_proportions.csv")
    parser.add_argument("--out", type=str, default=None, help="Output directory (default: pca_outputs/<dataset>_<range>)")
    parser.add_argument("--no-plot", action="store_true", help="Skip figure generation")
    parser.add_argument("--rotation", type=str, default="none", choices=["none", "varimax"], help="Optional rotation for interpretability")
    parser.add_argument("--biplot", action="store_true", help="Also generate biplot (PC1 vs PC2) with top feature vectors")
    parser.add_argument("--biplot-top", type=int, default=25, help="Number of top features to draw in biplot")
    parser.add_argument("--no-nat", action="store_true", help="Exclude national (NAT) column from margins datasets")
    #args = parser.parse_args()

    margins_path, ec_path = _find_default_paths(margins_csv, ec_csv)

    # Prepare data
    #include_nat = not getattr(parser.parse_args(), "no_nat", False)
    if dataset in ("margins_all", "margins_deltas"):
        if not margins_path:
            raise FileNotFoundError("presidential_margins.csv not found. Provide --margins-csv.")
        data = load_margins(margins_path, dataset, year_start, year_end, include_nat=include_nat)
    else:
        if not ec_path:
            raise FileNotFoundError("EC_proportions.csv not found under EC_proportion/. Provide --ec-csv.")
        data = load_ec(ec_path, dataset, year_start, year_end)

    if data.X.size == 0 or len(data.years) == 0:
        raise ValueError("No data after filtering; adjust year range or dataset choice.")

    # Run PCA
    scaler, pca, scores, loadings, explained = run_pca(data, n_components=n_components)

    # Optional rotation
    X_scaled = scaler.transform(data.X)
    scores_used = scores
    loadings_used = loadings
    explained_used = explained

    rot = (rotation or "none").lower()
    if rot != "none":
        scores_rot, loadings_rot, explained_rot = apply_rotation(X_scaled, loadings, rot)
        scores_used = scores_rot
        loadings_used = loadings_rot
        explained_used = explained_rot

    # Output directory (include rotation)
    yr_part = f"{data.years[0]}-{data.years[-1]}" if data.years else "all"
    label_rot = "" if rot == "none" else f"_{rot}"
    out_dir = out_dir or os.path.join("pca_outputs", f"{yr_part}/{label_rot}/{dataset}")
    _ensure_dir(out_dir)

    # Save CSVs
    _save_csvs(out_dir, data.years, data.feature_names, scores_used, loadings_used)
    # Save feature rankings per PC for quick lookup
    save_feature_rankings_csv(out_dir, data.feature_names, loadings_used)

    # Plots
    if not no_plot:
        plot_scree(explained_used, out_dir)
        plot_scores_2d(scores_used, data.years, out_dir)
        plot_scores_timeseries(scores_used, data.years, out_dir, k=min(3, scores_used.shape[1]))
        for c in range(1, min(3, scores_used.shape[1]) + 1):
            plot_loadings(loadings_used, data.feature_names, out_dir, component=c, top_n=12)
        if biplot and scores_used.shape[1] >= 2:
            plot_biplot(scores_used, loadings_used, data.feature_names, data.years, out_dir, component_pair=(1, 2), top_n=biplot_top)

    # Console summary
    total = explained_used.sum() * 100
    pc1 = f"{explained_used[0]*100:.1f}%" if len(explained_used) >= 1 else "n/a"
    pc2 = f"{explained_used[1]*100:.1f}%" if len(explained_used) >= 2 else "n/a"
    print(f"Explained variance by first {len(explained_used)} PCs: {total:.1f}% (PC1: {pc1}, PC2: {pc2})")
    save_interpretation_log(data.feature_names, loadings_used, explained_used, out_dir, top_n=8, include_nat=include_nat, years=data.years, scores=scores_used)
    print(f"\nSaved interpretation log to: {os.path.join(out_dir, 'interpretation_log.txt')}")
    print(f"\nSaved outputs to: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    datasets = [
        "margins_all", "margins_deltas", "ec", "ec_deltas"
    ]
    for dataset in datasets:
        print(f"\nRunning PCA for dataset: {dataset}")
        main(dataset=dataset, year_start=2000, year_end=2024, n_components=None, no_plot=False, rotation="none", biplot=True, biplot_top=25)
