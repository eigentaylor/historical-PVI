import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from statsmodels.tsa.api import VAR
from utils import lean_str

# -----------------------------
# Mode selector: choose one of ["random_walk", "matrix", "raw_votes"]
# -----------------------------
mode = "matrix"  # options: "random_walk", "matrix", "raw_votes"

# -----------------------------
# Option to include historical data in the future CSV
# -----------------------------
INCLUDE_HISTORICAL = True  # Set to True to include years <= 2024 in future CSV

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("presidential_margins.csv")
# check if DC is present
if 'DC' not in df['abbr'].unique():
    print("WARNING: DC missing from data!")
# Pivot into matrix: rows = years, columns = states, values = relative_margin

if mode == "raw_votes":
    pivot_D = df.pivot(index="year", columns="abbr", values="D_votes").dropna(axis=1)
    pivot_R = df.pivot(index="year", columns="abbr", values="R_votes").dropna(axis=1)
    Y_D = pivot_D.values  # T x S
    Y_R = pivot_R.values  # T x S
    years = pivot_D.index.tolist()
    states = pivot_D.columns.tolist()
    print(f"States after D_votes pivot: {states}")
    if 'DC' not in states:
        print("WARNING: DC missing after D_votes pivot!")
else:
    pivot = df.pivot(index="year", columns="abbr", values="relative_margin").dropna(axis=1)
    Y = pivot.values  # T x S
    years = pivot.index.tolist()
    states = pivot.columns.tolist()
    print(f"States after relative_margin pivot: {states}")
    if 'DC' not in states:
        print("WARNING: DC missing after relative_margin pivot!")

# -----------------------------

# PCA decomposition
# -----------------------------
K = 13  # Number of components

if mode == "matrix":
    # Add national margin as a column to the matrix
    nat_margin_hist = (
        df.drop_duplicates("year")
          .sort_values("year")["national_margin"]
          .values
    )
    print(f"States before matrix PCA: {states}")
    if 'DC' not in states:
        print("WARNING: DC missing before matrix PCA!")
    Y_ext = np.hstack([Y, nat_margin_hist.reshape(-1, 1)])
    pca = PCA(n_components=K)
    Z = pca.fit_transform(Y_ext)       # T x K
    W = pca.components_.T          # (S+1) x K
else:
    if mode == "raw_votes":
        # Use D_votes and R_votes as the matrix
        Y_votes = np.dstack([Y_D, Y_R])  # T x S x 2
        Y_flat = Y_votes.reshape(Y_D.shape[0], -1)  # T x (2S)
        if np.isnan(Y_flat).any() or np.isinf(Y_flat).any():
            print("NaNs or Infs detected in vote matrix!")
            exit()
        pca = PCA(n_components=K)
        Z = pca.fit_transform(Y_flat)
        W = pca.components_.T
    else:
        pca = PCA(n_components=K)
        Z = pca.fit_transform(Y)       # T x K
        W = pca.components_.T          # S x K

# -----------------------------

# Fit VAR(1) on factor scores
# -----------------------------
model = VAR(Z)
results = model.fit(maxlags=1)

# -----------------------------

# Simulate future latent states
# -----------------------------
future_steps = 6  # 6 elections (2028–2048)
Z_future = results.simulate_var(steps=future_steps)

if mode == "matrix":
    Y_future_ext = Z_future @ W.T  # shape: (future_steps x (S+1))
    Y_future_rel = Y_future_ext[:, :-1]
    nat_future = Y_future_ext[:, -1]
elif mode == "raw_votes":
    Y_future_flat = Z_future @ W.T  # shape: (future_steps x (2S))
    # Split back into D_votes and R_votes
    S = len(states)
    D_votes_future = Y_future_flat[:, :S]
    R_votes_future = Y_future_flat[:, S:]
else:
    Y_future_rel = Z_future @ W.T  # shape: (future_steps x S)

# -----------------------------

if mode == "random_walk":
    # Simulate national margins as random walk
    nat_margin_hist = (
        df.drop_duplicates("year")
          .sort_values("year")["national_margin"]
          .values
    )
    delta_nat = np.diff(nat_margin_hist)
    nat_sigma = delta_nat.std(ddof=1)

    np.random.seed(42)
    nat_future = [nat_margin_hist[-1]]
    for _ in range(future_steps - 1):
        nat_future.append(nat_future[-1] + np.random.normal(0, nat_sigma))
    nat_future = np.array(nat_future)

# -----------------------------

# Add national margin back to get full margin
# -----------------------------
if mode in ["random_walk", "matrix"]:
    Y_future_margin = Y_future_rel + nat_future[:, None]

# -----------------------------

# Format as DataFrame
# -----------------------------
future_years = [years[-1] + 4 * (i + 1) for i in range(future_steps)]

if mode in ["random_walk", "matrix"]:
    print(f"States used for future output: {states}")
    if 'DC' not in states:
        print("WARNING: DC missing in future output!")
    df_out = pd.DataFrame(Y_future_margin, columns=states)
    df_out["year"] = future_years
    df_final = df_out.melt(id_vars="year", var_name="abbr", value_name="pres_margin")
    # Assign correct national_margin for each year
    year_to_nat_margin = dict(zip(future_years, nat_future))
    df_final["national_margin"] = df_final["year"].map(year_to_nat_margin)
    # Add relative_margin (pres_margin - national_margin)
    df_final["relative_margin"] = df_final["pres_margin"] - df_final["national_margin"]
    df_final["pres_margin_str"] = df_final["pres_margin"].apply(lean_str)
    df_final["relative_margin_str"] = df_final["relative_margin"].apply(lean_str)
    df_final["national_margin_str"] = df_final["national_margin"].apply(lean_str)
elif mode == "raw_votes":
    # Build DataFrame from D_votes and R_votes
    df_list = []
    for i, year in enumerate(future_years):
        for j, state in enumerate(states):
            D = int(D_votes_future[i, j])
            R = int(R_votes_future[i, j])
            total = D + R
            pres_margin = (D - R) / total if total != 0 else np.nan
            national_margin = (D_votes_future[i].sum() - R_votes_future[i].sum()) / (D_votes_future[i].sum() + R_votes_future[i].sum()) if (D_votes_future[i].sum() + R_votes_future[i].sum()) != 0 else np.nan
            relative_margin = pres_margin - national_margin if not np.isnan(pres_margin) and not np.isnan(national_margin) else np.nan
            df_list.append({
                "year": year,
                "abbr": state,
                "D_votes": D,
                "R_votes": R,
                "pres_margin": pres_margin,
                "pres_margin_str": lean_str(pres_margin),
                "relative_margin": relative_margin,
                "relative_margin_str": lean_str(relative_margin),
                "national_margin": national_margin,
                "national_margin_str": lean_str(national_margin)
            })
    df_final = pd.DataFrame(df_list)

# -----------------------------

# Save to CSV
# -----------------------------
if INCLUDE_HISTORICAL:
    # Get historical data with relevant columns
    hist_cols = ["year", "abbr", "pres_margin", "pres_margin_str", "relative_margin", "relative_margin_str", "national_margin", "national_margin_str", "electoral_votes"]
    if mode == "raw_votes":
        hist_cols.extend(["D_votes", "R_votes"])
    df_hist = df[df["year"] <= 2024][hist_cols]
    # For future years, assign electoral_votes from 2024
    ev_2024 = df_hist[df_hist["year"] == 2024].set_index("abbr")["electoral_votes"].to_dict()
    df_final["electoral_votes"] = df_final["abbr"].map(ev_2024)
    df_combined = pd.concat([df_hist, df_final], ignore_index=True)
    df_combined = df_combined.sort_values(["year", "abbr"])
    df_combined.to_csv("presidential_margins_future.csv", index=False)
    print(f"Saved to presidential_margins_future.csv ✅ (mode: {mode}, with historical data)")
else:
    # For future years, assign electoral_votes from 2024
    if 'electoral_votes' not in df_final.columns:
        ev_2024 = df[df["year"] == 2024].set_index("abbr")["electoral_votes"].to_dict()
        df_final["electoral_votes"] = df_final["abbr"].map(ev_2024)
    df_final = df_final.sort_values(["year", "abbr"])
    df_final.to_csv("presidential_margins_future.csv", index=False)
    print(f"Saved to presidential_margins_future.csv ✅ (mode: {mode})")
