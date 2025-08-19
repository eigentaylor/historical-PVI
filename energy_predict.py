import os
import math
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from itertools import permutations
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import utils
import subprocess
import sys
import shutil

# Optional: import tipping point helpers if available
try:
    import historical_tipping_points as htp
    print("Using historical_tipping_points for detailed tipping analysis.")
except Exception:  # pragma: no cover
    htp = None
    print("Using default tipping point analysis (MODULE NOT FOUND)")


# ------------------------------------------------------------
# Tuning knobs (defaults) – can be overridden via main() args
# ------------------------------------------------------------
TAU = 0.08  # shock threshold
VAR_FRAC = 0.90  # fraction of variance to keep in factors
K_MIN = 3  # minimum number of factors
N_DRAW = 6000
N_KEEP = 1000
CLIP_D = 0.35  # soft bounds for delta proposals
SEED = 17
KMEANS_K = 5
CSV_PATH = "presidential_margins.csv"
OUT_DIR = "energy_predict"
NAT_CLIP = 0.25  # soft bounds for national margin delta proposals
HIST_BINS = np.linspace(-0.5, 0.5, 21)  # edges for margin histogram (20 bins)

# Default alpha knobs for energy components (tunable)
ALPHA_EV = 0.0
ALPHA_VAR = 0.0
ALPHA_SHOCK = 0.0
ALPHA_RESID = 0.0
ALPHA_EV_MARGIN = 1.0
ALPHA_EV_QUAD = 0.2
ALPHA_HIST_STATE = 0.2
ALPHA_HIST_EV = 0.0

# At-large groups: these at-large labels are derived from their districts and should
# be excluded from sampling; final predicted values will be set to the average of
# their districts. Adjust lists if your CSV uses different district names.
AT_LARGE_GROUPS = {
    "NE-AL": ["NE-01", "NE-02", "NE-03"],
    "ME-AL": ["ME-01", "ME-02"],
}

# Slightly downweight deltas for congressional districts (ME/NE) due to redistricting volatility
DISTRICT_WEIGHT = 0.85  # in (0,1]; 1.0 means no downweight, smaller reduces impact
DISTRICT_LABELS = {"ME-01", "ME-02", "NE-01", "NE-02", "NE-03"}


@dataclass
class Targets:
    states: List[str]
    years: List[int]
    state_sigma: np.ndarray  # (S,)
    state_weights: np.ndarray  # (S,) per-unit weights for energy terms
    mu_evd: float
    sd_evd: float
    mu_shock: float
    sd_shock: float
    F: np.ndarray  # (K, S) orthonormal rows
    score_sd: np.ndarray  # (K,)
    mean_Xc: np.ndarray  # (S,)
    resid_scale: float
    idio_scale: float
    # Historical margin histogram (bins, state-count mean+sd, ev-weight mean+sd)
    hist_bins: np.ndarray
    hist_state_mean: np.ndarray
    hist_state_sd: np.ndarray
    hist_ev_mean: np.ndarray
    hist_ev_sd: np.ndarray
    # EV-weighted margin statistics (across historical years)
    mu_ev_margin: float
    sd_ev_margin: float
    mu_ev_abs: float
    sd_ev_abs: float
    # National margin delta stats
    nat_mu: float  # mean 4-year national delta
    nat_sd: float  # std of 4-year national delta
    nat_idio_scale: float  # Laplace scale for national delta proposals
    nat_by_year: Dict[int, float]  # national margin per year


# ----------------------
# Data loading and prep
# ----------------------
def load_and_align(csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[int], pd.Series]:
    """Load CSV; return (pivot_m, pivot_ev, ev_w, states, years, nat_series).

    Requires columns: year, abbr, relative_margin, electoral_votes.
    If national_margin exists, it will be aggregated per year and returned as a Series.
    """
    df = pd.read_csv(csv_path)
    required = {"year", "abbr", "relative_margin", "electoral_votes"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    df = df.copy()
    df["year"] = df["year"].astype(int)
    df["abbr"] = df["abbr"].astype(str)
    df["relative_margin"] = pd.to_numeric(df["relative_margin"], errors="coerce")
    df["electoral_votes"] = pd.to_numeric(df["electoral_votes"], errors="coerce")

    # Wide pivots
    pivot_m = df.pivot_table(index="year", columns="abbr", values="relative_margin")
    pivot_ev = df.pivot_table(index="year", columns="abbr", values="electoral_votes", aggfunc="first")

    # EVs in the input may be missing for some recent cycles (e.g., 2024).
    # Forward/back-fill per column so every year has a value, then cast to float.
    pivot_ev = pivot_ev.ffill().bfill()

    years = sorted(int(y) for y in pivot_m.index if not pd.isna(y))
    # Keep any state that has at least one non-null margin and at least one electoral_votes entry.
    # This allows including ME/NE districts that start later in the record (e.g., 1992).
    valid_states = [s for s in pivot_m.columns if pivot_m[s].notna().any() and pivot_ev[s].notna().any()]
    pivot_m = pivot_m[valid_states].sort_index()
    pivot_ev = pivot_ev[valid_states].sort_index()

    # EV weights per year
    ev_tot = pivot_ev.sum(axis=1)
    # Normalized EV weights per year (no NaNs thanks to ffill/bfill)
    ev_w = pivot_ev.div(ev_tot, axis=0)

    # National margin per year (if present). If missing, fill with 0.0 to avoid crashes.
    if "national_margin" in df.columns:
        nat_series = (
            df.dropna(subset=["year"]).groupby("year")["national_margin"].first().astype(float)
        )
    else:
        nat_series = pd.Series({y: 0.0 for y in years}, name="national_margin")

    return pivot_m, pivot_ev, ev_w, valid_states, years, nat_series


# ------------------------------------
# Build historical delta-based targets
# ------------------------------------
def build_delta_targets(
    pivot_m: pd.DataFrame,
    ev_w: pd.DataFrame,
    years: List[int],
    states: List[str],
    nat_series: pd.Series,
    tau: float = TAU,
    var_frac: float = VAR_FRAC,
    K_min: int = K_MIN,
) -> Targets:
    """Compute deltas and all stats needed for energy scoring.

        Returns Targets with:
      - mu_evd, sd_evd: EV-weighted delta stats across cycles
      - state_sigma: per-state std of deltas
            - state_weights: per-state energy weights (districts downweighted)
      - mu_shock, sd_shock: avg fraction of |Δ| >= tau per cycle
      - F (K x S): factor directions (orthonormal rows), score_sd (K,),
      - mean_Xc (S,), resid_scale (float), idio_scale (float)
    """
    # Build 4-year deltas for all feasible cycles y where y-4 exists.
    # Now support partial-state histories: if a particular state is missing for a pair,
    # the delta for that state will be NaN for that row. Rows (pairs) with no usable states
    # are skipped. This allows including ME/NE districts that only appear from 1992 onward.
    y_pairs = [(y, y - 4) for y in years if (y - 4) in years]
    if not y_pairs:
        raise ValueError("Not enough years to compute deltas (need 4-year pairs).")

    deltas_list: List[np.ndarray] = []
    w_list: List[np.ndarray] = []
    skipped_pairs: List[Tuple[int, int]] = []
    for y, y0 in y_pairs:
        row_y = pivot_m.loc[y].reindex(states).to_numpy()
        row_y0 = pivot_m.loc[y0].reindex(states).to_numpy()
        d = row_y - row_y0  # contains NaN where either year/state is missing
        w = ev_w.loc[y].reindex(states).to_numpy()

        # mask of available (non-NaN) deltas for this pair
        mask = ~np.isnan(d)
        if not np.any(mask):
            skipped_pairs.append((y, y0))
            continue
        # If all available deltas are numerically zero, treat as missing and skip
        if np.allclose(d[mask], 0.0, atol=1e-12):
            skipped_pairs.append((y, y0))
            continue

        deltas_list.append(d)
        w_list.append(w)

    if not deltas_list:
        raise ValueError("No usable 4-year delta pairs after skipping empty/missing rows.")

    X = np.vstack(deltas_list)  # (T_eff, S) may contain NaNs
    W = np.vstack(w_list)       # (T_eff, S)
    T, S = X.shape

    if skipped_pairs:
        print(f"build_delta_targets: skipped {len(skipped_pairs)} 4-year pair(s) because they had no usable data: {skipped_pairs}")

    # EV-weighted delta distribution computed only over available states for each row.
    evd_list = []
    for i in range(T):
        row = X[i]
        w_row = W[i]
        mask = ~np.isnan(row)
        if not np.any(mask):
            continue
        ws = w_row[mask]
        xs = row[mask]
        denom = float(np.sum(ws)) if np.sum(ws) > 0 else float(np.sum(~np.isnan(xs)))
        if denom <= 0:
            evd_val = float(np.mean(xs))
        else:
            # normalize by available weights so the EV-weighted mean remains meaningful
            evd_val = float(np.sum(ws * xs) / denom)
        evd_list.append(evd_val)
    evd = np.array(evd_list)
    mu_evd = float(np.mean(evd))
    sd_evd = float(np.std(evd, ddof=1))
    if sd_evd <= 1e-8:
        sd_evd = 1e-6

    # Per-state volatility (nan-aware)
    state_sigma = np.nanstd(X, axis=0, ddof=1)
    # For states with a single observation, nanstd returns nan -- replace with small floor
    state_sigma = np.where(np.isnan(state_sigma), 1e-4, state_sigma)
    state_sigma = np.maximum(state_sigma, 1e-4)  # floor to avoid div by ~0

    # Per-state energy weights: slightly reduce contribution from ME/NE districts
    state_weights = np.ones(S, dtype=float)
    for j, s in enumerate(states):
        if s in DISTRICT_LABELS:
            state_weights[j] = DISTRICT_WEIGHT

    # Shock fraction stats (nan-aware per row), using state_weights on available states
    frac_shock_list = []
    for i in range(T):
        row = X[i]
        mask = ~np.isnan(row)
        if not np.any(mask):
            continue
        w_av = state_weights[mask]
        w_sum = float(np.sum(w_av)) if np.sum(w_av) > 0 else 1.0
        frac = float(np.sum((np.abs(row[mask]) >= tau).astype(float) * w_av) / w_sum)
        frac_shock_list.append(frac)
    frac_shock = np.array(frac_shock_list)
    mu_shock = float(np.mean(frac_shock))
    sd_shock = float(np.std(frac_shock, ddof=1))
    if sd_shock <= 1e-8:
        sd_shock = 1e-6

    # Factor subspace via SVD on column-centered deltas. Use nan-aware column means
    # and impute missing entries with the column mean (so Xc has zeros where missing).
    mean_Xc = np.nanmean(X, axis=0)
    # Center (will produce NaNs where X is NaN)
    Xc = X - mean_Xc
    # Replace NaNs in centered matrix with 0.0 (equivalent to imputing by column mean)
    Xc_filled = np.where(np.isnan(Xc), 0.0, Xc)
    U, Svals, Vt = np.linalg.svd(Xc_filled, full_matrices=False)
    # Explained variance by singular values
    var = Svals**2
    var_ratio = var / var.sum() if var.sum() > 0 else np.ones_like(var) / len(var)
    cum = np.cumsum(var_ratio)
    K = int(np.searchsorted(cum, var_frac) + 1)
    K = max(K, K_min)
    K = min(K, Vt.shape[0])
    F = Vt[:K, :]  # (K, S), rows orthonormal

    # Factor score stds from U[:, :K] * S[:K]
    scores = U[:, :K] * Svals[:K]
    score_sd = np.std(scores, axis=0, ddof=1)
    score_sd = np.maximum(score_sd, 1e-6)

    # Residual scale from remaining singular values
    if K < len(Svals):
        resid_mean_sq = float(np.sum(Svals[K:]**2) / (Xc_filled.shape[0] * Xc_filled.shape[1]))
    else:
        resid_mean_sq = 1e-6
    resid_scale = float(np.sqrt(max(resid_mean_sq, 1e-12)))

    # Idiosyncratic scale from median |delta| (nan-aware)
    idio_scale = 0.5 * float(np.nanmedian(np.abs(X)))
    if idio_scale <= 1e-8:
        idio_scale = 0.02

    # -------------------------------
    # Historical margin histograms
    # -------------------------------
    # For each historical year, compute two histograms over the state's relative margins:
    #  - state-count histogram (number of states per bin)
    #  - ev-weighted histogram (sum of normalized EV weights per bin)
    # Also compute EV-weighted mean margin and EV-weighted mean absolute margin per year.
    bin_edges = HIST_BINS
    nb = len(bin_edges) - 1
    state_hist_mat = []  # (T_hist, nb)
    ev_hist_mat = []
    ev_margin_list = []
    ev_abs_list = []
    for y in years:
        try:
            row = pivot_m.loc[y].reindex(states).to_numpy()
            w_row = ev_w.loc[y].reindex(states).to_numpy()
        except Exception:
            continue
        mask = ~np.isnan(row)
        if not np.any(mask):
            continue
        vals = row[mask]
        # state-count histogram
        counts, _ = np.histogram(vals, bins=bin_edges)
        state_hist_mat.append(counts.astype(float))
        # ev-weighted histogram: sum of normalized ev weights in each bin
        wvals = w_row[mask]
        # ensure normalized over available states
        wnorm = wvals / float(np.sum(wvals)) if np.sum(wvals) > 0 else np.ones_like(wvals) / len(wvals)
        ev_counts = np.zeros(nb, dtype=float)
        inds = np.digitize(vals, bin_edges) - 1
        inds = np.clip(inds, 0, nb - 1)
        for ii, wi in zip(inds, wnorm):
            ev_counts[ii] += wi
        ev_hist_mat.append(ev_counts)
        # EV-weighted mean and mean absolute
        ev_margin = float(np.sum(wvals * vals) / np.sum(wvals)) if np.sum(wvals) > 0 else float(np.mean(vals))
        ev_margin_list.append(ev_margin)
        ev_abs = float(np.sum(wvals * np.abs(vals)) / np.sum(wvals)) if np.sum(wvals) > 0 else float(np.mean(np.abs(vals)))
        ev_abs_list.append(ev_abs)

    if len(state_hist_mat) == 0:
        # fallback empty
        hist_state_mean = np.zeros(nb, dtype=float)
        hist_state_sd = np.ones(nb, dtype=float) * 1.0
        hist_ev_mean = np.zeros(nb, dtype=float)
        hist_ev_sd = np.ones(nb, dtype=float) * 1.0
        mu_ev_margin = 0.0
        sd_ev_margin = 0.02
        mu_ev_abs = 0.02
        sd_ev_abs = 0.01
    else:
        state_hist_arr = np.vstack(state_hist_mat)
        ev_hist_arr = np.vstack(ev_hist_mat)
        hist_state_mean = np.mean(state_hist_arr, axis=0)
        hist_state_sd = np.std(state_hist_arr, axis=0, ddof=1)
        hist_ev_mean = np.mean(ev_hist_arr, axis=0)
        hist_ev_sd = np.std(ev_hist_arr, axis=0, ddof=1)
        # floors
        hist_state_sd = np.where(hist_state_sd <= 1e-8, 1.0, hist_state_sd)
        hist_ev_sd = np.where(hist_ev_sd <= 1e-8, 1e-3, hist_ev_sd)
        # ev-margin stats
        ev_margin_arr = np.asarray(ev_margin_list, dtype=float)
        ev_abs_arr = np.asarray(ev_abs_list, dtype=float)
        mu_ev_margin = float(np.mean(ev_margin_arr))
        sd_ev_margin = float(np.std(ev_margin_arr, ddof=1)) if ev_margin_arr.size > 1 else float(np.std(ev_margin_arr))
        sd_ev_margin = max(sd_ev_margin, 1e-6)
        mu_ev_abs = float(np.mean(ev_abs_arr))
        sd_ev_abs = float(np.std(ev_abs_arr, ddof=1)) if ev_abs_arr.size > 1 else float(np.std(ev_abs_arr))
        sd_ev_abs = max(sd_ev_abs, 1e-6)

    # National 4-year deltas (y - (y-4)) using nat_series
    nat_pairs = [(y, y - 4) for y in years if (y - 4) in years and (y in nat_series.index) and ((y - 4) in nat_series.index)]
    nat_d = []
    for y, y0 in nat_pairs:
        try:
            ny = float(nat_series.loc[y])
            ny0 = float(nat_series.loc[y0])
            if not (np.isnan(ny) or np.isnan(ny0)):
                nat_d.append(ny - ny0)
        except Exception:
            continue
    if len(nat_d) == 0:
        # fallback small scales
        nat_mu = 0.0
        nat_sd = 0.02
        nat_idio = 0.01
    else:
        nat_d_arr = np.asarray(nat_d, dtype=float)
        nat_mu = float(np.mean(nat_d_arr))
        nat_sd = float(np.std(nat_d_arr, ddof=1)) if len(nat_d_arr) > 1 else float(np.std(nat_d_arr))
        nat_sd = max(nat_sd, 1e-6)
        med_abs = float(np.median(np.abs(nat_d_arr)))
        nat_idio = 0.5 * med_abs if med_abs > 1e-6 else 0.01

    return Targets(
        states=states,
        years=years,
        state_sigma=state_sigma,
        state_weights=state_weights,
        mu_evd=mu_evd,
        sd_evd=sd_evd,
        mu_shock=mu_shock,
        sd_shock=sd_shock,
        F=F,
        score_sd=score_sd,
        mean_Xc=mean_Xc,
        resid_scale=resid_scale,
        idio_scale=idio_scale,
    nat_mu=nat_mu,
    nat_sd=nat_sd,
    nat_idio_scale=nat_idio,
    # histogram & ev-margin stats
    hist_bins=HIST_BINS,
    hist_state_mean=hist_state_mean,
    hist_state_sd=hist_state_sd,
    hist_ev_mean=hist_ev_mean,
    hist_ev_sd=hist_ev_sd,
    mu_ev_margin=mu_ev_margin,
    sd_ev_margin=sd_ev_margin,
    mu_ev_abs=mu_ev_abs,
    sd_ev_abs=sd_ev_abs,
    nat_by_year={int(k): float(v) for k, v in nat_series.to_dict().items()},
    )


# ---------------------
# Energy for a delta d
# ---------------------
def _proj_residual(v: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Project v onto orthogonal complement of span(F rows). Return residual vector."""
    if F.size == 0:
        return v
    # Rows of F are orthonormal (from SVD Vt), so projection = F.T @ (F @ v)
    proj = F.T @ (F @ v)
    return v - proj


def energy_delta(
    d: np.ndarray,
    targets: Targets,
    w_target: np.ndarray,
    m_baseline: Optional[np.ndarray] = None,
    alpha_ev: float = 1.0,
    alpha_var: float = 1.0,
    alpha_shock: float = 1.0,
    alpha_resid: float = 1.0,
    alpha_ev_margin: float = 0.0,
    alpha_ev_quad: float = 0.0,
    alpha_hist_state: float = 0.0,
    alpha_hist_ev: float = 0.0,
) -> float:
    """Compute total energy for delta vector d by delegating to energy_components.

    Keeps a single canonical implementation (energy_components) and returns
    the scalar E_TOTAL to preserve existing callers. Note that several
    optional components (E_EV_MARGIN, E_EV_QUAD, E_HIST_STATE, E_HIST_EV)
    are controlled by their respective alpha_* knobs. If those alphas are
    left at their defaults (0.0) the components will be computed but
    multiplied by zero, producing 0.0 in the returned total.
    """
    comps = energy_components(
        d,
        targets,
        w_target,
        m_baseline=m_baseline,
        alpha_ev=alpha_ev,
        alpha_var=alpha_var,
        alpha_shock=alpha_shock,
        alpha_resid=alpha_resid,
        alpha_ev_margin=alpha_ev_margin,
        alpha_ev_quad=alpha_ev_quad,
        alpha_hist_state=alpha_hist_state,
        alpha_hist_ev=alpha_hist_ev,
    )
    return float(comps.get("E_TOTAL", comps.get("E_EV", 0.0)))


def energy_components(
    d: np.ndarray,
    targets: Targets,
    w_target: np.ndarray,
    m_baseline: Optional[np.ndarray] = None,
    alpha_ev: float = ALPHA_EV,
    alpha_var: float = ALPHA_VAR,
    alpha_shock: float = ALPHA_SHOCK,
    alpha_resid: float = ALPHA_RESID,
    alpha_ev_margin: float = ALPHA_EV_MARGIN,
    alpha_ev_quad: float = ALPHA_EV_QUAD,
    alpha_hist_state: float = ALPHA_HIST_STATE,
    alpha_hist_ev: float = ALPHA_HIST_EV,
) -> Dict[str, float]:
    """Return a dict of energy components for a delta vector d.

        This function computes a set of scalar diagnostics that together form a
        total "energy" score for a proposed vector of state deltas (d). The
        returned dictionary contains the following keys (E_*):

        - E_EV : EV-weighted national-delta term. Measures how the EV-weighted
            mean of the proposed delta deviates from the historical EV-weighted
            mean of 4-year deltas (targets.mu_evd). This is a z^2 term scaled by
            targets.sd_evd and multiplied by `alpha_ev`.

        - E_VAR : Per-state variance term. Computes a weighted average of
            (d_i / sigma_i)^2 across states (sigma_i from historical per-state
            volatility). `state_weights` can downweight certain units (e.g., ME/NE
            districts). This term penalizes unusually large per-state moves and is
            controlled by `alpha_var`.

        - E_SHOCK : Fraction-of-shocks term. Computes the weighted fraction of
            states whose |d_i| exceeds the shock threshold `TAU`, compares that
            fraction to the historical mean `targets.mu_shock` and returns a
            z^2-style penalty (controlled by `alpha_shock`). In the current
            implementation this term is disabled (set to 0) because shocks are
            observed frequently in history and we avoid penalizing them by default.

        - E_RESID : Residual term orthogonal to the learned factor subspace.
            Projects the centered delta vector onto the orthogonal complement of
            the factor rows (targets.F) and measures the norm of that residual
            relative to an empirical residual scale (targets.resid_scale). This
            discourages proposals that lie outside the historical factor subspace
            and is controlled by `alpha_resid`.

        The following components are optional histogram / margin concentration
        penalties that require `m_baseline` to be provided and are toggled by
        their alpha knobs (defaults are 0.0):

        - E_EV_MARGIN : EV-weighted margin term. Computes the EV-weighted mean of
            the final margins (m_baseline + d) and compares it to the historical
            EV-weighted mean margin (targets.mu_ev_margin) via a z^2 penalty.
            Multiply by `alpha_ev_margin` to enable.

        - E_EV_QUAD : Quadratic signed EV concentration. A signed, squared
            concentration index that rewards or penalizes concentration of margin
            signs and magnitudes across EV weights. Useful to encourage maps with
            historically-typical concentration patterns. Controlled by
            `alpha_ev_quad`.

        - E_HIST_STATE / E_HIST_EV : Histogram-match terms. These compare the
            distribution of simulated state margins (state-count histogram) and the
            EV-weighted histogram of simulated margins against historical means and
            standard deviations (targets.hist_state_mean / hist_ev_mean). They are
            useful to encourage maps whose distribution of margins resembles
            historical cycles. Enable with `alpha_hist_state` and `alpha_hist_ev`.

        Numerics and NaN handling:
            - For vectors containing NaNs (historical 4-year delta rows), the EV
                computations renormalize weights over available states. The VAR and
                SHOCK terms operate only over available states. The RESID term
                imputes missing entries with the column mean (so those positions
                contribute zero to the centered residual).

        Returns a dict containing all individual E_* terms plus `E_TOTAL`,
        `evd` (the EV-weighted delta), `ev_margin` (if m_baseline provided),
        `shock_frac` and `n_avail`.
        """
    d = np.asarray(d, dtype=float)
    S = d.size
    mask = ~np.isnan(d)
    n_avail = int(np.sum(mask))

    # EV term (renormalize weights over available states)
    if n_avail > 0:
        w_av = w_target[mask]
        d_av = d[mask]
        w_sum = float(np.sum(w_av))
        if w_sum > 0:
            evd = float(np.sum(w_av * d_av) / w_sum)
        else:
            evd = float(np.mean(d_av))
    else:
        evd = 0.0
    z_ev = (evd - targets.mu_evd) / targets.sd_evd
    E_EV = (z_ev * z_ev) * alpha_ev

    # Optional EV-margin and histogram components
    E_EV_MARGIN = 0.0
    E_EV_QUAD = 0.0
    E_HIST_STATE = 0.0
    E_HIST_EV = 0.0
    ev_margin = 0.0
    if m_baseline is not None and n_avail > 0:
        margins = np.array(m_baseline, dtype=float)
        margins[mask] = margins[mask] + d[mask]
        ev_margin = float(np.sum(w_target[mask] * margins[mask]) / float(np.sum(w_target[mask])) ) if np.sum(w_target[mask]) > 0 else float(np.mean(margins[mask]))
        z_em = (ev_margin - targets.mu_ev_margin) / targets.sd_ev_margin
        E_EV_MARGIN = (z_em * z_em) * alpha_ev_margin

        mu_abs = max(targets.mu_ev_abs, 1e-6)
        signed = np.sign(margins[mask]) * (np.abs(margins[mask]) / mu_abs) ** 2
        ev_quad_raw = float(np.sum((w_target[mask] / float(np.sum(w_target[mask]))) * signed)) if np.sum(w_target[mask]) > 0 else float(np.sum(signed) / len(signed))
        E_EV_QUAD = (ev_quad_raw * ev_quad_raw) * alpha_ev_quad

        try:
            sim_counts, _ = np.histogram(margins[mask], bins=targets.hist_bins)
            nb = len(targets.hist_bins) - 1
            sim_ev = np.zeros(nb, dtype=float)
            inds = np.digitize(margins[mask], targets.hist_bins) - 1
            inds = np.clip(inds, 0, nb - 1)
            wnorm = w_target[mask] / float(np.sum(w_target[mask])) if np.sum(w_target[mask]) > 0 else (w_target[mask] if w_target[mask].size>0 else np.ones_like(inds)/len(inds))
            for ii, wi in zip(inds, wnorm):
                sim_ev[ii] += wi
            z_state = (sim_counts - targets.hist_state_mean) / targets.hist_state_sd
            E_HIST_STATE = float(np.sum(z_state * z_state) / max(len(z_state), 1)) * alpha_hist_state
            z_ev_hist = (sim_ev - targets.hist_ev_mean) / targets.hist_ev_sd
            E_HIST_EV = float(np.sum(z_ev_hist * z_ev_hist) / max(len(z_ev_hist), 1)) * alpha_hist_ev
        except Exception:
            E_HIST_STATE = 0.0
            E_HIST_EV = 0.0

    # VAR term over available states (weighted)
    if n_avail > 0:
        z = (d[mask] / targets.state_sigma[mask])
        w = targets.state_weights[mask]
        den = float(np.sum(w)) if np.sum(w) > 0 else float(len(z))
        E_VAR = float(np.sum(w * (z * z)) / den) * alpha_var
    else:
        E_VAR = 0.0

    # SHOCK term over available states (weighted)
    if n_avail > 0:
        w = targets.state_weights[mask]
        den = float(np.sum(w)) if np.sum(w) > 0 else float(n_avail)
        frac = float(np.sum(((np.abs(d[mask]) >= TAU).astype(float)) * w) / den)
    else:
        frac = 0.0
    z_shock = (frac - targets.mu_shock) / targets.sd_shock
    E_SHOCK = (z_shock * z_shock) * alpha_shock

    # Residual (orthogonal to factor subspace), imputing missing with column mean
    v = np.zeros_like(d)
    if n_avail > 0:
        v[mask] = d[mask] - targets.mean_Xc[mask]
    # Where missing, v stays 0 (i.e., imputed by column mean)
    r = _proj_residual(v, targets.F)
    denom = math.sqrt(max(n_avail, 1)) * targets.resid_scale
    if denom <= 1e-9:
        denom = 1e-6
    E_RESID = float((np.linalg.norm(r) / denom) ** 2) * alpha_resid

    total = E_EV + E_VAR + E_SHOCK + E_RESID + E_EV_MARGIN + E_EV_QUAD + E_HIST_STATE + E_HIST_EV
    return {
        "E_EV": float(E_EV),
        "E_VAR": float(E_VAR),
        "E_SHOCK": float(E_SHOCK),
        "E_RESID": float(E_RESID),
        "E_EV_MARGIN": float(E_EV_MARGIN),
        "E_EV_QUAD": float(E_EV_QUAD),
        "E_HIST_STATE": float(E_HIST_STATE),
        "E_HIST_EV": float(E_HIST_EV),
        "E_TOTAL": float(total),
        "evd": float(evd),
        "ev_margin": float(ev_margin),
        "shock_frac": float(frac),
        "n_avail": float(n_avail),
    }


# -----------------
# Sampling deltas
# -----------------
def sample_deltas(
    targets: Targets,
    w_target: np.ndarray,
    m_baseline: Optional[np.ndarray] = None,
    ev_vec: Optional[np.ndarray] = None,
    N_draw: int = N_DRAW,
    N_keep: int = N_KEEP,
    seed: int = SEED,
    clip: float = CLIP_D,
    idio_scale: Optional[float] = None,
    alpha_ev: float = ALPHA_EV,
    alpha_var: float = ALPHA_VAR,
    alpha_shock: float = ALPHA_SHOCK,
    alpha_resid: float = ALPHA_RESID,
    # new knobs (defaults from top-level tuning)
    alpha_ev_margin: float = ALPHA_EV_MARGIN,
    alpha_ev_quad: float = ALPHA_EV_QUAD,
    # New selection options:
    # - selection_mode: 'lowest' (default) keep N_keep proposals with lowest E_TOTAL
    # - selection_mode: 'component_range' keep proposals that best match historical
    #   component ranges derived from hist_df (passed below). When using
    #   'component_range' provide hist_df or precomputed ranges.
    selection_mode: str = "component_range",
    hist_df: Optional[pd.DataFrame] = None,
    comp_q_low: float = 0.05,
    comp_q_high: float = 0.95,
    alpha_hist_state: float = ALPHA_HIST_STATE,
    alpha_hist_ev: float = ALPHA_HIST_EV,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw factor+noise proposals, score by energy, return lowest-energy subset.

    Returns (keep_d, energies) where keep_d is (N_keep, S) and energies is (N_keep,).
    """
    rng = np.random.default_rng(seed)
    S = len(targets.states)
    K = targets.F.shape[0]

    if idio_scale is None:
        idio_scale = targets.idio_scale

    # Factor draws: coeffs ~ N(0, score_sd)
    coeffs = rng.normal(loc=0.0, scale=targets.score_sd, size=(N_draw, K)) if K > 0 else np.zeros((N_draw, 0))
    d_factor = coeffs @ targets.F  # (N_draw, S)

    # Idiosyncratic Laplace noise (heavy tails)
    eps = rng.laplace(loc=0.0, scale=idio_scale, size=(N_draw, S))
    proposals = d_factor + eps
    proposals = np.clip(proposals, -clip, clip)

    # Compute full energy components for each proposal (we'll always need E_TOTAL
    # but may also use the individual components when selection_mode != 'lowest')
    comps_list: List[Dict[str, float]] = [
        energy_components(proposals[i], targets, w_target, m_baseline=m_baseline)
        for i in range(N_draw)
    ]
    energies = np.array([float(c.get("E_TOTAL", 0.0)) for c in comps_list])

    if selection_mode == "component_range":
        # Need historical ranges per component. Prefer hist_df if provided; otherwise
        # attempt to read from OUT_DIR/historical_energies.csv if present in cwd.
        if hist_df is None:
            try:
                hist_path = os.path.join(os.getcwd(), OUT_DIR, "historical_energies.csv")
                if os.path.exists(hist_path):
                    hist_df = pd.read_csv(hist_path, index_col=0)
            except Exception:
                hist_df = None

        # Determine which energy keys to consider (exclude E_TOTAL itself)
        sample_keys = [k for k in comps_list[0].keys() if k.startswith("E_") and k != "E_TOTAL"] if comps_list else []

        if hist_df is not None and not hist_df.empty:
            # Use quantile-based ranges per component
            ranges = {}
            for k in sample_keys:
                if k in hist_df.columns:
                    lo = float(hist_df[k].quantile(comp_q_low))
                    hi = float(hist_df[k].quantile(comp_q_high))
                else:
                    # fallback: use min/max from historical values we have (or +-inf)
                    vals = hist_df.get(k)
                    if vals is None:
                        lo, hi = (-np.inf, np.inf)
                    else:
                        lo = float(vals.min())
                        hi = float(vals.max())
                ranges[k] = (lo, hi)
        else:
            # No historical info -> treat everything as acceptable range
            ranges = {k: (-np.inf, np.inf) for k in sample_keys}

        # Score proposals by fraction of components that lie within historical ranges
        comp_scores = np.zeros(N_draw, dtype=float)
        for i, cdict in enumerate(comps_list):
            cnt = 0
            tot = 0
            for k, (lo, hi) in ranges.items():
                if k in cdict:
                    tot += 1
                    val = float(cdict[k])
                    if val >= lo and val <= hi:
                        cnt += 1
            comp_scores[i] = (cnt / tot) if tot > 0 else 0.0

        # Keep proposals with highest comp_scores (prefer more components inside hist range).
        # Tie-breaker: lower E_TOTAL
        order = np.lexsort((energies, -comp_scores))
        idx_keep = order[::-1][:N_keep]  # highest scores first
        keep = proposals[idx_keep]
        keep_E = comp_scores[idx_keep]
        return keep, keep_E, idx_keep

    # Default behavior: keep N_keep lowest by E_TOTAL
    idx = np.argsort(energies)[:N_keep]
    keep = proposals[idx]
    keep_E = energies[idx]
    return keep, keep_E, idx


def maps_from_deltas(m_baseline: np.ndarray, keep_d: np.ndarray) -> np.ndarray:
    return m_baseline[None, :] + keep_d


def expand_maps_with_atlarge(maps_sample: np.ndarray, sample_states: List[str], full_states: List[str], atlarge_groups: Dict[str, List[str]]):
    """Expand maps defined over sample_states into full_states by inserting
    at-large entries computed as the mean of their districts.

    - maps_sample: (N, S_sample)
    - returns maps_full: (N, S_full) with the same ordering as full_states
    """
    S_full = len(full_states)
    N = maps_sample.shape[0]
    maps_full = np.full((N, S_full), np.nan, dtype=float)

    # build index maps
    idx_sample = {s: i for i, s in enumerate(sample_states)}
    idx_full = {s: i for i, s in enumerate(full_states)}

    # copy over sample-state values
    for s in sample_states:
        if s in idx_full:
            maps_full[:, idx_full[s]] = maps_sample[:, idx_sample[s]]

    # compute at-large values as mean of available districts
    for al, districts in atlarge_groups.items():
        if al not in idx_full:
            continue
        # gather district indices that exist in full_states
        avail = [d for d in districts if d in idx_full]
        # if no districts are present, leave NaN
        if not avail:
            continue
        # For each district, prefer sample value if present, otherwise fall back to full index
        district_cols = []
        for d in avail:
            if d in idx_sample:
                district_cols.append(maps_sample[:, idx_sample[d]])
            else:
                # district was not part of sampling (unlikely) but may exist in baseline/full;
                # leave as NaN column of length N
                district_cols.append(np.full((N,), np.nan))
        # stack and take nanmean across columns
        stacked = np.vstack(district_cols)
        maps_full[:, idx_full[al]] = np.nanmean(stacked, axis=0)

    # For any remaining full-state columns that are still nan (e.g., states that were neither
    # in sample_states nor at-large groups), fill with 0 to avoid downstream issues.
    nan_mask = np.isnan(maps_full)
    if np.any(nan_mask):
        maps_full[nan_mask] = 0.0
    return maps_full


def cluster_maps(maps: np.ndarray, k: int = KMEANS_K, seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    km = KMeans(n_clusters=k, n_init=20, random_state=seed)
    labels = km.fit_predict(maps)
    return labels, km.cluster_centers_


def ev_counts_for_map(m_vec: np.ndarray, ev_vec: np.ndarray, pv_shift: float = 0.0, tie_to: str = "R") -> Tuple[int, int]:
    state_m = m_vec + pv_shift
    # D gets state if margin > 0; ties to tie_to
    d_wins = (state_m > 0).astype(int)
    tie_mask = (state_m == 0)
    if tie_to.upper() == "D":
        d_wins = np.where(tie_mask, 1, d_wins)
    # else ties to R → leave as 0 for D
    D_EV = int(np.sum(ev_vec * d_wins))
    total = int(np.sum(ev_vec))
    R_EV = total - D_EV
    return D_EV, R_EV


def format_centroid(centroid: np.ndarray, states: List[str], sort_ascending: bool = True) -> List[str]:
    pairs = list(zip(states, centroid.tolist()))
    pairs.sort(key=lambda x: x[1], reverse=not sort_ascending)
    lines = [f"{abbr},{val:+.3f}" for abbr, val in pairs]
    return lines


# -----------------------------
# Tipping point report (simple)
# -----------------------------
def tipping_report_for_map(states: List[str], margins: np.ndarray, evs: np.ndarray, title: str) -> str:
    """Build a compact tipping report using historical_tipping_points helpers if available."""
    rows = [{"abbr": s, "relative_margin": float(m), "evs": int(e)} for s, m, e in zip(states, margins, evs)]

    lines: List[str] = [title, ""]

    if htp is not None:
        try:
            tps, ordered = htp.compute_threshold_tipping_points(rows)
            mismatch = htp.compute_ev_pv_mismatch_ranges(rows)
            ties = htp.compute_ec_tie_ranges(rows)

            lines.append("Tipping thresholds (by national margin):")
            r_keys = ["R_all", "R_sweep", "R_blowout", "R_landslide", "R_solid", "R_squeak"]
            d_keys = ["D_squeak", "D_solid", "D_landslide", "D_blowout", "D_sweep", "D_all"]
            for key in r_keys + d_keys:
                if key in tps:
                    tp = tps[key]
                    state_emoji = utils.emoji_from_lean(tp['margin'])
                    lines.append(f"  {key}: {utils.lean_str(tp['margin'])} via {tp['state']} (D: {tp['D_evs_after']} EV, R: {tp['R_evs_after']} EV)")

            # Mismatch ranges
            if mismatch:
                lines.append("")
                for k, (lo, hi) in mismatch.items():
                    desc = 'D PV / R EC' if k == 'D_PV_R_EC' else 'R PV / D EC'
                    lines.append(f"Mismatch {desc}: ({utils.lean_str(lo)},{utils.lean_str(hi)})")

            # Tie ranges
            if ties:
                lines.append("")
                lines.append("Potential EC Ties (269-269):")
                for tr in ties:
                    start = tr.get('start'); end = tr.get('end'); st = tr.get('state')
                    if end is None:
                        lines.append(f"  at {utils.lean_str(start)} via {st}")
                    else:
                        lines.append(f"  {utils.lean_str(start)} → {utils.lean_str(end)} via {st}")

            # Ordered states by relative margin
            lines.append("")
            lines.append("States (R→D order):")
            for abbr, ev, rm in ordered:
                state_emoji = utils.emoji_from_lean(rm, use_swing=True)
                lines.append(f"  {state_emoji}{abbr}:\t\t\t{utils.lean_str(rm)} ({int(ev)} EV)")
        except Exception as e:  # pragma: no cover
            lines.append(f"[tipping computation failed: {e}]")
    else:
        lines.append("[historical_tipping_points not available; skipping detailed tipping analysis]")

    return "\n".join(lines)


def compute_historical_energies(pivot_m: pd.DataFrame, ev_w: pd.DataFrame, states: List[str], targets: Targets, out_root: str) -> pd.DataFrame:
    """Compute energy components for each historical year map (relative margins).

    Returns a DataFrame with per-year components and saves CSV + a plot PNG to out_root.
    """
    rows = []
    os.makedirs(out_root, exist_ok=True)
    for y in pivot_m.index:
        # build margin vector for this year aligned to states
        try:
            m = pivot_m.loc[y].reindex(states).to_numpy(dtype=float)
        except Exception:
            continue
        # compute delta relative to mean_Xc (or zero?) --- we want to score the map itself as a "delta" from baseline=0
        # Use d = m - mean_Xc so residual term behaves as for deltas
        d = m - targets.mean_Xc
        # weights for EV term: use ev_w for that year if available
        if y in ev_w.index:
            w = ev_w.loc[y].reindex(states).to_numpy(dtype=float)
        else:
            # fallback to uniform
            w = np.ones(len(states), dtype=float) / len(states)

        # Use top-level alpha knobs (defaults applied inside energy_components)
        comps = energy_components(d, targets, w, m_baseline=m)
        comps['year'] = int(y)
        rows.append(comps)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index('year').sort_index()
    csv_out = os.path.join(out_root, 'historical_energies.csv')
    df.to_csv(csv_out)

    # Plot all energy components (any column starting with 'E_'), including E_TOTAL.
    plt.figure(figsize=(10, 4.5))
    energy_cols = [c for c in df.columns if c.startswith('E_')]
    # remove any which are all zero
    energy_cols = [c for c in energy_cols if df[c].sum() != 0]
    # sensible default markers to help differentiate short series
    markers = ['o', 's', '^', 'x', 'd', '*', '+', 'v', '>', '<']
    for i, col in enumerate(energy_cols):
        mk = markers[i % len(markers)]
        plt.plot(df.index, df[col], marker=mk, label=col)
    plt.xlabel('year')
    # set x ticks to every presidential election year
    plt.xticks([yr for yr in df.index if yr % 4 == 0])
    plt.ylabel('energy')
    plt.title('Historical energies per year')
    plt.legend(ncol=2, fontsize='small')
    plt.grid(True)
    png_out = os.path.join(out_root, 'historical_energies.png')
    plt.tight_layout()
    plt.savefig(png_out)
    plt.close()
    return df


# ---------------------------
# Yearly simulation pipeline
# ---------------------------
def run_sampler_for_year(
    year_label: int,
    m_baseline: np.ndarray,
    ev_vec: np.ndarray,
    w_target: np.ndarray,
    states: List[str],
    targets: Targets,
    out_root: str,
    seed: int = SEED,
    n_draw: int = N_DRAW,
    n_keep: int = N_KEEP,
    kmeans_k: int = KMEANS_K,
    example_margin: Optional[float] = None,
    nat_baseline: Optional[float] = None,
    nat_party: Optional[str] = None,
    # sensible defaults for new energy terms (tunables taken from top-level knobs)
    alpha_ev_margin: float = ALPHA_EV_MARGIN,
    alpha_ev_quad: float = ALPHA_EV_QUAD,
    alpha_hist_state: float = ALPHA_HIST_STATE,
    alpha_hist_ev: float = ALPHA_HIST_EV,
    # option to use medoids instead of centroids
    use_medoids: bool = False,
) -> Dict:
    os.makedirs(out_root, exist_ok=True)

    # Exclude at-large labels from sampling; we'll set them as district averages later.
    sample_states = [s for s in states if s not in AT_LARGE_GROUPS]
    if len(sample_states) != len(states):
        # Build reduced baseline, ev_vec, and w_target for sampling
        idx_full = {s: i for i, s in enumerate(states)}
        m_baseline_sample = np.array([m_baseline[idx_full[s]] for s in sample_states])
        ev_vec_sample = np.array([ev_vec[idx_full[s]] for s in sample_states])
        w_target_sample = np.array([w_target[idx_full[s]] for s in sample_states])
        # Build temporary targets that reflect the reduced state vector. We'll reuse the same
        # targets.F by selecting the columns corresponding to sample_states, and similarly for mean_Xc/state_sigma.
        col_idx = [idx_full[s] for s in sample_states]
        # Make a shallow copy of targets with reduced-dimension arrays
        targets_sample = Targets(
            states=sample_states,
            years=targets.years,
            state_sigma=targets.state_sigma[col_idx],
            state_weights=targets.state_weights[col_idx],
            mu_evd=targets.mu_evd,
            sd_evd=targets.sd_evd,
            mu_shock=targets.mu_shock,
            sd_shock=targets.sd_shock,
            F=targets.F[:, col_idx] if targets.F.size else targets.F,
            score_sd=targets.score_sd,
            mean_Xc=targets.mean_Xc[col_idx],
            resid_scale=targets.resid_scale,
            idio_scale=targets.idio_scale,
            nat_mu=targets.nat_mu,
            nat_sd=targets.nat_sd,
            nat_idio_scale=targets.nat_idio_scale,
            # histogram & ev-margin stats are shared (not per-state)
            hist_bins=targets.hist_bins,
            hist_state_mean=targets.hist_state_mean,
            hist_state_sd=targets.hist_state_sd,
            hist_ev_mean=targets.hist_ev_mean,
            hist_ev_sd=targets.hist_ev_sd,
            mu_ev_margin=targets.mu_ev_margin,
            sd_ev_margin=targets.sd_ev_margin,
            mu_ev_abs=targets.mu_ev_abs,
            sd_ev_abs=targets.sd_ev_abs,
            nat_by_year=targets.nat_by_year,
        )
        # Draw proposals on reduced-dimension targets
        keep_d, keep_E, idx = sample_deltas(
            targets_sample,
            w_target_sample,
            m_baseline=m_baseline_sample,
            ev_vec=ev_vec_sample,
            N_draw=n_draw,
            N_keep=n_keep,
            seed=seed,
        )
        maps_sample = maps_from_deltas(m_baseline_sample, keep_d)
        # Expand sample maps back to full-state vector by filling at-large entries with district averages
        maps = expand_maps_with_atlarge(maps_sample, sample_states, states, AT_LARGE_GROUPS)
    else:
        keep_d, keep_E, idx = sample_deltas(
            targets,
            w_target,
            m_baseline=m_baseline,
            ev_vec=ev_vec,
            N_draw=n_draw,
            N_keep=n_keep,
            seed=seed,
        )
        maps = maps_from_deltas(m_baseline, keep_d)

    # Sample national margin deltas independently (optionally constrained by nat_party) and align with kept proposals
    rng_nat = np.random.default_rng(seed + 101)
    # Determine constraint thresholds
    constraint = None
    nat_party_norm = (nat_party or "").strip().upper() if nat_party else None
    min_delta = None
    max_delta = None
    if nat_party_norm in ("D", "R"):
        # Require the resulting national margin to favor the requested party
        # national_margin_final = nat_baseline + delta; positive favors D, negative favors R
        if nat_party_norm == "D":
            # delta >= -nat_baseline (e.g., nat_baseline=-0.016 -> delta>=0.016)
            min_delta = max(0.0 - float(nat_baseline or 0.0), 0.0)
            constraint = ("min", min_delta)
        else:
            # R: delta <= -nat_baseline (e.g., nat_baseline=+0.01 -> delta<=-0.01)
            max_delta = min(0.0 - float(nat_baseline or 0.0), 0.0)
            constraint = ("max", max_delta)

    def _sample_nat_batch(sz: int) -> np.ndarray:
        arr = rng_nat.laplace(loc=targets.nat_mu, scale=targets.nat_idio_scale, size=(sz,))
        arr = np.clip(arr, -NAT_CLIP, NAT_CLIP)
        if constraint is None:
            return arr
        kind, thr = constraint
        if kind == "min":
            return arr[arr >= thr]
        else:
            return arr[arr <= thr]

    if constraint is None:
        nat_proposals = _sample_nat_batch(n_draw)
        # In theory size should be n_draw; if not, resample simply
        while nat_proposals.size < n_draw:
            nat_proposals = np.concatenate([nat_proposals, _sample_nat_batch(n_draw)])
        nat_proposals = nat_proposals[:n_draw]
    else:
        # Rejection sample until we have n_draw respecting the constraint
        acc: List[float] = []
        max_iters = 50
        it = 0
        while len(acc) < n_draw and it < max_iters:
            batch = _sample_nat_batch(n_draw)
            if batch.size:
                acc.extend(batch.tolist())
            it += 1
        if len(acc) < n_draw:
            # Fallback: enforce threshold with small tail noise in allowed direction
            rem = n_draw - len(acc)
            noise = np.abs(rng_nat.laplace(loc=0.0, scale=max(1e-3, targets.nat_idio_scale * 0.25), size=(rem,)))
            if constraint[0] == "min":
                fill = (min_delta if min_delta is not None else 0.0) + noise
            else:
                fill = (max_delta if max_delta is not None else 0.0) - noise
            acc.extend(fill.tolist())
        nat_proposals = np.array(acc[:n_draw], dtype=float)

    nat_keep = nat_proposals[idx]

    # Diagnostics: energy percentiles and shock fraction of kept
    pct = np.percentile(keep_E, [5, 25, 50, 75, 95])
    shock_fracs = (np.abs(keep_d) >= TAU).mean(axis=1)
    shock_avg = float(np.mean(shock_fracs))

    labels, centroids = cluster_maps(maps, k=kmeans_k, seed=seed)
    cluster_sizes = [int(np.sum(labels == i)) for i in range(kmeans_k)]
    mean_E_by_cluster = [float(np.mean(keep_E[labels == i])) if np.any(labels == i) else float("nan") for i in range(kmeans_k)]

    # Optionally replace centroids with medoids (a cluster member closest to centroid)
    if use_medoids:
        try:
            new_centroids = np.zeros_like(centroids)
            for i in range(kmeans_k):
                mask = labels == i
                if np.any(mask):
                    members = maps[mask]
                    center = centroids[i]
                    dists = np.linalg.norm(members - center, axis=1)
                    best = np.argmin(dists)
                    new_centroids[i] = members[best]
                else:
                    new_centroids[i] = centroids[i]
            centroids = new_centroids
        except Exception:
            # fallback to original centroids on any failure
            pass

    # Per-cluster national PV shift (baseline + mean sampled nat delta for that cluster)
    if nat_baseline is None:
        # default to 0 baseline if not provided
        nat_baseline = 0.0
    pv_shift_by_cluster = []
    for i in range(kmeans_k):
        mask = (labels == i)
        if np.any(mask):
            pv_shift = float(nat_baseline + np.mean(nat_keep[mask]))
        else:
            pv_shift = float(nat_baseline)
        pv_shift_by_cluster.append(pv_shift)

    # Save diagnostics
    diag = {
        "year": year_label,
        "energy_percentiles": {"p5": pct[0], "p25": pct[1], "p50": pct[2], "p75": pct[3], "p95": pct[4]},
        "avg_shock_frac": shock_avg,
        "cluster_sizes": cluster_sizes,
        "mean_energy_by_cluster": mean_E_by_cluster,
        "K_factors": int(targets.F.shape[0]),
        "sd_evd": float(targets.sd_evd),
        "mu_shock": float(targets.mu_shock),
        "sd_shock": float(targets.sd_shock),
        "resid_scale": float(targets.resid_scale),
        "example_margin": example_margin,
        "nat_baseline": float(nat_baseline),
        "nat_mu": float(targets.nat_mu),
        "nat_sd": float(targets.nat_sd),
        "nat_idio_scale": float(targets.nat_idio_scale),
        "pv_shift_by_cluster": pv_shift_by_cluster,
    "nat_party_constraint": nat_party_norm,
    "nat_min_delta_required": float(min_delta) if min_delta is not None else None,
    "nat_max_delta_required": float(max_delta) if max_delta is not None else None,
    }
    with open(os.path.join(out_root, f"{year_label}_diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2)

    # Per-centroid outputs
    results = []
    for i, c in enumerate(centroids):

        # Determine example PV shift to display for this centroid
        pv_shift_this = pv_shift_by_cluster[i]
        ex_margin = pv_shift_this if example_margin is None else example_margin
        
        # Sorted CSV and TXT
        pairs = list(zip(states, c.tolist(), ev_vec.astype(int).tolist(), [ex_margin] * len(states), [ex_margin+c[idx_full[s]] for s in states]))
        pairs.sort(key=lambda x: x[1])  # ascending by margin (R→D)

        # CSV
        df_c = pd.DataFrame(pairs, columns=["abbr", "relative_margin", "electoral_votes", "national_margin", "final_margin"])
        csv_path = os.path.join(out_root, f"{year_label}_csv_centroid_{i+1}.csv")
        df_c.to_csv(csv_path, index=False)

        # TXT lines
        txt_lines = [f"Centroid {i+1} for {year_label}", "abbr,relative_margin"]
        # Append an example final-margin string per state (uses pv shift)
        for a, m, e, _, fm in pairs:
            if ex_margin is None:
                example_str = ""
            else:
                #fm = m + ex_margin
                example_str = f"\t{utils.final_margin_color_key(fm)}\n\tfinal ({utils.lean_str(ex_margin)}):\t{utils.emoji_from_lean(fm)} {utils.lean_str(fm)}"
            margin_change = f"\tchange: {utils.lean_str(m - m_baseline[idx_full[a]])} (Δ from baseline), was {utils.lean_str(m_baseline[idx_full[a]])}"
            txt_lines.append(f"{utils.emoji_from_lean(m, use_swing=True)}{a}\t\t{utils.lean_str(m)},\t{int(e)}{example_str}\n{margin_change}")

        # Shock log (|predicted - baseline| >= TAU)
        deltas = c - m_baseline
        shock_mask = np.abs(deltas) >= TAU
        shock_idx = np.where(shock_mask)[0]
        if shock_idx.size > 0:
            # Sort shocks by magnitude descending
            shock_sorted = sorted(((idx, float(deltas[idx])) for idx in shock_idx), key=lambda t: abs(t[1]), reverse=True)
            txt_lines.append("")
            txt_lines.append(f"Shocks (|predicted - baseline| ≥ {TAU:.3f}): {len(shock_sorted)} states ({len(shock_sorted)/len(states):.3f})")
            txt_lines.append("abbr, Δ(pred-baseline), baseline → predicted")
            for idx, dval in shock_sorted:
                abbr = states[idx]
                base = float(m_baseline[idx])
                pred = float(c[idx])
                swing_emoji = utils.emoji_from_lean(dval, use_swing=True)
                txt_lines.append(f"{swing_emoji}{abbr}\t\t{utils.lean_str(dval)},\t{utils.lean_str(base)} → {utils.lean_str(pred)}")

        txt_path = os.path.join(out_root, f"{year_label}_centroid_{i+1}.txt")

        # EV counts under PV shifts for this centroid
        D0, R0 = ev_counts_for_map(c, ev_vec, pv_shift=0.0)
        Dnat_D, Dnat_R = ev_counts_for_map(c, ev_vec, pv_shift=pv_shift_this)
        Dp3, Rp3 = ev_counts_for_map(c, ev_vec, pv_shift=0.03)

        # Flips relative to baseline and shock metadata
        flips = int(np.sum(np.sign(c) != np.sign(m_baseline)))
        shock_states = [states[s_idx] for s_idx in shock_idx]

        # Tipping point log — prefer centralized historical_tipping_points saver when available
        tp_title = f"Tipping points for {year_label} centroid {i+1}"
        tp_path = os.path.join(out_root, f"{year_label}_tipping_centroid_{i+1}.txt")
        if htp is not None:
            # Build state dicts expected by historical_tipping_points helpers
            rows = [{"abbr": s, "relative_margin": float(m), "evs": int(e), "national_margin": 0.0}
                    for s, m, e in zip(states, c, ev_vec)]
            try:
                # pass extra_margins=None to use htp defaults; callers can change this later
                saved = htp.save_tipping_report(rows, out_root, os.path.basename(tp_path), year=year_label, source=f"{year_label}_centroid_{i+1}", extra_margins=None)
                tp_path = saved
            except Exception:
                # fallback to local formatter if centralized saver fails
                tp_report = tipping_report_for_map(states, c, ev_vec, tp_title)
                with open(tp_path, "w", encoding="utf-8") as f:
                    f.write(tp_report)
        else:
            tp_report = tipping_report_for_map(states, c, ev_vec, tp_title)
            with open(tp_path, "w", encoding="utf-8") as f:
                f.write(tp_report)

        # Energy diagnostics for this centroid: compute components for delta = centroid - baseline
        try:
            delta_for_energy = c - m_baseline
            comps = energy_components(delta_for_energy, targets, w_target, m_baseline=m_baseline)
        except Exception:
            comps = {}

        # Load historical energies if available to compute ranges
        hist_df = None
        try:
            hist_path = os.path.join(out_root, "historical_energies.csv")
            if os.path.exists(hist_path):
                hist_df = pd.read_csv(hist_path, index_col=0)
            else:
                hist_path2 = os.path.join(os.getcwd(), OUT_DIR, "historical_energies.csv")
                if os.path.exists(hist_path2):
                    hist_df = pd.read_csv(hist_path2, index_col=0)
        except Exception:
            hist_df = None

        # Determine sample keys and ranges
        sample_keys = [k for k in comps.keys() if k.startswith("E_")]
        ranges = {}
        if hist_df is not None and not hist_df.empty:
            for k in sample_keys:
                if k in hist_df.columns:
                    lo = float(hist_df[k].quantile(0.05))
                    hi = float(hist_df[k].quantile(0.95))
                else:
                    vals = hist_df.get(k)
                    if vals is None:
                        lo, hi = (float("-inf"), float("inf"))
                    else:
                        lo = float(vals.min())
                        hi = float(vals.max())
                ranges[k] = (lo, hi)
        else:
            ranges = {k: (float("-inf"), float("inf")) for k in sample_keys}

        # Append energy diagnostics to txt_lines
        txt_lines.append("")
        txt_lines.append("Energy diagnostics (value vs historical 5%-95% range):")
        for k in sample_keys:
            val = comps.get(k, float("nan"))
            lo, hi = ranges.get(k, (float("-inf"), float("inf")))
            in_range = "IN" if (not np.isnan(val) and val >= lo and val <= hi) else "OUT"
            # format safely
            def fmt(x):
                try:
                    if np.isinf(x):
                        return str(x)
                    return f"{x:.6g}"
                except Exception:
                    return str(x)
            txt_lines.append(f"{k}: {fmt(val)}  hist[{fmt(lo)},{fmt(hi)}]  {in_range}")

        # Write the centroid TXT with diagnostics
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(txt_lines))
        except Exception:
            # ignore write errors (non-fatal)
            pass

        results.append({
            "centroid_index": i + 1,
            "csv": csv_path,
            "txt": txt_path,
            "tipping": tp_path,
            "D_EV@0": D0, "R_EV@0": R0,
            "pv_shift": pv_shift_this,
            "D_EV@nat": Dnat_D, "R_EV@nat": Dnat_R,
            "D_EV@+3": Dp3, "R_EV@+3": Rp3,
            "flips_vs_baseline": flips,
            "shock_count": int(len(shock_states)),
            "shock_states": shock_states,
        })
    # After finishing 'results' & having 'centroids' ready:
    # if example_margin is not None:
    #     write_yapms_color_tables(states, centroids, example_margin, out_root, year_label)
    # else:
    #     # Use per-centroid PV shifts for Yapms tables
    #     write_yapms_color_tables(states, centroids, None, out_root, year_label, col_offsets=pv_shift_by_cluster)
    return {"diagnostics": diag, "results": results, "labels": labels.tolist(), "centroids": centroids}

# ----------
# Main flow
# ----------
def main(
    csv_path: str = CSV_PATH,
    out_dir: str = OUT_DIR,
    seed: int = SEED,
    n_draw: int = N_DRAW,
    n_keep: int = N_KEEP,
    kmeans_k: int = KMEANS_K,
):
    os.makedirs(out_dir, exist_ok=True)

    # 1) Load CSV and align
    pivot_m, pivot_ev, ev_w, states, years, nat_series = load_and_align(csv_path)

    # 2) Targets from full history
    targets = build_delta_targets(pivot_m, ev_w, years, states, nat_series, tau=TAU, var_frac=VAR_FRAC, K_min=K_MIN)

    # Compute and save historical energies for the real-world maps
    try:
        compute_historical_energies(pivot_m, ev_w, states, targets, out_dir)
    except Exception as e:  # pragma: no cover
        print(f"compute_historical_energies failed: {e}")

    # Helper: baseline and weights (use 2024 values)
    if 2024 not in pivot_m.index:
        raise ValueError("CSV must include 2024 to define 2024 baseline.")
    m_2024 = pivot_m.loc[2024].reindex(states).to_numpy()
    ev_2024 = pivot_ev.loc[2024].reindex(states).to_numpy()
    w_2024 = ev_w.loc[2024].reindex(states).to_numpy()

    # Baseline national margin for 2024
    if 2024 not in targets.nat_by_year:
        raise ValueError("CSV must include national_margin for 2024.")
    nat_2024 = float(targets.nat_by_year[2024])

    # If user wants multi-year generation, run the generic driver which will
    # produce a `final_maps` folder containing chosen centroid CSVs/TXT and a manifest.
    def generate_multi_year_maps(
        start_year: int,
        end_year: int,
        step: int,
        baseline_margin: np.ndarray,
        ev_vec_local: np.ndarray,
        w_target_local: np.ndarray,
        states_local: List[str],
        targets_local: Targets,
        out_root_local: str,
        seed_local: int = SEED,
        n_draw_local: int = N_DRAW,
        n_keep_local: int = N_KEEP,
        kmeans_k_local: int = KMEANS_K,
        pick: str = "largest",
        use_medoids: bool = False,
    alpha_ev_margin_local: float = ALPHA_EV_MARGIN,
    alpha_ev_quad_local: float = ALPHA_EV_QUAD,
    alpha_hist_state_local: float = ALPHA_HIST_STATE,
    alpha_hist_ev_local: float = ALPHA_HIST_EV,
        nat_party: Optional[str] = None
    ) -> Dict:
        final_dir = os.path.join(out_root_local, "final_maps")
        os.makedirs(final_dir, exist_ok=True)
        manifest = {"generated": [], "start_year": int(start_year), "end_year": int(end_year), "step": int(step)}

        cur_baseline = baseline_margin.copy()
        cur_nat = float(targets_local.nat_by_year.get(start_year - step, 0.0)) if start_year - step in targets_local.nat_by_year else float(targets_local.nat_by_year.get(start_year, 0.0))
        cur_seed = int(seed_local)

        years_seq = list(range(start_year, end_year + 1, step))
        for yr in years_seq:
            ydir_local = os.path.join(out_root_local, str(yr))
            os.makedirs(ydir_local, exist_ok=True)
            out = run_sampler_for_year(
                year_label=yr,
                m_baseline=cur_baseline,
                ev_vec=ev_vec_local,
                w_target=w_target_local,
                states=states_local,
                targets=targets_local,
                out_root=ydir_local,
                seed=cur_seed,
                n_draw=n_draw_local,
                n_keep=n_keep_local,
                kmeans_k=kmeans_k_local,
                example_margin=None,
                nat_baseline=cur_nat,
                use_medoids=use_medoids,
                # alpha knobs intentionally omitted here; run_sampler_for_year
                # will use its defaults (set from top-level ALPHA_* constants)
                nat_party='D' if yr < 2036 else None
            )

            labels_arr = np.array(out["labels"]) if isinstance(out["labels"], list) else out["labels"]
            sizes = [int(np.sum(labels_arr == i)) for i in range(kmeans_k_local)]
            if pick == "largest":
                chosen_idx_local = int(np.argmax(sizes))
            else:
                # fallback to cluster 0
                chosen_idx_local = 0

            # chosen centroid and metadata
            chosen_centroid = out["centroids"][chosen_idx_local]
            pv_shifts = out["diagnostics"].get("pv_shift_by_cluster", [cur_nat] * kmeans_k_local)
            chosen_pv = float(pv_shifts[chosen_idx_local]) if pv_shifts else float(cur_nat)

            # Copy centroid CSV/TXT/TIPPING to final_maps with standardized names
            res = out["results"][chosen_idx_local]
            src_csv = res.get("csv")
            src_txt = res.get("txt")
            src_tp = res.get("tipping")
            dest_csv = os.path.join(final_dir, f"{yr}_chosen_centroid.csv")
            dest_txt = os.path.join(final_dir, f"{yr}_chosen_centroid.txt")
            dest_tp = os.path.join(final_dir, f"{yr}_chosen_tipping.txt")
            try:
                if src_csv and os.path.exists(src_csv):
                    shutil.copy(src_csv, dest_csv)
                if src_txt and os.path.exists(src_txt):
                    shutil.copy(src_txt, dest_txt)
                if src_tp and os.path.exists(src_tp):
                    shutil.copy(src_tp, dest_tp)
            except Exception:
                # non-fatal
                pass

            manifest_entry = {
                "year": int(yr),
                "chosen_centroid_index": int(chosen_idx_local + 1),
                "csv": dest_csv if os.path.exists(dest_csv) else src_csv,
                "txt": dest_txt if os.path.exists(dest_txt) else src_txt,
                "tipping": dest_tp if os.path.exists(dest_tp) else src_tp,
                "pv_shift": chosen_pv,
                "cluster_size": int(sizes[chosen_idx_local]),
            }
            manifest["generated"].append(manifest_entry)

            # prepare next iteration: use chosen centroid as baseline
            cur_baseline = np.array(chosen_centroid, dtype=float)
            cur_nat = chosen_pv
            cur_seed += 1

    # save manifest
        manifest_path = os.path.join(final_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, indent=2)

        # Attempt to render PNGs from the chosen centroid CSVs using the local run_batch.py
        try:
            script = os.path.join(os.path.dirname(__file__), "yapms-map-export", "run_batch.py")
            if os.path.exists(script):
                # run in inline mode to avoid starting a dev server
                subprocess.run([sys.executable, script, final_dir, "--inline"], check=False)
        except Exception as e:
            print(f"Rendering final_maps with yapms run_batch failed: {e}")

        return manifest

    # run default multi-year generation (2028->2032) and write final_maps
    try:
        manifest = generate_multi_year_maps(
            start_year=2028,
            end_year=2040,
            step=4,
            baseline_margin=m_2024,
            ev_vec_local=ev_2024,
            w_target_local=w_2024,
            states_local=states,
            targets_local=targets,
            out_root_local=out_dir,
            seed_local=seed,
            n_draw_local=n_draw,
            n_keep_local=n_keep,
            kmeans_k_local=kmeans_k
        )
        # write a short summary next to main outputs
        with open(os.path.join(out_dir, "final_maps_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    except Exception as e:
        print(f"generate_multi_year_maps failed: {e}")

    # Console prints (brief)
    print(f"Targets: K={targets.F.shape[0]}, sd_evd={targets.sd_evd:.4f}, mu_shock={targets.mu_shock:.4f}, sd_shock={targets.sd_shock:.4f}, resid_scale={targets.resid_scale:.4f}, nat_mu={targets.nat_mu:.4f}, nat_sd={targets.nat_sd:.4f}")
    # print(f"2028: clusters={summary['2028']['cluster_sizes']}, avg_shock_keep={summary['2028']['avg_shock_frac']:.3f}")
    # print(f"2032: clusters={summary['2032']['cluster_sizes']}, avg_shock_keep={summary['2032']['avg_shock_frac']:.3f}")


if __name__ == "__main__":
    # Simple CLI via environment variables is possible; for now just run defaults
    main()
    
    import subprocess
    # run ./yapms-map-export/run_batch.py
    #print("Running yapms-map-export batch script...")
    #subprocess.run(["python", "./yapms-map-export/run_batch.py"], check=True)

