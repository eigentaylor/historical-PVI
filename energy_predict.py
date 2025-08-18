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
import utils

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
NAT_CLIP = 0.12  # soft bounds for national margin delta proposals

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
    alpha_ev: float = 1.0,
    alpha_var: float = 1.0,
    alpha_shock: float = 1.0,
    alpha_resid: float = 1.0,
) -> float:
    """Compute E_TOTAL(d) with components as specified.

    Returns scalar energy.
    """
    # EV term
    evd = float(np.dot(w_target, d))
    z_ev = (evd - targets.mu_evd) / targets.sd_evd
    E_EV = (z_ev * z_ev) * alpha_ev

    # Variance term (weighted; downweight ME/NE districts)
    z = d / targets.state_sigma
    w = targets.state_weights
    den = float(np.sum(w)) if np.sum(w) > 0 else float(len(z))
    E_VAR = float(np.sum(w * (z * z)) / den) * alpha_var

    # Shock fraction term (weighted by state_weights)
    shock_mask = (np.abs(d) >= TAU).astype(float)
    frac = float(np.sum(w * shock_mask) / den)
    z_shock = (frac - targets.mu_shock) / targets.sd_shock
    E_SHOCK = 0 #(z_shock * z_shock) * alpha_shock
    # we are trying not to penalize shocks, as they happen actually quite often

    # Residual (orthogonal to factor subspace)
    v = d - targets.mean_Xc
    r = _proj_residual(v, targets.F)
    denom = math.sqrt(len(d)) * targets.resid_scale
    if denom <= 1e-9:
        denom = 1e-6
    E_RESID = float((np.linalg.norm(r) / denom) ** 2) * alpha_resid

    return E_EV + E_VAR + E_SHOCK + E_RESID


def energy_components(
    d: np.ndarray,
    targets: Targets,
    w_target: np.ndarray,
    alpha_ev: float = 1.0,
    alpha_var: float = 1.0,
    alpha_shock: float = 1.0,
    alpha_resid: float = 1.0,
) -> Dict[str, float]:
    """Return a dict of energy components for a delta vector d.

    This is numerically aligned with energy_delta for fully-observed vectors.
    For deltas containing NaNs (as can happen with historical 4-year pairs for
    states that didn't exist yet), computations are done in a NaN-aware manner:
      - EV term uses available-state weights renormalized to their own sum.
      - VAR term averages z^2 over available states only.
      - SHOCK term uses the fraction over available states only.
      - RESID term imputes missing entries with the column mean (so v = 0 there),
        and uses sqrt(n_avail) in the denominator.
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
    E_SHOCK = 0 #(z_shock * z_shock) * alpha_shock

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

    total = E_EV + E_VAR + E_SHOCK + E_RESID
    return {
        "E_EV": float(E_EV),
        "E_VAR": float(E_VAR),
        "E_SHOCK": float(E_SHOCK),
        "E_RESID": float(E_RESID),
        "E_TOTAL": float(total),
        "evd": float(evd),
        "shock_frac": float(frac),
        "n_avail": float(n_avail),
    }


# -----------------
# Sampling deltas
# -----------------
def sample_deltas(
    targets: Targets,
    w_target: np.ndarray,
    N_draw: int = N_DRAW,
    N_keep: int = N_KEEP,
    seed: int = SEED,
    clip: float = CLIP_D,
    idio_scale: Optional[float] = None,
    alpha_ev: float = 1.0,
    alpha_var: float = 1.0,
    alpha_shock: float = 1.0,
    alpha_resid: float = 1.0,
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

    # Score energies
    energies = np.array([
        energy_delta(proposals[i], targets, w_target, alpha_ev, alpha_var, alpha_shock, alpha_resid)
        for i in range(N_draw)
    ])

    # Keep N_keep lowest
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
            nat_by_year=targets.nat_by_year,
        )

        keep_d, keep_E, idx = sample_deltas(targets_sample, w_target_sample, N_draw=n_draw, N_keep=n_keep, seed=seed)
        maps_sample = maps_from_deltas(m_baseline_sample, keep_d)
        # Expand sample maps back to full-state vector by filling at-large entries with district averages
        maps = expand_maps_with_atlarge(maps_sample, sample_states, states, AT_LARGE_GROUPS)
    else:
        keep_d, keep_E, idx = sample_deltas(targets, w_target, N_draw=n_draw, N_keep=n_keep, seed=seed)
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
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(txt_lines))

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
    if example_margin is not None:
        write_yapms_color_tables(states, centroids, example_margin, out_root, year_label)
    else:
        # Use per-centroid PV shifts for Yapms tables
        write_yapms_color_tables(states, centroids, None, out_root, year_label, col_offsets=pv_shift_by_cluster)
    return {"diagnostics": diag, "results": results, "labels": labels.tolist(), "centroids": centroids}


# ---------------------------------------------
# Color tables for Yapms (constants + highlights)
# ---------------------------------------------
def _color_token(margin: float) -> str:
    """
    Map a final margin to color token used in TXT outputs:
    one of {'RED','Red','red','lred','BLUE','Blue','blue','lblue'}.
    Uses your existing utils.final_margin_color_key.
    """
    return utils.final_margin_color_key(margin)

def _build_colors_df(states, centroids, example_margin: Optional[float], col_names: Optional[List[str]] = None, col_offsets: Optional[List[float]] = None) -> "pd.DataFrame":
    """
    Rows = states, Cols = centroid_1..centroid_5, Values = color tokens at (m + example_margin).
    centroids: np.ndarray shape (5, S)
    """
    if col_names is None:
        cols = [f"centroid_{i+1}" for i in range(len(centroids))]
    else:
        # use provided column names (preserve original cluster labels if centroids reordered)
        cols = list(col_names)
    data = {s: {} for s in states}
    # Determine offsets per column
    if example_margin is not None:
        offsets = [float(example_margin)] * len(centroids)
    else:
        offsets = [0.0] * len(centroids) if not col_offsets else [float(x) for x in col_offsets]
    for j, c in enumerate(centroids):
        for s_idx, s in enumerate(states):
            fm = float(c[s_idx]) + float(offsets[j])
            data[s][cols[j]] = _color_token(fm)
    df = pd.DataFrame.from_dict(data, orient="index")
    # nice alphabetical row order
    df = df.loc[sorted(df.index)]
    return df

def _constant_states_df(colors_df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Return only states whose colors are identical across all centroids.
    Sorted/grouped by color family and shade.
    """
    same_mask = (colors_df.nunique(axis=1) == 1)
    const_df = colors_df.loc[same_mask].copy()
    if const_df.empty:
        return const_df

    # First column represents the constant color
    const_df["__color__"] = const_df.iloc[:, 0]

    # Sort by color in a deterministic family/shade order (Blue family first)
    order = {
        "BLUE":  0, "Blue":  1, "blue":  2, "lblue": 3,
        "RED":  10, "Red":  11, "red":  12, "lred":  13,
    }
    const_df["__rank__"] = const_df["__color__"].map(order).fillna(999)

    # Reset the index to a column so we can sort by rank then by the state label (works whether
    # the index has a name or not and avoids passing an Index object to sort_values).
    df_reset = const_df.reset_index()
    idx_col = df_reset.columns[0]
    df_reset = df_reset.sort_values(["__rank__", idx_col])
    df_reset = df_reset.set_index(idx_col)
    const_df = df_reset.drop(columns=["__rank__"])
    return const_df

def _highlight_changes_df(colors_df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Keep the first column; for subsequent centroids, show the color only if it differs
    from the previous centroid for that state. Otherwise leave blank.
    Includes ALL states that vary at least once. States that never vary are excluded.
    """
    # Identify varying rows first
    varying = colors_df.loc[colors_df.nunique(axis=1) > 1].copy()
    if varying.empty:
        return varying

    cols = list(varying.columns)
    out = varying.copy()
    # first column stays visible
    for r in out.index:
        prev = out.at[r, cols[0]]
        # subsequent columns only when different vs previous col
        for c in cols[1:]:
            cur = out.at[r, c]
            if cur == prev:
                out.at[r, c] = ""
            else:
                # keep the token and update prev
                prev = cur
    return out

def _write_markdown_and_csvs(year_label: int, out_root: str,
                             const_df: "pd.DataFrame",
                             colors_df: "pd.DataFrame",
                             highlight_df: "pd.DataFrame") -> None:
    """
    Save:
      - {year}_color_tables.md (markdown with constant groups + change-only highlight)
      - {year}_colors_by_centroid.csv (full table)
      - {year}_constant_colors.csv (state + constant colors)
      - {year}_color_changes_only.csv (change-only highlight table)
    """
    # CSVs
    colors_csv = os.path.join(out_root, f"{year_label}_colors_by_centroid.csv")
    colors_df.to_csv(colors_csv, index=True)

    const_csv = os.path.join(out_root, f"{year_label}_constant_colors.csv")
    if not const_df.empty:
        # compress constant table to a single 'color' column
        const_export = const_df.iloc[:, [0]].rename(columns={const_df.columns[0]: "color"})
        const_export.to_csv(const_csv, index=True)
    else:
        # write empty placeholder
        pd.DataFrame(columns=["color"]).to_csv(const_csv, index=True)

    highlight_csv = os.path.join(out_root, f"{year_label}_color_changes_only.csv")
    if not highlight_df.empty:
        highlight_df.to_csv(highlight_csv, index=True)
    else:
        pd.DataFrame().to_csv(highlight_csv, index=True)

    # Markdown
    md_lines = []
    md_lines.append(f"# {year_label} Color Tables\n")

    # Constant states grouped by color
    md_lines.append("## Constant-color states (same in all centroids)\n")
    if const_df.empty:
        md_lines.append("_None_\n")
    else:
        # Group sections by color value in first column
        # Ensure family/shade order
        family_order = ["BLUE", "Blue", "blue", "lblue", "RED", "Red", "red", "lred"]
        for tok in family_order:
            group = const_df[const_df.iloc[:, 0] == tok]
            if group.empty:
                continue
            md_lines.append(f"**{tok}**\n")
            md_lines.append(group.drop(columns=[]).to_markdown())
            md_lines.append("")

    # Change-only highlight (first column always included)
    md_lines.append("## Change-only highlight (blank = same as previous centroid)\n")
    if highlight_df.empty:
        md_lines.append("_No changes across centroids._\n")
    else:
        md_lines.append(highlight_df.to_markdown())
        md_lines.append("")

    # (Optional) full table at the end if you ever want it visible:
    md_lines.append("<details><summary>Full color table by centroid</summary>\n\n")
    md_lines.append(colors_df.to_markdown())
    md_lines.append("\n</details>\n")

    md_path = os.path.join(out_root, f"{year_label}_color_tables.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

def write_yapms_color_tables(states, centroids, example_margin, out_root, year_label, col_offsets: Optional[List[float]] = None) -> None:
    """
    Public helper: call this AFTER you've computed centroids for a year.
    Only writes tables if example_margin is not None.
    """
    if example_margin is None and not col_offsets:
        return

    # Reorder centroids to minimize the total number of highlighted changes across
    # states (i.e., minimize transitions between adjacent centroid columns).
    # k is typically small (default 5) so brute-force over permutations is fine.
    def _order_centroids_by_min_changes(states, centroids, example_margin):
        k = len(centroids)
        if k <= 1:
            return centroids, list(range(k))

        S = len(states)
        # tokens[r][j] = color token for state r under centroid j
        # use empty-string placeholders to keep types consistent
        tokens = [[""] * k for _ in range(S)]
        for j, c in enumerate(centroids):
            for s_idx in range(S):
                fm = float(c[s_idx]) + float(example_margin)
                tokens[s_idx][j] = _color_token(fm)
        best_cost = float("inf")
        best_perm = tuple(range(k))
        for perm in permutations(range(k)):
            cost = 0
            # count transitions between adjacent columns across all states
            for r in range(S):
                prev = tokens[r][perm[0]]
                for idx in perm[1:]:
                    cur = tokens[r][idx]
                    if cur != prev:
                        cost += 1
                        prev = cur
            if cost < best_cost:
                best_cost = cost
                best_perm = perm

        ordered = [centroids[i] for i in best_perm]
        return ordered, list(best_perm)

    centroids, chosen_order = _order_centroids_by_min_changes(states, centroids, example_margin if example_margin is not None else 0.0)
    # small debug trace written to stdout so users know the chosen ordering
    print(f"Reordered centroids to minimize changes: order={chosen_order}")
    # Build column names that preserve original cluster indices (1-based)
    orig_cols = [f"centroid_{i+1}" for i in chosen_order]
    # Reorder offsets to match centroid order
    offsets = None
    if example_margin is not None:
        offsets = [float(example_margin)] * len(centroids)
    elif col_offsets:
        offsets = [col_offsets[i] for i in chosen_order]
    colors_df = _build_colors_df(states, centroids, example_margin, col_names=orig_cols, col_offsets=offsets)
    const_df = _constant_states_df(colors_df)
    # Only varying rows appear in highlight; first column always included
    highlight_df = _highlight_changes_df(colors_df)
    _write_markdown_and_csvs(year_label, out_root, const_df, colors_df, highlight_df)

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

    # 3) YEAR = 2028
    year = 2028
    ydir = os.path.join(out_dir, str(year))
    os.makedirs(ydir, exist_ok=True)
    out_2028 = run_sampler_for_year(
        year_label=year,
        m_baseline=m_2024,
        ev_vec=ev_2024,
        w_target=w_2024,
        states=states,
        targets=targets,
        out_root=ydir,
        seed=seed,
        n_draw=n_draw,
        n_keep=n_keep,
        kmeans_k=kmeans_k,
        example_margin=None,
        nat_party="D",
        nat_baseline=nat_2024,
    )

    # Choose a 2028 centroid: largest cluster
    labels_2028 = np.array(out_2028["labels"]) if isinstance(out_2028["labels"], list) else out_2028["labels"]
    sizes = [np.sum(labels_2028 == i) for i in range(kmeans_k)]
    chosen_idx = int(np.argmax(sizes))
    m_2028 = out_2028["centroids"][chosen_idx]
    # Choose national margin for 2028 from the same centroid
    pv_shifts_2028 = out_2028["diagnostics"].get("pv_shift_by_cluster", [nat_2024] * kmeans_k)
    nat_2028 = float(pv_shifts_2028[chosen_idx])

    # 4) YEAR = 2032 (iterate using chosen 2028 centroid as baseline)
    year = 2032
    ydir = os.path.join(out_dir, str(year))
    os.makedirs(ydir, exist_ok=True)
    
    # create a little txt file in the 2032 folder which says which centroid we used for 2028
    with open(os.path.join(ydir, "centroid_2028.txt"), "w", encoding="utf-8") as f:
        f.write(f"Chosen centroid for 2028: {chosen_idx + 1}\n")

    out_2032 = run_sampler_for_year(
        year_label=year,
        m_baseline=m_2028,
        ev_vec=ev_2024,  # use 2024 EVs unless 2030s apportionment provided
        w_target=w_2024,
        states=states,
        targets=targets,  # stationary targets assumption
        out_root=ydir,
        seed=seed + 1,  # slight change
        n_draw=n_draw,
        n_keep=n_keep,
        kmeans_k=kmeans_k,
        example_margin=None,
        nat_baseline=nat_2028,
    )

    # Summary log
    summary = {
        "2028": out_2028["diagnostics"],
        "2028_results": out_2028["results"],
        "chosen_centroid_index_2028": int(chosen_idx + 1),
        "chosen_pv_shift_2028": float(nat_2028),
        "2032": out_2032["diagnostics"],
        "2032_results": out_2032["results"],
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Console prints (brief)
    print(f"Targets: K={targets.F.shape[0]}, sd_evd={targets.sd_evd:.4f}, mu_shock={targets.mu_shock:.4f}, sd_shock={targets.sd_shock:.4f}, resid_scale={targets.resid_scale:.4f}, nat_mu={targets.nat_mu:.4f}, nat_sd={targets.nat_sd:.4f}")
    print(f"2028: clusters={summary['2028']['cluster_sizes']}, avg_shock_keep={summary['2028']['avg_shock_frac']:.3f}")
    print(f"2032: clusters={summary['2032']['cluster_sizes']}, avg_shock_keep={summary['2032']['avg_shock_frac']:.3f}")


if __name__ == "__main__":
    # Simple CLI via environment variables is possible; for now just run defaults
    main()
    
    # NOTE: put the following prompts into powershell
