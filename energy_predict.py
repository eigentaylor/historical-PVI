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


@dataclass
class Targets:
    states: List[str]
    years: List[int]
    state_sigma: np.ndarray  # (S,)
    mu_evd: float
    sd_evd: float
    mu_shock: float
    sd_shock: float
    F: np.ndarray  # (K, S) orthonormal rows
    score_sd: np.ndarray  # (K,)
    mean_Xc: np.ndarray  # (S,)
    resid_scale: float
    idio_scale: float


# ----------------------
# Data loading and prep
# ----------------------
def load_and_align(csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[int]]:
    """Load CSV; return (pivot_m, pivot_ev, ev_w, states, years).

    Requires columns: year, abbr, relative_margin, electoral_votes
    Restricts to states that appear in all years present in the file.
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

    years = sorted(int(y) for y in pivot_m.index if not pd.isna(y))
    # Restrict to states present in all years
    valid_states = [s for s in pivot_m.columns if pivot_m[s].notna().all() and pivot_ev[s].notna().all()]
    pivot_m = pivot_m[valid_states].sort_index()
    pivot_ev = pivot_ev[valid_states].sort_index()

    # EV weights per year
    ev_tot = pivot_ev.sum(axis=1)
    ev_w = pivot_ev.div(ev_tot, axis=0)

    return pivot_m, pivot_ev, ev_w, valid_states, years


# ------------------------------------
# Build historical delta-based targets
# ------------------------------------
def build_delta_targets(
    pivot_m: pd.DataFrame,
    ev_w: pd.DataFrame,
    years: List[int],
    states: List[str],
    tau: float = TAU,
    var_frac: float = VAR_FRAC,
    K_min: int = K_MIN,
) -> Targets:
    """Compute deltas and all stats needed for energy scoring.

    Returns Targets with:
      - mu_evd, sd_evd: EV-weighted delta stats across cycles
      - state_sigma: per-state std of deltas
      - mu_shock, sd_shock: avg fraction of |Δ| >= tau per cycle
      - F (K x S): factor directions (orthonormal rows), score_sd (K,),
      - mean_Xc (S,), resid_scale (float), idio_scale (float)
    """
    # Build 4-year deltas for all feasible cycles y where y-4 exists
    y_pairs = [(y, y - 4) for y in years if (y - 4) in years]
    if not y_pairs:
        raise ValueError("Not enough years to compute deltas (need 4-year pairs).")

    deltas_list: List[np.ndarray] = []
    w_list: List[np.ndarray] = []
    for y, y0 in y_pairs:
        d = (pivot_m.loc[y].reindex(states).to_numpy() - pivot_m.loc[y0].reindex(states).to_numpy())
        w = ev_w.loc[y].reindex(states).to_numpy()
        deltas_list.append(d)
        w_list.append(w)

    X = np.vstack(deltas_list)  # (T, S)
    W = np.vstack(w_list)       # (T, S)
    T, S = X.shape

    # EV-weighted delta distribution
    evd = np.sum(W * X, axis=1)
    mu_evd = float(np.mean(evd))
    sd_evd = float(np.std(evd, ddof=1))
    if sd_evd <= 1e-8:
        sd_evd = 1e-6

    # Per-state volatility
    state_sigma = np.std(X, axis=0, ddof=1)
    state_sigma = np.maximum(state_sigma, 1e-4)  # floor to avoid div by ~0

    # Shock fraction stats
    frac_shock = (np.abs(X) >= tau).mean(axis=1)
    mu_shock = float(np.mean(frac_shock))
    sd_shock = float(np.std(frac_shock, ddof=1))
    if sd_shock <= 1e-8:
        sd_shock = 1e-6

    # Factor subspace via SVD on column-centered deltas
    mean_Xc = np.mean(X, axis=0)
    Xc = X - mean_Xc
    U, Svals, Vt = np.linalg.svd(Xc, full_matrices=False)
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
        resid_mean_sq = float(np.sum(Svals[K:]**2) / (Xc.shape[0] * Xc.shape[1]))
    else:
        resid_mean_sq = 1e-6
    resid_scale = float(np.sqrt(max(resid_mean_sq, 1e-12)))

    # Idiosyncratic scale from median |delta|
    idio_scale = 0.5 * float(np.median(np.abs(X)))
    if idio_scale <= 1e-8:
        idio_scale = 0.02

    return Targets(
        states=states,
        years=years,
        state_sigma=state_sigma,
        mu_evd=mu_evd,
        sd_evd=sd_evd,
        mu_shock=mu_shock,
        sd_shock=sd_shock,
        F=F,
        score_sd=score_sd,
        mean_Xc=mean_Xc,
        resid_scale=resid_scale,
        idio_scale=idio_scale,
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

    # Variance term
    z = d / targets.state_sigma
    E_VAR = float(np.mean(z * z)) * alpha_var

    # Shock fraction term
    frac = float((np.abs(d) >= TAU).mean())
    z_shock = (frac - targets.mu_shock) / targets.sd_shock
    E_SHOCK = (z_shock * z_shock) * alpha_shock

    # Residual (orthogonal to factor subspace)
    v = d - targets.mean_Xc
    r = _proj_residual(v, targets.F)
    denom = math.sqrt(len(d)) * targets.resid_scale
    if denom <= 1e-9:
        denom = 1e-6
    E_RESID = float((np.linalg.norm(r) / denom) ** 2) * alpha_resid

    return E_EV + E_VAR + E_SHOCK + E_RESID


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
) -> Tuple[np.ndarray, np.ndarray]:
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
    return keep, keep_E


def maps_from_deltas(m_baseline: np.ndarray, keep_d: np.ndarray) -> np.ndarray:
    return m_baseline[None, :] + keep_d


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
                lines.append(f"  {state_emoji}{abbr}: {utils.lean_str(rm)} ({ev} EV)")
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
    example_margin: Optional[float] = None
) -> Dict:
    os.makedirs(out_root, exist_ok=True)

    keep_d, keep_E = sample_deltas(targets, w_target, N_draw=n_draw, N_keep=n_keep, seed=seed)
    maps = maps_from_deltas(m_baseline, keep_d)

    # Diagnostics: energy percentiles and shock fraction of kept
    pct = np.percentile(keep_E, [5, 25, 50, 75, 95])
    shock_fracs = (np.abs(keep_d) >= TAU).mean(axis=1)
    shock_avg = float(np.mean(shock_fracs))

    labels, centroids = cluster_maps(maps, k=kmeans_k, seed=seed)
    cluster_sizes = [int(np.sum(labels == i)) for i in range(kmeans_k)]
    mean_E_by_cluster = [float(np.mean(keep_E[labels == i])) if np.any(labels == i) else float("nan") for i in range(kmeans_k)]

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
    }
    with open(os.path.join(out_root, f"{year_label}_diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2)

    # Per-centroid outputs
    results = []
    for i, c in enumerate(centroids):
        # Sorted CSV and TXT
        pairs = list(zip(states, c.tolist(), ev_vec.tolist()))
        pairs.sort(key=lambda x: x[1])  # ascending by margin (R→D)

        # CSV
        df_c = pd.DataFrame(pairs, columns=["abbr", "relative_margin", "electoral_votes"])
        csv_path = os.path.join(out_root, f"{year_label}_centroid_{i+1}.csv")
        df_c.to_csv(csv_path, index=False)

        # TXT lines
        txt_lines = [f"Centroid {i+1} for {year_label}", "abbr,relative_margin"]
        # If example_margin provided, append an example final-margin string per state
        for a, m, e in pairs:
            if example_margin is None:
                example_str = ""
            else:
                fm = m + example_margin
                example_str = f",\tfinal ({utils.lean_str(example_margin)}):\t{utils.emoji_from_lean(fm)} {utils.lean_str(fm)}\t{utils.final_margin_color_key(fm)}"
            txt_lines.append(f"{utils.emoji_from_lean(m, use_swing=True)}{a},\t{utils.lean_str(m)},\t{e}{example_str}")

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
                txt_lines.append(f"{swing_emoji}{abbr},\t{utils.lean_str(dval)},\t{utils.lean_str(base)} → {utils.lean_str(pred)}")
        txt_path = os.path.join(out_root, f"{year_label}_centroid_{i+1}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(txt_lines))

        # EV counts under PV shifts
        D0, R0 = ev_counts_for_map(c, ev_vec, pv_shift=0.0)
        Dp3, Rp3 = ev_counts_for_map(c, ev_vec, pv_shift=0.03)

        # Flips relative to baseline and shock metadata
        flips = int(np.sum(np.sign(c) != np.sign(m_baseline)))
        shock_states = [states[idx] for idx in shock_idx]

        # Tipping point log — prefer centralized historical_tipping_points saver when available
        tp_title = f"Tipping points for {year_label} centroid {i+1}"
        tp_path = os.path.join(out_root, f"{year_label}_centroid_{i+1}_tipping.txt")
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
            "D_EV@+3": Dp3, "R_EV@+3": Rp3,
            "flips_vs_baseline": flips,
            "shock_count": int(len(shock_states)),
            "shock_states": shock_states,
        })
    # After finishing 'results' & having 'centroids' ready:
    if example_margin is not None:
        write_yapms_color_tables(states, centroids, example_margin, out_root, year_label)
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

def _build_colors_df(states, centroids, example_margin: float, col_names: Optional[List[str]] = None) -> "pd.DataFrame":
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
    for j, c in enumerate(centroids):
        for s_idx, s in enumerate(states):
            fm = float(c[s_idx]) + float(example_margin)
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

def write_yapms_color_tables(states, centroids, example_margin, out_root, year_label) -> None:
    """
    Public helper: call this AFTER you've computed centroids for a year.
    Only writes tables if example_margin is not None.
    """
    if example_margin is None:
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

    centroids, chosen_order = _order_centroids_by_min_changes(states, centroids, example_margin)
    # small debug trace written to stdout so users know the chosen ordering
    print(f"Reordered centroids to minimize changes: order={chosen_order}")
    # Build column names that preserve original cluster indices (1-based)
    orig_cols = [f"centroid_{i+1}" for i in chosen_order]
    colors_df = _build_colors_df(states, centroids, example_margin, col_names=orig_cols)
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
    pivot_m, pivot_ev, ev_w, states, years = load_and_align(csv_path)

    # 2) Targets from full history
    targets = build_delta_targets(pivot_m, ev_w, years, states, tau=TAU, var_frac=VAR_FRAC, K_min=K_MIN)

    # Helper: baseline and weights (use 2024 values)
    if 2024 not in pivot_m.index:
        raise ValueError("CSV must include 2024 to define 2024 baseline.")
    m_2024 = pivot_m.loc[2024].reindex(states).to_numpy()
    ev_2024 = pivot_ev.loc[2024].reindex(states).to_numpy()
    w_2024 = ev_w.loc[2024].reindex(states).to_numpy()

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
        example_margin=0.03
    )

    # Choose a 2028 centroid: largest cluster
    labels_2028 = np.array(out_2028["labels"]) if isinstance(out_2028["labels"], list) else out_2028["labels"]
    sizes = [np.sum(labels_2028 == i) for i in range(kmeans_k)]
    chosen_idx = int(np.argmax(sizes))
    m_2028 = out_2028["centroids"][chosen_idx]

    # 4) YEAR = 2032 (iterate using chosen 2028 centroid as baseline)
    year = 2032
    ydir = os.path.join(out_dir, str(year))
    os.makedirs(ydir, exist_ok=True)
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
        example_margin=0.07
    )

    # Summary log
    summary = {
        "2028": out_2028["diagnostics"],
        "2028_results": out_2028["results"],
        "chosen_centroid_index_2028": int(chosen_idx + 1),
        "2032": out_2032["diagnostics"],
        "2032_results": out_2032["results"],
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Console prints (brief)
    print(f"Targets: K={targets.F.shape[0]}, sd_evd={targets.sd_evd:.4f}, mu_shock={targets.mu_shock:.4f}, sd_shock={targets.sd_shock:.4f}, resid_scale={targets.resid_scale:.4f}")
    print(f"2028: clusters={summary['2028']['cluster_sizes']}, avg_shock_keep={summary['2028']['avg_shock_frac']:.3f}")
    print(f"2032: clusters={summary['2032']['cluster_sizes']}, avg_shock_keep={summary['2032']['avg_shock_frac']:.3f}")


if __name__ == "__main__":
    # Simple CLI via environment variables is possible; for now just run defaults
    main()
