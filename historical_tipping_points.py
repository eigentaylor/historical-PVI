import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import utils

# Try to read SWING_MARGIN from params if available; otherwise default to 0.03
try:
    import params
    SWING_MARGIN = float(getattr(params, 'SWING_MARGIN', 0.03))
except Exception:
    SWING_MARGIN = 0.03

# Toggle for including future projections (>2024) in addition to historical
USE_FUTURE = False
HIST_CSV_PATH = 'presidential_margins.csv'
FUTURE_CSV_PATH = 'presidential_margins_future.csv'
CSV_PATH = HIST_CSV_PATH
OUTPUT_DIR = 'tipping_points'
if USE_FUTURE:
    #OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'future')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
# clear all files and subdirectories in the output directory
for root, dirs, files in os.walk(OUTPUT_DIR, topdown=False):
    for name in files:
        os.remove(os.path.join(root, name))
    for name in dirs:
        os.rmdir(os.path.join(root, name))

# Outcome thresholds (total EVs for a party)
TARGET_EVS = {
    #538: 'all',
    500: 'sweep',
    400: 'blowout',
    350: 'landslide',
    300: 'solid',
    270: 'squeak',
}

ORDERED_KEYS = [
    'R_sweep', 'R_blowout', 'R_landslide', 'R_solid', 'R_squeak',
    'D_squeak', 'D_solid', 'D_landslide', 'D_blowout', 'D_sweep',
]


def margin_to_str(m: float) -> str:
    """Convert numeric national margin to 'D+X.Y' or 'R+X.Y' format (percentage points)."""
    return utils.lean_str(m)  # Assuming utils.lean_str handles the formatting correctly


def rel_margin_to_str(m: float) -> str:
    return utils.lean_str(m)  # Assuming utils.lean_str handles the formatting correctly


def read_presidential_margins(csv_path: str) -> Dict[int, List[Dict]]:
    by_year: Dict[int, List[Dict]] = defaultdict(list)
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                year = int(row['year'])
                if year >= 2024:
                    pass
                abbr = row['abbr']
                evs = int(float(row['electoral_votes']))
                pres_margin = float(row.get('pres_margin', '0') or 0.0)
                rel_margin = float(row['relative_margin'])  # D positive, R negative
                nat_margin = float(row.get('national_margin', '0') or 0.0)
            except (KeyError, ValueError):
                # Skip malformed rows
                continue
            by_year[year].append({
                'abbr': abbr,
                'evs': evs,
                'pres_margin': pres_margin,
                'relative_margin': rel_margin,
                'national_margin': nat_margin,
            })
    return by_year


def read_data(use_future: bool) -> Dict[int, List[Dict]]:
    """Read historical data and optionally append future data (years > 2024)."""
    by_year = read_presidential_margins(HIST_CSV_PATH) if not use_future else read_presidential_margins(FUTURE_CSV_PATH)
    # if use_future and os.path.exists(FUTURE_CSV_PATH):
    #     future = read_presidential_margins(FUTURE_CSV_PATH)
    #     for year, states in future.items():
    #         if year > 2024:
    #             by_year[year].extend(states)
    return by_year


def compute_threshold_tipping_points(states: List[Dict]) -> Tuple[Dict[str, Dict], List[Tuple[str, int, float]]]:
    """Compute tipping points for thresholds using relative margins ordering.

    Returns:
        tipping_points: dict keyed by 'R_sweep', 'D_solid', etc. with info
        ordering: list of (abbr, evs, relative_margin) sorted R->D
    """
    # Sort by deviation (relative margin), most R (most negative) to most D (most positive)
    ordered = sorted(((s['abbr'], s['evs'], s['relative_margin']) for s in states), key=lambda x: x[2])

    total_evs = sum(s['evs'] for s in states)
    assert total_evs == 538, f"Total EVs should be 538, got {total_evs}"
    current_r_evs = 0
    current_d_evs = total_evs

    tipping_points: Dict[str, Dict] = {}
    
    

    # Iterate across states from R->D, awarding EVs to R as national margin shifts R-ward
    for state, evs, deviation in ordered:
        current_margin = -deviation  # Required national margin to make this state 50/50
        if 'D_all' not in tipping_points:
            tipping_points['D_all'] = {
                'margin': current_margin,
                'state': state,
                'evs': evs,
                'D_evs_after': current_d_evs,
                'R_evs_after': current_r_evs,
            }
        # Check both R and D thresholds at the point of this pivot
        for goal, outcome_type in sorted(TARGET_EVS.items(), key=lambda x: x[0], reverse=True):
            r_key = f"R_{outcome_type}"
            d_key = f"D_{outcome_type}"
            # If adding this state to R crosses the R goal
            if current_r_evs + evs >= goal and r_key not in tipping_points:
                tipping_points[r_key] = {
                    'margin': current_margin,
                    'state': state,
                    'evs': evs,
                    'D_evs_after': current_d_evs - evs,
                    'R_evs_after': current_r_evs + evs,
                }
            # If D would fall below the D goal after R takes this state
            if current_d_evs - evs < goal and d_key not in tipping_points:
                tipping_points[d_key] = {
                    'margin': current_margin,
                    'state': state,
                    'evs': evs,
                    'D_evs_after': current_d_evs,
                    'R_evs_after': current_r_evs,
                }
        current_d_evs -= evs
        current_r_evs += evs

    tipping_points['R_all'] = {
        'margin': current_margin,
        'state': state,
        'evs': evs,
        'D_evs_after': current_d_evs,
        'R_evs_after': current_r_evs,
    }
    return tipping_points, ordered


def compute_actual_tipping_point(states: List[Dict]) -> Dict:
    """Compute the actual election tipping point for the winner using presidential margins."""
    total_evs = sum(s['evs'] for s in states)
    r_evs = sum(s['evs'] for s in states if s['pres_margin'] < 0)
    d_evs = total_evs - r_evs

    if r_evs >= 270:
        winner = 'R'
        # Order from most R to least R (ascending pres_margin)
        ordered = sorted((s for s in states), key=lambda x: x['pres_margin'])
        cum = 0
        tipping_state = None
        for s in ordered:
            if s['pres_margin'] < 0:  # R-won state contributes to R total
                cum += s['evs']
                if cum >= 270:
                    tipping_state = s
                    break
        if tipping_state is None:
            # Fallback: last R state in order (should not happen with valid data)
            tipping_state = next((s for s in reversed(ordered) if s['pres_margin'] < 0), ordered[-1])
        d_after = total_evs - cum
        r_after = cum
        margin = tipping_state['pres_margin']
    else:
        winner = 'D'
        # Order from most D to least D (descending pres_margin)
        ordered = sorted((s for s in states), key=lambda x: x['pres_margin'], reverse=True)
        cum = 0
        tipping_state = None
        for s in ordered:
            if s['pres_margin'] > 0:  # D-won state contributes to D total
                cum += s['evs']
                if cum >= 270:
                    tipping_state = s
                    break
        if tipping_state is None:
            # Fallback: last D state in order
            tipping_state = next((s for s in reversed(ordered) if s['pres_margin'] > 0), ordered[-1])
        d_after = cum
        r_after = total_evs - cum
        margin = tipping_state['pres_margin']

    return {
        'winner': winner,
        'state': tipping_state['abbr'],
        'evs': tipping_state['evs'],
        'margin': margin,
        'D_evs_after': d_after,
        'R_evs_after': r_after,
    }


# --- New helpers for EV/PV mismatch and actual-at-national-margin ---

def compute_ec_totals_at_margin(states: List[Dict], national_margin: float) -> Tuple[int, int]:
    """Return (D_evs, R_evs) using state outcome at given national margin m where
    state_margin = national_margin + relative_margin.
    """
    d_evs = 0
    r_evs = 0
    for s in states:
        sm = national_margin + s['relative_margin']
        if sm > 0:
            d_evs += s['evs']
        else:
            r_evs += s['evs']
    return d_evs, r_evs


def find_closest_state_for_margin(states: List[Dict], national_margin: float) -> Tuple[str, float, int]:
    """Find the closest state on the winning PV side at the given national margin.
    If D wins PV (m > 0): pick the blue state with the smallest positive state margin.
    If R wins PV (m < 0): pick the red state with the smallest absolute negative state margin.
    If m == 0: we pick the logic based on which party has more electoral votes (ex. if D has more EVs, pick the blue state with the smallest positive margin).
    Returns (abbr, state_margin, evs).
    """
    best = None
    if national_margin > 0 or national_margin == 0 and sum(s['evs'] for s in states if s['relative_margin'] > 0) > sum(s['evs'] for s in states if s['relative_margin'] < 0):
        # Closest blue state: minimal positive state margin
        for s in states:
            sm = national_margin + s['relative_margin']
            if sm > 0:
                key = sm  # minimize
                if best is None or key < best[1]:
                    best = (s['abbr'], sm, s['evs'])
    else:
        # Closest red state: maximal (least negative) state margin
        for s in states:
            sm = national_margin + s['relative_margin']
            if sm <= 0:
                key = sm  # minimize absolute negativity by maximizing sm
                if best is None or key > best[1]:
                    best = (s['abbr'], sm, s['evs'])
        if best is None:
            # Fallback: pick the least positive (if all states are blue at this m, unlikely)
            for s in states:
                sm = national_margin + s['relative_margin']
                if best is None or abs(sm) < abs(best[1]):
                    best = (s['abbr'], sm, s['evs'])
    # As a final guard
    if best is None:
        any_state = states[0]
        return any_state['abbr'], national_margin + any_state['relative_margin'], any_state['evs']
    return best


def find_state_for_national_margin(states: List[Dict], national_margin: float) -> Tuple[str, float, int]:
    """Compatibility wrapper naming for external callers: returns (abbr, state_margin, evs)

    Uses the same logic as find_closest_state_for_margin so callers can import this
    function and rely on stable behavior.
    """
    return find_closest_state_for_margin(states, national_margin)


def compute_ev_pv_mismatch_ranges(states: List[Dict]) -> Dict[str, Tuple[float, float]]:
    """Compute ranges of national margin where EC and PV winners differ.

    Returns a dict possibly containing keys:
        'D_PV_R_EC': (low, high) for m in (low, high) where D wins PV, R wins EC
        'R_PV_D_EC': (low, high) for m in (low, high) where R wins PV, D wins EC
    """
    rel_evs = [(s['relative_margin'], s['evs']) for s in states]
    total_evs = sum(e for _, e in rel_evs)

    # R EVs at m = 0
    r0 = sum(e for r, e in rel_evs if r < 0)

    # Positive side: D wins PV (m > 0) but R wins EC (R >= 270)
    pos_range = None
    if r0 >= 270:
        R = r0
        # Pivots where rel < 0 (these flip from R to D as m increases past p=-rel > 0)
        for p, ev in sorted(((-r, e) for r, e in rel_evs if r < 0)):
            if p <= 0:
                continue
            if R >= 270 and R - ev < 270:
                pos_range = (0.0, p)
                break
            R -= ev

    # Negative side: R wins PV (m < 0) but D wins EC (D >= 270 => R <= 268)
    neg_range = None
    if r0 <= total_evs - 270:  # R <= 268
        R = r0
        # Pivots where rel > 0 (these flip from D to R as m decreases past p=-rel < 0)
        for p, ev in sorted(((-r, e) for r, e in rel_evs if r > 0), reverse=True):
            if p >= 0:
                continue
            if R <= total_evs - 270 and R + ev > total_evs - 270:
                neg_range = (p, 0.0)
                break
            R += ev

    out: Dict[str, Tuple[float, float]] = {}
    if pos_range is not None:
        out['D_PV_R_EC'] = pos_range
    if neg_range is not None:
        out['R_PV_D_EC'] = neg_range
    return out


# --- New: Detect potential EC tie (269-269) ranges ---

def compute_ec_tie_ranges(states: List[Dict]) -> List[Dict]:
    """Return a list of tie ranges where the EC splits 269-269.

    Each item: {
        'start': float,         # national margin where tie starts (inclusive)
        'end': Optional[float], # national margin where tie ends (exclusive); None if open-ended
        'state': str,           # state whose flip creates the tie at 'start'
    }

    We model national margin decreasing from very D+ toward very R- and track
    when R's EV total first equals 269 after a flip. Between that pivot and the
    next pivot, the EVs remain at 269-269.
    """
    # Order states by relative margin, most R (most negative) to most D (most positive)
    ordered = sorted(((s['abbr'], s['evs'], s['relative_margin']) for s in states), key=lambda x: x[2])

    ties: List[Dict] = []
    current_r = 0
    n = len(ordered)

    for i, (abbr, evs, rel) in enumerate(ordered):
        pivot = -rel
        next_r = current_r + evs
        if next_r == 269:
            # Tie begins at this pivot (inclusive) and lasts until the next pivot
            next_pivot = -ordered[i + 1][2] if i + 1 < n else None
            ties.append({
                'start': float(pivot),
                'end': (float(next_pivot) if next_pivot is not None else None),
                'state': abbr,
            })
        current_r = next_r

    return ties


def ensure_output_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def build_report(year: int, tipping_points: Dict[str, Dict], ordered_rel: List[Tuple[str, int, float]], actual_info: Dict, mismatch_ranges: Dict[str, Tuple[float, float]], tie_ranges: List[Dict], extra_margins: Optional[List[float]] = None, source: str = 'presidential_margins.csv') -> str:
    lines: List[str] = []
    lines.append(f"Year {year} Tipping Point Analysis")
    lines.append("")
    lines.append("Tipping Points by National Margin:")

    # Prepare a unified list of tipping entries (thresholds + Actual + extra margins)
    entries: List[Tuple[float, str]] = []
    # Add threshold entries from tipping_points for both R and D keys
    threshold_keys = ['R_all', 'R_sweep', 'R_blowout', 'R_landslide', 'R_solid', 'R_squeak',
                      'D_squeak', 'D_solid', 'D_landslide', 'D_blowout', 'D_sweep', 'D_all']
    for key in threshold_keys:
        tp = tipping_points.get(key)
        if not tp:
            continue
        line = f"  {key}: {margin_to_str(tp['margin'])} via {tp['state']} (D: {tp['D_evs_after']} EVs, R: {tp['R_evs_after']} EVs)"
        entries.append((float(tp['margin']), line))

    # Actual national margin entry
    am = float(actual_info.get('margin', 0.0))
    actual_line = f"  Actual: {margin_to_str(am)} via {actual_info['state']} (D: {actual_info['D_evs']} EVs, R: {actual_info['R_evs']} EVs)"
    entries.append((am, actual_line))

    # Add extra margins as entries (use same formatting style)
    if extra_margins:
        # prepare inline states list from ordered_rel
        states_list = [{'abbr': a, 'evs': e, 'relative_margin': rm} for a, e, rm in ordered_rel]
        seen_m = set()
        for m in extra_margins:
            if m in seen_m:
                continue
            seen_m.add(m)
            d_evs, r_evs = compute_ec_totals_at_margin(states_list, m)
            st_abbr, st_margin, st_evs = find_closest_state_for_margin(states_list, m)
            line = f"  {margin_to_str(m)}: D {d_evs} EV, R {r_evs} EV via {st_abbr} ({rel_margin_to_str(st_margin)})"
            entries.append((float(m), line))

    # Sort all entries by numeric margin (R-most negative first -> D-most positive last)
    entries.sort(key=lambda x: x[0])
    # Emit entries in order
    for _, text in entries:
        lines.append(text)

    # Insert extra margins into the tipping list itself (use same state dict shape as other helpers)
    if extra_margins:
        # prepare states list from ordered_rel tuples
        states_list = [{'abbr': a, 'evs': e, 'relative_margin': rm} for a, e, rm in ordered_rel]
        # Unique margins preserving order
        seen_m = set()
        uniq_m = []
        for m in extra_margins:
            if m not in seen_m:
                seen_m.add(m)
                uniq_m.append(m)
        # Add as additional tipping entries
        lines.append("")
        lines.append("  Selected national margins:")
        for m in uniq_m:
            d_evs, r_evs = compute_ec_totals_at_margin(states_list, m)
            st_abbr, st_margin, st_evs = find_closest_state_for_margin(states_list, m)
            lines.append(f"    {margin_to_str(m)}: D {d_evs} EV, R {r_evs} EV via {st_abbr} ({rel_margin_to_str(st_margin)})")

    # EV/PV mismatch ranges
    if mismatch_ranges:
        for k, (lo, hi) in mismatch_ranges.items():
            if k == 'D_PV_R_EC':
                desc = 'D wins PV, R wins EC'
            else:
                desc = 'R wins PV, D wins EC'
            lines.append("")
            lines.append(f"EV/PV Mismatch Range: ({margin_to_str(lo)},{margin_to_str(hi)}) ({desc})")

    # Potential EC tie ranges
    if tie_ranges:
        lines.append("")
        lines.append("Potential EC Tie (269-269):")
        for tr in tie_ranges:
            start = tr['start']
            end = tr.get('end')
            state = tr['state']
            if end is not None:
                lines.append(f"  From {margin_to_str(start)} to {margin_to_str(end)} (via {state}) (D: 269 EVs, R: 269 EVs)")
            else:
                lines.append(f"  At {margin_to_str(start)} (via {state}) (D: 269 EVs, R: 269 EVs)")

    lines.append("")
    lines.append("States sorted by relative margins (R to D):")
    for abbr, evs, rm in ordered_rel:
        state_emoji = utils.emoji_from_lean(rm, use_swing=True)
        lines.append(f"  {state_emoji} {abbr}: {rel_margin_to_str(rm)} ({evs} EV)")
    lines.append("")
    lines.append(f"Source: {source}")

    return "\n".join(lines)


def process_year(year: int, states: List[Dict], extra_margins: Optional[List[float]] = None, use_future: bool = False) -> str:
    """Compatibility wrapper: build and save a tipping report for a single year.

    This now delegates to the generic saver `save_tipping_report` so other modules
    can call that function directly with a list of state dicts and a chosen path.
    """
    fname = f"{year}_tipping_points.txt"
    out_dir = OUTPUT_DIR if not use_future else os.path.join(OUTPUT_DIR, 'future')
    return save_tipping_report(states, out_dir, fname, year=year, extra_margins=extra_margins)


def tipping_report_text_from_states(states: List[Dict], year: int = 0, source: str = 'presidential_margins.csv', extra_margins: Optional[List[float]] = None) -> str:
    """Generate tipping report text from a list of state dicts.

    states: list of dicts each containing at least: 'abbr', 'evs', 'relative_margin'.
    year: optional year to include in the header (0 means no-year).
    source: textual source name to include in the report footer.

    Returns the report as a string (does not write files).
    """
    tipping_points, ordered_rel = compute_threshold_tipping_points(states)

    # Actual national margin from provided rows (optional)
    am_val = next((s.get('national_margin') for s in states if 'national_margin' in s), 0.0)
    am = float(am_val or 0.0)
    d_evs, r_evs = compute_ec_totals_at_margin(states, am)
    actual_state, _, _ = find_closest_state_for_margin(states, am)
    actual_info = {
        'margin': am,
        'state': actual_state,
        'D_evs': d_evs,
        'R_evs': r_evs,
    }

    mismatch_ranges = compute_ev_pv_mismatch_ranges(states)
    tie_ranges = compute_ec_tie_ranges(states)

    # Default extra margins if client did not pass any
    if extra_margins is None:
        extra_margins = [0.0, SWING_MARGIN, -SWING_MARGIN, 0.1, -0.1]

    report = build_report(year, tipping_points, ordered_rel, actual_info, mismatch_ranges, tie_ranges, extra_margins=extra_margins, source=source)
    return report


def save_tipping_report(states: List[Dict], out_dir: str, filename: str, year: Optional[int] = None, source: str = 'presidential_margins.csv', extra_margins: Optional[List[float]] = None) -> str:
    """Generate a tipping report from states and save it to out_dir/filename.

    Returns the absolute path to the written file.
    """
    report = tipping_report_text_from_states(states, year=(year or 0), source=source, extra_margins=extra_margins)
    ensure_output_dir(out_dir)
    out_path = os.path.join(out_dir, filename)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(report)
    return out_path


def main(use_future=False):
    by_year = read_data(use_future)
    if not by_year:
        print(f"No data found in {CSV_PATH}")
        return
    years = sorted(by_year.keys())
    for y in years:
        if use_future and y <= 2024:
            continue
        out_path = process_year(y, by_year[y], extra_margins=[0.0, SWING_MARGIN, -SWING_MARGIN, 0.1, -0.1], use_future=use_future)
        print(f"Wrote {out_path}")


if __name__ == '__main__':
    main()
    main(use_future=True)
