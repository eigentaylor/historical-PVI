import base64
import json
import math
import pandas as pd
from pathlib import Path

# Minimal SavedMap builder for USA presidential blank map (ME/NE split)
# Inputs:
# - csv_path: path to CSV with columns [abbr, pres_margin or relative_margin, electoral_votes]
# - use_relative: if True, use relative_margin; else pres_margin
# - title_year_tag: string used for map.year metadata; must match SVG asset year token (2024312)
# - margin_bins: thresholds for shade indices (abs margin). Default maps to 4 shades.
# Output: prints base64-encoded SavedMap JSON to stdout.

US_CANDIDATES = [
    {
        "id": "0",
        "name": "Democrat",
        "defaultCount": 0,
        "margins": [{"color": "#1C408C"}, {"color": "#577CCC"}, {"color": "#8AAFFF"}, {"color": "#949BB3"}],
    },
    {
        "id": "1",
        "name": "Republican",
        "defaultCount": 0,
        "margins": [{"color": "#BF1D29"}, {"color": "#FF5865"}, {"color": "#FF8B98"}, {"color": "#CF8980"}],
    },
]

TOSSUP = {"id": "", "name": "Tossup", "defaultCount": 0, "margins": [{"color": "#cccccc"}]}

ME_NE_MAP = {
    "ME-AL": "me-al",
    "ME-01": "me-01",
    "ME-02": "me-02",
    "NE-AL": "ne-al",
    "NE-01": "ne-01",
    "NE-02": "ne-02",
    "NE-03": "ne-03",
}

STATE_ABBR_FIX = {"DC": "dc"}

EV_FALLBACK = {
    # Fallbacks if not present in CSV
    "DC": 3,
}

MARGIN_BINS_DEFAULT = [0.05, 0.10, 0.20, math.inf]  # 0-5, 5-10, 10-20, 20+


def shade_index(abs_margin: float, bins):
    for i, b in enumerate(bins):
        if abs_margin < b:
            return i
    return len(bins) - 1


def load_ev_lookup(df: pd.DataFrame) -> dict:
    ev = {}
    for _, row in df.iterrows():
        abbr = str(row["abbr"]).upper()
        ev_val = row.get("electoral_votes")
        if pd.notna(ev_val):
            ev[abbr] = int(ev_val)
    ev.update(EV_FALLBACK)
    return ev


def build_saved_map(df: pd.DataFrame, use_relative: bool, year_tag: str, bins) -> dict:
    ev_lookup = load_ev_lookup(df)
    regions = []
    margin_col = "relative_margin" if use_relative and "relative_margin" in df.columns else "pres_margin"

    for _, row in df.iterrows():
        abbr_raw = str(row["abbr"]).upper()
        # Skip non-states if needed except DC/ME/NE districts
        expanded_regions = []
        if abbr_raw.startswith("ME-") or abbr_raw.startswith("NE-"):
            rid = ME_NE_MAP.get(abbr_raw)
            if not rid:
                continue
            expanded_regions = [(rid, None)]
        elif abbr_raw == "ME":
            # Expand ME statewide into at-large + 2 districts with same margin
            expanded_regions = [("me-al", 2), ("me-01", 1), ("me-02", 1)]
        elif abbr_raw == "NE":
            expanded_regions = [("ne-al", 2), ("ne-01", 1), ("ne-02", 1), ("ne-03", 1)]
        else:
            expanded_regions = [(STATE_ABBR_FIX.get(abbr_raw, abbr_raw.lower()), None)]
        try:
            margin = float(row[margin_col])
        except Exception:
            continue
        base_ev = int(row.get("electoral_votes") or ev_lookup.get(abbr_raw, 0))
        # Determine per-region EV when expanded; if None provided, fall back to state EV
        for region_id, ev_override in expanded_regions:
            ev = int(ev_override if ev_override is not None else base_ev)
            if ev <= 0:
                continue
            # Decide winner and shade index
            if margin > 0:  # D leads
                candidates = [{"id": "0", "count": ev, "margin": shade_index(abs(margin), bins)}]
            elif margin < 0:  # R leads
                candidates = [{"id": "1", "count": ev, "margin": shade_index(abs(margin), bins)}]
            else:
                candidates = []  # Tossup; counts filled by Regions store to value
            regions.append({
                "id": region_id,
                "value": ev,
                "permaVal": ev,
                "locked": False,
                "permaLocked": False,
                "disabled": False,
                "candidates": candidates,
            })

    saved = {
        "map": {"country": "usa", "type": "presidential", "year": year_tag, "variant": "blank"},
        "tossup": TOSSUP,
        "candidates": US_CANDIDATES,
        "regions": regions,
    }
    return saved


def encode_base64_json(obj: dict) -> str:
    s = json.dumps(obj, separators=(",", ":"))
    return base64.b64encode(s.encode("utf-8")).decode("ascii")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build base64 SavedMap JSON for YAPMS embed route")
    parser.add_argument("csv", type=Path, help="CSV with columns abbr, pres_margin or relative_margin, electoral_votes")
    parser.add_argument("--relative", action="store_true", help="Use relative_margin column")
    parser.add_argument("--filter-year", type=int, help="Filter CSV to a specific election year (e.g., 2024)")
    parser.add_argument("--year-tag", default="2024312", help="Map year token to match SVG asset (default 2024312)")
    parser.add_argument("--bins", nargs="*", type=float, help="Margin bins like 0.05 0.10 0.20")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.filter_year and "year" in df.columns:
        df = df[df["year"] == args.filter_year]
    bins = args.bins if args.bins else MARGIN_BINS_DEFAULT
    if bins[-1] != math.inf:
        bins = [*bins, math.inf]
    saved = build_saved_map(df, args.relative, args.year_tag, bins)
    print(encode_base64_json(saved))


if __name__ == "__main__":
    main()
