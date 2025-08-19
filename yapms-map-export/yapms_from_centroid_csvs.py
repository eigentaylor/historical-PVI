import base64
import json
from pathlib import Path
import sys

import pandas as pd

# Ensure we can import project-level modules when running from this subfolder
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Use repo utilities for token mapping
import utils  # type: ignore
import params  # type: ignore

from export_yapms_png import export_png  # async function
import asyncio


# Build candidates with colors using the requested palette (lightest -> darkest)
US_CANDIDATES = [
    {
        "id": "0",
        "name": "Democrat",
        "defaultCount": 0,
        # index 0..3 corresponds to tilt, lean, likely, called
        "margins": [
            {"color": "#b3c1f9"},  # D_tilt
            {"color": "#7a91ef"},  # D_lean
            {"color": "#425fe3"},  # D_likely
            {"color": "#183ebf"},  # D_called
        ],
    },
    {
        "id": "1",
        "name": "Republican",
        "defaultCount": 0,
        # index 0..3 corresponds to tilt, lean, likely, called
        "margins": [
            {"color": "#f4b6ba"},  # R_tilt
            {"color": "#e97c83"},  # R_lean
            {"color": "#d9424b"},  # R_likely
            {"color": "#bf1d29"},  # R_called
        ],
    },
]

TOSSUP = {"id": "", "name": "Tossup", "defaultCount": 0, "margins": [{"color": params.CATEGORY_COLORS.get("swing", "#C3B1E1")}]}  # light purple

TOKEN_TO_PARTY_AND_INDEX = {
    # Democrats
    "lblue": ("0", 0),
    "blue":  ("0", 1),
    "Blue":  ("0", 2),
    "BLUE":  ("0", 3),
    # Republicans
    "lred": ("1", 0),
    "red":  ("1", 1),
    "Red":  ("1", 2),
    "RED":  ("1", 3),
}


def token_for_margin(m: float) -> str:
    return utils.final_margin_color_key(m)


def build_saved_map_from_centroid_df(df: pd.DataFrame, year_tag: str = "2024312") -> dict:
    required = {"abbr", "final_margin", "electoral_votes"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in centroid CSV: {missing}")

    regions = []
    # For national popular vote estimate, compute EV-weighted average of final_margin
    total_ev_for_pv = 0
    weighted_margin_sum = 0.0
    for _, row in df.iterrows():
        abbr = str(row["abbr"]).upper()
        rid = abbr.lower()
        try:
            fm = float(row["final_margin"])
        except Exception:
            continue
        try:
            ev = int(row["electoral_votes"]) if pd.notna(row["electoral_votes"]) else 0
        except Exception:
            ev = 0
        if ev <= 0:
            continue

        # accumulate for national PV
        total_ev_for_pv += ev
        weighted_margin_sum += fm * ev

        tok = token_for_margin(fm)
        if tok in TOKEN_TO_PARTY_AND_INDEX:
            cand_id, shade_idx = TOKEN_TO_PARTY_AND_INDEX[tok]
            cand = [{"id": cand_id, "count": ev, "margin": int(shade_idx)}]
        else:
            cand = []  # tossup/swing

        regions.append({
            "id": rid,
            "value": ev,
            "permaVal": ev,
            "locked": False,
            "permaLocked": False,
            "disabled": False,
            "candidates": cand,
        })

    saved = {
        "map": {"country": "usa", "type": "presidential", "year": year_tag, "variant": "blank"},
        "tossup": TOSSUP,
        "candidates": US_CANDIDATES,
        "regions": regions,
    # national_margin is D-R as a float in [-1,1]
    "national_margin": (weighted_margin_sum / total_ev_for_pv) if total_ev_for_pv > 0 else 0.0,
    }
    return saved


def to_base64_json(obj: dict) -> str:
    s = json.dumps(obj, separators=(",", ":"))
    return base64.b64encode(s.encode("utf-8")).decode("ascii")


async def process_year_folder(year_dir: Path, out_dir: Path, base_url: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    csvs = sorted(year_dir.glob(f"{year_dir.name}_csv_centroid_*.csv"))
    # if we're in the final_maps folder, we look for {year}_chosen_centroid.csv for all years
    if year_dir.name == "final_maps":
        csvs += sorted(year_dir.glob(f"*_chosen_centroid.csv"))
    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        saved = build_saved_map_from_centroid_df(df)
        b64 = to_base64_json(saved)
        if year_dir.name == "final_maps":
            yr = csv_path.stem.split('_')[0]
            png_path = out_dir / f"img_{yr}_centroid.png"
        else:
            yr = year_dir.name
            png_path = out_dir / f"{yr}_img_centroid_{csv_path.stem.split('_')[-1]}.png"
        await export_png(b64, png_path, base_url, year=yr)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Render YAPMS maps from centroid CSVs (final_margin)")
    parser.add_argument("year_folder", type=Path, help="Folder containing e.g., 2028_csv_centroid_*.csv")
    parser.add_argument("--out", type=Path, default=None, help="Output folder for PNGs (defaults to year_folder)")
    parser.add_argument("--url", default="inline", help="YAPMS base URL (e.g., http://127.0.0.1:8081) or 'inline'")
    args = parser.parse_args()

    year_dir = args.year_folder
    out_dir = args.out if args.out else year_dir
    asyncio.run(process_year_folder(year_dir, out_dir, args.url))


if __name__ == "__main__":
    main()
