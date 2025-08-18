"""Debug helper: colorize SVG for a single centroid CSV and write out the modified SVG

Run: python debug_dump_colored_svg.py <csv_path> <out_svg_path>
"""
import sys
from pathlib import Path
import pandas as pd

from yapms_from_centroid_csvs import build_saved_map_from_centroid_df
from export_yapms_png import _colorize_svg


def main():
    if len(sys.argv) < 3:
        print("Usage: python debug_dump_colored_svg.py <csv_path> <out_svg_path>")
        return
    csv_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    df = pd.read_csv(csv_path)
    saved = build_saved_map_from_centroid_df(df)
    # load SVG for year in saved (or fallback)
    year = str(saved.get("map", {}).get("year", "2024312"))
    svg_path = Path(__file__).resolve().parent / "yapms" / "apps" / "yapms" / "src" / "lib" / "assets" / "maps" / "usa" / f"usa-presidential-{year}-blank.svg"
    svg_text = svg_path.read_text(encoding="utf-8")
    colored = _colorize_svg(svg_text, saved)
    out_path.write_text(colored, encoding="utf-8")
    print(f"Wrote colored SVG to {out_path}")


if __name__ == '__main__':
    main()
