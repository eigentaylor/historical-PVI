import subprocess
import sys
from pathlib import Path

# One-shot helper: CSV (centroid) -> base64 SavedMap -> PNG via running dev server URL
# Usage:
#   python csv_to_png.py path/to/2028_csv_centroid_1.csv --out map.png --url http://localhost:5173


def run(cmd):
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{res.stderr}")
    return res.stdout.strip()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="CSV to YAPMS PNG (uses embed route)")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--out", type=Path, default=Path("map.png"))
    parser.add_argument("--url", default="http://localhost:5173")
    parser.add_argument("--bins", nargs="*", type=float, help="Margin bins like 0.05 0.10 0.20")
    parser.add_argument("--year-tag", default="2024312")
    parser.add_argument("--margin-col", default="final_margin")
    args = parser.parse_args()

    bins_args = []
    if args.bins:
        bins_args = ["--bins", *[str(x) for x in args.bins]]

    base64_json = run([
        sys.executable,
        str(Path(__file__).with_name("build_centroid_map_json.py")),
        str(args.csv),
        "--year-tag", args.year_tag,
        "--margin-col", args.margin_col,
        *bins_args,
    ])

    run([
        sys.executable,
        str(Path(__file__).with_name("export_yapms_png.py")),
        base64_json,
        "--out", str(args.out),
        "--url", args.url,
    ])


if __name__ == "__main__":
    main()
