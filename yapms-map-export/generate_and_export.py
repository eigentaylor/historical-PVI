import subprocess
import sys
from pathlib import Path

# Convenience script: build base64 JSON then export PNG in one go.
# Usage:
#   python yapms-map-export/generate_and_export.py presidential_margins_future.csv out.png --relative --filter-year 2024 --url inline

def run(cmd):
    res = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{res.stderr}")
    return res.stdout.strip()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--relative", action="store_true")
    parser.add_argument("--filter-year", type=int)
    parser.add_argument("--year-tag", default="2024312")
    parser.add_argument("--bins", nargs="*")
    parser.add_argument("--url", default="inline")
    args = parser.parse_args()

    build_cmd = [
        sys.executable,
        str(Path(__file__).with_name("build_centroid_map_json.py")),
        str(args.csv),
    ]
    if args.relative:
        build_cmd.append("--relative")
    if args.filter_year:
        build_cmd += ["--filter-year", str(args.filter_year)]
    if args.year_tag:
        build_cmd += ["--year-tag", args.year_tag]
    if args.bins:
        build_cmd += ["--bins", *args.bins]

    base64_json = run(build_cmd)

    export_cmd = [
        sys.executable,
        str(Path(__file__).with_name("export_yapms_png.py")),
        base64_json,
        "--out",
        str(args.out),
        "--url",
        args.url,
    ]
    _ = run(export_cmd)


if __name__ == "__main__":
    main()
