"""Generate per-year state shift summaries (raw and relative) from
`presidential_margins.csv`.

Creates directories:
  - state_shifts/raw_shifts/
  - state_shifts/rel_shifts/

For each year in the CSV it writes two text files:
  - state_shifts/raw_shifts/{year}.txt
  - state_shifts/rel_shifts/{year}.txt

Each file lists:
  - national margin delta and direction (left/right/none)
  - states that shifted with the nation (same sign)
  - states that shifted against the nation (opposite sign)

This script uses only the Python standard library.
"""

from __future__ import annotations
import csv
from pathlib import Path
from typing import Dict, List, Optional
import utils


ROOT = Path(__file__).resolve().parent
CSV = ROOT / "presidential_margins.csv"
OUT_DIR = ROOT / "state_shifts"


def parse_float(s: str) -> Optional[float]:
    if s is None:
        return None
    s = s.strip()
    if s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None


def sign(v: Optional[float]) -> int:
    """Return the sign of v: 1 for positive (left), -1 for negative (right), 0 for zero or None."""
    if v is None:
        return 0
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


def fmt(v: Optional[float]) -> str:
    if v is None:
        return "NA"
    return utils.lean_str(v)


def load_data(csv_path: Path) -> Dict[int, List[Dict]]:
    years: Dict[int, List[Dict]] = {}
    with csv_path.open(newline="", encoding="utf-8") as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            try:
                year = int(row["year"])
            except Exception:
                continue
            rec = {
                "abbr": row.get("abbr", "").strip(),
                "pres_margin_delta": parse_float(row.get("pres_margin_delta", "")),
                "relative_margin_delta": parse_float(row.get("relative_margin_delta", "")),
                "national_margin_delta": parse_float(row.get("national_margin_delta", "")),
                "relative_margin": parse_float(row.get("relative_margin", "")),
            }
            years.setdefault(year, []).append(rec)
    return years


def categorize(year: int, rows: List[Dict], use_relative: bool = False) -> str:
    # determine national delta for the year from any row that has it
    national_vals = [r["national_margin_delta"] for r in rows if r.get("national_margin_delta") is not None]
    national_delta = national_vals[0] if national_vals else 0.0
    nsign = sign(national_delta)
    if nsign > 0:
        n_dir = "left"
    elif nsign < 0:
        n_dir = "right"
    else:
        n_dir = "none"

    # pick delta key
    key = "relative_margin_delta" if use_relative else "pres_margin_delta"

    with_same = {"left": [], "right": [], "none": []}
    against = {"left": [], "right": [], "none": []}
    # collect shocks (based on relative margin changes) when requested
    shocks: List[tuple] = []

    for r in rows:
        abbr = r.get("abbr") or ""
        len_abbr = len(abbr)
        if len_abbr == 2:
            abbr = f"{abbr}\t"
        val = r.get(key)
        rel_margin = r.get("relative_margin")
        s = sign(val)
        # check for shocks based on relative margin delta (>= 8 points)
        if use_relative:
            rv = r.get("relative_margin_delta")
            if rv is not None and abs(rv) >= 0.08:
                shocks.append((abbr, rv, rel_margin))
        if nsign == 0:
            # national didn't shift: categorize by state direction
            if s > 0:
                with_same["left"].append((abbr, val))
            elif s < 0:
                with_same["right"].append((abbr, val))
            else:
                with_same["none"].append((abbr, val))
        else:
            if s == 0:
                with_same["none"].append((abbr, val))
            elif s == nsign:
                # same sign as national
                if s > 0:
                    with_same["left"].append((abbr, val))
                else:
                    with_same["right"].append((abbr, val))
            else:
                # opposite sign
                if s > 0:
                    against["left"].append((abbr, val))
                else:
                    against["right"].append((abbr, val))

    # build textual output
    lines: List[str] = []
    lines.append(f"Year: {year}")
    lines.append(f"National margin delta: {fmt(national_delta)} ({n_dir})")
    lines.append("")

    if nsign == 0:
        lines.append("Note: national margin delta is zero (no national shift). States are listed by their own direction:")
        lines.append("")
        for dir_label in ("left", "right", "none"):
            group = with_same[dir_label]
            lines.append(f"States shifted {dir_label} ({len(group)}):")
            if group:
                lines.append(', '.join(f"{abbr} ({fmt(val)})" for abbr, val in group))
            else:
                lines.append("(none)")
            lines.append("")
        return "\n".join(lines)

    for dir_label in ("left", "right", "none"):
        group = with_same[dir_label]
        if len(group) == 0:
            continue
        lines.append(f"States that shifted WITH the nation {dir_label} (same sign as national_margin_delta):")
        #lines.append(f"  {dir_label.capitalize()} ({len(group)}):")
        if group:
            lines.extend(f"\t{abbr}\t{fmt(val)}" for abbr, val in group)
        else:
            lines.append("    (none)")
    lines.append("")

    for dir_label in ("left", "right"):
        group = against[dir_label]
        if len(group) == 0:
            continue
        lines.append(f"States that shifted AGAINST the nation {dir_label} (opposite sign to national_margin_delta):")
        if group:
            lines.extend(f"\t{abbr}\t{fmt(val)}" for abbr, val in group)
        else:
            lines.append("    (none)")
    if against["none"]:
        lines.append("")
        lines.append("States with no change:")
        lines.append('  ' + ', '.join(f"{abbr}\t\t{fmt(val)}" for abbr, val in against["none"]))

    # If this categorize call was for relative margins, also write a shocks file for the year.
    if use_relative:
        shocks_dir = OUT_DIR / "shocks"
        shocks_dir.mkdir(exist_ok=True)
        shock_lines: List[str] = []
        shock_lines.append(f"Year: {year}")
        shock_lines.append("Shocks (|relative_margin_delta| >= 8.0):")
        shock_lines.append("")
        if shocks:
            for abbr, rv, rel_margin in sorted(shocks, key=lambda x: x[1]):
                shock_emoji = utils.emoji_from_lean(rv)
                rel_emoji = utils.emoji_from_lean(rel_margin, use_swing=True)
                shock_lines.append(f"{rel_emoji}{abbr} {'\t'*(1 + 1)}{fmt(rel_margin-rv)}->{fmt(rel_margin)}\t{shock_emoji}{fmt(rv)}")
        else:
            shock_lines.append("(none)")
        (shocks_dir / f"{year}.txt").write_text("\n".join(shock_lines), encoding="utf-8")

    return "\n".join(lines)


def raw_shifts_away(rows: List[Dict]) -> List[tuple]:
    """Return list of (abbr, pres_margin_delta) for states that shifted against the
    national raw (presidential) margin delta. If the national margin delta is zero or
    missing, return an empty list.
    """
    national_vals = [r["national_margin_delta"] for r in rows if r.get("national_margin_delta") is not None]
    national_delta = national_vals[0] if national_vals else 0.0
    nsign = sign(national_delta)
    if nsign == 0:
        return []
    out: List[tuple] = []
    for r in rows:
        abbr = r.get("abbr") or ""
        # keep two-letter abbr alignment consistent with other outputs
        if len(abbr) == 2:
            abbr = f"{abbr}\t"
        val = r.get("pres_margin_delta")
        s = sign(val)
        if s != 0 and s != nsign:
            out.append((abbr, val))
    # sort by abbreviation
    return sorted(out, key=lambda x: x[0])


def main() -> None:
    if not CSV.exists():
        print(f"Expected CSV at {CSV!s} but file not found.")
        return
    years = load_data(CSV)
    OUT_DIR.mkdir(exist_ok=True)
    raw_dir = OUT_DIR / "raw_shifts"
    rel_dir = OUT_DIR / "rel_shifts"
    raw_dir.mkdir(exist_ok=True)
    rel_dir.mkdir(exist_ok=True)

    # sort years and skip the first (earliest) year because it has no delta data
    years_items = sorted(years.items())
    if years_items:
        years_to_process = years_items[1:]
    else:
        years_to_process = []

    for year, rows in years_to_process:
        raw_text = categorize(year, rows, use_relative=False)
        rel_text = categorize(year, rows, use_relative=True)

        (raw_dir / f"{year}.txt").write_text(raw_text, encoding="utf-8")
        (rel_dir / f"{year}.txt").write_text(rel_text, encoding="utf-8")
        # collect raw-away shifts for summary
        away = raw_shifts_away(rows)
        # write per-year short summary in the raw_shifts dir as well for quick glance
        summary_lines = [f"Year: {year}"]
        if away:
            summary_lines.extend(
                [f"\t{abbr.strip()} ({fmt(val)})\n" for abbr, val in away]
            )
        else:
            summary_lines.append("(none)")
        #(raw_dir / f"{year}_away.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    # write an aggregated year-by-year summary of states that shifted against the national raw shift
    away_summary_path = OUT_DIR / "away_from_nation.txt"
    away_lines: List[str] = []
    for year, rows in years_to_process:
        away = raw_shifts_away(rows)
        away_lines.append(f"Year: {year} (National delta: {fmt(rows[0].get('national_margin_delta'))})")
        if away:
            away_lines.extend(
            [
                f"\t{abbr.strip()} ({fmt(val)}) [Relative margin: {fmt(r.get('relative_margin') - val)} -> {fmt(r.get('relative_margin'))}]"
                for abbr, val in away
                for r in rows if r.get("abbr") == abbr.strip()
            ]
            )
        else:
            away_lines.append("(none)")
        away_lines.append("")
    away_summary_path.write_text("\n".join(away_lines), encoding="utf-8")

    print(f"Wrote shift files to: {raw_dir} and {rel_dir}")


if __name__ == "__main__":
    main()
