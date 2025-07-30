import pandas as pd
import os
import numpy as np
from utils import lean_str, emoji_from_lean

# Read the future margins file
df = pd.read_csv("presidential_margins_future.csv")

# Output directory
output_dir = "future_elections"
os.makedirs(output_dir, exist_ok=True)
# Clear all files in the output directory
for file in os.listdir(output_dir):
    file_path = os.path.join(output_dir, file)
    if os.path.isfile(file_path):
        os.remove(file_path)

# Get all years
years = sorted(df["year"].unique())

for year in years:
    year_df = df[df["year"] == year].copy()
    # Calculate EC totals
    dem_ec = year_df[year_df["pres_margin"] > 0]["electoral_votes"].sum()
    rep_ec = year_df[year_df["pres_margin"] < 0]["electoral_votes"].sum()

    # Sort states by pres_margin descending
    year_df = year_df.sort_values("pres_margin", ascending=False)

    # Prepare lines for output
    lines = []
    lines.append(f"Year: {year}")
    lines.append(f"Democratic EC: {dem_ec}")
    lines.append(f"Republican EC: {rep_ec}")
    # Popular vote lean from national_margin
    nat_margin = year_df["national_margin"].iloc[0]
    nat_lean = lean_str(nat_margin)
    lines.append(f"Popular Vote: {nat_lean}")
    lines.append("")
    lines.append("States sorted by presidential margin:")
    for _, row in year_df.iterrows():
        lean = lean_str(row["pres_margin"])
        emoji = emoji_from_lean(row["pres_margin"], use_swing=False)
        state = row["abbr"]
        ev = row["electoral_votes"]
        lines.append(f"{state}: {lean} {emoji} ({ev} EV)")


    # Write to file
    out_path = os.path.join(output_dir, f"{year}_EC.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# -----------------------------
# State summary log
# -----------------------------
summary_lines = []
summary_lines.append("State Summary Log\n")
state_types = {}
for state in sorted(df["abbr"].unique()):
    margins = df[df["abbr"] == state].sort_values("year")["pres_margin"].values
    years = df[(df["abbr"] == state) & (df["year"] > 2024)].sort_values("year")["year"].values
    margins = df[(df["abbr"] == state) & (df["year"] > 2024)].sort_values("year")["pres_margin"].values
    signs = np.sign(margins)
    safe_blue = np.all(margins > 0.03)
    safe_red = np.all(margins < -0.03)
    swing_mask = (np.abs(margins) <= 0.03)
    always_swing = np.all(swing_mask)
    ever_swing = np.any(swing_mask)
    crosses_zero = np.any(np.diff(signs) != 0)
    # Detect flips
    first_sign = signs[0]
    last_sign = signs[-1]
    ever_flip = (first_sign != 0 and last_sign != 0 and first_sign != last_sign)
    # Classification
    if safe_blue:
        state_types[state] = "Safely Blue"
    elif safe_red:
        state_types[state] = "Safely Red"
    elif always_swing and not crosses_zero:
        state_types[state] = "Consistent Swing (never swings)"
    elif always_swing and crosses_zero:
        state_types[state] = "Swing (swings between parties)"
    elif ever_swing and not crosses_zero:
        state_types[state] = "Almost swing (never swings)"
    elif ever_flip:
        state_types[state] = f"Flips from {'Blue' if first_sign > 0 else 'Red'} to {'Red' if last_sign < 0 else 'Blue'}"
    else:
        state_types[state] = "Mixed/Other"

# List states by type
for t in ["Safely Blue", "Safely Red", "Consistent Swing (never swings)", "Swing (swings between parties)", "Almost swing (never swings)"]:
    summary_lines.append(f"\n{t} states:")
    for state, typ in state_types.items():
        if typ == t:
            summary_lines.append(f"  {state}")

# Flips
summary_lines.append("\nStates that flip:")
for state, typ in state_types.items():
    if typ.startswith("Flips"):
        margins = df[(df["abbr"] == state) & (df["year"] > 2024)].sort_values("year")["pres_margin"].values
        years = df[(df["abbr"] == state) & (df["year"] > 2024)].sort_values("year")["year"].values
        first_sign = np.sign(margins[0])
        flip_years = [years[i] for i in range(1, len(margins)) if np.sign(margins[i]) != np.sign(margins[i-1])]
        summary_lines.append(f"  {state}: {typ} (flips in {', '.join(map(str, flip_years))})")

# Swing states that swing: list years and party
summary_lines.append("\nSwing states that swing (list years and party):")
for state, typ in state_types.items():
    if typ == "Swing (swings between parties)":
        margins = df[df["abbr"] == state].sort_values("year")["pres_margin"].values
        years = df[df["abbr"] == state].sort_values("year")["year"].values
        swings = [(y, "Blue" if m > 0 else "Red" if m < 0 else "Tie") for y, m in zip(years, margins)]
        summary_lines.append(f"  {state}: " + ", ".join([f"{y}: {p}" for y, p in swings]))

# Consistent swing states
summary_lines.append("\nConsistent swing states (never swing):")
for state, typ in state_types.items():
    if typ == "Consistent Swing (never swings)":
        summary_lines.append(f"  {state}")


# Write summary log
summary_path = os.path.join(output_dir, "state_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

# -----------------------------
# Election summary log (PV and EC totals for future years)
# -----------------------------
summary_election_lines = []
summary_election_lines.append("Election Summary (Future Years >2024)\n")
future_years = sorted([y for y in df["year"].unique() if y > 2024])
for year in future_years:
    year_df = df[df["year"] == year]
    dem_ec = year_df[year_df["pres_margin"] > 0]["electoral_votes"].sum()
    rep_ec = year_df[year_df["pres_margin"] < 0]["electoral_votes"].sum()
    nat_margin = year_df["national_margin"].iloc[0]
    nat_lean = lean_str(nat_margin)
    summary_election_lines.append(f"Year: {year}")
    summary_election_lines.append(f"  Popular Vote: {nat_lean}")
    summary_election_lines.append(f"  Democratic EC: {dem_ec}")
    summary_election_lines.append(f"  Republican EC: {rep_ec}")
    summary_election_lines.append("")

election_summary_path = os.path.join(output_dir, "election_summary.txt")
with open(election_summary_path, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_election_lines))
