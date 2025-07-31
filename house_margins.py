import csv
from collections import defaultdict

import numpy as np
import utils

def run_import_house(start_year=1976):
    input_file = "data/1976-2022-house.csv"
    output_file = "house_margins.csv"

    full_results = {}
    mark_for_uncontested_scaling = {}
    number_of_contested_districts = defaultdict(int)

    # Import House data
    with open(input_file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        # Skip the header row
        header_row = next(reader)
        next(reader)
        for row in reader:
            year = int(row[0])
            if year < start_year:
                continue
            state_po = row[2]
            district = row[7]
            total_votes = row[16]
            party = row[12]
            if total_votes == '1':
                mark_for_uncontested_scaling[(year, state_po, district)] = party
            votes = int(row[15])

            # Only consider Democrat and Republican
            if "DEMOCRAT" in party:
                key = (year, state_po, district)
                full_results.setdefault(key, {}).setdefault("D", 0)
                full_results[key]["D"] += votes
            elif "REPUBLICAN" in party:
                key = (year, state_po, district)
                full_results.setdefault(key, {}).setdefault("R", 0)
                full_results[key]["R"] += votes

    # Aggregate contested results by (year, state_po)
    aggregated_contested_results = defaultdict(lambda: {"D": 0, "R": 0})
    for (year, state_po, district), votes_dict in full_results.items():
        if "D" in votes_dict and "R" in votes_dict and votes_dict["D"] > 0 and votes_dict["R"] > 0:
            aggregated_contested_results[(year, state_po)]["D"] += votes_dict["D"]
            aggregated_contested_results[(year, state_po)]["R"] += votes_dict["R"]

    # Prepare output rows
    output_rows = []
    for (year, state_po), votes_dict in aggregated_contested_results.items():
        D = votes_dict.get("D", 0)
        R = votes_dict.get("R", 0)
        total = D + R
        margin = (D - R) / total if total > 0 else ""
        margin_str = utils.lean_str(margin)

        output_rows.append([year, state_po, D, R, margin, margin_str])

    # Calculate distance from national average
    years = sorted(list(set(row[0] for row in output_rows)))
    for year in years:
        total_D = sum(row[2] for row in output_rows if row[0] == year)
        total_R = sum(row[3] for row in output_rows if row[0] == year)
        national_margin = (total_D - total_R) / (total_D + total_R) if (total_D + total_R) > 0 else 0.0

        for row in output_rows:
            if row[0] == year:
                state_margin = (row[2] - row[3]) / (row[2] + row[3]) if (row[2] + row[3]) > 0 else 0.0
                row.append(state_margin - national_margin)  # Margin delta
                row.append(utils.lean_str(state_margin - national_margin))  # Margin delta string
                row.append(national_margin)
                row.append(utils.lean_str(national_margin))

    # Sort by state abbreviation and district
    output_rows.sort(key=lambda x: (x[1], int(x[0])))

    # Write to output CSV
    with open(output_file, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["year", "abbr", "D_votes", "R_votes", "house_margin", "house_margin_str", "relative_margin", "relative_margin_str", "national_margin", "national_margin_str"])
        writer.writerows(output_rows)

    print(f"Done! Output written to {output_file}")

if __name__ == "__main__":
    run_import_house()
