import csv
import utils

def run_import_csv(start_year=1976):
    input_file = "data/1976-2020-president.csv"
    output_file = "presidential_margins.csv"

    results = {}

    # Import pre-1976 data if needed
    if start_year < 1976:
        with open("data/1900_2024_election_results.csv", newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                year = int(row['year'])
                if year >= 1976 or year < start_year:
                    continue
                state_po = row['state_po']
                D_votes = int(row['D_votes']) if row['D_votes'] else 0
                R_votes = int(row['R_votes']) if row['R_votes'] else 0
                key = (str(year), state_po)
                results.setdefault(key, {})['D'] = D_votes
                results.setdefault(key, {})['R'] = R_votes

    # Import post-1976 data as before
    with open(input_file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        # Skip the header row
        next(reader)
        for row in reader:
            if row[9] == "TRUE": # write in
                continue  # Skip write-in candidates
            year = row[0]
            if int(year) < start_year:
                continue
            state_po = row[2]
            if year == "2020" and state_po == "DC":
                pass
            party = row[14]
            votes = int(row[10])
            if state_po == "AZ" and year == "2016":
                print(f"Warning: 2016 AZ D_votes is {votes}, expected 1161167")
            # Only consider Democrat and Republican
            if party == "DEMOCRAT":
                key = (year, state_po)
                results.setdefault(key, {}).setdefault("D", 0)
                results[key]["D"] += votes
            elif party == "REPUBLICAN":
                key = (year, state_po)
                results.setdefault(key, {}).setdefault("R", 0)
                results[key]["R"] += votes

    # Load electoral votes from 1900_2024_election_results.csv
    ev_map = {}
    with open("data/1900_2024_election_results.csv", newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ev_map[(int(row['year']), row['state_po'])] = int(row['electoral_votes']) if row['electoral_votes'] else 0
            if row['state_po'] in ["ME", "NE"]:
                # add an extra EV
                ev_map[(int(row['year']), row['state_po'])] += 1

    # Prepare output rows
    output_rows = []
    for (year, state_po), votes_dict in results.items():
        D = votes_dict.get("D", 0)
        if state_po == "AZ" and year == "2016":
            print(f"Warning: 2016 AZ D_votes is {D}, expected 1161167")
        R = votes_dict.get("R", 0)
        total = D + R
        margin = (D - R) / total if total > 0 else ""
        margin_str = utils.lean_str(margin)
        ev = ev_map.get((int(year), state_po), 0)
        output_rows.append([int(year), state_po, D, R, margin, margin_str, ev])

    # Read 2024 margins from the trends file
    trends_file = "data/State_Two-Party_Votes__2024_.csv"
    state_2024_margin = {}
    with open(trends_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            state_po = row['abbr']
            D = int(row['D_votes'])
            R = int(row['R_votes'])
            margin = (D - R) / (D + R) if (D + R) > 0 else ""
            state_2024_margin[state_po] = margin
            margin_str = utils.lean_str(margin)
            ev = ev_map.get((2024, state_po), 0)
            # Update the output rows with 2024 margin
            output_rows.append([2024, state_po, D, R, margin, margin_str, ev])
            
    # check what 2016 AZ D_votes were
    az_2016_d_votes = next((row[2] for row in output_rows if row[0] == 2016 and row[1] == "AZ"), None)
            
    years = sorted(list(set(row[0] for row in output_rows)))

    # calculate distance from national average
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
                # electoral votes already present

    # Sort by state abbreviation
    output_rows.sort(key=lambda x: (x[1], x[0]))

    # Write to output CSV
    with open(output_file, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["year", "abbr", "D_votes", "R_votes", "pres_margin", "pres_margin_str", "electoral_votes", "relative_margin", "relative_margin_str", "national_margin", "national_margin_str"])
        writer.writerows(output_rows)

    print(f"Done! Output written to {output_file}")