import csv
import utils

# Abbreviation to full state name mapping for Electoral_College.csv
ABBR_TO_NAME = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
    'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'DC': 'D.C.', 'FL': 'Florida',
    'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana',
    'IA': 'Iowa', 'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine',
    'MD': 'Maryland', 'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
    'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire',
    'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota',
    'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island',
    'SC': 'South Carolina', 'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
    'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming',
    'ME-01': 'Maine First District', 'ME-02': 'Maine Second District', 'ME-AL': 'Maine At-Large',
    'NE-01': 'Nebraska First District', 'NE-02': 'Nebraska Second District', 'NE-03': 'Nebraska Third District', 'NE-AL': 'Nebraska At-Large'
}

SEPARATE_BY_DISTRICT = True  # Set to True to separate Maine and Nebraska by district

def load_electoral_college_map(path: str = 'Electoral_College.csv'):
    """Load Electoral College votes by (year, full_state_name) from CSV.
    Prints warnings for malformed rows.
    Returns dict[(int year, str state_name)] = int evs
    """
    ec_map = {}
    try:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 3:
                    continue
                try:
                    year = int(row[0])
                except ValueError:
                    # likely a header or malformed row
                    continue
                state_name = row[1].strip()
                evs_str = row[2].strip() if len(row) > 2 else ''
                if evs_str == '':
                    # some years intentionally blank; skip
                    continue
                try:
                    evs = int(evs_str)
                except ValueError:
                    print(f"Warning: invalid EV '{evs_str}' for {state_name} {year} in {path}")
                    continue
                ec_map[(year, state_name)] = evs
    except FileNotFoundError:
        print(f"Warning: {path} not found. EVs will default to 0.")
    return ec_map


def get_ev_from_map(year: int, state_po: str, ec_map) -> int:
    name = ABBR_TO_NAME.get(state_po, None)
    if name is None:
        print(f"Warning: unknown state abbreviation '{state_po}' for year {year}")
        return 0
    ev = ec_map.get((int(year), name))
    if ev is None:
        print(f"Warning: missing EV for {name} ({state_po}) in {year} in Electoral_College.csv")
        return 0
    return ev


def load_existing_2024_evs(path: str = 'presidential_margins.csv'):
    """Load 2024 EVs from an existing presidential_margins.csv to preserve them.
    Returns dict[str abbr] = int evs. Silently ignores missing file.
    """
    evs_2024 = {}
    try:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if not reader or not reader.fieldnames:
                return evs_2024
            year_field = 'year' if 'year' in reader.fieldnames else None
            abbr_field = 'abbr' if 'abbr' in reader.fieldnames else ('state_po' if 'state_po' in reader.fieldnames else None)
            ev_field = 'electoral_votes' if 'electoral_votes' in reader.fieldnames else None
            if not (year_field and abbr_field and ev_field):
                return evs_2024
            for row in reader:
                if str(row.get(year_field, '')).strip() == '2024':
                    abbr = (row.get(abbr_field) or '').strip()
                    ev_str = (row.get(ev_field) or '').strip()
                    if abbr and ev_str:
                        try:
                            # handle possible float-like strings
                            evs_2024[abbr] = int(float(ev_str))
                        except ValueError:
                            print(f"Warning: invalid 2024 EV '{ev_str}' for {abbr} in {path}")
    except FileNotFoundError:
        # OK if the file doesn't exist yet
        pass
    return evs_2024


def run_import_csv(start_year=1976, separate_by_district: bool = True, me_ne_enhanced_path: str = 'ME_NE_enhanced.csv'):
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

    # Load electoral votes mapping from Electoral_College.csv
    ec_map = load_electoral_college_map('Electoral_College.csv')
    # Preserve existing 2024 EVs from current presidential_margins.csv
    existing_2024_evs = load_existing_2024_evs(output_file)

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
        ev = get_ev_from_map(int(year), state_po, ec_map)
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
            # Use EVs from existing presidential_margins.csv for 2024, not Electoral_College.csv
            ev_2024 = existing_2024_evs.get(state_po)
            if ev_2024 is None:
                print(f"Warning: missing 2024 EV for {state_po} in {output_file}; leaving blank in new output")
            # Update the output rows with 2024 margin
            output_rows.append([2024, state_po, D, R, margin, margin_str, (ev_2024 if ev_2024 is not None else '')])
            
    # check what 2016 AZ D_votes were
    az_2016_d_votes = next((row[2] for row in output_rows if row[0] == 2016 and row[1] == "AZ"), None)

    # Optionally replace ME/NE rows with district/AL rows from ME_NE_enhanced.csv
    if separate_by_district:
        # Load enhanced rows
        enhanced = []
        try:
            with open(me_ne_enhanced_path, newline='', encoding='utf-8') as f:
                ereader = csv.DictReader(f)
                for er in ereader:
                    try:
                        y = int((er.get('year') or '').strip())
                    except ValueError:
                        continue
                    abbr = (er.get('abbr') or '').strip()
                    if not (abbr.startswith('ME-') or abbr.startswith('NE-')):
                        continue
                    # Parse fields we need; tolerate blanks
                    def parse_int(s):
                        s = (s or '').replace(',', '').strip()
                        try:
                            return int(s)
                        except ValueError:
                            return 0
                    def parse_float(s):
                        s = (s or '').strip()
                        try:
                            return float(s)
                        except ValueError:
                            return ''
                    Dv = parse_int(er.get('D_votes'))
                    Rv = parse_int(er.get('R_votes'))
                    pm = parse_float(er.get('pres_margin'))
                    ev = parse_int(er.get('electoral_votes'))
                    enhanced.append({'year': y, 'abbr': abbr, 'D': Dv, 'R': Rv, 'pres_margin': pm, 'ev': ev})
        except FileNotFoundError:
            enhanced = []

        # Index enhanced by (year, state_prefix)
        enh_by_year_state = {}
        for r in enhanced:
            state_prefix = r['abbr'].split('-')[0]
            enh_by_year_state.setdefault((r['year'], state_prefix), []).append(r)

        def replace_state(rows, state_po: str):
            new_rows = []
            for row in rows:
                y, ab, D, R, margin, margin_str, ev = row[:7]
                if ab == state_po:
                    group = enh_by_year_state.get((y, state_po), None)
                    if group:
                        # Require an -AL row to do full replacement; else rename fallback
                        has_al = any(gr['abbr'].endswith('-AL') for gr in group)
                        if has_al:
                            # Replace with all district and AL rows for that state/year
                            for gr in sorted(group, key=lambda g: g['abbr']):
                                g_margin = gr['pres_margin']
                                g_margin_str = utils.lean_str(g_margin) if g_margin != '' else ''
                                new_rows.append([y, gr['abbr'], gr['D'], gr['R'], g_margin, g_margin_str, gr['ev']])
                        else:
                            # Fallback: just rename the state row to -AL for consistency
                            al_abbr = f"{state_po}-AL"
                            new_rows.append([y, al_abbr, D, R, margin, margin_str, ev])
                    else:
                        # No enhanced data; leave as-is but rename to -AL for consistency
                        al_abbr = f"{state_po}-AL"
                        new_rows.append([y, al_abbr, D, R, margin, margin_str, ev])
                else:
                    new_rows.append(row)
            return new_rows

        # First replace ME, then NE
        output_rows = replace_state(output_rows, 'ME')
        output_rows = replace_state(output_rows, 'NE')
            
    years = sorted(list(set(row[0] for row in output_rows)))

    # calculate distance from national average and relative margin delta
    for year in years:
        total_D = sum(row[2] for row in output_rows if row[0] == year)
        total_R = sum(row[3] for row in output_rows if row[0] == year)
        national_margin = (total_D - total_R) / (total_D + total_R) if (total_D + total_R) > 0 else 0.0

        for row in output_rows:
            if row[0] == year:
                state_margin = (row[2] - row[3]) / (row[2] + row[3]) if (row[2] + row[3]) > 0 else 0.0
                relative_margin = state_margin - national_margin
                row.append(relative_margin)  # Margin delta
                row.append(utils.lean_str(relative_margin))  # Margin delta string
                row.append(national_margin)
                row.append(utils.lean_str(national_margin))
    # sort output rows by abbr, year
    output_rows.sort(key=lambda x: (x[1], x[0]))
    # calculate relative margin delta
    for row in output_rows:
        year = row[0]
        state_po = row[1]
        if year > min(years):
            prev_year = year - 4
            prev_row = next((r for r in output_rows if r[0] == prev_year and r[1] == state_po), None)
            if prev_row:
                # relative margin delta = current relative_margin - previous relative_margin
                relative_margin_delta = row[7] - prev_row[7]  # relative_margin - previous relative_margin
                # Insert relative_margin_delta and its string between entries 8 and 9
                row.insert(9, relative_margin_delta)
                row.insert(10, utils.lean_str(relative_margin_delta))
                # national margin delta
                national_margin_delta = row[11] - prev_row[11]  # national_margin - previous national_margin
                row.append(national_margin_delta)
                row.append(utils.lean_str(national_margin_delta))
                # presidential margin delta (use pres_margin at index 4; not affected by prior inserts)
                try:
                    pres_margin_delta = row[4] - prev_row[4]
                except Exception:
                    pres_margin_delta = 0
                row.append(pres_margin_delta)
                row.append(utils.lean_str(pres_margin_delta))
            else:
                row.insert(9, '0')  # No previous year data
                row.insert(10, '0')
                # keep placeholders for national and pres deltas to maintain column count
                # national margin delta placeholders
                row.append('0')
                row.append('0')
                # presidential margin delta placeholders
                row.append('0')
                row.append('0')
        else:
            row.insert(9, '0')  # No previous year data
            row.insert(10, '0')
            # keep placeholders for national and pres deltas to maintain column count
            row.append('0')
            row.append('0')
            row.append('0')
            row.append('0')

    # Sort by state abbreviation
    output_rows.sort(key=lambda x: (x[1], x[0]))

    # Write to output CSV
    with open(output_file, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["year", "abbr", "D_votes", "R_votes", "pres_margin", "pres_margin_str", "electoral_votes", "relative_margin", "relative_margin_str", "relative_margin_delta", "relative_margin_delta_str", "national_margin", "national_margin_str", "national_margin_delta", "national_margin_delta_str", "pres_margin_delta", "pres_margin_delta_str"])
        writer.writerows(output_rows)

    print(f"Done! Output written to {output_file}")

if __name__ == "__main__":
    run_import_csv()