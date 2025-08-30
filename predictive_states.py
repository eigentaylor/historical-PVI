import os
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt


OUT_DIR = "predictive_states"

EC_winners = {
    1976: "D",
    1980: "R",
    1984: "R",
    1988: "R",
    1992: "D",
    1996: "D",
    2000: "R",
    2004: "R",
    2008: "D",
    2012: "D",
    2016: "R",
    2020: "D",
    2024: "R",
}

# When True, read the longer 1900-2024 file under data/ and use its columns
USE_ALL_YEARS = True
ALL_YEARS_CSV = os.path.join("data", "1900_2024_election_results.csv")
STATE_KEY = "state_po" if USE_ALL_YEARS else "abbr"
# Minimum streak length (in elections) to list in state_streaks.txt
STREAK_MIN_LENGTH = 4

TOTAL_ELECTIONS = (2024 - 1900) // 4 + 1 if USE_ALL_YEARS else (2024 - 1976) // 4 + 1

def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def load_data(csv_path="presidential_margins.csv"):
    if USE_ALL_YEARS:
        df = pd.read_csv(ALL_YEARS_CSV)
        # normalize column names: expect year, D_votes, R_votes, electoral_votes, overall_winner, winner_state
        if "D_votes" in df.columns:
            df["D_votes"] = pd.to_numeric(df["D_votes"], errors="coerce").fillna(0).astype(int)
        else:
            df["D_votes"] = 0
        if "R_votes" in df.columns:
            df["R_votes"] = pd.to_numeric(df["R_votes"], errors="coerce").fillna(0).astype(int)
        else:
            df["R_votes"] = 0
        if "electoral_votes" in df.columns:
            df["electoral_votes"] = pd.to_numeric(df["electoral_votes"], errors="coerce").fillna(0).astype(int)
        else:
            df["electoral_votes"] = 0
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
        # winner_state might be boolean or string; normalize to bool
        if "winner_state" in df.columns:
            df["winner_state"] = df["winner_state"].astype(str).str.lower().isin(["true", "1", "t", "yes"])
        else:
            df["winner_state"] = False
        # overall_winner: normalize to 'D' or 'R' when present
        if "overall_winner" in df.columns:
            def norm_overall(x):
                try:
                    s = str(x).strip().upper()
                    if "D" in s:
                        return "D"
                    if "R" in s:
                        return "R"
                except Exception:
                    pass
                return None

            df["overall_winner"] = df["overall_winner"].apply(norm_overall)
        else:
            df["overall_winner"] = None

        # When reading the long file, ignore congressional-district rows for
        # Maine and Nebraska (e.g. 'ME-1', 'NE-2'). We only want the state-level
        # rows 'ME' and 'NE' here.
        if STATE_KEY in df.columns:
            df[STATE_KEY] = df[STATE_KEY].astype(str)
            mask = ~df[STATE_KEY].str.contains(r'^(NE|ME)-', regex=True, na=False)
            df = df[mask].copy()

        return df

    # default path (presidential_margins.csv)
    df = pd.read_csv(csv_path)
    # ensure numeric
    df["D_votes"] = pd.to_numeric(df["D_votes"], errors="coerce").fillna(0).astype(int)
    df["R_votes"] = pd.to_numeric(df["R_votes"], errors="coerce").fillna(0).astype(int)
    df["electoral_votes"] = pd.to_numeric(df["electoral_votes"], errors="coerce").fillna(0).astype(int)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    if "national_margin" in df.columns:
        df["national_margin"] = pd.to_numeric(df["national_margin"], errors="coerce").fillna(0.0)
    else:
        df["national_margin"] = 0.0
    return df


def compute_year_winners(df):
    years = sorted(df["year"].unique())
    pv_winner = {}
    ec_winner = {}

    for y in years:
        sub = df[df["year"] == y]
        if USE_ALL_YEARS:
            # overall_winner (from file) is the EC winner for the year
            overall = sub["overall_winner"].iloc[0] if "overall_winner" in sub.columns else None
            ec_winner[y] = overall
            # PV winner: user specified exceptions 2000 and 2016 where PV winner differs
            if y in (2000, 2016):
                if overall == "D":
                    pv_winner[y] = "R"
                elif overall == "R":
                    pv_winner[y] = "D"
                else:
                    pv_winner[y] = None
            else:
                pv_winner[y] = overall
        else:
            # popular vote winner: sign of national_margin (should be same for all rows)
            nat_margin = float(sub["national_margin"].iloc[0])
            pv_winner[y] = "D" if nat_margin > 0 else "R"

            # electoral college: sum electoral votes by which candidate carried each state
            # determine state winners
            sub = sub.copy()
            sub["state_winner"] = sub.apply(lambda r: "D" if r["D_votes"] > r["R_votes"] else "R", axis=1)
            ev_by = sub.groupby("state_winner")["electoral_votes"].sum().to_dict()
            d_ev = ev_by.get("D", 0)
            r_ev = ev_by.get("R", 0)
            if d_ev > r_ev:
                ec_winner[y] = "D"
            elif r_ev > d_ev:
                ec_winner[y] = "R"
            else:
                ec_winner[y] = None

    return pv_winner, ec_winner


def tally_state_matches(df, pv_winner, ec_winner):
    states = sorted(df[STATE_KEY].unique())
    years = sorted(df["year"].unique())

    pv_matches = {s: [] for s in states}
    pv_misses = {s: [] for s in states}
    ec_matches = {s: [] for s in states}
    ec_misses = {s: [] for s in states}

    for y in [int(year) for year in years]:
        sub = df[df["year"] == y].set_index(STATE_KEY)
        for s in states:
            if s not in sub.index:
                continue
            row = sub.loc[s]
            # determine which candidate the state voted for in that year
            if "D_votes" in row and "R_votes" in row:
                state_winner = "D" if int(row["D_votes"]) > int(row["R_votes"]) else "R"
            elif "pres_margin" in row:
                state_winner = "D" if row["pres_margin"] > 0 else "R"
            else:
                # fallback
                state_winner = "D"

            # PV: for non-ALL_YEARS compare state_winner to pv_winner here.
            # For ALL_YEARS, PV handling is performed later (uses party_win/overall rules).
            if not USE_ALL_YEARS:
                if pv_winner[y] is not None and state_winner == pv_winner[y]:
                    pv_matches[s].append(y)
                else:
                    pv_misses[s].append(y)

            # If using the long CSV, determine EC match via party_win == overall_winner
            if USE_ALL_YEARS:
                party = None
                if "party_win" in sub.columns:
                    party = sub.loc[s, "party_win"]
                overall_ec = ec_winner.get(y)

                # EC: state voted for EC winner if party_win equals overall EC winner
                if party is not None and overall_ec is not None and str(party).strip().upper() == str(overall_ec).upper():
                    ec_matches[s].append(y)
                else:
                    ec_misses[s].append(y)

                # PV: normally same as EC (party == overall), except 2000 and 2016 where PV winner was 'D'
                if y in (2000, 2016):
                    if s == "NV":
                        pass
                    if party is not None and str(party).strip().upper() == "D":
                        pv_matches[s].append(y)
                    else:
                        pv_misses[s].append(y)
                else:
                    if party is not None and overall_ec is not None and str(party).strip().upper() == str(overall_ec).upper():
                        pv_matches[s].append(y)
                    else:
                        pv_misses[s].append(y)
            else:
                # prefer the explicit EC_winners mapping when available, fall back to computed ec_winner
                overall_ec = EC_winners.get(y, ec_winner.get(y))
                if overall_ec is not None and state_winner == overall_ec:
                    ec_matches[s].append(y)
                else:
                    ec_misses[s].append(y)

    return pv_matches, pv_misses, ec_matches, ec_misses


def plot_counts(pv_matches, ec_matches):
    # counts per state
    import matplotlib

    states = sorted(pv_matches.keys())
    pv_counts = {s: len(pv_matches[s]) for s in states}
    ec_counts = {s: len(ec_matches[s]) for s in states}

    # Sort separately; create two horizontal bar charts (1x2)
    pv_sorted = sorted(pv_counts.items(), key=lambda x: x[1], reverse=True)
    ec_sorted = sorted(ec_counts.items(), key=lambda x: x[1], reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 12))

    pv_states, pv_vals = zip(*pv_sorted)
    ec_states, ec_vals = zip(*ec_sorted)

    axes[0].barh(pv_states, pv_vals, color="tab:blue")
    axes[0].invert_yaxis()
    axes[0].set_title("Times state voted for Popular Vote winner")
    axes[0].set_xlabel("Count")
    axes[0].set_xticks(range(0, max(pv_vals) + 1, 1))

    axes[1].barh(ec_states, ec_vals, color="tab:green")
    axes[1].invert_yaxis()
    axes[1].set_title("Times state voted for Electoral College winner")
    axes[1].set_xlabel("Count")
    axes[1].set_xticks(range(0, max(ec_vals) + 1, 1))

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "predictive_match_counts.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def write_summary(states, pv_matches, pv_misses, ec_matches, ec_misses):
    # Sort states by PV right count, then EC right count (both descending)
    sorted_states = sorted(
        states,
        key=lambda s: (len(pv_matches[s]), len(ec_matches[s])),
        reverse=True,
    )

    path = os.path.join(OUT_DIR, "state_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Summary of state matches across {TOTAL_ELECTIONS} total elections\n\n")
        for s in sorted_states:
            f.write(f"STATE {s}\n")
            f.write(f"PV: right {len(pv_matches[s])}: {pv_matches[s]}\n")
            f.write(f"PV: wrong {len(pv_misses[s])}: {pv_misses[s]}\n")
            f.write(f"EC: right {len(ec_matches[s])}: {ec_matches[s]}\n")
            f.write(f"EC: wrong {len(ec_misses[s])}: {ec_misses[s]}\n")
            f.write("\n")
    return path


def consecutive_ranges(years_list):
    if not years_list:
        return []
    years = sorted(years_list)
    ranges = []
    start = years[0]
    end = years[0]
    for y in years[1:]:
        # elections occur every 4 years, treat years separated by 4 as consecutive
        if y == end + 4:
            end = y
        else:
            ranges.append((start, end))
            start = y
            end = y
    ranges.append((start, end))
    return ranges


def find_longest_streak(years_list):
    ranges = consecutive_ranges(years_list)
    if not ranges:
        return 0, []
    max_len = 0
    best = []
    for (a, b) in ranges:
        length = (b - a) // 4 + 1
        if length > max_len:
            max_len = length
            best = [(int(a), int(b))]
        elif length == max_len:
            best.append((int(a), int(b)))
    return max_len, best


def list_all_streaks(years_list):
    """Return list of streaks as tuples (start_year, end_year, length_in_elections).
    Consecutive elections are 4 years apart.
    """
    ranges = consecutive_ranges(sorted(set(years_list)))
    result = []
    for (a, b) in ranges:
        length = (b - a) // 4 + 1
        result.append((int(a), int(b), int(length)))
    return result


def write_streaks(states, pv_matches, ec_matches, min_len=STREAK_MIN_LENGTH):
    """Write all streaks of length >= min_len (in elections) to the output file.
    The file lists PV and EC streaks per state and includes the start/end years and length.
    """
    # Prepare and sort states by the sum of all qualifying streaks (PV + EC)
    # where qualifying means length >= min_len (measured in elections).
    sortable = []
    for s in states:
        pv_streaks_all = list_all_streaks(pv_matches[s])
        ec_streaks_all = list_all_streaks(ec_matches[s])
        pv_sum = sum(st[2] for st in pv_streaks_all if st[2] >= min_len)
        ec_sum = sum(st[2] for st in ec_streaks_all if st[2] >= min_len)
        sortable.append((s, pv_sum, ec_sum))
    # sort by EC then PV descending
    sortable.sort(key=lambda x: (x[2], x[1]), reverse=True)

    path = os.path.join(OUT_DIR, "state_streaks.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Listing all streaks with length >= {min_len} elections across {TOTAL_ELECTIONS} total elections\n\n")
        for s, _, _ in sortable:
            f.write(f"STATE {s}\n")

            pv_streaks = [st for st in list_all_streaks(pv_matches[s]) if st[2] >= min_len]
            if pv_streaks:
                f.write(f"PV streaks (total {sum(st[2] for st in pv_streaks)}) (start,end,length): {pv_streaks}\n")
            else:
                f.write(f"PV streaks (>= {min_len}): none\n")

            ec_streaks = [st for st in list_all_streaks(ec_matches[s]) if st[2] >= min_len]
            if ec_streaks:
                f.write(f"EC streaks (total {sum(st[2] for st in ec_streaks)}) (start,end,length): {ec_streaks}\n")
            else:
                f.write(f"EC streaks (>= {min_len}): none\n")

            f.write("\n")
    return path


def main():
    ensure_out_dir()
    df = load_data()
    pv_winner, ec_winner = compute_year_winners(df)
    pv_matches, pv_misses, ec_matches, ec_misses = tally_state_matches(df, pv_winner, ec_winner)
    states = sorted(pv_matches.keys())

    plot_path = plot_counts(pv_matches, ec_matches)
    summary_path = write_summary(states, pv_matches, pv_misses, ec_matches, ec_misses)
    streaks_path = write_streaks(states, pv_matches, ec_matches)

    print(f"Wrote plot: {plot_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote streaks: {streaks_path}")


if __name__ == "__main__":
    main()
