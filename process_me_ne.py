import csv
import os
from typing import List, Dict, Any, Tuple, DefaultDict
from collections import defaultdict

# Input and output paths
ROOT = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(ROOT, "ME_NE_District_Data.csv")
OUTPUT_FILE = os.path.join(ROOT, "ME_NE_enhanced.csv")
NATL_FILE = os.path.join(ROOT, "presidential_margins.csv")


def to_float(val: Any):
    if val is None:
        return None
    val = str(val).strip()
    if val == "":
        return None
    try:
        return float(val)
    except ValueError:
        # Try stripping any stray percent signs or commas just in case
        return float(val.replace("%", "").replace(",", ""))


def to_int_or_empty(val: Any):
    if val is None:
        return ""
    v = str(val).strip()
    if v == "":
        return ""
    try:
        # handle values like "1,234" or "1234.0"
        return int(float(v.replace(",", "")))
    except ValueError:
        return ""


def fmt(val: Any):
    if val == "" or val is None:
        return ""
    if isinstance(val, float):
        # 6 decimal places for normalized shares and margin
        return f"{val:.6f}"
    return str(val)


def main():
    rows: List[Dict[str, Any]] = []

    with open(INPUT_FILE, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            year_raw = (row.get("year") or "").strip()
            abbr = (row.get("abbr") or "").strip()
            d_raw = to_float(row.get("D_pct_raw"))
            r_raw = to_float(row.get("R_pct_raw"))
            d_votes = to_int_or_empty(row.get("D_votes"))
            r_votes = to_int_or_empty(row.get("R_votes"))

            # Prefer using explicit district vote counts when available.
            # If both D_votes and R_votes are integers, compute shares and margin from them.
            d_pct = r_pct = pres_margin = ""
            if isinstance(d_votes, int) and isinstance(r_votes, int):
                denom_votes = d_votes + r_votes
                if denom_votes > 0:
                    d_pct = d_votes / denom_votes
                    r_pct = r_votes / denom_votes
                    pres_margin = d_pct - r_pct
            else:
                # Fallback: normalize from raw percentage columns if present
                if d_raw is not None and r_raw is not None:
                    denom = d_raw + r_raw
                    if denom > 0:
                        d_pct = d_raw / denom
                        r_pct = r_raw / denom
                        pres_margin = d_pct - r_pct

            # Parse year for sorting but keep original for output
            try:
                year_num = int(year_raw)
            except ValueError:
                # Skip malformed year rows
                continue

            rows.append({
                "year": year_num,
                "abbr": abbr,
                "D_pct_raw": d_raw if d_raw is not None else "",
                "R_pct_raw": r_raw if r_raw is not None else "",
                "D_votes": d_votes,
                "R_votes": r_votes,
                "D_pct": d_pct,
                "R_pct": r_pct,
                "pres_margin": pres_margin,
            })

    # Build maps from presidential_margins.csv
    natl_margin_by_year: Dict[int, float] = {}
    state_totals: Dict[Tuple[int, str], Dict[str, Any]] = {}
    try:
        with open(NATL_FILE, "r", newline="", encoding="utf-8") as nf:
            nreader = csv.DictReader(nf)
            for nrow in nreader:
                try:
                    y = int((nrow.get("year") or "").strip())
                except ValueError:
                    continue
                nm = nrow.get("national_margin")
                if nm is None or nm == "":
                    continue
                try:
                    nmf = float(str(nm))
                except ValueError:
                    continue
                # Only set if not already present; first seen per year is fine
                if y not in natl_margin_by_year:
                    natl_margin_by_year[y] = nmf
                # Collect state totals for ME and NE (and generally any state if needed)
                abbr = (nrow.get("abbr") or "").strip()
                if not abbr:
                    continue
                try:
                    dv = int((nrow.get("D_votes") or "").replace(",", ""))
                    rv = int((nrow.get("R_votes") or "").replace(",", ""))
                except ValueError:
                    dv = rv = None
                pm_raw = nrow.get("pres_margin")
                try:
                    pm = float(pm_raw) if pm_raw not in (None, "") else None
                except ValueError:
                    pm = None
                state_totals[(y, abbr)] = {
                    "D_votes": dv,
                    "R_votes": rv,
                    "pres_margin": pm,
                }
    except FileNotFoundError:
        # If file not found, leave dict empty; downstream will leave blanks
        pass

    # Add electoral_votes = 1 to every district row and attach national_margin
    for r in rows:
        r["electoral_votes"] = 1
        nm = natl_margin_by_year.get(r["year"])  # may be None
        r["national_margin"] = nm if nm is not None else ""

    # Group by (state, year)
    by_state_year: DefaultDict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        state = r["abbr"].split("-")[0]
        if state in ("ME", "NE"):
            by_state_year[(state, r["year"])].append(r)

    # Track whether all districts originally had concrete votes per (state, year)
    all_concrete_original: Dict[Tuple[str, int], bool] = {}
    for (state, year), drows in by_state_year.items():
        all_concrete_original[(state, year)] = all(
            isinstance(dr.get("D_votes"), int) and isinstance(dr.get("R_votes"), int)
            for dr in drows
        )

    # Distribute missing district votes using state totals while matching margins as closely as possible
    def distribute_votes(state: str, year: int, drows: List[Dict[str, Any]]):
        st = state_totals.get((year, state))
        if not st or st.get("D_votes") is None or st.get("R_votes") is None:
            return  # Can't distribute without state totals

        D_total = st["D_votes"]
        R_total = st["R_votes"]

        # Sum existing concrete votes
        D_exist = sum((dr.get("D_votes") or 0) for dr in drows if isinstance(dr.get("D_votes"), int))
        R_exist = sum((dr.get("R_votes") or 0) for dr in drows if isinstance(dr.get("R_votes"), int))

        D_rem = D_total - D_exist
        R_rem = R_total - R_exist

        # Identify rows needing allocation
        need = [dr for dr in drows if not (isinstance(dr.get("D_votes"), int) and isinstance(dr.get("R_votes"), int))]
        if not need:
            return

        # Guard against negative remainder
        if D_rem < 0 or R_rem < 0:
            return

        # Build weights from target shares (d_pct and r_pct)
        d_weights = []
        r_weights = []
        for dr in need:
            d_share = dr.get("D_pct") if isinstance(dr.get("D_pct"), float) else None
            r_share = dr.get("R_pct") if isinstance(dr.get("R_pct"), float) else None
            if d_share is None or r_share is None or d_share + r_share <= 0:
                d_share = 0.5
                r_share = 0.5
            d_weights.append(max(0.0, float(d_share)))
            r_weights.append(max(0.0, float(r_share)))

        sum_dw = sum(d_weights)
        sum_rw = sum(r_weights)
        if sum_dw <= 0:
            d_weights = [1.0] * len(need)
            sum_dw = float(len(need))
        if sum_rw <= 0:
            r_weights = [1.0] * len(need)
            sum_rw = float(len(need))

        # Largest remainder allocation for D and R
        def allocate(total: int, weights: List[float]) -> List[int]:
            if total <= 0:
                return [0] * len(weights)
            s = sum(weights)
            floats = [total * w / s for w in weights]
            bases = [int(x) for x in map(float, map(lambda v: int(v), floats))]  # temporary placeholder
            # Correct bases to floor properly and track remainders
            bases = [int(x) for x in floats]
            remainders = [fx - bx for fx, bx in zip(floats, bases)]
            short = total - sum(bases)
            order = sorted(range(len(weights)), key=lambda i: (-remainders[i], i))
            for i in range(short):
                bases[order[i % len(weights)]] += 1
            return bases

        D_alloc = allocate(D_rem, d_weights)
        R_alloc = allocate(R_rem, r_weights)

        for dr, d_add, r_add in zip(need, D_alloc, R_alloc):
            dr["D_votes"] = d_add
            dr["R_votes"] = r_add

    # Apply distribution for ME and NE groups
    for (state, year), drows in by_state_year.items():
        distribute_votes(state, year, drows)

    al_rows: List[Dict[str, Any]] = []
    for (state, year), drows in by_state_year.items():
        # Decide AL based on original concreteness
        originally_all_concrete = all_concrete_original.get((state, year), False)

        if originally_all_concrete and len(drows) > 0:
            # Sum concrete district votes
            D_sum = sum(dr.get("D_votes", 0) or 0 for dr in drows)
            R_sum = sum(dr.get("R_votes", 0) or 0 for dr in drows)
            denom = D_sum + R_sum
            if denom > 0:
                d_pct = D_sum / denom
                r_pct = 1.0 - d_pct
                pres_margin = d_pct - r_pct
                d_raw = d_pct * 100.0
                r_raw = r_pct * 100.0
            else:
                d_pct = r_pct = pres_margin = ""
                d_raw = r_raw = ""
            D_votes = D_sum if denom > 0 else ""
            R_votes = R_sum if denom > 0 else ""
        else:
            # Use exact state info from presidential_margins.csv
            st = state_totals.get((year, state))
            if st and st.get("D_votes") is not None and st.get("R_votes") is not None:
                D_votes = int(st["D_votes"])  # state totals
                R_votes = int(st["R_votes"])  # state totals
                denom = D_votes + R_votes
                if denom > 0:
                    d_pct = D_votes / denom
                    r_pct = 1.0 - d_pct
                    d_raw = d_pct * 100.0
                    r_raw = r_pct * 100.0
                else:
                    d_pct = r_pct = d_raw = r_raw = ""
                pres_margin = st.get("pres_margin") if st.get("pres_margin") is not None else (d_pct - r_pct if isinstance(d_pct, float) and isinstance(r_pct, float) else "")
            else:
                # Fallback to averaging shares if state totals missing
                valid = [dr for dr in drows if isinstance(dr.get("D_pct"), float) and isinstance(dr.get("R_pct"), float)]
                if valid:
                    d_pct = sum(dr["D_pct"] for dr in valid) / len(valid)
                    r_pct = 1.0 - d_pct
                    pres_margin = d_pct - r_pct
                    d_raw = d_pct * 100.0
                    r_raw = r_pct * 100.0
                else:
                    d_pct = r_pct = pres_margin = d_raw = r_raw = ""
                D_votes = R_votes = ""

        abbr_al = f"{state}-AL"
        nm = natl_margin_by_year.get(year)
        al_rows.append({
            "year": year,
            "abbr": abbr_al,
            "D_pct_raw": d_raw if d_raw != "" else "",
            "R_pct_raw": r_raw if r_raw != "" else "",
            "D_votes": D_votes,
            "R_votes": R_votes,
            "D_pct": d_pct,
            "R_pct": r_pct,
            "pres_margin": pres_margin,
            "electoral_votes": 2,
            "national_margin": nm if nm is not None else "",
        })

    # Combine rows
    rows.extend(al_rows)

    # Compute relative margins and deltas per abbr
    # First sort by abbr, year
    rows.sort(key=lambda x: (x["abbr"], x["year"]))

    # Attach relative_margin now that national_margin is present
    for r in rows:
        nm = r.get("national_margin")
        if isinstance(r.get("pres_margin"), float) and isinstance(nm, float):
            r["relative_margin"] = r["pres_margin"] - nm
        else:
            r["relative_margin"] = ""

    # Compute deltas
    by_abbr: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_abbr[r["abbr"]].append(r)

    for abbr, rlist in by_abbr.items():
        rlist.sort(key=lambda x: x["year"])  # ensure chronological
        prev_rel = None
        for r in rlist:
            rel = r.get("relative_margin")
            if isinstance(rel, float) and isinstance(prev_rel, float):
                r["relative_margin_delta"] = rel - prev_rel
            else:
                # No previous cycle or missing values
                r["relative_margin_delta"] = 0 if isinstance(rel, float) else ""
            prev_rel = rel if isinstance(rel, float) else prev_rel

    fieldnames = [
        "year",
        "abbr",
        "electoral_votes",
        "D_pct_raw",
        "R_pct_raw",
        "D_votes",
        "R_votes",
        "D_pct",
        "R_pct",
        "pres_margin",
        "national_margin",
        "relative_margin",
        "relative_margin_delta",
    ]

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: fmt(r[k]) for k in fieldnames})

    print(f"Wrote {len(rows)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
