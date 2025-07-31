import pandas as pd
import numpy as np
from utils import lean_str
import argparse

def run_fourier_future(end_year=2048, REDO_EACH_STEP=False, INCLUDE_HISTORICAL=True, USE_FFT2=False):
    # -----------------------------
    # Parameters
    # -----------------------------
    future_steps = (end_year - 2024) // 4  # Number of future elections (ex. 2028–2048)

    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_csv("presidential_margins.csv")
    years = sorted(df["year"].unique())
    states = sorted(df["abbr"].unique())

    # Pivot relative margins: rows = years, columns = states
    pivot_rel = df.pivot(index="year", columns="abbr", values="relative_margin").sort_index()
    # National margin time series
    nat_margin_hist = (
        df.drop_duplicates("year").sort_values("year")["national_margin"].values
    )


    # -----------------------------
    # FFT-based extension function
    # -----------------------------
    def fft_extend(ts, future_steps, redo_each_step=False):
        N = len(ts)
        if not redo_each_step:
            freq = np.fft.fft(ts)
            freq_pad = np.concatenate([freq, np.zeros(future_steps)])
            ts_extended = np.fft.ifft(freq_pad).real
            return ts_extended[N:]
        else:
            # Iteratively extend one step at a time, redoing FFT each time
            ts_current = ts.copy()
            extended = []
            for _ in range(future_steps):
                freq = np.fft.fft(ts_current)
                freq_pad = np.concatenate([freq, np.zeros(1)])
                ts_next = np.fft.ifft(freq_pad).real
                next_val = ts_next[-1]
                extended.append(next_val)
                ts_current = np.append(ts_current, next_val)
            return np.array(extended)

    # -----------------------------
    # FFT2-based extension function
    # -----------------------------
    def fft2_extend(data, future_steps):
        N, M = data.shape
        freq = np.fft.fft2(data)
        freq_pad = np.pad(freq, ((0, future_steps), (0, 0)), mode='constant')
        data_extended = np.fft.ifft2(freq_pad).real
        return data_extended[N:, :]

    # -----------------------------
    # Extend national margin and relative margins
    # -----------------------------
    if USE_FFT2:
        # Combine relative margins and national margin into a single matrix
        nat_margin_hist_np = np.array(nat_margin_hist)  # Ensure it's a NumPy array
        combined_data = np.hstack([pivot_rel.values, nat_margin_hist_np.reshape(-1, 1)])

        if REDO_EACH_STEP:
            # Iteratively extend one step at a time, redoing FFT2 each time
            extended_data = combined_data.copy()
            for _ in range(future_steps):
                freq = np.fft.fft2(extended_data)
                freq_pad = np.pad(freq, ((0, 1), (0, 0)), mode='constant')
                next_step = np.fft.ifft2(freq_pad).real[-1:, :]
                extended_data = np.vstack([extended_data, next_step])
            extended_data = extended_data[-future_steps:, :]
        else:
            # Extend all at once
            extended_data = fft2_extend(combined_data, future_steps)

        # Split extended data back into relative margins and national margin
        rel_future = {state: extended_data[:, i] for i, state in enumerate(states)}
        nat_future = extended_data[:, -1]
    else:
        nat_future = fft_extend(nat_margin_hist, future_steps, redo_each_step=REDO_EACH_STEP)
        rel_future = {}
        for state in states:
            ts = pivot_rel[state].values
            rel_future[state] = fft_extend(ts, future_steps, redo_each_step=REDO_EACH_STEP)

    # -----------------------------
    # Ensure future_years is defined before use
    # -----------------------------
    if end_year is not None:
        # Calculate how many future elections to reach end_year
        last_hist_year = years[-1]
        future_steps = (end_year - last_hist_year) // 4
        future_years = [last_hist_year + 4 * (i + 1) for i in range(future_steps)]
    else:
        future_steps = 6
        future_years = [years[-1] + 4 * (i + 1) for i in range(future_steps)]

    # -----------------------------
    # Compute pres_margin for each state/year
    # -----------------------------
    df_list = []
    for i, year in enumerate(future_years):
        nat = nat_future[i]
        for state in states:
            rel = rel_future[state][i]
            pres = rel + nat
            df_list.append({
                "year": year,
                "abbr": state,
                "pres_margin": pres,
                "pres_margin_str": lean_str(pres),
                "relative_margin": rel,
                "relative_margin_str": lean_str(rel),
                "national_margin": nat,
                "national_margin_str": lean_str(nat)
            })
    df_final = pd.DataFrame(df_list)

    # -----------------------------
    # Add electoral votes
    # -----------------------------
    ev_2024 = df[df["year"] == 2024].set_index("abbr")["electoral_votes"].to_dict()
    df_final["electoral_votes"] = df_final["abbr"].map(ev_2024)

    # -----------------------------
    # Optionally include historical data
    # -----------------------------
    if INCLUDE_HISTORICAL:
        hist_cols = [
            "year", "abbr", "pres_margin", "pres_margin_str", "relative_margin", "relative_margin_str", "national_margin", "national_margin_str", "electoral_votes"
        ]
        df_hist = df[df["year"] <= 2024][hist_cols]
        df_combined = pd.concat([df_hist, df_final], ignore_index=True)
        df_combined = df_combined.sort_values(["abbr", "year"])
        df_combined.to_csv("presidential_margins_future.csv", index=False)
        print(f"Saved to presidential_margins_future.csv ✅ (FFT, with historical data, end_year={end_year})")
    else:
        df_final = df_final.sort_values(["year", "abbr"])
        df_final.to_csv("presidential_margins_future.csv", index=False)
        print(f"Saved to presidential_margins_future.csv ✅ (FFT, end_year={end_year})")
