import os
import pandas as pd
import matplotlib.pyplot as plt
import utils

def calculate_pvi(start_year, end_year):
    # Ensure the PVIs folder exists
    os.makedirs("PVIs", exist_ok=True)
    # clear files in the PVIs directory
    for file in os.listdir("PVIs"):
        file_path = os.path.join("PVIs", file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Load data
    house_data = pd.read_csv("house_margins.csv")
    presidential_data = pd.read_csv("presidential_margins.csv")

    # Filter data for the relevant years
    house_data = house_data[(house_data['year'] >= max(start_year, 1976)) & (house_data['year'] <= min(end_year, 2024))]
    presidential_data = presidential_data[(presidential_data['year'] >= max(start_year, 1976)) & (presidential_data['year'] <= min(end_year, 2024))]

    # Group house data by state and calculate average relative margin for the last four elections
    house_data['cycle'] = house_data['year'] // 2 * 2  # Ensure even years
    house_averages = house_data.groupby(['abbr', 'cycle'])['relative_margin'].mean().reset_index()

    # Prepare results
    results = {}

    for year in range(max(start_year, 1984), min(end_year, 2024) + 1, 2):
        year_results = {}
        for state in presidential_data['abbr'].unique():
            # get the last presidential year 
            last_presidential_year = year if year % 4 == 0 else year - 2
            # Get presidential margins
            pres_margin_current = presidential_data[(presidential_data['year'] == last_presidential_year) & (presidential_data['abbr'] == state)]['relative_margin'].mean()
            pres_margin_previous = presidential_data[(presidential_data['year'] == last_presidential_year - 4) & (presidential_data['abbr'] == state)]['relative_margin'].mean()

            # Get house average margin
            house_margins = house_averages[(house_averages['cycle'] <= year) & (house_averages['cycle'] > year - 8) & (house_averages['abbr'] == state)]['relative_margin']
            house_margin_avg = house_margins.mean() if not house_margins.empty else 0

            # Calculate PVI
            pvi = 0.50 * (pres_margin_current if not pd.isna(pres_margin_current) else 0) + \
                  0.25 * (pres_margin_previous if not pd.isna(pres_margin_previous) else 0) + \
                  0.25 * house_margin_avg

            year_results[state] = pvi

        results[year] = year_results

    # Write results to a text file
    with open("PVIs/state_pvis.txt", "w") as f:
        for state in presidential_data['abbr'].unique():
            f.write(f"State: {state}\n")
            for year in sorted(results.keys()):
                pvi = results[year].get(state, "N/A")
                f.write(f"{year}: {utils.lean_str(pvi)}\n")
            f.write("\n")

    # Generate plots
    for state in presidential_data['abbr'].unique():
        state_pvis = {year: results[year].get(state, None) for year in results}
        years = list(state_pvis.keys())
        pvis = [float(pvi) for pvi in state_pvis.values()]

        plt.figure(figsize=(10, 6))
        plt.plot(years, pvis, marker='o', label=f"PVI ({state})")
        plt.title(f"PVI Over Time for {state}", color='white')
        plt.xlabel("Year", color='white')
        plt.ylabel("PVI", color='white')
        plt.grid(True, linestyle='--', alpha=0.6)
        # plot red dashed line at y=0
        plt.axhline(0, color='red', linestyle='--', linewidth=1, label='National Average')
        plt.xticks(years, rotation=45, color='white')
        plt.legend()
        plt.gca().set_facecolor("#2E2E2E")
        plt.gca().tick_params(colors='white')
        plt.gca().spines['bottom'].set_color('white')
        plt.gca().spines['left'].set_color('white')
        plt.gca().spines['top'].set_color('white')
        plt.gca().spines['right'].set_color('white')
        plt.savefig(f"PVIs/{state}_pvi_plot.png", facecolor="#2E2E2E")
        plt.close()

if __name__ == "__main__":
    import sys
    start_year = 1984
    end_year = 2024
    calculate_pvi(start_year, end_year)
