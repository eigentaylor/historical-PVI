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
            evs_current = presidential_data[(presidential_data['year'] == last_presidential_year) & (presidential_data['abbr'] == state)]['electoral_votes'].mean()
            pres_margin_current = presidential_data[(presidential_data['year'] == last_presidential_year) & (presidential_data['abbr'] == state)]['relative_margin'].mean()
            pres_margin_previous = presidential_data[(presidential_data['year'] == last_presidential_year - 4) & (presidential_data['abbr'] == state)]['relative_margin'].mean()

            # Get house average margin
            house_margins = house_averages[(house_averages['cycle'] <= year) & (house_averages['cycle'] > year - 8) & (house_averages['abbr'] == state)]['relative_margin']
            house_margin_avg = house_margins.mean() if not house_margins.empty else 0

            # Calculate PVI
            pvi = 0.50 * (pres_margin_current if not pd.isna(pres_margin_current) else 0) + \
                  0.25 * (pres_margin_previous if not pd.isna(pres_margin_previous) else 0) + \
                  0.25 * house_margin_avg

            year_results[state] = (pvi, evs_current)

        results[year] = year_results

    # Write results to a text file
    with open("PVIs/state_pvis.txt", "w") as f:
        for state in presidential_data['abbr'].unique():
            f.write(f"State: {state}\n")
            for year in sorted(results.keys()):
                pvi = results[year].get(state, "N/A")[0]
                evs_current = int(results[year].get(state, "N/A")[1])
                f.write(f"{year}: {utils.lean_str(pvi)} (EVs: {evs_current})\n")
            f.write("\n")

    # Create rankings folder
    rankings_folder = "PVIs/rankings"
    os.makedirs(rankings_folder, exist_ok=True)
    # clear files in the rankings directory
    for file in os.listdir(rankings_folder):
        file_path = os.path.join(rankings_folder, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Generate rankings for each year
    for year in results:
        rankings = []
        for state, pvi in results[year].items():
            rankings.append((state, pvi))

        # Sort rankings by PVI value (descending for D, ascending for R)
        rankings.sort(key=lambda x: x[1][0])
        EV_dot = 0.0
        # Write rankings to a text file
        with open(f"{rankings_folder}/{year}_rankings.txt", "w") as f:
            # Add a summary line
            f.write(f"\nSUMMARY:\n")
            # write the state with the smallest magnitude PVI
            min_state = min(rankings, key=lambda x: abs(x[1][0]))
            f.write(f"State with smallest magnitude PVI: {min_state[0]} {utils.lean_str(min_state[1][0])} (EVs: {int(min_state[1][1])})\n")
            f.write(f"EVs dot PVI: {EV_dot:.2f}\n\n")
            
            for state, pvi in rankings:
                f.write(f"{state} {utils.lean_str(pvi[0])} (EVs: {int(pvi[1])})\n")
                EV_dot += pvi[0] * pvi[1]

    # Generate plots
    for state in presidential_data['abbr'].unique():
        state_pvis = {year: results[year].get(state, None) for year in results}
        years = list(state_pvis.keys())
        pvis = [float(pvi[0]) for pvi in state_pvis.values()]

        plt.figure(figsize=(10, 6), dpi=300)
        plt.plot(years, pvis, marker='o', label=f"PVI ({state})")
        plt.title(f"PVI Over Time for {state}", color='white')
        plt.xlabel("Year", color='white')
        plt.ylabel("PVI", color='white')
        plt.grid(True, linestyle='--', alpha=0.6)
        # plot red dashed line at y=0
        plt.axhline(0, color='red', linestyle='--', linewidth=1, label='National Average')
        plt.xticks(years, rotation=45, color='white')
        
        # Update y-axis tick labels to use utils.lean_str
        y_vals = plt.gca().get_yticks()
        plt.gca().set_yticklabels([utils.lean_str(y_val) for y_val in y_vals], color='white')

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
