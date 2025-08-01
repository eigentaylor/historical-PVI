import os
import pandas as pd
import matplotlib.pyplot as plt
import utils

def calculate_pvi(start_year, end_year, pres_weight=0.5, last_pres_weight=0.25, house_weight=0.25):
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
            pvi = pres_weight * (pres_margin_current if not pd.isna(pres_margin_current) else 0) + \
                  last_pres_weight * (pres_margin_previous if not pd.isna(pres_margin_previous) else 0) + \
                  house_weight * house_margin_avg

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
            for state, pvi in rankings:
                EV_dot += pvi[0] * pvi[1]
            # Add a summary line
            f.write(f"\nSUMMARY:\n")
            # write the state with the smallest magnitude PVI
            min_state = min(rankings, key=lambda x: abs(x[1][0]))
            f.write(f"State with smallest magnitude PVI: {min_state[0]} {utils.lean_str(min_state[1][0])} (EVs: {int(min_state[1][1])})\n")
            f.write(f"EVs dot PVI: {EV_dot:.2f}\n\n")
            
            for state, pvi in rankings:
                f.write(f"{state} {utils.lean_str(pvi[0])} (EVs: {int(pvi[1])})\n")
                #EV_dot += pvi[0] * pvi[1]
    # create a csv for the 2024 PVIs
    pvi_2024 = {state: results[2024].get(state, (0, 0))[0] for state in presidential_data['abbr'].unique()}
    
    # Gather additional data for the 2024 CSV
    pvi_2024_data = []

    # Calculate the national margin for 2024
    national_margin_2024 = presidential_data[presidential_data['year'] == 2024]['national_margin'].mean()

    for state in presidential_data['abbr'].unique():
        pvi_value = pvi_2024.get(state, 0)
        state_data = presidential_data[(presidential_data['year'] == 2024) & (presidential_data['abbr'] == state)]
        
        # Extract relevant data
        margin_2024 = state_data['relative_margin'].mean() + national_margin_2024 if not state_data.empty else 0
        evs_2024 = state_data['electoral_votes'].mean().astype(int) if not state_data.empty else 0
        relative_margin_2024 = state_data['relative_margin'].mean()
        pvi_label = utils.lean_str(pvi_value)

        # Append to the data list
        pvi_2024_data.append({
            'abbreviation': state,
            'pvi_2024': pvi_value,
            '2024_margin': margin_2024,
            'evs_2024': evs_2024,
            '2024_relative_margin': relative_margin_2024,
            '2024_national_margin': national_margin_2024,
            'pvi_2024_label': pvi_label
        })

    # Create the DataFrame
    pvi_2024_df = pd.DataFrame(pvi_2024_data)

    # Save to CSV
    pvi_2024_df.to_csv("PVIs/state_pvi_2024.csv", index=False)

    # Generate plots
    for state in presidential_data['abbr'].unique():
        state_pvis = {year: results[year].get(state, None) for year in results}
        years = list(state_pvis.keys())
        pvis = [float(pvi[0]) for pvi in state_pvis.values()]

        # Determine colors for points based on the sign of the pres_margin of that year
        pres_margins = {}
        for year in years:
            df = presidential_data[(presidential_data['year'] == year) & (presidential_data['abbr'] == state)]
            if not df.empty and 'pres_margin' in df.columns:
                pres_margins[year] = df['pres_margin'].values[0]
            else:
                pres_margins[year] = 0  # or np.nan if you prefer
        colors = ['deepskyblue' if pres_margins[year] > 0 else 'red' for year in years if year % 4 == 0]

        plt.figure(figsize=(10, 6), dpi=300)
        plt.plot(years, pvis, color='green', linestyle='-', linewidth=1, label=f"PVI Line ({state})")
        plt.scatter([year for year in years if year % 4 == 0], [pvis[years.index(year)] for year in years if year % 4 == 0], color=colors, edgecolor='black', zorder=5, label=f"Pres. Election Results ({state})")

        plt.title(f"PVI Over Time for {state}", color='white')
        plt.xlabel("Year", color='white')
        plt.ylabel("PVI", color='white')
        plt.grid(True, linestyle='--', alpha=0.6)
        # plot purple dashed line at y=0
        plt.axhline(0, color='purple', linestyle='--', linewidth=1, label='National Average')
        plt.xticks(years, rotation=45, color='white')
        
        # Update y-axis tick labels to use utils.lean_str
        y_vals = plt.gca().get_yticks()
        plt.gca().set_yticklabels([utils.lean_str(y_val) if y_val != 0 else "NATIONAL AVG" for y_val in y_vals], color='white')

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
    calculate_pvi(start_year, end_year, pres_weight=0.96, last_pres_weight=0.03, house_weight=0.01)
