import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import os
import matplotlib.pyplot as plt
from datetime import datetime
import shutil

END_YEAR = 2048

# Define the sine function for fitting
def sine_function(x, A, B, C, D):
    return A * np.sin(B * x + C) + D

def main():
    # Set the future years to predict (2028-END_YEAR in 4-year increments)
    future_years = list(range(2028, END_YEAR + 1, 4))
    print(f"Predicting for years: {future_years}")
    
    # Load the population data
    pop_data = pd.read_csv('data/2024_info.csv')
    population_dict = dict(zip(pop_data['abbreviation'], pop_data['Population']))
    
    # Load the existing presidential margins data
    pres_margins_df = pd.read_csv('presidential_margins.csv')
    
    # Create a backup of presidential_margins_future.csv if it exists
    if os.path.exists('presidential_margins_future.csv'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f'backups/presidential_margins_future_backup_{timestamp}.csv'
        os.makedirs(os.path.dirname(backup_file), exist_ok=True)
        shutil.copy('presidential_margins_future.csv', backup_file)
        print(f"Created backup at {backup_file}")
    
    # Get the list of states
    states = sorted(list(set(pres_margins_df['abbr'].unique())))
    print(f"Processing {len(states)} states")
    
    # Create a structure to hold our predictions
    predictions = []
    
    # Process each state
    for state in states:
        print(f"Processing state: {state}")
        
        # Get historical data for this state
        state_data = pres_margins_df[pres_margins_df['abbr'] == state].sort_values('year')
        
        # Extract years and relative margins
        years = state_data['year'].values
        relative_margins = state_data['relative_margin'].values
        
        # Get electoral votes for this state in 2024 (to use for future predictions)
        ev_2024 = state_data[state_data['year'] == 2024]['electoral_votes'].values[0] if 2024 in state_data['year'].values else 3
        
        if len(years) < 3:
            print(f"Not enough data points for state {state}, skipping")
            continue
        
        # Convert years to indices for fitting
        x_indices = np.arange(len(years))
        
        # Try to fit sine wave first
        try:
            # Fit the sine function to the data
            params, _ = curve_fit(sine_function, x_indices, relative_margins, p0=[0.1, 0.5, 0, np.mean(relative_margins)], 
                                 maxfev=10000)
            A, B, C, D = params
            
            # Generate the sine fit for existing years
            sine_fit = sine_function(x_indices, A, B, C, D)
            
            # Calculate the R-squared to see how good the fit is
            ss_tot = np.sum((relative_margins - np.mean(relative_margins))**2)
            ss_res = np.sum((relative_margins - sine_fit)**2)
            r_squared_sine = 1 - (ss_res / ss_tot)
            
            print(f"Sine fit R-squared for {state}: {r_squared_sine}")
            
            # If R-squared is too low, use linear fit instead
            if r_squared_sine < 0.5:
                raise Exception("Sine fit R-squared too low, using linear fit instead")
            
            # Use sine function to predict future values
            last_idx = len(years) - 1
            future_indices = np.array([last_idx + (fut_year - 2024)/4 + 1 for fut_year in future_years])
            future_values = sine_function(future_indices, A, B, C, D)
            
            use_sine = True
            print(f"Using sine fit for {state}")
            
        except Exception as e:
            print(f"Could not use sine fit for {state}: {e}")
            
            # Use linear fit instead
            z = np.polyfit(x_indices, relative_margins, 1)
            p = np.poly1d(z)
            
            # Calculate the R-squared for the linear fit
            linear_fit = p(x_indices)
            ss_tot = np.sum((relative_margins - np.mean(relative_margins))**2)
            ss_res = np.sum((relative_margins - linear_fit)**2)
            r_squared_linear = 1 - (ss_res / ss_tot)
            
            print(f"Linear fit R-squared for {state}: {r_squared_linear}")
            
            # Use linear function to predict future values
            last_idx = len(years) - 1
            future_indices = np.array([last_idx + (fut_year - 2024)/4 + 1 for fut_year in future_years])
            future_values = p(future_indices)
            
            use_sine = False
            print(f"Using linear fit for {state}")
        
        # Create entries for each future year
        for i, year in enumerate(future_years):
            relative_margin = future_values[i]           
            
            # Add this prediction to our list
            predictions.append({
                'year': year,
                'abbr': state,
                'relative_margin': relative_margin,
                'fit_type': 'sine' if use_sine else 'linear',
                'electoral_votes': ev_2024
            })
    
    # Extract national margin data only once per year (e.g., using California's row)
    national_margin_data = pres_margins_df[pres_margins_df['abbr'] == 'CA'][['year', 'national_margin']].dropna().sort_values('year')
    national_years = national_margin_data['year'].to_numpy()
    national_margins = national_margin_data['national_margin'].to_numpy()

    # Convert years to indices for fitting
    national_x_indices = np.arange(len(national_years))

    # Try to fit sine wave for national margin
    try:
        # Fit the sine function to the data
        national_params, _ = curve_fit(sine_function, national_x_indices, national_margins, 
                                       p0=[0.1, 0.5, 0, np.mean(national_margins)], maxfev=10000)
        A, B, C, D = national_params

        # Generate the sine fit for existing years
        national_sine_fit = sine_function(national_x_indices, A, B, C, D)

        # Calculate the R-squared to see how good the fit is
        ss_tot = np.sum((national_margins - np.mean(national_margins))**2)
        ss_res = np.sum((national_margins - national_sine_fit)**2)
        r_squared_sine = 1 - (ss_res / ss_tot)

        print(f"Sine fit R-squared for national margin: {r_squared_sine}")

        # If R-squared is too low, use FFT instead
        if r_squared_sine < 0.4:
            raise Exception("Sine fit R-squared too low, using FFT instead")

        # Use sine function to predict future national margins
        last_idx = len(national_years) - 1
        future_indices = np.array([last_idx + (fut_year - 2024)/4 + 1 for fut_year in future_years])
        future_national_margins = sine_function(future_indices, A, B, C, D)

        print("Using sine fit for national margin")

    except Exception as e:
        print(f"Could not use sine fit for national margin: {e}")

        # Use FFT as a fallback
        fft_coeffs = np.fft.rfft(national_margins)
        fft_freqs = np.fft.rfftfreq(len(national_margins), d=1)

        # Reconstruct the signal using the dominant frequencies
        dominant_freqs = np.argsort(np.abs(fft_coeffs))[-3:]  # Use top 3 frequencies
        fft_reconstructed = np.zeros_like(national_x_indices, dtype=float)
        for freq in dominant_freqs:
            fft_reconstructed += np.real(fft_coeffs[freq] * np.exp(2j * np.pi * fft_freqs[freq] * national_x_indices))

        # Predict future national margins using FFT
        future_national_margins = []
        for fut_idx in future_indices:
            value = np.sum([
                np.real(fft_coeffs[freq] * np.exp(2j * np.pi * fft_freqs[freq] * fut_idx))
                for freq in dominant_freqs
            ])
            future_national_margins.append(value)

        print("Using FFT for national margin")

    # Add national margin predictions to the predictions list
    for i, year in enumerate(future_years):
        national_margin = future_national_margins[i]
        for prediction in predictions:
            if prediction['year'] == year:
                prediction['national_margin'] = national_margin
    
    # Convert predictions to DataFrame
    if predictions:
        pred_df = pd.DataFrame(predictions)
        
        # Group by year to get all state predictions for each year
        for year in future_years:
            year_preds = pred_df[pred_df['year'] == year]
            
            # Get the population vector
            pop_vector = np.array([population_dict.get(state, 0) for state in year_preds['abbr']])
            total_pop = np.sum(pop_vector)
            pop_vector = pop_vector / total_pop  # Normalize
            
            # Get the extended margin vector
            margin_vector = year_preds['relative_margin'].values
            
            # Orthogonalize against population to get national margin estimate
            dot_product = np.sum(pop_vector * margin_vector)
            #national_margin = dot_product
            
            #print(f"Estimated national margin for {year}: {national_margin:.4f}")
            
            # Update the predictions with the orthogonalized values
            for idx, row in year_preds.iterrows():
                state = row['abbr']
                relative_margin = row['relative_margin'] - dot_product
                pres_margin = relative_margin + national_margin
                pres_margin = np.clip(pres_margin, -1, 1)  # Ensure margin is within [-1, 1]
                
                # Update the dataframe
                #pred_df.at[idx, 'national_margin'] = national_margin
                pred_df.at[idx, 'pres_margin'] = pres_margin
                
                # Add string representations
                pred_df.at[idx, 'pres_margin_str'] = f"{'D' if pres_margin > 0 else 'R'}+{abs(pres_margin)*100:.1f}"
                pred_df.at[idx, 'relative_margin_str'] = f"{'D' if relative_margin > 0 else 'R'}+{abs(relative_margin)*100:.1f}"
                pred_df.at[idx, 'national_margin_str'] = f"{'D' if national_margin > 0 else 'R'}+{abs(national_margin)*100:.1f}"
        
        # Now include historical data
        # First, let's create a DataFrame with all the data we need from the historical data
        historical_data = pres_margins_df.copy()
        
        # Combine historical and prediction data
        columns_to_keep = ['year', 'abbr', 'pres_margin', 'pres_margin_str', 'relative_margin', 
                           'relative_margin_str', 'national_margin', 'national_margin_str', 'electoral_votes']
        
        result_df = pd.concat([historical_data[columns_to_keep], pred_df[columns_to_keep]], ignore_index=True)
        
        # Sort by year and state
        result_df = result_df.sort_values(['year', 'abbr']).reset_index(drop=True)
        
        # Save to CSV
        result_df.to_csv('presidential_margins_future.csv', index=False)
        print(f"Successfully wrote predictions to presidential_margins_future.csv")
    else:
        print("No predictions were generated")
    
    # Extract national margin data
    national_margin_data = pres_margins_df[pres_margins_df['abbr'] == 'CA'][['year', 'national_margin']].dropna().sort_values('year')
    national_years = national_margin_data['year'].to_numpy()
    national_margins = national_margin_data['national_margin'].to_numpy()

    # Convert years to indices for fitting
    national_x_indices = np.arange(len(national_years))

    # Try to fit sine wave for national margin
    try:
        # Fit the sine function to the data
        national_params, _ = curve_fit(sine_function, national_x_indices, national_margins, 
                                       p0=[0.1, 0.5, 0, np.mean(national_margins)], maxfev=10000)
        A, B, C, D = national_params

        # Generate the sine fit for existing years
        national_sine_fit = sine_function(national_x_indices, A, B, C, D)

        # Calculate the R-squared to see how good the fit is
        ss_tot = np.sum((national_margins - np.mean(national_margins))**2)
        ss_res = np.sum((national_margins - national_sine_fit)**2)
        r_squared_sine = 1 - (ss_res / ss_tot)

        print(f"Sine fit R-squared for national margin: {r_squared_sine}")

        # If R-squared is too low, use FFT instead
        if r_squared_sine < 0.5:
            raise Exception("Sine fit R-squared too low, using FFT instead")

        # Use sine function to predict future national margins
        last_idx = len(national_years) - 1
        future_indices = np.array([last_idx + (fut_year - 2024)/4 + 1 for fut_year in future_years])
        future_national_margins = sine_function(future_indices, A, B, C, D)

        print("Using sine fit for national margin")

    except Exception as e:
        print(f"Could not use sine fit for national margin: {e}")

        # Use FFT as a fallback
        fft_coeffs = np.fft.rfft(national_margins)
        fft_freqs = np.fft.rfftfreq(len(national_margins), d=1)

        # Reconstruct the signal using the dominant frequencies
        dominant_freqs = np.argsort(np.abs(fft_coeffs))[-3:]  # Use top 3 frequencies
        fft_reconstructed = np.zeros_like(national_x_indices, dtype=float)
        for freq in dominant_freqs:
            fft_reconstructed += np.real(fft_coeffs[freq] * np.exp(2j * np.pi * fft_freqs[freq] * national_x_indices))

        # Predict future national margins using FFT
        future_national_margins = []
        for fut_idx in future_indices:
            value = np.sum([
                np.real(fft_coeffs[freq] * np.exp(2j * np.pi * fft_freqs[freq] * fut_idx))
                for freq in dominant_freqs
            ])
            future_national_margins.append(value)

        print("Using FFT for national margin")

    # Update predictions with national margin
    for i, year in enumerate(future_years):
        national_margin = future_national_margins[i]  # Ensure this is defined here
        year_preds = pred_df[pred_df['year'] == year]

        # Get the population vector
        pop_vector = np.array([population_dict.get(state, 0) for state in year_preds['abbr']])
        total_pop = np.sum(pop_vector)
        pop_vector = pop_vector / total_pop  # Normalize

        # Get the extended margin vector
        margin_vector = year_preds['relative_margin'].values

        # Orthogonalize against population to get state-level margins
        dot_product = np.sum(pop_vector * margin_vector)

        for idx, row in year_preds.iterrows():
            state = row['abbr']
            relative_margin = row['relative_margin'] - dot_product
            pres_margin = relative_margin + national_margin

            # Update the dataframe
            pred_df.at[idx, 'national_margin'] = national_margin
            pred_df.at[idx, 'pres_margin'] = pres_margin

            # Add string representations
            pred_df.at[idx, 'pres_margin_str'] = f"{'D' if pres_margin > 0 else 'R'}+{abs(pres_margin)*100:.1f}"
            pred_df.at[idx, 'relative_margin_str'] = f"{'D' if relative_margin > 0 else 'R'}+{abs(relative_margin)*100:.1f}"
            pred_df.at[idx, 'national_margin_str'] = f"{'D' if national_margin > 0 else 'R'}+{abs(national_margin)*100:.1f}"
        
        # Combine historical and prediction data
        columns_to_keep = ['year', 'abbr', 'pres_margin', 'pres_margin_str', 'relative_margin', 
                           'relative_margin_str', 'national_margin', 'national_margin_str', 'electoral_votes']
        
        result_df = pd.concat([historical_data[columns_to_keep], pred_df[columns_to_keep]], ignore_index=True)
        
        # Sort by year and state
        result_df = result_df.sort_values(['year', 'abbr']).reset_index(drop=True)
        
        # Save to CSV
        result_df.to_csv('presidential_margins_future.csv', index=False)
        print(f"Successfully wrote predictions to presidential_margins_future.csv")
    else:
        print("No predictions were generated")

if __name__ == "__main__":
    main()
