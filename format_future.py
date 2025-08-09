import pandas as pd
import numpy as np
from datetime import datetime

def save_predictions_to_csv(predictions, historical_data_path, output_suffix):
    """
    Save predictions to a CSV file, combining them with historical data.

    Parameters:
        predictions (list of dict): List of prediction dictionaries with keys:
            - 'year': int, the year of the prediction
            - 'abbr': str, state abbreviation
            - 'relative_margin': float, relative margin for the state
            - 'fit_type': str, type of fit used ('sine' or 'linear')
            - 'electoral_votes': int, electoral votes for the state
        historical_data_path (str): Path to the historical data CSV file.
        output_suffix (str): Suffix to append to the output file name.

    Returns:
        None
    """
    # Load historical data
    historical_data = pd.read_csv(historical_data_path)

    # Convert predictions to DataFrame
    pred_df = pd.DataFrame(predictions)

    # Add string representations for margins
    pred_df['pres_margin_str'] = pred_df['relative_margin'].apply(
        lambda x: f"{'D' if x > 0 else 'R'}+{abs(x)*100:.1f}"
    )
    pred_df['relative_margin_str'] = pred_df['relative_margin'].apply(
        lambda x: f"{'D' if x > 0 else 'R'}+{abs(x)*100:.1f}"
    )

    # Add placeholder for national_margin and its string representation
    pred_df['national_margin'] = np.nan
    pred_df['national_margin_str'] = ''

    # Combine historical and prediction data
    columns_to_keep = ['year', 'abbr', 'pres_margin', 'pres_margin_str', 'relative_margin', 
                       'relative_margin_str', 'national_margin', 'national_margin_str', 'electoral_votes']

    # Ensure all columns exist in predictions
    for col in columns_to_keep:
        if col not in pred_df.columns:
            pred_df[col] = np.nan

    result_df = pd.concat([historical_data[columns_to_keep], pred_df[columns_to_keep]], ignore_index=True)

    # Sort by year and state
    result_df = result_df.sort_values(['year', 'abbr']).reset_index(drop=True)

    # Generate output file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'presidential_margins_future_{output_suffix}_{timestamp}.csv'

    # Save to CSV
    result_df.to_csv(output_file, index=False)
    print(f"Successfully wrote predictions to {output_file}")
