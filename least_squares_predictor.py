import pandas as pd
import numpy as np
import utils
import os

def load_data(file_path):
    """Load presidential margins data from a CSV file."""
    data = pd.read_csv(file_path)
    return data

def prepare_matrices(data, num_past_elections=3):
    """Prepare the M matrix and b vector for least squares computation with a variable number of past elections."""
    states = data['abbr'].unique()
    years = sorted(data['year'].unique())

    M = []
    b = []

    for state in states:
        state_data = data[data['abbr'] == state].sort_values('year')
        margins = state_data['relative_margin'].values
        state_years = state_data['year'].values

        for i in range(len(state_years) - num_past_elections):
            year = state_years[i]
            if year == 2016:
                pass
            if state_years[i + num_past_elections] - state_years[i] == 4 * num_past_elections:  # Ensure consecutive 4-year intervals
                M.append(margins[i:i + num_past_elections][::-1])  # Reverse to match the order of most recent to oldest
                b.append(margins[i + num_past_elections])

    return np.array(M), np.array(b)

def predict_future(margins, coefficients):
    """Predict future margins using the least squares coefficients."""
    return np.dot(margins, coefficients)

def print_coefficients(coefficients):
    """Print the coefficients in a readable format."""
    text = "TRAINED COEFFICIENTS:\n\trel_margin[year + 4] ~\n"
    for i, coef in enumerate(coefficients):
        text += f"\t\t{coef} * rel_margin[year - {4 * i}]\n"
    return text

def EC_text(data_year, rel_predictions_year, national_margin_year, actual_pres_margins_year, electoral_votes_year, actual_data=True):
    EC_prediction = {'D': 0, 'R': 0}
    EC_actual = {'D': 0, 'R': 0}
    for abbr, predicted_relative in zip(data_year['abbr'], rel_predictions_year):
        predicted_margin = predicted_relative + national_margin_year
        actual_margin = actual_pres_margins_year[abbr] if actual_data else None
        if predicted_margin > 0:
            EC_prediction['D'] += electoral_votes_year[abbr]
        else:
            EC_prediction['R'] += electoral_votes_year[abbr]
        if actual_margin and actual_margin > 0:
            EC_actual['D'] += electoral_votes_year[abbr]
        else:
            EC_actual['R'] += electoral_votes_year[abbr]
    EC_prediction_winner = 'D' if EC_prediction['D'] > EC_prediction['R'] else 'R'
    text = f"Predicted Electoral College: D: {EC_prediction['D']}, R: {EC_prediction['R']} (Winner: {EC_prediction_winner})\n"
    if actual_data:
        EC_actual_winner = 'D' if EC_actual['D'] > EC_actual['R'] else 'R'
        EC_correct = EC_prediction_winner == EC_actual_winner
        text += f"Actual Electoral College: D: {EC_actual['D']}, R: {EC_actual['R']} (Winner: {EC_actual_winner})\n"
        text += f"Prediction Correct: {'(EC CORRECT)' if EC_correct else '(EC INCORRECT)'}\n"
    return text

def prediction_calculation(coefficients, rel_margins, start_year):
    """Write the calculation for the relative margin calculation for the state"""
    predicted_margin = sum(coefficients[i] * rel_margins[0][i] for i in range(len(coefficients)))
    text = f"\t\t{predicted_margin:3f}~\n"
    for i, coeff in enumerate(coefficients):
        text += f"\t\t\t{coeff:3f} * {rel_margins[0][i]:3f}({start_year - 4 * i})\n"
    return text
def verify_predictions_for_year(data, coefficients, year, output_dir, num_past_elections=3):
    """Verify predictions for a specific year and save results to a file with a variable number of past elections."""
    if year == 2024:
        pass
    data_year = data[data['year'] == year].sort_values('abbr')
    past_data = [data[data['year'] == year - 4 * i].sort_values('abbr') for i in range(1, num_past_elections + 1)]

    if data_year.empty or any(d.empty for d in past_data):
        print(f"Skipping year {year} due to insufficient data.")
        return

    rel_margins_past = [d['relative_margin'].values for d in past_data]
    rel_margins_year = data_year['relative_margin'].values

    M_year = np.column_stack(rel_margins_past)  # Reverse to match the order of most recent to oldest
    # for debugging match the abbrs to the M_year row
    debug_M = {abbr: {i: M_year[i]} for i, abbr in enumerate(data_year['abbr'])}
    rel_predictions_year = predict_future(M_year, coefficients)

    national_margin_year = data_year['national_margin'].values[0] if not data_year.empty else None
    actual_pres_margins_year = {abbr: data_year[data_year['abbr'] == abbr]['pres_margin'].values[0] for abbr in data_year['abbr']}
    electoral_votes_year = {abbr: data_year[data_year['abbr'] == abbr]['electoral_votes'].values[0] for abbr in data_year['abbr']}
    
    output_path = os.path.join(output_dir, f"{year}_prediction_verification.txt")
    with open(output_path, 'w') as f:
        f.write(print_coefficients(coefficients=coefficients))
        f.write(f"Prediction Verification for {year}\n")
        f.write("=================================\n")
        f.write(EC_text(data_year, rel_predictions_year, national_margin_year, actual_pres_margins_year, electoral_votes_year))
        f.write("=================================\n")        

        for abbr, predicted_relative, actual_relative in zip(data_year['abbr'], rel_predictions_year, rel_margins_year):
            predicted_margin = predicted_relative + national_margin_year
            actual_margin = actual_pres_margins_year[abbr] if not data_year.empty else None
            got_correct = actual_margin is not None and np.sign(predicted_margin) == np.sign(actual_margin)
            ev = electoral_votes_year[abbr]
            upset_str = " (INCORRECT)" if not got_correct else " (CORRECT)"
            f.write(f"{abbr}: Predicted final margin {utils.lean_str(predicted_margin)} (Actual {utils.lean_str(actual_margin)}), EVs: {ev}{upset_str}\n{prediction_calculation(coefficients=coefficients, rel_margins=list(debug_M[abbr].values()), start_year=year)}\t\tPredicted relative: {utils.lean_str(predicted_relative)} + national {utils.lean_str(national_margin_year)} = {utils.lean_str(predicted_margin)}\n\t\tActual relative: {utils.lean_str(actual_relative)} + national {utils.lean_str(national_margin_year)} = {utils.lean_str(actual_margin)}\n")

    print(f"Verification for {year} saved to {output_path}")

def analyze_2028_prediction(data, coefficients, output_dir, num_past_elections=3):
    """Perform 2028 prediction and tipping point analysis and save results to a file."""
    data_2024 = data[data['year'] == 2024].sort_values('abbr')
    past_data = [data[data['year'] == 2024 - 4 * i].sort_values('abbr') for i in range(1, num_past_elections)]

    if any(d.empty for d in past_data):
        print("Skipping 2028 prediction due to insufficient data.")
        return

    rel_margins_past = [d['relative_margin'].values for d in past_data]
    rel_margins_2024 = data_2024['relative_margin'].values

    M_2028 = np.column_stack([rel_margins_2024] + rel_margins_past[::-1])  # Include 2024 and reverse past margins
    debug_M = {abbr: {i: M_2028[i]} for i, abbr in enumerate(data_2024['abbr'])}
    rel_predictions_2028 = predict_future(M_2028, coefficients)

    electoral_votes_2024 = {abbr: data_2024[data_2024['abbr'] == abbr]['electoral_votes'].values[0] for abbr in data_2024['abbr']}

    def find_tipping_point(predictions, electoral_votes):
        for national_margin in np.arange(-20, 20, 0.005):
            EC_count = {'D': 0, 'R': 0}
            for abbr, margin in zip(data_2024['abbr'], predictions):
                adjusted_margin = margin + national_margin
                if adjusted_margin > 0:
                    EC_count['D'] += electoral_votes[abbr]
                else:
                    EC_count['R'] += electoral_votes[abbr]
            if EC_count['D'] >= 270:
                return national_margin, EC_count
        return None, None

    tipping_point_margin_2028, EC_at_tipping_point_2028 = find_tipping_point(rel_predictions_2028, electoral_votes_2024)

    output_path = os.path.join(output_dir, "2028_prediction.txt")
    with open(output_path, 'w') as f:
        f.write("2028 Prediction and Tipping Point Analysis\n")
        f.write(print_coefficients(coefficients=coefficients))
        f.write("=================================\n")

        # Calculate Electoral College prediction
        EC_prediction = {'D': 0, 'R': 0}
        for abbr, margin in zip(data_2024['abbr'], rel_predictions_2028):
            if margin > 0:
                EC_prediction['D'] += electoral_votes_2024[abbr]
            else:
                EC_prediction['R'] += electoral_votes_2024[abbr]

        #f.write(f"Predicted Electoral College: D: {EC_prediction['D']}, R: {EC_prediction['R']}\n")
        #f.write("=================================\n")

        for abbr, predicted_relative, prev_rel_margin in zip(data_2024['abbr'], rel_predictions_2028, rel_margins_2024):
            ev = electoral_votes_2024[abbr]
            f.write(f"{abbr}: Predicted relative {utils.lean_str(predicted_relative)}, EVs: {ev}\n{prediction_calculation(coefficients=coefficients, rel_margins=list(debug_M[abbr].values()), start_year=2024)}\t\tFrom 2024 relative: {utils.lean_str(prev_rel_margin)} -> {utils.lean_str(predicted_relative)} (difference: {utils.lean_str(predicted_relative - prev_rel_margin)}, TRENDED {'LEFT' if predicted_relative > prev_rel_margin else 'RIGHT'})\n")

        f.write(f"\nTipping Point National Margin for 2028: {utils.lean_str(tipping_point_margin_2028)}\n")
        if EC_at_tipping_point_2028 is not None:
            f.write(f"Electoral College at Tipping Point:\n\tD:{EC_at_tipping_point_2028['D']}\n\tR:{EC_at_tipping_point_2028['R']}\n")
        else:
            f.write("No tipping point found within the search range.\n")

        sorted_states = sorted(
            [(abbr, margin + tipping_point_margin_2028, margin, electoral_votes_2024[abbr]) for abbr, margin in zip(data_2024['abbr'], rel_predictions_2028)],
            key=lambda x: x[1]
        )

        f.write("\nStates sorted by relative margin under tipping point:\n")
        for abbr, final_margin, rel_margin, ev in sorted_states:
            f.write(f"{abbr}: Final Margin {utils.lean_str(final_margin)}, EVs: {ev}\n\tPredicted relative: {utils.lean_str(rel_margin)} + national {utils.lean_str(tipping_point_margin_2028)} = {utils.lean_str(final_margin)}\n")

    print(f"2028 prediction and tipping point analysis saved to {output_path}")

def main():
    # Load the data
    file_path = 'presidential_margins.csv'
    data = load_data(file_path)

    # Number of past elections to use
    num_past_elections = 2  # Change this value to use a different number of past elections

    # Prepare the matrices
    M, b = prepare_matrices(data, num_past_elections=num_past_elections)

    # Solve the least squares problem
    coefficients, _, _, _ = np.linalg.lstsq(M, b, rcond=None)

    print("Least squares solution (coefficients):", coefficients)

    # Create output directory
    output_dir = "least_squares_predict"
    os.makedirs(output_dir, exist_ok=True)

    # Perform prediction verification for all years starting from 1984
    for year in range(1984, 2025, 4):
        verify_predictions_for_year(data, coefficients, year, output_dir, num_past_elections=num_past_elections)

    # Perform 2028 prediction and tipping point analysis
    analyze_2028_prediction(data, coefficients, output_dir, num_past_elections=num_past_elections)

if __name__ == "__main__":
    main()
