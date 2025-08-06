import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import os

# Define the sine function for fitting
def sine_function(x, A, B, C, D):
    return A * np.sin(B * x + C) + D

# Load the population data
pop_data = pd.read_csv('data/2024_info.csv')
population_dict = dict(zip(pop_data['abbreviation'], pop_data['Population']))

# Load the Excel file
excel_file = 'state_tables/state_margins_2000_2024.xlsx'
excel_data = pd.read_excel(excel_file, sheet_name=None)  # Read all sheets

# For debugging, print the sheet names
print(f"Sheets in {excel_file}: {list(excel_data.keys())}")

# For debugging, print the first few rows of the first sheet
first_sheet_name = list(excel_data.keys())[0]
print(f"First few rows of sheet '{first_sheet_name}':")
print(excel_data[first_sheet_name].head())
