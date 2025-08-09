import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# clear output directory
import os
output_dir = 'dendrogram'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
#clear previous files
for file in os.listdir(output_dir):
    file_path = os.path.join(output_dir, file)
    if os.path.isfile(file_path):
        os.remove(file_path)
        
# Load the data
df = pd.read_csv('presidential_margins.csv', index_col=0)

# Reset index to ensure 'year' is a column in the DataFrame
df = df.reset_index()

# Ensure dark mode for plots
plt.style.use('dark_background')

# Ensure consistent ordering by state abbreviation
state_columns = sorted(df['abbr'].unique())

# 1. Hierarchical clustering of election years
# Each year is a vector of relative margins for all states (sorted by state abbreviation)
year_vectors = df.pivot(index='year', columns='abbr', values='relative_margin').fillna(0).values  # shape: (years, states)
year_labels = df['year'].unique().astype(str).tolist()

Z_years = linkage(year_vectors, method='ward')

plt.figure(figsize=(12, 6))
dendrogram(Z_years, labels=year_labels, leaf_rotation=90)
plt.title('Hierarchical Clustering of Election Years', color='white')
plt.xlabel('Year', color='white')
plt.ylabel('Distance', color='white')
plt.tight_layout()
plt.savefig('dendrogram/dendrogram_years_dark.png', facecolor='black')
plt.close()

# 2. Hierarchical clustering of states
# Each state is a vector of its relative margins across years
state_vectors = df.pivot(index='year', columns='abbr', values='relative_margin').fillna(0).T.values  # shape: (states, years)
state_labels = sorted(df['abbr'].unique())

Z_states = linkage(state_vectors, method='ward')

plt.figure(figsize=(20, 12), dpi=300)  # Increased figure size and resolution
dendrogram(Z_states, labels=state_labels, leaf_rotation=90)
plt.title('Hierarchical Clustering of States', color='white')
plt.xlabel('State', color='white')
plt.ylabel('Distance', color='white')
plt.tight_layout()
plt.savefig('dendrogram/dendrogram_states_dark.png', facecolor='black')
plt.close()

print("Dendrograms saved as 'dendrogram/dendrogram_years_dark.png' and 'dendrogram/dendrogram_states_dark.png'.")

# Function to perform clustering for specific year ranges
def cluster_by_year_ranges(df, interval=None):
    years = sorted(df['year'].unique(), reverse=True)  # Sort years in descending order
    if interval is None:
        interval = len(years)
    for i in range(0, len(years), interval):
        end_year = years[i]
        start_year = years[min(i + interval - 1, len(years) - 1)]  # Adjust to ensure valid range
        if start_year == end_year:
            continue
        
        # Filter data for the year range
        df_range = df[(df['year'] >= start_year) & (df['year'] <= end_year)]
        
        # Perform clustering for states
        state_vectors = df_range.pivot(index='year', columns='abbr', values='relative_margin').fillna(0).T.values
        state_labels = sorted(df_range['abbr'].unique())

        Z_states = linkage(state_vectors, method='ward')

        # Save dendrogram
        plt.figure(figsize=(20, 12), dpi=300)
        dendrogram(Z_states, labels=state_labels, leaf_rotation=90)
        plt.title(f'Hierarchical Clustering of States ({start_year}-{end_year})', color='white')
        plt.xlabel('State', color='white')
        plt.ylabel('Distance', color='white')
        plt.tight_layout()
        filename = f'dendrogram/{start_year}_{end_year}_state_dendrogram.png'
        plt.savefig(filename, facecolor='black')
        plt.close()
        print(f"Dendrogram saved as '{filename}'.")

# do for all year
cluster_by_year_ranges(df, interval=None)
# Call the function with a specific interval (e.g., 3 means 8 years from 3 elections)
cluster_by_year_ranges(df, interval=3)