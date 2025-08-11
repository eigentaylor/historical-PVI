import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, to_tree

# clear output directory
import os
output_dir = 'dendrogram'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
#clear previous files
# Clear previous files in output_dir and all its subdirectories
for root, dirs, files in os.walk(output_dir):
    for file in files:
        file_path = os.path.join(root, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
        
# Load the data
df = pd.read_csv('presidential_margins.csv', index_col=0)

# Reset index to ensure 'year' is a column in the DataFrame
df = df.reset_index()

# Compute change in relative margins per state (current year minus previous election year)
# Sort to ensure proper diff ordering
_df_sorted = df.sort_values(['abbr', 'year'])
_df_sorted['relative_margin_change'] = _df_sorted.groupby('abbr')['relative_margin'].diff()
# Merge back to original df order
df = _df_sorted

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

plt.figure(figsize=(20, 12), dpi=600)  # Increased figure size and resolution
dendrogram(Z_states, labels=state_labels, leaf_rotation=90)
plt.title('Hierarchical Clustering of States', color='white')
plt.xlabel('State', color='white')
plt.ylabel('Distance', color='white')
plt.tight_layout()
plt.savefig('dendrogram/dendrogram_states_dark.png', facecolor='black')
plt.close()

print("Dendrograms saved as 'dendrogram/dendrogram_years_dark.png' and 'dendrogram/dendrogram_states_dark.png'.")

# Function to perform clustering for specific year ranges with optional sliding windows and change mode
def cluster_by_year_ranges(df, interval=None, sliding=True, mode='margin'):
    """
    Perform hierarchical clustering of states for year windows.
    - interval: number of elections in each window (e.g., 3 -> 3 elections -> ~8 years)
    - sliding: if True, use sliding windows (e.g., 2016-2024, 2012-2020, ...). If False, use disjoint windows.
    - mode: 'margin' for relative margins, 'change' for change in relative margins (diff to previous election).
    """
    years = sorted(df['year'].unique(), reverse=True)  # Descending years (e.g., [2024, 2020, ..., 1976])
    if mode == 'change':
        # Drop the earliest year (last element in descending order) since change isn't defined there
        years = years[:-1]

    # Determine the set of windows (start_year, end_year)
    windows = []
    if interval is None or interval >= len(years):
        # Single window covering all years
        windows = [(min(years), max(years))]
    else:
        if sliding:
            for i in range(0, len(years) - interval + 1):
                end_year = years[i]
                start_year = years[i + interval - 1]
                windows.append((start_year, end_year))
        else:
            for i in range(0, len(years), interval):
                end_year = years[i]
                start_year = years[min(i + interval - 1, len(years) - 1)]
                if start_year == end_year:
                    continue
                windows.append((start_year, end_year))

    value_col = 'relative_margin' if mode == 'margin' else 'relative_margin_change'
    mode_title = 'Relative Margins' if mode == 'margin' else 'Change in Relative Margins'
    mode_suffix = 'margin' if mode == 'margin' else 'change'

    for (start_year, end_year) in windows:
        # Filter data for the year range
        df_range = df[(df['year'] >= start_year) & (df['year'] <= end_year)].copy()
        if mode == 'change':
            df_range = df_range.dropna(subset=[value_col])  # drop first election per state where diff is NaN

        # Build matrix: rows=states, cols=years
        pivot = df_range.pivot(index='year', columns='abbr', values=value_col).fillna(0)
        state_vectors = pivot.T.values
        state_labels = pivot.columns.tolist()

        if state_vectors.shape[0] == 0 or state_vectors.shape[1] == 0:
            continue  # skip empty windows

        Z_states = linkage(state_vectors, method='ward')

        # Save dendrogram
        plt.figure(figsize=(20, 12), dpi=500)
        dendrogram(Z_states, labels=state_labels, leaf_rotation=90)
        plt.title(f'Hierarchical Clustering of States ({mode_title}) ({start_year}-{end_year})', color='white')
        plt.xlabel('State', color='white')
        plt.ylabel('Distance', color='white')
        plt.tight_layout()
        filename = f'dendrogram/{mode}/{start_year}_{end_year}_state_dendrogram_{mode_suffix}.png'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, facecolor='black')
        plt.close()
        print(f"Dendrogram saved as '{filename}'.")

        # Save tree structure to a text file
        tree = to_tree(Z_states)
        tree_filename = f'dendrogram/{mode}/{start_year}_{end_year}_state_tree_{mode_suffix}.txt'
        with open(tree_filename, 'w') as f:
            def write_tree(node, depth=0):
                if node.is_leaf():
                    f.write(f"{'  ' * depth}{state_labels[node.id]}\n")
                else:
                    f.write(f"{'  ' * depth}Cluster (dist: {node.dist:.3f})\n")
                    write_tree(node.get_left(), depth + 1)
                    write_tree(node.get_right(), depth + 1)
            write_tree(tree)
        print(f"Tree structure saved as '{tree_filename}'.")

# do for all years (one window over full range) using margins
cluster_by_year_ranges(df, interval=None, sliding=False, mode='margin')
# do for all years (one window over full range) using change in margins
cluster_by_year_ranges(df, interval=None, sliding=False, mode='change')
# Sliding windows of size 3 elections for margins
cluster_by_year_ranges(df, interval=3, sliding=True, mode='margin')
# Sliding windows of size 3 elections for change in margins
cluster_by_year_ranges(df, interval=3, sliding=True, mode='change')