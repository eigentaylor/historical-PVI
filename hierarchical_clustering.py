import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

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