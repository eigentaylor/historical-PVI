import pandas as pd
import utils

def process_csv(file_path, output_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # remove the 'index' and 'index_1' columns if they exist
    if 'index' in df.columns:
        df = df.drop(columns=['index'])
    if 'index_1' in df.columns:
        df = df.drop(columns=['index_1'])
    if 'pres_pvi_2024' in df.columns:
        df = df.drop(columns=['pres_pvi_2024'])
    if 'pres_pvi_2024_label' in df.columns:
        df = df.drop(columns=['pres_pvi_2024_label'])
        
    # get the presidential margins from presidential_margins.csv
    pres_margins = pd.read_csv('presidential_margins.csv')
    # set the 2024_pres_margin column in the main dataframe from the presidential margins 
    for index, row in df.iterrows():
        state = row['abbreviation']
        year = 2024
        margin_row = pres_margins[(pres_margins['abbr'] == state) & (pres_margins['year'] == year)]
        if not margin_row.empty:
            df.at[index, '2024_pres_margin'] = margin_row['pres_margin'].values[0]
            df.at[index, '2024_pres_margin_str'] = utils.lean_str(margin_row['pres_margin'].values[0])
            df.at[index, '2024_pres_relative_margin'] = margin_row['relative_margin'].values[0]
            df.at[index, '2024_pres_relative_margin_str'] = utils.lean_str(margin_row['relative_margin'].values[0])
        else:
            df.at[index, '2024_pres_margin'] = None

    # we add the following columns:
    # 'gov_relative': gov_2024_margin - 2024_pres_margin
    # 'gov_relative_str': utils.lean_str(gov_relative)
    df['gov_relative'] = df['gov_2024_margin'] - df['2024_pres_margin']
    df['gov_relative_str'] = df['gov_relative'].apply(utils.lean_str)
    # 'sen_relative': sen_2024_margin - 2024_pres_margin
    # 'sen_relative_str': utils.lean_str(sen_relative)
    # Only compute sen_relative if 'sen_2024_margin' is not "NO_ELECTION"
    mask = df['sen_2024_margin'] != "NO_ELECTION"
    df.loc[mask, 'sen_relative'] = pd.to_numeric(df.loc[mask, 'sen_2024_margin'], errors='coerce') - df.loc[mask, '2024_pres_margin']
    df.loc[mask, 'sen_relative_str'] = df.loc[mask, 'sen_relative'].apply(utils.lean_str)
    # we manually get the margins for "NO_ELECTION" (states with a class II and III seat)
    # AK, AL, AR, CO, GA, ID, IL, IA, KS, KY, LA, NH, NC, OK, OR, SC, SD 
    sen_2022_margins = {
        'AK': -0.074,
        'AL': -0.357,
        'AR': -0.346,
        'CO': 0.146,
        'GA': 0.028,
        'ID': -0.32,
        'IL': 0.153,
        'IA': -0.122,
        'KS': -0.23,
        'KY': -0.236,
        'LA': -0.3291,
        'NH': 0.091,
        'NC': -0.032,
        'OK': -0.322,
        'OR': 0.149,
        'SC': -0.259,
        'SD': -0.434
    }
    for state, margin in sen_2022_margins.items():
        if state in df['abbreviation'].values:
            #df.loc[df['abbreviation'] == state, 'sen_2022_margin'] = margin
            pres_2024_margin = df.loc[df['abbreviation'] == state, '2024_pres_margin'].values[0]
            df.loc[df['abbreviation'] == state, 'sen_relative'] = margin - pres_2024_margin
            df.loc[df['abbreviation'] == state, 'sen_relative_str'] = utils.lean_str(margin - pres_2024_margin)

    # reorder the columns to match the desired output
    columns_order = [
        'abbreviation', 'state', 'Population', 'evs_2024',
        '2024_pres_margin', '2024_pres_relative_margin', '2024_national_margin', 
        'gov_relative', 'sen_relative',
        'gov_2024_margin', 'sen_2024_margin', 'sen_2024_party', 'sen_2022_party', 'sen_2020_party',
        'senate_class_1', 'senate_class_2', '2022_senate_margin',
        '2024_pres_margin_str', '2024_pres_relative_margin_str', 
        'gov_relative_str', 'sen_relative_str'
    ]
    df = df[columns_order]
    # Sort the data by the 'abbreviation' column and reset the index
    df = df.sort_values(by='abbreviation').reset_index(drop=True)
    df.index.name = 'index'

    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_path, index=True)

# File paths
input_file = 'data/2024_info_wip.csv'
output_file = 'data/2024_info.csv'

# Process the CSV
process_csv(input_file, output_file)
