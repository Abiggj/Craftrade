import os
import pandas as pd
from datetime import datetime

# Paths to CSV and folder containing overview text files
csv_path = './shares.csv'
overviews_folder = 'overviews'

# Load the stock values CSV, assuming there's a 'Date' column with the date for each entry
stock_data = pd.read_csv(csv_path, parse_dates=['Date'])

# Prepare a list to store merged data
merged_data = []

# Get a list of overview text files, and sort them by date (assuming the date format in filenames is 'YYYY-MM-DD.txt')
overview_files = sorted(os.listdir(overviews_folder))
overview_files_dates = [datetime.strptime(f.split('.')[0], '%d_%m_%Y') for f in overview_files]

# Loop over each row in the stock data
for index, row in stock_data.iterrows():
    # Find the corresponding overview file based on the date range
    stock_date = row['Date']
    matching_file = None

    # Iterate over overview file dates to find the appropriate date range
    for i in range(len(overview_files_dates) - 1):
        start_date = overview_files_dates[i]
        end_date = overview_files_dates[i + 1]

        # Check if the stock date falls within this date range
        if start_date <= stock_date < end_date:
            matching_file = overview_files[i]
            break
    # If the stock date is after the last date range, use the last file
    if not matching_file and stock_date >= overview_files_dates[-1]:
        matching_file = overview_files[-1]

    # Read the matching overview file
    if matching_file:
        with open(os.path.join(overviews_folder, matching_file), 'r') as file:
            overview_text = file.read()

        # Combine the stock values with the overview
        row_data = row.to_dict()  # Convert row to a dictionary
        row_data['overview'] = overview_text  # Add the overview text

        # Append the combined data to the list
        merged_data.append(row_data)

# Convert merged data to a DataFrame
merged_df = pd.DataFrame(merged_data)

# Save to a new CSV or JSON file
merged_df.to_csv('stock_overviews.csv', index=False)
# Or save as JSON
# merged_df.to_json('merged_stock_overviews.json', orient='records')

print("Data merging complete. Saved to 'stock_overviews.csv'.")
