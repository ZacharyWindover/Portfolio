import numpy as np
import pandas as pd
import geopandas as gpd
from geodatasets import get_path
import matplotlib.pyplot as plt
import os
import re

# Directory path
path = "C://Users//Zachary Windover//Documents//Climate Series//"

# Save Info
save_file = "processed_climate_summaries.csv"
save_location = "C://Users//Zachary Windover//Documents//Climate Series//"
save = os.path.join(path, save_file)



# List of file names in the directory
dir_list = sorted(os.listdir(path))

# List of dataframes for each csv file
df_list = []

# Regex to extract <month>-<year> from file names
file_format = r"en_climate_summaries_All_(\d+)-(\d+).csv"

# Put each CSV into a dataframe
for file in dir_list:

    file_path = os.path.join(path, file)

    # Match month and year using regex
    match = re.search(file_format, file)

    if match:
        month, year = int(match.group(1)), int(match.group(2))
    else:
        print(f"Skipping {file}: Filename does not match expected format.")
        continue

    try:

        df = pd.read_csv(file_path)

        df['Year'] = year
        df['Month'] = month

        df_list.append(df)

    except Exception as e:
        print(f"Error reading {file}: {e}")

# Concatenate all dataframes into one
climate_df = pd.concat(df_list, ignore_index=True)

# Filter to limit to Canada only
climate_df = climate_df[climate_df['Long'] < -46]
climate_df = climate_df[climate_df['Lat'] > 40]

climate_df.to_csv(save, index=False)


