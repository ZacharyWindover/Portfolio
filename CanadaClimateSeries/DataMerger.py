import os
import re
import datetime
import pandas as pd

from common import *


# Merge function specifically for countries
def merge_summaries(path, country_name):

    # List of file names in the directory
    dir_list = sorted(os.listdir(path))

    # List of dataframes for each csv file
    df_list = []

    # Regex to extract <month>-<year> from file names
    #climate_summary_format = r"en_climate_summaries_All_(\d+)-(\d+).csv"

    # Put each CSV into a dataframe
    for file in dir_list:

        file_path = os.path.join(path, file)

        # Match month and year using regex
        match = re.search(climate_summary_format, file)

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

    if country_name == "Canada":
        # Filter to limit to Canada only (some data points are out of Canada)
        climate_df = climate_df[climate_df['Long'] < -46]
        climate_df = climate_df[climate_df['Lat'] > 40]

    # Save dataframe as csv
    save = "merged_climate_summaries_" + country_name + ".csv"
    save_path = merged_data_path + save

    climate_df.to_csv(save_path, index=False)


# Merge function specifically to merge country summaries
def merge_countries(path):

    # List of file names in the directory
    dir_list = sorted(os.listdir(path))

    # List of dataframes for each csv file
    df_list = []

    # Regex to extract <country> from file name
    #country_summary_format = r"merged_climate_summaries_(\D[a-zA-Z]+).csv"

    # Put each CSV into a dataframe
    for file in dir_list:

        file_path = os.path.join(path, file)

        # Match country using regex
        match = re.search(country_summary_format, file)

        if match:
            country_name = re.findall(country_summary_format, file)
        else:
            print(f"Skipping {file}: File name does not match expected format.")
            continue

        try:

            df = pd.read_csv(file_path)
            df['Country'] = country_name
            df_list.append(df)

        except Exception as e:
            print(f"Error reading {file}: {e}")


    # Concatenate all dataframes into one
    climate_df = pd.concat(df_list, ignore_index=True)

    # Get current datetime for saving merged to csv file
    current_time = datetime.datetime.now()
    ymd = str(current_time.year) + "_" + str(current_time.month) + "_" + str(current_time.day)
    hms = str(current_time.hour) + "_" + str(current_time.minute) + "_" + str(current_time.second)

    # Save dataframe as csv
    save = "concatenated_climate_summaries_" + ymd + "_" + hms
    save_path = merged_data_path + save

    climate_df.to_csv(save, index=False)


