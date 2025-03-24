import pandas as pd

import shutil
import os
import re

from DataScraper import *
from common import *


# Import countries.csv
new_df = pd.read_csv((source_path + source_file)) # Import countries.csv
new_df.dropna(inplace=True) # Drop all countries that don't have a url
new_list = new_df.values.tolist() # Convert to list

# Check if prev_countries.csv exists
prev_file = source_path + prev_source
new_file = source_path + source_file

# If prev_countries.csv doesn't exist
# Duplicate countries.csv as the new prev_countries.csv
if not os.path.isfile(prev_file):

    # Duplicate countries.csv as prev_countries.csv
    shutil.copy(new_file, prev_file)

    # Check if Raw Data directory is empty
    contents = os.listdir(raw_data_path)

    # If Raw Data directory isn't empty
    # Purge all files in folder
    # (should be empty if no prev_ file)
    if len(contents) != 0:
        purge(raw_data_path)

    # For all countries with a url
    # Create respective folder in Raw Data
    # Then download all files from national climate summary website source
    for country in new_list:

        # Create the folder the data will be downloaded to
        new_dir = raw_data_path + country[0] + "\\"

        try:
            os.mkdir(new_dir)
        except FileExistsError:
            print(f"Directory '{new_dir}' already exists.")
        except PermissionError:
            print(f"Permission denied: Unable to create '{new_dir}'")
        except Exception as e:
            print(f"An error occurred: {e}")

        # Get data from country website
        scrape_all_country_data(country[1], new_dir)

    # Data Manager is Done
    # Move onto Data Merger to concatenate all files

# If prev_countries.csv does exist
# Compare with countries.csv
# To check for changes
else:

    # Import prev_countries.csv
    prev_df = pd.read_csv((source_path + prev_source))
    prev_df.dropna(inplace=True) # Drop all countries that don't have a url
    prev_list = prev_df.values.tolist() # Convert to list

    # Check if there are country additions or deletions
    if not prev_list == new_list:

        deletions = []
        additions = []
        unchanged = []

        # If user doesn't want to delete prev_ exclusive sources
        # Or doesn't want to add new countries.csv exclusive sources
        # Record changes using mod_list to modify countries.csv
        mod_list = []

        # Check for deletions
        for country in prev_list:
            if country not in new_list: deletions.append(country)
            if country in new_list: unchanged.append(country)

        # Check for additions
        for country in new_list:
            if country not in prev_list: additions.append(country)


        print(f"There are {len(deletions)} deletions and {len(additions)} additions "
              f"\nfrom previous source to current source.")
        print(f"deletions: {[deletion[0] for deletion in deletions]}")
        print(f"additions: {[addition[0] for addition in additions]}")


        # Ask user if they want to delete sources not in new countries.csv
        delete_choice = input(f"\nWould you like to delete {len(deletions)} sources? (yes/no)\n")

        # Ask user if they want to delete sources not in new countries.csv until they give proper answer
        while not delete_choice('yes', 'no'):
            delete_choice = input(f"\nWould you like to delete {len(deletions)} sources? (yes/no)\n")

        # Ask user if they want to add new sources not in prev_countries.csv
        addition_choice = input(f"\n\nWould you like to add {len(additions)} sources? (yes/no)\n")

        # Ask user if they want to add new sources not in prev_countries.csv until they give proper answer
        while not addition_choice('yes', 'no'):
            addition_choice = input(f"\n\nWould you like to add {len(additions)} sources? (yes/no)\n")


        # If user wants to delete sources not in new countries.csv
        if delete_choice == 'yes':
            for deletion in deletions:
                delete_path = raw_data_path + deletion[0]
                os.rmdir(delete_path)

        elif delete_choice == 'no':
            mod_list.extend(deletions)

        # If user wants to add new sources not in prev_
        if addition_choice == 'yes':
            for addition in additions:

                mod_list.extend(additions)

                new_dir = raw_data_path + country[0] + "\\"
                os.mkdir(new_dir)

                scrape_all_country_data(country[1], new_dir)


        # If user doesn't want to delete sources from prev_ not in countries.csv
        # Or if user doesn't want to add sources from countries.csv not in prev_
        # Overwrite countries.csv with full source list
        if delete_choice == 'no' or addition_choice == 'no':

            mod_list.extend(unchanged)

            # Create a dataframe from mod_list
            mod_df = pd.DataFrame(mod_list, columns=["Country", "URL"])
            mod_df.sort_values("Country")

            # Delete original countries.csv and overwrite with mod_
            file = source_path + source_file
            os.remove(file)

            mod_df.to_csv(file)

        additions = []

        print("Checking for updates...\n")

        for country in unchanged:

            print(f"Checking for updates for {country[0]}...\n")

            check_for_updates()

    else:

        unchanged = prev_list

        print("Checking for updates...\n")

        # Will store which months / years need to be added
        additions = []

        for country in unchanged:

            print(f"Checking for updates for {country[0]}...\n")

            check_for_updates(country)
































