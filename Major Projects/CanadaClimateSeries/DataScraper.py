from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
import os
import re

from common import *


#
# Inputs:
# url - takes a url as a string
# download_dir - takes download folder path as a string
#
# Function:
# Using selenium, downloads all available climate summaries
# from the url, and stores them in download_dir
#
def scrape_all_country_data(url, download_dir):

    # Set up Chrome options
    chrome_options = webdriver.ChromeOptions()

    prefs = {

        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True

    }

    chrome_options.add_experimental_option("prefs", prefs)

    # Start Selenium WebDriver
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)

    # WebDriverWait setup
    wait = WebDriverWait(driver, 5)  # Wait up to 5 seconds

    # Get all available years
    year_dropdown = Select(driver.find_element(By.ID, "intYear"))
    years = [option.text for option in year_dropdown.options]
    years = [int(y) for y in years if y.isdigit()]  # Convert to integers


    # Iterate through all available years
    for year in years:

        # Select the year
        year_dropdown.select_by_value(str(year))

        # Wait for month dropdown to refresh
        wait.until(lambda driver: len(Select(driver.find_element(By.ID, "intMonth")).options) > 1)

        # Get all available months for the year
        month_dropdown = Select(driver.find_element(By.ID, "intMonth"))

        # print(f"UPDATED DEBUG: Available months for {year}: {[option.text for option in month_dropdown.options]}")

        # months = [option.text for option in month_dropdown.options]
        months = [option.get_attribute("value") for option in month_dropdown.options]
        months = [int(m) for m in months if m.isdigit()]  # Convert to integers

        # print(f"Available months for {year}: {months}")

        # Iterate through all available months for the year
        for month in months:
            
            # print(f"Downloading: Year {year}, Month {month}")

            # Select the month
            month_dropdown.select_by_value(str(month))

            # Wait for province dropdown to refresh
            wait.until(lambda driver: len(Select(driver.find_element(By.ID, "prov")).options) > 0)

            # Select "All" for Province dropdown
            province_dropdown = Select(driver.find_element(By.ID, "prov"))
            province_dropdown.select_by_index(0)

            # Select CSV file format
            format_checkbox = driver.find_element(By.ID, "csv")
            format_checkbox.click()

            # Click Download button
            download_button = driver.find_element(By.NAME, "btnSubmit")
            download_button.click()

    # Close the browser
    driver.quit()


def scrape_country_data_for_month(driver, month):

    month_dropdown = Select(driver.find_element(By.ID, "intMonth"))
    month_dropdown.select_by_value(str(month))

    # Wait for province dropdown to refresh
    wait = WebDriverWait(driver, 5)
    wait.until(lambda driver: len(Select(driver.find_element(By.ID, "prov")).options) > 0)

    # Select All from province dropdown
    province_dropdown = Select(driver.find_element(By.ID, "prov"))
    province_dropdown.select_by_index(0)

    # Select CSV file format
    format_checkbox = driver.find_element(By.ID, "csv")
    format_checkbox.click()

    # Click Download button
    download_button = driver.find_element(By.NAME, "btnSubmit")
    download_button.click()


#
# Inputs:
# country - list containing url and country name
#
# Function:
# Checks files in Raw Data folder for latest year and month of data
# Then checks the country website to see if it is missing any
# And downloads them using selenium
#
def check_for_updates(country):

    country_name = country[0]
    country_url = country[1]

    path = raw_data_path + country[0] + "\\"
    files = os.listdir(path)

    # Recording most recent year and month in Raw Data
    latest_year, latest_month = 0


    # Find latest year and month in Raw Data
    for file in files:

        # Match month and year using regex
        match = re.search(climate_summary_format, file)

        if match:

            month, year = int(match.group(1)), int(match.group(2))

            if year > latest_year:
                latest_year = year
                latest_month = month
            else:
                if month > latest_month:
                    latest_month = month

    # Stating download directory for default download location in chrome
    download_dir = raw_data_path + country_name + "\\"

    # Set up Chrome options
    chrome_options = webdriver.ChromeOptions()

    prefs = {

        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True

    }

    chrome_options.add_experimental_option("prefs", prefs)

    # Start Selenium WebDriver
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(country_url)

    # WebDriverWait setup
    wait = WebDriverWait(driver, 5)  # Wait up to 5 seconds

    # Get latest year from year dropdown
    year_dropdown = Select(driver.find_element(By.ID, "intYear"))   # Select year dropdown
    years = [option.text for option in year_dropdown.options]       # Get all years in dropdown
    years = [int(y) for y in years if y.isdigit()]                  # Convert to integers
    max_year = max(years)                                           # Record latest year in dropdown


    # Iterate through every year between latest year
    # in Raw Data and latest year on website
    for year in range(latest_year, max_year + 1):

        # Select year
        year_dropdown.select_by_value(str(year))

        # Wait for month dropdown to refresh
        wait.until(lambda driver: len(Select(driver.find_element(By.ID, "intMonth")).options) > 1)

        # Get latest month for the year
        month_dropdown = Select(driver.find_element(By.ID, "intMonth"))                 # Select month dropdown
        months = [option.get_attribute("value") for option in month_dropdown.options]   # Get all months in dropdown
        months = [int(m) for m in months if m.isdigit()]                                # Convert to integers
        max_month = max(months)                                                         # Get max month

        # print(f"UPDATED DEBUG: Available months for {year}: {[option.text for option in month_dropdown.options]}")

        if year == latest_year:

            for month in range(latest_month + 1, max_month):

                # print(f"Downloading: Year {year}, Month {month}")

                scrape_country_data_for_month(driver, month)

        else:

            for month in months:

                # print(f"Downloading: Year {year}, Month {month}")

                scrape_country_data_for_month(driver, month)

    # Close the browser
    driver.quit()








#
# Inputs:
# directory - target folder path as a string
#
# Function:
# Deletes all files and folders in given directory
#
def purge(directory):

    for entry in os.listdir(directory):

        full_path = os.path.join(directory, entry)

        if os.path.isdir(full_path):
            os.rmdir(full_path)
        elif os.path.isfile(full_path):
            os.remove(full_path)

