from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
import os

class Country:
    def __init__(self, name, dir, url):
        self.name = name
        self.dir = dir
        self.url = url


#def clean_slate_protocol(dir):


#def append_protocol():



def download_climate_data(country):

    # Set up Chrome options
    chrome_options = webdriver.ChromeOptions()

    prefs = {

        "download.default_directory": country.dir,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True

    }

    chrome_options.add_experimental_option("prefs", prefs)

    # Start Selenium WebDriver
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(country.url)

    # Get all available years
    year_dropdown = Select(driver.find_element(By.ID, "intYear"))
    years = [option.text for option in year_dropdown.options]
    years = [int(y) for y in years if y.isdigit()] # Convert to integers

    # WebDriverWait setup
    wait = WebDriverWait(driver, 5) # Wait up to 2 seconds

    # Iterate through all available years
    for year in years:

        # Select the year
        year_dropdown.select_by_value(str(year))

        # Wait for month dropdown to refresh
        wait.until(lambda driver: len(Select(driver.find_element(By.ID, "intMonth")).options) > 1)

        # Get all available months for the year
        month_dropdown = Select(driver.find_element(By.ID, "intMonth"))

        # print(f"UPDATED DEBUG: Available months for {year}: {[option.text for option in month_dropdown.options]}")

        months = [option.text for option in month_dropdown.options]
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


# First, import CSV of countries, URLs, and directories and put into list
main_dir = "C:\\Users\\Zachary Windover\\Documents\\Climate Series Test\\"
source_path = "C:\\Users\\Zachary Windover\\Documents\\Climate Series Test\\Source\\"
raw_data_path = "C:\\Users\\Zachary Windover\\Documents\\Climate Series Test\\Raw Data\\"
file_name = "Countries.csv"

# Import source.csv into pandas dataframe
file = source_path + file_name
countries_df = pd.read_csv(file)

# Convert dataframe to list of Country objects
countries_list = countries_df.values.tolist()

countries = []

for country in countries_list:
    countries.append(Country(country[0], country[1], country[2]))

for country in countries:
    #print(f"'{country.name}' \t '{country.url}' \t '{country.dir}'")
    print(f"'{country.name}': '{country.url}'")


# Purge & Clean Slate, or Append
runtype = input("\nRun Options: \n(1) Purge & Clean Slate \n(2) Append \n\n")

# Purge & Clean Slate
# Deletes all current files in the raw data folder
# And re-downloads all files from all websites again
#if int(runtype) == 1:
#    clean_slate_protocol(raw_data_path)

#elif int(runtype) == 2:
#    append_protocol()




# Set up Chrome options to set download directory
#raw_data_dir = "C:\\Users\\Zachary Windover\\Documents\\Climate Series Test\\Raw Data\\Canada\\"
#website_url_dir = "https://climate.weather.gc.ca/prods_servs/cdn_climate_summary_e.html"
#chrome_options = webdriver.ChromeOptions()

#prefs = {

#    "download.default_directory": raw_data_dir,
#    "download.prompt_for_download": False,
#    "download.directory_upgrade": True,
#    "safebrowsing.enabled": True

#}

#chrome_options.add_experimental_option("prefs", prefs)

# Start Selenium WebDriver
#driver = webdriver.Chrome(options=chrome_options)
#driver.get("https://climate.weather.gc.ca/prods_servs/cdn_climate_summary_e.html")

# Get all available years
#year_dropdown = Select(driver.find_element(By.ID, "intYear"))
#years = [option.text for option in year_dropdown.options]
#years = [int(y) for y in years if y.isdigit()] # Convert to integers

#print(f"Available years: {years}")

# WebDriverWait setup
#wait = WebDriverWait(driver, 5) # Wait up to 2 seconds

# Iterate through all available years
#for year in years:

#    # Select the year
#    year_dropdown.select_by_value(str(year))

    # Wait for month dropdown to refresh
#    wait.until(lambda driver: len(Select(driver.find_element(By.ID, "intMonth")).options) > 1)

    # Get all available months for the year
#    month_dropdown = Select(driver.find_element(By.ID, "intMonth"))

    #print(f"UPDATED DEBUG: Available months for {year}: {[option.text for option in month_dropdown.options]}")

#    months = [option.text for option in month_dropdown.options]
#    months = [option.get_attribute("value") for option in month_dropdown.options]
#    months = [int(m) for m in months if m.isdigit()] # Convert to integers

    #print(f"Available months for {year}: {months}")

    # Iterate through all available months for the year
#    for month in months:

        #print(f"Downloading: Year {year}, Month {month}")

        # Select the month
#        month_dropdown.select_by_value(str(month))

        # Wait for province dropdown to refresh
#        wait.until(lambda driver: len(Select(driver.find_element(By.ID, "prov")).options) > 0)

        # Select "All" for Province dropdown
#        province_dropdown = Select(driver.find_element(By.ID, "prov"))
#        province_dropdown.select_by_index(0)

        # Select CSV file format
#        format_checkbox = driver.find_element(By.ID, "csv")
#        format_checkbox.click()

        # Click Download button
#        download_button = driver.find_element(By.NAME, "btnSubmit")
#        download_button.click()

        # Wait for download to complete
        #time.sleep(10)


# Select Year
#year_dropdown = Select(driver.find_element(By.ID, "intYear"))
#year = 1840
#year_dropdown.select_by_value(str(year))

# Select Month
#month_dropdown = Select(driver.find_element(By.ID, "intMonth"))
#month = 1
#month_dropdown.select_by_value(str(month))

# Select Province
#province_dropdown = Select(driver.find_element(By.ID, "prov"))
#province_dropdown.select_by_index(0)

# Select CSV Format
#format_checkbox = driver.find_element(By.ID, "csv")
#format_checkbox.click()

# Click Download Button
#download_button = driver.find_element(By.NAME, "btnSubmit")
#download_button.click()

# Wait for download to complete
#time.sleep(3)

# Close the browser
#driver.quit(v)