import numpy as np
import pandas as pd
import geopandas as gpd
from geodatasets import get_path
import matplotlib.pyplot as plt
import os
import re

# Path to climate series directory
path = "C://Users//Zachary Windover//Documents//Climate Series//"
file = "processed_climate_summaries.csv"
file_path = os.path.join(path, file)

climate_df = pd.read_csv(file_path)

print(climate_df.head())