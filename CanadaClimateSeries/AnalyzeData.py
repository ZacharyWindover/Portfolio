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





#plt.figure(figsize=(10, 5))
#plt.plot(climate_df.groupby('Year')['Tm'].mean(), marker='o', linestyle='-')
#plt.xlabel("Year")
#plt.ylabel("Avg Temperature")
#plt.title("Average Temperature Over Time In Canada")
#plt.grid(True)
#plt.show()

geoclimate_df = gpd.GeoDataFrame(
    climate_df, geometry=gpd.points_from_xy(climate_df.Long, climate_df.Lat), crs="EPSG:4326"
)

#print(geoclimate_df.head())

world = gpd.read_file(get_path("naturalearth.land"))

# Restrict to Canada
ax = world.clip([-150, 40, -30, 90]).plot(color="white", edgecolor="black")
# point 1 = x direction left
# point 2 = y direcion down


geoclimate_df.plot(ax=ax, color="red")
plt.show()

# if Y < 40 or x < -46