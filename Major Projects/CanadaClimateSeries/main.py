import numpy as np
import pandas as pd
import geopandas as gpd
from geodatasets import get_path
import matplotlib.pyplot as plt
import os
import re


# Run Data Manager script
with open("DataManager.py") as file:
    exec(file.read())

