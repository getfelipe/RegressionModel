import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import geopandas as gpd
from folium import plugins
from shapely.geometry import Point
import json


df = pd.read_parquet('housing_manipulated.parquet')
gdf_counties = gpd.read_file('california_counties.geojson')

points = [Point(long, lat) for long, lat in zip(df['longitude'], df['latitude'])]
gdf = gpd.GeoDataFrame(df, geometry=points)

gdf = gdf.set_crs(epsg=4326)
gdf_counties = gdf_counties.to_crs(epsg=4326)
gdf_counties['centroid'] = gdf_counties.centroid

gdf_joined = gpd.sjoin(gdf, gdf_counties, how='left', predicate='within')
gdf_joined[gdf_joined.isnull().any(axis=1)]

null_rows = gdf_joined[gdf_joined[['index_right']].isnull().any(axis=1)].index

def county_nearlier(row):
    point = row["geometry"]
    distances = gdf_counties["centroid"].distance(point)
    idx_county_nearlier = distances.idxmin()
    county_nearlier = gdf_counties.loc[idx_county_nearlier]
    return county_nearlier[["name", "abbrev"]]


gdf_joined.loc[null_rows, ["name", "abbrev"]] = gdf_joined.loc[null_rows].apply(county_nearlier, axis=1)
gdf_joined.isnull().sum()