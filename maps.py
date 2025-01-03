import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


from IPython.display import display
import geopandas as gpd
import folium
from folium import plugins
from shapely.geometry import Point
import json

size_map = {'width': 500, 'height': 500}
fig = folium.Figure(**size_map)

df = pd.read_parquet('housing_manipulated.parquet')
map_center = [df['latitude'].mean(), df['longitude'].mean()]
map_ = folium.Map(location=map_center, 
           zoom_start=5,
           ).add_to(fig)

gdf_counties = pd.read_parquet('gdf_counties.parquet')



with open('california_counties.geojson', 'r') as file:
    geojson = json.load(file)


fig = folium.Figure(**size_map)
map_ = folium.Map(
    location=map_center, 
    zoom_start=5,
    tiles='cartodb positron',
    control_scale=True,
).add_to(fig)

# folium.GeoJson(
#     geojson,
#     name='geojson',
#     key_on='features.name',
#     tooltip=folium.GeoJsonTooltip(fields=['name']),
# ).add_to(map_)




folium.Choropleth(
    geo_data=geojson,
    name='choropleth',
    fill_color='YlGn',
    data=gdf_counties,
    fill_opacity=0.7,
    legend_name='Median Income',
    columns=['abbrev', 'median_income'],
    key_on='feature.properties.abbrev',
    line_opacity=0.3,
).add_to(map_)

folium.LayerControl().add_to(map_)
folium.LatLngPopup().add_to(map_)
plugins.MousePosition().add_to(map_)
display(map_)


