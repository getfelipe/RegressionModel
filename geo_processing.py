import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import geopandas as gpd
from folium import plugins
from shapely.geometry import Point
import json


# fig, ax = plt.subplots()
# sns.scatterplot(data=df, x='longitude', y='latitude', ax=ax, alpha=0.2, hue='ocean_proximity')
# plt.show()

# fig, ax = plt.subplots()
# sns.jointplot(data=df, x='longitude', y='latitude', ax=ax, alpha=0.2)
# plt.show()

# fig, ax = plt.subplots()
# sns.scatterplot(data=df, x='longitude', y='latitude', ax=ax, hue='median_income_cat', palette='coolwarm')
# plt.show()

# # Valores contínuos com barra de coloração

# fig, ax = plt.subplots(figsize=(10, 6))

# norm_median_house_value = plt.Normalize(df['median_house_value'].min(), df['median_house_value'].max())
# sm_median_house_value = plt.cm.ScalarMappable(norm=norm_median_house_value, cmap='coolwarm')

# sns.scatterplot(data=df, x='longitude', y='latitude', ax=ax, hue='median_house_value', palette='coolwarm')
# ax.get_legend().remove()
# fig.colorbar(sm_median_house_value, ax=ax)
# plt.show()

df = pd.read_parquet('housing_manipulated.parquet')
gdf_counties = gpd.read_file('california_counties.geojson')

gdf_counties.head()

# Convertendo as latitudes e longitudes no formato GeoDataFrame
points = [Point(long, lat) for long, lat in zip(df['longitude'], df['latitude'])]
gdf = gpd.GeoDataFrame(df, geometry=points)


# Convertendo os pontos do GeoDataFrame no mesmo formato que os polígonos dos condados da Califórnia.
gdf = gdf.set_crs(epsg=4326)
gdf_counties = gdf_counties.to_crs(epsg=4326)


# O objetivo é separar os pontos do GeoDataFrame dentro dos condados da Califórnia.
# Como os pontos estão dentro dos condados, o parâmetro 'within' será escolhido.
gdf_joined = gpd.sjoin(gdf, gdf_counties, how='left', predicate='within')

# Existem alguns pontos que não foram encontrados os respectivos condados, isso pode ser devido algum tratamento feito antes nas colunas de latitude e longitude, 
# que fizeram alterar esses valores, seja aplicação de média, sistema de referência errado, coordenadas arredondas ou o próprio equipamento que coleta esses dados.
# Entretanto, é possível executar técnicas que de certa forma tente recuperar o valor original dessas linhas.
gdf_joined[gdf_joined.isnull().any(axis=1)]

# Vamos coletar o índice dessas linhas para tratá-las
null_rows = gdf_joined[gdf_joined[['index_right']].isnull().any(axis=1)].index
gdf_joined = gdf_joined.drop(columns=["index_right", "fullname", "abcode", "ansi"])
# Então, uma forma de resolver isso é utilizar o centróide, que é representar vários pontos em um único ponto.
# Ele representa o centro da área, que resume essa área. 
# E então, através do centróide dos condados realizar a diferença, que será a distância, entre os pontos faltantes e definir a menor distância encontrada dos condados em relação ao ponto. 
gdf_counties['centroid'] = gdf_counties.centroid

def county_nearlier(row):
    point = row["geometry"]
    distances = gdf_counties["centroid"].distance(point)
    idx_county_nearlier = distances.idxmin()
    county_nearlier = gdf_counties.loc[idx_county_nearlier]
    return county_nearlier[["name", "abbrev"]]


gdf_joined.loc[null_rows, ["name", "abbrev"]] = gdf_joined.loc[null_rows].apply(county_nearlier, axis=1)
gdf_joined.isnull().sum()

gdf_joined.dropna(inplace=True)
gdf_joined.loc[null_rows, ['name', 'abbrev']].value_counts()


gdf_counties = gdf_counties.merge(
    gdf_joined.groupby('name').median(numeric_only=True),
    left_on='name',
    right_index=True,

)


gdf_counties.head()

county_ocean_prox = gdf_joined[['name', 'ocean_proximity']].groupby('name').agg(pd.Series.mode)
gdf_counties = gdf_counties.merge(
    county_ocean_prox,
    left_on='name',
    right_index=True
)

gdf_counties.info()




fig, axis = plt.subplots(1,2, figsize=(20,8))

gdf_counties.plot(ax=axis[0], 
                edgecolor='black',
                column='median_house_value',
                cmap='coolwarm',
                legend=True,
                legend_kwds={'label': 'Median House Value', 'orientation': 'vertical'})

# axis[0].scatter(
#     gdf_joined['longitude'],
#     gdf_joined['latitude'],
#     color='red',
#     s=1,
#     alpha=0.2,
# )


gdf_counties.plot(ax=axis[1], 
                edgecolor='black',
                column='median_income',
                cmap='Yro',
                legend=True,
                legend_kwds={'label': 'Median Income', 'orientation': 'vertical'})



for x, y, abbrev in zip(gdf_counties['centroid'].x, gdf_counties['centroid'].y, gdf_counties['abbrev']):
    axis[0].text(x, y, abbrev, ha='center', va='center', fontsize=8)
    axis[1].text(x, y, abbrev, ha='center', va='center', fontsize=8)

plt.show()

gdf_counties.info()

gdf_counties.to_parquet('gdf_counties.parquet')
