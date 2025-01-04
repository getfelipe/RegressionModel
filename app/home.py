import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import pydeck as pdk
import shapely
from joblib import load


housing_clean = 'housing_manipulated.parquet'
gdf_counties = 'gdf_counties.parquet'
ridge_poly_model = 'ridge_polyfeat_target_quantile.joblib'


@st.cache_data
def load_data_housing():
    return pd.read_parquet(housing_clean)

@st.cache_data
def load_data_geo():
    gdf = gpd.read_parquet(gdf_counties)
    gdf = gdf.explode(ignore_index=True)


    def fix_orient_geometry(geometry):
        if not geometry.is_valid:
            geometry = geometry.buffer(0)
        if isinstance(geometry, (shapely.geometry.Polygon, shapely.geometry.MultiPolygon)):
            geometry = shapely.geometry.polygon.orient(geometry, sign=1.0)

        return geometry
    

    gdf['geometry'] = gdf['geometry'].apply(fix_orient_geometry)

    def get_polygon_coordinates(geometry):
        return (
            [[[x, y] for x, y in geometry.exterior.coords]]
            if isinstance(geometry, shapely.geometry.Polygon)
            else [
                [[x, y] for x, y in polygon.exterior.coords]
                for polygon in geometry.geoms
            ]
        )
    
    gdf['geometry'] = gdf['geometry'].apply(get_polygon_coordinates)

    return gdf

@st.cache_resource
def load_model():
    return load(ridge_poly_model)

df = load_data_housing()
df.drop(columns=['housing_median_age_cat'], inplace=True)
gdf = load_data_geo()
model = load_model()


st.title("Property Price Forecasts")
column1, column2 = st.columns(2)


with column1:
    with st.form(key='form'):
        counties = sorted(set(list(gdf['name'].sort_values())))
        county_select = st.selectbox('County', counties)



        latitude = gdf.query('name == @county_select')['latitude'].values
        longitude = gdf.query('name == @county_select')['longitude'].values

        housing_median_age = st.number_input('Property Age', value=10, min_value=1, max_value=50)
        total_rooms = gdf.query('name == @county_select')['total_rooms'].values
        total_bedrooms = gdf.query('name == @county_select')['total_bedrooms'].values
        population =gdf.query('name == @county_select')['population'].values
        households = gdf.query('name == @county_select')['households'].values

        median_income = st.slider("Average Income ($1,000s)", 5.0, 60.0, 45.0, 5.0)
        median_income_scale = median_income / 10
        ocean_proximity = gdf.query('name == @county_select')['ocean_proximity'].values
        bins_income = [0, 1.5, 3, 4.5, 6, np.inf]
        median_income_cat = np.digitize(median_income_scale, bins=bins_income)

        rooms_per_household = gdf.query('name == @county_select')['rooms_per_household'].values
        bedrooms_per_room = gdf.query('name == @county_select')['bedrooms_per_room'].values
        population_per_household = gdf.query('name == @county_select')['population_per_household'].values
        housing_median_age_cat = gdf.query('name == @county_select')['housing_median_age_cat'].values

        model_input = {
            'longitude': longitude,
            'latitude': latitude,
            'housing_median_age': housing_median_age,
            'total_rooms': total_rooms,
            'total_bedrooms': total_bedrooms,
            'population': population,
            'households': households,
            'median_income': median_income_scale,
            'ocean_proximity': ocean_proximity,
            'median_income_cat': median_income_cat,
            'rooms_per_household': rooms_per_household,
            'bedrooms_per_room': bedrooms_per_room,
            'population_per_household': population_per_household,
            'housing_median_age_cat': 3,
        }

        df_inputs = pd.DataFrame(model_input)#, index=[0])
        prevision_button = st.form_submit_button('Predict Price')
        if prevision_button:
            price = model.predict(df_inputs)
            st.metric(label=f"Predict Price: US$", value=f"{price[0][0]:.2f}")


with column2:
    view_state = pdk.ViewState(
        latitude=float(latitude[0]),
        longitude=float(longitude[0]),
        zoom=5,
        min_zoom=5,
        max_zoom=15,
    )


    county_selected = gdf.query('name == @county_select') [['name', 'geometry']]

    polygon_layer = pdk.Layer(
        'PolygonLayer',
        data=gdf[['name', 'geometry']],
        get_polygon='geometry',
        get_fill_color=[173, 216, 230, 100],  # Light teal
        get_line_color=[0, 0, 0],  # Black borders
        get_line_width=20,
        pickable=True,
        auto_highlight=True,
    )

    highlight_layer = pdk.Layer(
        'PolygonLayer',
        data=county_selected,
        get_polygon='geometry',
        get_fill_color=[220, 20, 60, 150],  # Crimson red
        get_line_color=[0, 0, 0],  # Black borders
        get_line_width=20,
        pickable=True,
        auto_highlight=True,
    )


    tooltip = {
        'html': '<b>{name}</b>',
        'style': {'backgroundColor': 'steelblue', 'color': 'white', 'fontsize': '10px'},
    }

    map_ = pdk.Deck(
        initial_view_state=view_state,
        map_style='light',
        layers=[
            polygon_layer,
            highlight_layer
            ],
        tooltip=tooltip,
    )

    st.pydeck_chart(map_)