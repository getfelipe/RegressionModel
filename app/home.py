import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd


from joblib import load 



housing_clean = r'/home/felipe/Documents/RegressionModel/housing_manipulated.parquet'
gdf_counties = r'/home/felipe/Documents/RegressionModel/gdf_counties.parquet'
housing_manipulated = r'/home/felipe/Documents/RegressionModel/housing_manipulated.parquet'
california_counties = r'/home/felipe/Documents/RegressionModel/california_counties.geojson'
ridge_poly_model = r'/home/felipe/Documents/RegressionModel/ridge_polyfeat_target_quantile.joblib'


@st.cache_data
def load_data_housing():
    return pd.read_parquet(housing_clean)

@st.cache_data
def load_data_geo():
    return gpd.read_parquet(gdf_counties)

@st.cache_resource
def load_model():
    return load(ridge_poly_model)

df = load_data_housing()
df.drop(columns=['housing_median_age_cat'], inplace=True)
gdf = load_data_geo()
model = load_model()




st.title("Property Price Forecasts")

longitude = st.number_input('Longitude', value=-122.33,)
latitude = st.number_input('latitude', value=37.88)
housing_median_age = st.number_input('Property Age', value=10)
total_rooms = st.number_input('Total of Rooms', value=5)
total_bedrooms = st.number_input('Total of bedrooms', value=2)
population = st.number_input('Population', value=3),
households = st.number_input('Households', value=3)

median_income = st.slider("Average Income, multiples of 10K USD", 0.5, 15.0, 4.5, 0.5)
ocean_proximity = st.selectbox('Ocean Proximity', df['ocean_proximity'].unique())
median_income_cat = st.number_input('Income Category', value=4)

rooms_per_household = st.number_input('Rooms per Households', value=5)
bedrooms_per_room = st.number_input('Bedrooms per Room', value=0.2)
population_per_household = st.number_input('Population per Household', value=5)

housing_median_age_cat = st.number_input('Average Age Category', value=3)

model_input = {
    'longitude': longitude,
    'latitude': latitude,
    'housing_median_age': housing_median_age,
    'total_rooms': total_rooms,
    'total_bedrooms': total_bedrooms,
    'population': population,
    'households': households,
    'median_income': median_income,
    'ocean_proximity': ocean_proximity,
    'median_income_cat': median_income_cat,
    'rooms_per_household': rooms_per_household,
    'bedrooms_per_room': bedrooms_per_room,
    'population_per_household': population_per_household,
    'housing_median_age_cat': housing_median_age_cat,
}

df_inputs = pd.DataFrame(model_input, index=[0])
prevision_button = st.button('Predict Psrice')
if prevision_button:
    price = model.predict(df_inputs)
    st.write('Preço Previsto: ', price)
