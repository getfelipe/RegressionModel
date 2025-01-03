from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OrdinalEncoder,
    OneHotEncoder,
    PolynomialFeatures,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
    QuantileTransformer
)

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
from train_model import *

df = pd.read_parquet('housing_manipulated.parquet')
df.dropna(inplace=True)
gdf_counties = pd.read_parquet('gdf_counties.parquet')


target_column = ['median_house_value']
one_hot_encoder_column = [ 'ocean_proximity']
ordinal_encoder_column = ['median_income_cat']


x = df.drop(columns=target_column)
y = df[target_column]
robust_columns = df.columns.difference(target_column + one_hot_encoder_column + ordinal_encoder_column)

pipeline_robust = Pipeline(
    steps=[
        ('robust_scaler', RobustScaler()),
        ('poly', PolynomialFeatures(degree=1, include_bias=False))
    ]
)

preprocessing = ColumnTransformer(
    transformers=[
        ('ordinal_encoder', OrdinalEncoder(categories='auto'), ordinal_encoder_column),
        ('one_hot', OneHotEncoder(drop='first'), one_hot_encoder_column),
        ('robust_scaler_poly', pipeline_robust, robust_columns),

    ],
    remainder='passthrough'
)


param_grid = {
    'regressor__preprocessor__robust_scaler_poly__poly__degree': [1, 2, 3],
    'regressor__reg__alpha': [1E-2, 5E-2, 0.1, 0.5, 1, 5, 10, 20],
}


grid_search = grid_search_cv_regressor(
    regressor=Ridge(),
    preprocessor=preprocessing,
    target_transformer=QuantileTransformer(output_distribution='normal'),
    param_grid=param_grid
)


grid_search.fit(x, y)
print(grid_search.best_params_)
print(grid_search.best_score_)


coefs = coeficients(
    grid_search.best_estimator_.regressor_['reg'].coef_,
    grid_search.best_estimator_.regressor_['preprocessor'].get_feature_names_out()
)

plot_coefs(coefs)


regressors = {
    'DummyRegressor': {
        'preprocessor': None,
        'regressor': DummyRegressor(strategy='mean'),
        'target_transformer': None
    },
    'LinearRegressor': {
        'preprocessor': preprocessing,
        'regressor': LinearRegression(),
        'target_transformer': None
    },
    'Ridge_grid': {
        'preprocessor': grid_search.best_estimator_.regressor_['preprocessor'],
        'regressor': grid_search.best_estimator_.regressor_['reg'],
        'target_transformer': grid_search.best_estimator_.transformer_,
    },
}


results = {
    model: train_validate_model(x, y, **attributes)
    for model, attributes in regressors.items()

}

df_results = arrange_results(results)
df_results.groupby('model').mean()

metrics(df_results)

plot_residues_estimator(grid_search.best_estimator_, x, y)

coefs[coefs['coeficients'].between(-0.2, 0.2) & (coefs['coeficients'] != 0)]

