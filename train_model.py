import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter

import seaborn as sns
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import KFold, cross_validate 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_validate, GridSearchCV
from sklearn.metrics import PredictionErrorDisplay


def train_validate_model(x, y, regressor, preprocessor=None, 
                         target_transformer=None, n_splits=5):
    
    model = build_pipeline(regressor, preprocessor, target_transformer)

    kf = KFold(n_splits=n_splits, shuffle=True)

    scores = cross_validate(
        model, x, y, cv=kf, scoring=[
            'r2',
            'neg_mean_absolute_error',
            'neg_root_mean_squared_error',
        ]
    )

    return scores


def build_pipeline(regressor, preprocessor, target_transformer):
    if preprocessor:
        pipeline = Pipeline([('preprocessor', preprocessor), ('reg', regressor)])
    else:
        pipeline = Pipeline([('reg', regressor)])


    if target_transformer:
        model = TransformedTargetRegressor(
            regressor=pipeline, transformer=target_transformer
        )
    
    else:
        model = pipeline 

    return model



def arrange_results(results):
    for key, value in results.items():
        results[key]['time_seconds'] = (
            results[key]['fit_time'] + results[key]['score_time']
        )

    df_results = (
        pd.DataFrame(results).T.reset_index().rename(columns={'index': 'model'})
    )

    df_results_extended = df_results.explode(df_results.columns[1:].to_list()).reset_index(drop=True)


    try:
        df_results_extended = df_results_extended.apply(pd.to_numeric)
    except ValueError:
        pass

    return df_results_extended


def metrics(df):
    fig, axis = plt.subplots(2,2, figsize=(8,8), sharex=True)

    params = [
        'time_seconds',
        'test_r2',
        'test_neg_mean_absolute_error',
        'test_neg_root_mean_squared_error'
    ]

    types = [
        'Time (s)',
        'R2',
        'MAE',
        'RMSE'
    ]

    for ax, metric, name in zip(axis.flatten(), params, types):
        sns.boxplot(
            x='model',
            y=metric,
            data=df,
            ax=ax,
            showmeans=True
        )
        
        ax.set_title(name)
        ax.set_ylabel(name)
        ax.tick_params(axis='x', rotation=90)  # Fixed the typo here


    plt.tight_layout()
    


    return plt.show()


def coeficients(coefs, columns):
    return pd.DataFrame(
        data=coefs,
        index=columns, 
        columns=['coeficients'],
    ).sort_values(by='coeficients')

def plot_coefs(df, title='Coeficients'):
    df.plot.barh()
    plt.title(title)
    plt.axvline(x=0, color='0.5')
    plt.xlabel('Coeficients')
    plt.gca().get_legend().remove()
    plt.show()


def grid_search_cv_regressor(
    regressor,
    param_grid,
    preprocessor=None,
    target_transformer=None,
    n_splits=5,
    return_train_score=False,
):
    model = build_pipeline(
        regressor, preprocessor, target_transformer
    )

    kf = KFold(n_splits=n_splits, shuffle=True,)

    grid_search = GridSearchCV(
        model,
        cv=kf,
        param_grid=param_grid,
        scoring=["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"],
        refit="neg_root_mean_squared_error",
        n_jobs=-1,
        return_train_score=return_train_score,
        verbose=1,
    )

    return grid_search


def build_pipeline(
    regressor, preprocessor=None, target_transformer=None
):
    if preprocessor:
        pipeline = Pipeline([("preprocessor", preprocessor), ("reg", regressor)])
    else:
        pipeline = Pipeline([("reg", regressor)])

    if target_transformer:
        model = TransformedTargetRegressor(
            regressor=pipeline, transformer=target_transformer
        )
    else:
        model = pipeline
    return model


def plot_residuos(y_true, y_pred):
    residues = y_true - y_pred
    fig, axis = plt.subplots(1, 3, figsize=(12, 6))
    sns.histplot(residues, kde=True, ax=axis[0])

    error_display_01 = PredictionErrorDisplay.from_predictions(
        y_true=y_true, y_pred=y_pred, kind='residual_vs_predicted', ax=axis[1]
    )

    error_display_02 = PredictionErrorDisplay.from_predictions(
        y_true=y_true, y_pred=y_pred, kind='actual_vs_predicted', ax=axis[2]
    )

    plt.tight_layout()
    plt.show()


def plot_residues_estimator(estimator, x, y, eng_formatter=False, subsample=0.25):

    fix, axis = plt.subplots(1, 3, figsize=(12, 6))

    error_display_01 = PredictionErrorDisplay.from_estimator(
        estimator, x, y, kind='residual_vs_predicted', ax=axis[1],
        scatter_kwargs={'alpha': 0.2}, subsample=subsample
    )


    error_display_02 = PredictionErrorDisplay.from_estimator(
        estimator, x, y, kind='actual_vs_predicted', ax=axis[2],
        scatter_kwargs={'alpha': 0.2}, subsample=subsample
    )

    residues = error_display_01.y_true - error_display_01.y_pred
    sns.histplot(residues, kde=True, ax=axis[0])

    if eng_formatter:
        for ax in axis:
            ax.xaxis.set_major_formatter(EngFormatter())
            ax.yaxis.set_major_formatter(EngFormatter())

    plt.tight_layout()
    plt.show()