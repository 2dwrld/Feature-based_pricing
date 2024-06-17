from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from mlxtend.regressor import StackingRegressor


def tune_hyperparameters(X, y):
    rf = RandomForestRegressor(random_state=42)
    gb = GradientBoostingRegressor(random_state=42)
    meta_regressor = LinearRegression()

    stacking_regressor = StackingRegressor(
        regressors=[('rf', rf), ('gb', gb)],
        meta_regressor=meta_regressor
    )

    param_grid = {
        'regressors__rf__n_estimators': [50, 100],  # [50, 100]
        'regressors__rf__max_depth': [5, 10],  # [5, 10]
        'regressors__rf__min_samples_split': [2, 5],  # [2, 5]
        'regressors__rf__min_samples_leaf': [1, 2],  # [1, 2]

        'regressors__gb__n_estimators': [50, 100],  # [50, 100]
        'regressors__gb__learning_rate': [0.05, 0.1],  # [0.05, 0.1]
        'regressors__gb__max_depth': [5, 10],  # [5, 10]
        'regressors__gb__min_samples_split': [2, 5],  # [2, 5]
        'regressors__gb__min_samples_leaf': [1, 2],  # [1, 2]

        'meta_regressor__fit_intercept': [True, False]
    }

    grid_search = GridSearchCV(estimator=stacking_regressor, param_grid=param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X, y)

    return grid_search.best_estimator_
