import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scripts.model import CustomStackingRegressor

def tune_hyperparameters(X, y):
    # Определяем базовые модели
    base_models = [
        ('rf', RandomForestRegressor()),
        ('gb', GradientBoostingRegressor()),
        ('lr', LinearRegression())
    ]

    # Извлечение моделей из кортежей для StackingRegressor
    base_models_only = [model for name, model in base_models]

    # Определяем мета модель
    meta_model = GradientBoostingRegressor()

    # Создаем стековую модель
    stacked_model = CustomStackingRegressor(
        regressors=base_models_only,
        meta_regressor=meta_model
    )

    # Определяем параметры для подбора
    param_grid = {
        'model__regressors__0__n_estimators': [50, 100],
        'model__regressors__0__max_depth': [5, 10],
        'model__regressors__1__n_estimators': [50, 100],
        'model__regressors__1__learning_rate': [0.01, 0.1],
        'model__regressors__2__fit_intercept': [True, False],
        'model__meta_regressor__n_estimators': [50, 100],
        'model__meta_regressor__learning_rate': [0.01, 0.1]
    }

    # Настройка предварительной обработки данных
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), X.columns)
        ],
        remainder='passthrough'
    )

    # Создание конвейера с предварительной обработкой и стековой моделью
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', stacked_model)
    ])

    # Настройка GridSearchCV
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid,
                               cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

    # Обучение модели
    grid_search.fit(X, y)

    # Выводим лучшие параметры и лучшую оценку
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score (MSE): {np.sqrt(-grid_search.best_score_)}")

    return grid_search.best_estimator_
