import numpy as np
import json
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from scripts.model import CustomStackingRegressor


def tune_hyperparameters(X, y, param_file='best_params.json'):
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
        'model__regressors__0__n_estimators': [50, 100, 150, 200],
        'model__regressors__0__max_depth': [5, 10, 15, 20],
        'model__regressors__1__n_estimators': [50, 100, 150, 200],
        'model__regressors__1__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'model__regressors__2__fit_intercept': [True, False],
        'model__meta_regressor__n_estimators': [50, 100, 150, 200],
        'model__meta_regressor__learning_rate': [0.01, 0.05, 0.1, 0.2]
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

    try:
        # Попытка загрузки лучших параметров из файла
        with open(param_file, 'r') as file:
            best_params = json.load(file)
            print("Загружены лучшие параметры из файла")
            # Установим лучшие параметры в pipeline
            pipeline.set_params(**best_params)
            # Фитинг pipeline с данными, чтобы обеспечить соответствие ColumnTransformer
            pipeline.fit(X, y)
            return pipeline
    except (FileNotFoundError, json.JSONDecodeError):
        # Если файл не найден или содержит ошибки, запускаем GridSearchCV
        print("Запуск поиска гиперпараметров...")
        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid,
                                   cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X, y)

        # Сохранение лучших параметров в файл
        with open(param_file, 'w') as file:
            json.dump(grid_search.best_params_, file)

        print(f"Лучшие параметры: {grid_search.best_params_}")
        print(f"Лучший скор (MSE): {np.sqrt(-grid_search.best_score_)}")

        return grid_search.best_estimator_
