import numpy as np
import pandas as pd
import xgboost as xgb
from numpy import ndarray
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from ml_handlers.load_data import load
from ml_handlers.get_azimuth import get_azimuth_func
from ml_handlers.print_metrics import print_metrics_func


class DataProceccing:

    @staticmethod
    def data_processing(df: DataFrame) -> DataFrame:
        """
        Обрабатывает данные в переданном датафрейме, добавляя новые признаки и фильтруя строки.

        :param df: Датафрейм с данными о квартирах.
        """
        
        # Рассчитываем цену за квадратный метр для каждой квартиры
        df['price_per_sqm'] = df['price'] / df['totalArea']

        # Задаем широту и долготу центра города и рассчитываем для каждой квартиры расстояние от центра и азимут
        city_center_coordinates = [55.7522, 37.6156]
        df['distance'] = (
            list(map(lambda lat, lon: 
            geodesic(city_center_coordinates, [lat, lon]).meters, df['latitude'], df['longitude']))
        )
        df['azimuth'] = (
            list(map(lambda lat, lon: 
            get_azimuth_func(lat, lon, city_center_coordinates), df['latitude'], df['longitude']))
        )

        # Фильтруем датафрейм, оставляя только те квартиры, которые расположены не дальше 40 км от центра города
        df = df.loc[df['distance'] < 40000]

        # Округляем значения столбцов стоимости метра, расстояния и азимута
        df['price_per_sqm'] = df['price_per_sqm'].round(0)
        df['distance'] = df['distance'].round(0)
        df['azimuth'] = df['azimuth'].round(0)

        # Выводим сводную информацию о датафрейме и его столбцах
        return df
    

    @staticmethod
    def remove_outliers(df: DataFrame, num_outliers: int = 3000) -> DataFrame:
        """
        Удаляет выбросы из датафрейма на основе межквартильного размаха (IQR).

        :param df: Датафрейм с данными.
        :param num_outliers: Количество строк, подходящих под критерии выбросов, которые будут удалены.
        """

        for column in df.columns:
            if not pd.api.types.is_numeric_dtype(df[column]):
                raise ValueError(f"Столбец {column} содержит нечисловые данные.")
        
        # Вычисляем первый и третий квартили для каждого столбца
        first_quartile = df.quantile(q=0.25)
        third_quartile = df.quantile(q=0.75)
        
        # Рассчитываем межквартильный размах (IQR)
        IQR = third_quartile - first_quartile
        
        # Вычисляем количество выбросов в каждой строке
        outliers = (
            df[((df > (third_quartile + 1.5 * IQR)) | (df < (first_quartile - 1.5 * IQR)))
            .any(axis=1)].count(axis=1)
        )
        
        # Сортируем строки по количеству выбросов в убывающем порядке
        outliers.sort_values(ascending=False, inplace=True)

        # Выбираем первые num_outliers строк с наибольшим количеством выбросов
        outliers = outliers.head(num_outliers)
        
        # Удаляем выбранные строки из датафрейма
        df.drop(outliers.index, inplace=True)

        return df
    

    @staticmethod
    def encode_categorical_columns(df: DataFrame) -> DataFrame:
        """
        Вычисляет столбцы с категорийными признаками и заменяет их на числовые значения.

        :param df: Датафрейм с данными.
        """
        
        # Вычисляем столбцы с категорийными признаками
        categorical_columns = df.columns[df.dtypes == 'object']
        
        # Инициализируем кодировщик меток
        labelencoder = LabelEncoder()
        
        # Преобразуем категорийные признаки в числовые
        for column in categorical_columns:
            # Преобразуем категорийный столбец
            df[column] = labelencoder.fit_transform(df[column])

        return df
    

    @staticmethod
    def prepare_data_for_training(df: DataFrame) -> tuple[list, DataFrame, DataFrame, Series, Series]:
        """
        Подготавливает данные для обучения модели.

        :param df: Датафрейм с данными.
        :return: Кортеж с признаками обучения, данными обучения и валидации, 
        а также целевыми переменными обучения и валидации.
        """

        # Назначаем целевой переменной цену 1 кв. метра, а можно и цену всей квартиры, тогда будет y = df['price']
        y = df['price_per_sqm']

        # Создаем список признаков, на основании которых будем строить модели
        features = [
                    'wallsMaterial', 
                    'floorNumber', 
                    'floorsTotal', 
                    'totalArea', 
                    'kitchenArea',
                    'distance',
                    'azimuth'
                ]

        # Создаем датафрейм, состоящий из признаков, выбранных ранее
        X = df[features]

        # Проводим случайное разбиение данных на выборки для обучения (train) и валидации (val)
        # по умолчанию в пропорции 0.75/0.25
        train_X, val_X, train_Y, val_Y = train_test_split(X, y, random_state=1)
        return (features, train_X, val_X, train_Y, val_Y)


    @staticmethod
    def random_forest_model(
        train_X: DataFrame, train_Y: DataFrame, 
        val_X: Series, val_Y: Series
        ) -> tuple[RandomForestRegressor, ndarray]:
        # Создаем регрессионную модель случайного леса 
        rf_model = RandomForestRegressor(
            n_estimators=2000, n_jobs=-1, bootstrap=False,
            criterion='friedman_mse', max_features=3, random_state=1,
            max_depth=55, min_samples_split=5
        )
        """
        Обучает модель случайного леса и выполняет предсказание на валидационных данных.

        :param train_X: Признаки обучения.
        :param train_Y: Целевая переменная обучения.
        :param val_X: Признаки валидации.
        :param val_Y: Целевая переменная валидации.
        :return: Кортеж с обученной моделью и предсказанными значениями на валидационных данных.
        """

        # Проводим подгонку модели на обучающей выборке 
        rf_model.fit(train_X, train_Y)

        # Вычисляем предсказанные значения цен на основе валидационной выборки
        rf_prediction = rf_model.predict(val_X).round(0)

        # Вычисляем и печатаем величины ошибок при сравнении известных цен квартир 
        # из валидационной выборки с предсказанными моделью
        print_metrics_func(rf_prediction, val_Y)
        return (rf_model, rf_prediction)


    @staticmethod
    def xgboost_model(
        train_X: DataFrame, train_Y: DataFrame,
        val_X: Series, val_Y: Series
        ) -> tuple[xgb.XGBRegressor, ndarray]:
        """
        Обучает модель XGBoost и выполняет предсказание на валидационных данных.

        :param train_X: Признаки обучения.
        :param train_Y: Целевая переменная обучения.
        :param val_X: Признаки валидации.
        :param val_Y: Целевая переменная валидации.
        :return: Кортеж с обученной моделью и предсказанными значениями на валидационных данных.
        """
        #Создаем регрессионную модель XGBoost
        xgb_model = xgb.XGBRegressor(
            objective ='reg:gamma', learning_rate = 0.01, max_depth = 45, 
            n_estimators = 2000, nthread = -1, eval_metric = 'gamma-nloglik', 
        )
        # Проводим подгонку модели на обучающей выборке 
        xgb_model.fit(train_X, train_Y)

        # Вычисляем предсказанные значения цен на основе валидационной выборки
        xgb_prediction = xgb_model.predict(val_X).round(0)

        # Вычисляем и печатаем величины ошибок при сравнении известных цен квартир 
        # из валидационной выборки с предсказанными моделью
        print_metrics_func(xgb_prediction, val_Y)
        return (xgb_model, xgb_prediction)

    
    @staticmethod
    def average_predictions_and_evaluate(rf_prediction: ndarray, xgb_prediction: ndarray, val_Y: Series) -> None:
        """
        Усредняет предсказания двух моделей и вычисляет метрики качества предсказаний.

        :param rf_prediction: Предсказания модели случайного леса.
        :param xgb_prediction: Предсказания модели XGBoost.
        :param val_Y: Целевая переменная валидации.
        """
        # Усредняем предсказания обоих моделей
        prediction = rf_prediction * 0.5 + xgb_prediction * 0.5
        
        # Вычисляем и печатаем величины ошибок для усредненного предсказания
        print_metrics_func(prediction, val_Y)

    
    @staticmethod
    def plot_feature_importances(filepath: str, rf_model: RandomForestRegressor, features: list):
        """
        Визуализирует важность признаков в модели случайного леса.

        :param filepath: Путь к файлу с данными.
        :param rf_model: Обученная модель случайного леса.
        :param features: Список признаков.
        """
        # Рассчитываем важность признаков в модели Random Forest
        importances = rf_model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in rf_model.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]

        # Печатаем рейтинг признаков
        df = load(filepath)
        X = df[features]
        print("Рейтинг важности признаков:")
        for f in range(X.shape[1]):
            print("%d. %s (%f)" % (f + 1, features[indices[f]], importances[indices[f]]))

        # Строим столбчатую диаграмму важности признаков
        plt.figure()
        plt.title("Важность признаков")
        plt.bar(range(X.shape[1]), importances[indices], color="g", yerr=std[indices], align="center")
        plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
        plt.xlim([-1, X.shape[1]])
        plt.tight_layout()  # Для лучшего расположения элементов
        plt.show()
