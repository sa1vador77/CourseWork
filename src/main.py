#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
import pandas as pd
from xgboost import XGBRegressor
from geopy.distance import geodesic
from sklearn.ensemble import RandomForestRegressor

from ml_handlers.load_data import load
from ml_handlers.get_azimuth import get_azimuth_func
from ml_handlers.processing import DataProceccing


def predict_flat_price(
    rf_model: RandomForestRegressor, xgb_model: XGBRegressor, 
    city_center_coordinates: list[float], flat_params: dict[str: list[float]]
    ) -> None:
    """
    Предсказывает цену предложения квартиры на основе параметров квартиры и обученных моделей.
    
    :param rf_model: Обученная модель RandomForestRegressor.
    :param xgb_model: Обученная модель XGBRegressor.
    :param city_center_coordinates: Координаты центра города в виде [широта, долгота].
    :param flat_params: Словарь с параметрами квартиры, где ключи - названия параметров, значения - списки значений.
    """
    # Создаем датафрейм с параметрами квартиры
    flat = pd.DataFrame(flat_params)

    # Рассчитываем недостающие параметры квартиры - расстояние от центра города и азимут
    flat['distance'] = (
        list(map(lambda x, y: 
        geodesic(city_center_coordinates, [x, y]).meters, flat['latitude'], flat['longitude']))
    )
    flat['azimuth'] = (
        list(map(lambda x, y: 
        get_azimuth_func(x, y, city_center_coordinates), flat['latitude'], flat['longitude']))
    )
    flat['distance'] = flat['distance'].round(0)
    flat['azimuth'] = flat['azimuth'].round(0)

    # Удаляем ненужные столбцы с широтой и долготой
    flat = flat.drop(['latitude', 'longitude'], axis=1)

    # Вычисляем предсказанное значение стоимости по двум моделям
    rf_prediction_flat = rf_model.predict(flat).round(0)
    xgb_prediction_flat = xgb_model.predict(flat).round(0)

    # Усредняем полученные значения и умножаем на общую площадь квартиры
    price = (rf_prediction_flat * 0.5 + xgb_prediction_flat * 0.5) * flat['totalArea'][0]

    # Печатаем предсказанное значение цены предложения
    print(f'Предсказанная моделью цена предложения: {int(price[0].round(-3))} рублей')


def main() -> None:
    train_X, train_Y, val_X, val_Y = DataProceccing.prepare_data_for_training(
        df=DataProceccing.remove_outliers(
            df=DataProceccing.encode_categorical_columns(
                df=DataProceccing.data_processing(
                    df=load('data/data.csv')
                )
            )
        )
    )[1:]
    predict_flat_price(
        rf_model=DataProceccing.random_forest_model(
            train_X, val_X, train_Y, val_Y
        )[0],
        xgb_model=DataProceccing.xgboost_model(
            train_X, val_X, train_Y, val_Y
        )[0],
        city_center_coordinates=[55.751244, 37.618423],
        flat_params={
            'wallsMaterial': [6], 
            'floorNumber': [7],
            'floorsTotal': [7],
            'totalArea': [101.2],
            'kitchenArea': [16.7],
            'latitude': [55.858817],
            'longitude': [37.638755]
        }
    )


if __name__ == "__main__":
    main()
