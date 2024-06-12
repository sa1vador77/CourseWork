import pandas as pd
from pandas import DataFrame


def load(filepath: str) -> DataFrame:
    """
    Загружает данные из CSV файла в DataFrame.

    :param filepath: Путь к файлу с данными.
    :return: DataFrame с загруженными данными.
    """

    df = pd.read_csv(filepath)
    return df
