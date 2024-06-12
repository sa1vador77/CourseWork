import numpy as np

def mean_absolute_percentage_error(y_true: list[float] | np.ndarray, y_pred: list[float] | np.ndarray) -> float:
    """
    Вычисляет среднюю абсолютную процентную ошибку (MAPE) между истинными и предсказанными значениями.

    :param y_true: Список или массив истинных значений.
    :param y_pred: Список или массив предсказанных значений.
    :return: Средняя абсолютная процентная ошибка (MAPE), выраженная в процентах.
    """
    
    # Преобразуем списки в numpy массивы для удобства математических операций
    y_true_array = np.array(y_true)
    y_pred_array = np.array(y_pred)
    
    # Вычисляем абсолютные процентные ошибки
    absolute_percentage_errors = np.abs((y_true_array - y_pred_array) / y_true_array)
    
    # Вычисляем среднее значение абсолютных процентных ошибок и умножаем на 100 для преобразования в проценты
    mape = np.mean(absolute_percentage_errors) * 100
    
    return mape