import numpy as np

def median_absolute_percentage_error(y_true: list[float] | np.ndarray, y_pred: list[float] | np.ndarray) -> float:
    """
    Вычисляет медианную абсолютную процентную ошибку (MdAPE) между истинными и предсказанными значениями.

    :param y_true: Список или массив истинных значений.
    :param y_pred: Список или массив предсказанных значений.
    :return: Медианная абсолютная процентная ошибка (MdAPE), выраженная в процентах.
    """
    
    # Преобразуем списки в numpy массивы для удобства математических операций
    y_true_array = np.array(y_true)
    y_pred_array = np.array(y_pred)
    
    # Вычисляем абсолютные процентные ошибки
    absolute_percentage_errors = np.abs((y_true_array - y_pred_array) / y_true_array)
    
    # Вычисляем медианное значение абсолютных процентных ошибок и умножаем на 100 для преобразования в проценты
    mdape = np.median(absolute_percentage_errors) * 100
    
    return mdape