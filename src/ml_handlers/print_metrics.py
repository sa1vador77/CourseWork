import numpy as np
from sklearn.metrics import r2_score

from ml_handlers import mape, mdape

def print_metrics_func(prediction: list[float] | np.ndarray, val_y: list[float] | np.ndarray) -> None:
    """
    Печатает рассчитанные значения коэффициента детерминации, средней и медианной абсолютных ошибок.

    :param prediction: Список или массив предсказанных значений.
    :param val_y: Список или массив истинных значений.
    """
    
    # Рассчитываем метрики
    r2 = r2_score(val_y, prediction)
    mape_ = mape.mean_absolute_percentage_error(val_y, prediction)
    mdape_ = mdape.median_absolute_percentage_error(val_y, prediction)

    # Печатаем результаты
    text = (
        f"R\u00b2: {r2:.2f}\n\n"
        f"Средняя абсолютная ошибка: {mape_:.3f} %\n"
        f"Медианная абсолютная ошибка: {mdape_:.3f} %"
    )
    print(text)
