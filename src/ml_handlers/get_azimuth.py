import math

def get_azimuth_func(latitude: float, longitude: float, city_center_coordinates: tuple[float, float]) -> float:
    """
    Вычисляет азимут (угол) между двумя точками на поверхности Земли, заданными их широтой и долготой.

    :param latitude: Широта конечной точки в градусах.
    :param longitude: Долгота конечной точки в градусах.
    :param city_center_coordinates: Кортеж, содержащий широту и долготу начальной точки (например, центра города).
    :return: Азимут в градусах, округленный до двух знаков после запятой.
    """
    
    # Извлекаем широту и долготу начальной точки
    start_latitude, start_longitude = city_center_coordinates
    
    # Переводим координаты из градусов в радианы для математических вычислений
    start_lat_rad = math.radians(start_latitude)
    end_lat_rad = math.radians(latitude)
    start_long_rad = math.radians(start_longitude)
    end_long_rad = math.radians(longitude)

    # Вычисляем синусы и косинусы широт
    cos_start_lat = math.cos(start_lat_rad)
    cos_end_lat = math.cos(end_lat_rad)
    sin_start_lat = math.sin(start_lat_rad)
    sin_end_lat = math.sin(end_lat_rad)
    
    # Вычисляем разность долготы и её синус и косинус
    delta_long_rad = end_long_rad - start_long_rad
    cos_delta_long = math.cos(delta_long_rad)
    sin_delta_long = math.sin(delta_long_rad)

    # Вычисляем вспомогательные переменные для определения азимута
    x = cos_start_lat * sin_end_lat - sin_start_lat * cos_end_lat * cos_delta_long
    y = sin_delta_long * cos_end_lat
    
    # Рассчитываем начальный азимут в градусах
    initial_azimuth_deg = math.degrees(math.atan(-y / x))

    # Корректируем азимут, если x отрицателен
    if x < 0:
        initial_azimuth_deg += 180

    # Преобразуем азимут в диапазон от -180 до 180 градусов
    adjusted_azimuth_rad = (initial_azimuth_deg + 180) % 360 - 180
    adjusted_azimuth_rad = -math.radians(adjusted_azimuth_rad)
    
    # Преобразуем азимут в радианы и затем обратно в градусы
    normalized_azimuth_rad = (
        adjusted_azimuth_rad - 
        (2 * math.pi * math.floor(adjusted_azimuth_rad / (2 * math.pi)))
        )
    normalized_azimuth_deg = math.degrees(normalized_azimuth_rad)

    # Возвращаем окончательный азимут, округленный до двух знаков после запятой
    return round(normalized_azimuth_deg, 2)
