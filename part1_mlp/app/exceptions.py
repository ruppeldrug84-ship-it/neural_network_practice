"""Кастомные исключения для модуля многослойного перцептрона."""


class NeuralNetworkError(Exception):
    """Базовое исключение для ошибок нейронной сети."""
    pass


class InvalidLayerSizeError(NeuralNetworkError):
    """Исключение при некорректном размере слоя."""
    pass


class MismatchedDataError(NeuralNetworkError):
    """Исключение при несовпадении размерности данных и входного слоя."""
    pass


class ModelNotTrainedError(NeuralNetworkError):
    """Исключение при попытке предсказания на неинициализированной модели."""
    pass


class DataFormatError(NeuralNetworkError):
    """Исключение при неверном формате или повреждении файла данных."""
    pass