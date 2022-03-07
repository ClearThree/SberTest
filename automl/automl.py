from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from autosklearn.classification import AutoSklearnClassifier
from sklearn.model_selection import train_test_split

from automl.errors import NotFittedError, WrongMetricError
from automl.metrics import METRICS, SKLEARN_METRICS


class AutoML:
    def __init__(self, metric: str, **kwargs):
        """
        Инициализирует объект класса AutoML
        :param metric: строка, в которой содержится название метрики, на её основе происходит выбор модели и оптимизация
         ансамбля.
        :param kwargs: опциональные аргументы для инициализации класса AutoSklearnClassifier.
        """
        if metric not in METRICS:
            raise WrongMetricError(metric)
        metric_scorer = METRICS[metric]
        self.auto_sklearn = AutoSklearnClassifier(metric=metric_scorer, **kwargs)
        self.fitted = False
        self.metric = metric
        self.score = None

    def fit(
        self,
        x: Union[List, np.array, pd.DataFrame],
        y: Union[List, np.array, pd.DataFrame],
        split_kwargs: Dict[str, Any] = None,
        fit_kwargs: Dict[str, Any] = None,
    ):
        """
        Делит полученный датасет на обучающую и тестовую выборку, выполняет оптимизацию и обучение ансамбля из моделей.
        Выполняет тестирование на тестовой выборке и сохраняет результат тестирования.
        :param x: Данные для обучения.
        :param y: Разметка для данных.
        :param split_kwargs: опциональные аргументы для функции разбиения данных на обучающую и тестовую выборку.
        :param fit_kwargs: опциональные аргументы для функции fit класса AutoSklearnClassifier.
        """
        split_kwargs = split_kwargs if split_kwargs else {}
        fit_kwargs = fit_kwargs if fit_kwargs else {}
        x_train, x_test, y_train, y_test = train_test_split(x, y, **split_kwargs)
        self.auto_sklearn.fit(x_train, y_train, **fit_kwargs)
        self.fitted = True
        self.score = self.test(x_test, y_test)

    def predict(self, x: Union[List, np.array, pd.DataFrame]) -> np.array:
        """
        Выполняет классификацию для полученного набора данных.
        :param x: Данные для классификации.
        :return: Результат классификации.
        :raises NotFittedError в случае, когда используется еще необученный ансамбль моделей.
        """
        if not self.fitted:
            raise NotFittedError
        return self.auto_sklearn.predict(x)

    def test(self, x: Union[List, np.array, pd.DataFrame], y: Union[List, np.array, pd.DataFrame]) -> float:
        """
        Выполняет тестирование для полученного набора данных.
        :param x: Данные для выполнения классификации.
        :param y: Верная разметка для переданных данных.
        :return: Полученное значение метрики.
        :raises NotFittedError в случае, когда используется еще необученный ансамбль моделей.
        """
        if not self.fitted:
            raise NotFittedError
        predictions = self.auto_sklearn.predict(x)
        return SKLEARN_METRICS[self.metric](y, predictions)

    @property
    def leaderboard(self) -> pd.DataFrame:
        """
        Метод-геттер для возвращения таблицы полученных моделей.
        :return: Таблица моделей.
        :raises NotFittedError в случае, когда используется еще необученный ансамбль моделей.
        """
        if not self.fitted:
            raise NotFittedError
        return self.auto_sklearn.leaderboard()

    @property
    def ensemble(self) -> Dict[str, Any]:
        """
        Метод-геттер для возвращения полного описания ансамбля.
        :return: Словарь с описанием ансамбля.
        :raises NotFittedError в случае, когда используется еще необученный ансамбль моделей.
        """
        if not self.fitted:
            raise NotFittedError
        return self.auto_sklearn.show_models()

    def __repr__(self):
        return f"AutoML(metric={self.metric}, fitted={self.fitted}, score={self.score})"

    def __str__(self):
        return self.__repr__()
