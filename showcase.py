from sklearn.datasets import load_breast_cancer

from automl import AutoML


def main():
    automl = AutoML(
        metric='accuracy',
        time_left_for_this_task=120,  # ограничивает общее время поиска оптимальной архитектуры переданным кол-вом
        # секунд. Чем больше время, тем более "качественная" архитектура может быть найдена.
        # Для демо просто передается две минуты.
        per_run_time_limit=30,  # ограничивает время на одну итерацию поиска
        memory_limit=None  # снимает ограничение по используемой памяти (можно задать кол-во мегабайт).
    )
    print("Unfitted AutoML object: ", automl)
    x, y = load_breast_cancer(return_X_y=True)  # Используем для тестирования стандартный датасет из sklearn.
    print("Dataset loaded. Fitting...")
    automl.fit(x, y)  # Запускаем процесс оптимизации и обучения моделей/ансамбля.
    print("Fitted AutoML object: ", automl)
    print("Score: ", automl.score)  # Выводим точность модели в метрике, с которой инициализировалась модель.
    print("Leaderboard: ", automl.leaderboard)  # Выводим таблицу найденных моделей
    print("Ensemble: ", automl.ensemble)  # Выводим подробное описание созданного ансамбля.
    prediction = automl.predict(x)  # Используем модель для классификации.


if __name__ == '__main__':
    main()
