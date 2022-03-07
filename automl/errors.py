from automl.metrics import METRICS


class NotFittedError(Exception):
    def __init__(self):
        super().__init__(
            "Trying to use not fitted model. Please, fit the model by .fit(...) method"
        )


class WrongMetricError(Exception):
    def __init__(self, metric: str, message: str = None):
        self.metric = metric
        self.message = message if message else f"Passed metric '{self.metric}' is not supported." \
                                               f"Options are: {METRICS.keys()}"
        super().__init__(self.message)
