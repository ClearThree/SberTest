from autosklearn.metrics import (
    accuracy, f1, log_loss, mean_absolute_error, mean_squared_error, median_absolute_error, precision, r2, recall,
    roc_auc
) # noqa
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss as sk_log_loss, mean_absolute_error as sk_mae, mean_squared_error as sk_mse,
    median_absolute_error as sk_medae, precision_score, r2_score, recall_score, roc_auc_score)  # noqa

METRICS = {
    "accuracy": accuracy,
    "f1": f1,
    "precision": precision,
    "recall": recall,
    "roc_auc": roc_auc,
    "log_loss": log_loss,
    "r2": r2,
    "mae": mean_absolute_error,
    "mse": mean_squared_error,
    "medae": median_absolute_error,
}

SKLEARN_METRICS = {
    "accuracy": accuracy_score,
    "f1": f1_score,
    "precision": precision_score,
    "recall": recall_score,
    "roc_auc": roc_auc_score,
    "log_loss": sk_log_loss,
    "r2": r2_score,
    "mae": sk_mae,
    "mse": sk_mse,
    "medae": sk_medae,
}
