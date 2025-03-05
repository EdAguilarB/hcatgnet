from math import sqrt

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from hcatgnet.options.enums import ProblemTypes


def calculate_metrics(y_true: list, y_predicted: list, task=ProblemTypes.Regression):

    if task == ProblemTypes.Regression:
        r2 = r2_score(y_true=y_true, y_pred=y_predicted)
        mae = mean_absolute_error(y_true=y_true, y_pred=y_predicted)
        rmse = sqrt(mean_squared_error(y_true=y_true, y_pred=y_predicted))
        error = [(y_predicted[i] - y_true[i]) for i in range(len(y_true))]
        prctg_error = [
            abs(error[i] / y_true[i]) for i in range(len(error)) if y_true[i] != 0
        ]
        mbe = np.mean(error)
        mape = np.mean(prctg_error)
        error_std = np.std(error)
        metrics = [r2, mae, rmse, mbe, mape, error_std]
        metrics_names = [
            "R2",
            "MAE",
            "RMSE",
            "Mean Bias Error",
            "Mean Absolute Percentage Error",
            "Error Standard Deviation",
        ]

    elif task == ProblemTypes.Classification:
        accuracy = accuracy_score(y_true=y_true, y_pred=y_predicted)
        precision = precision_score(y_true=y_true, y_pred=y_predicted)
        recall = recall_score(y_true=y_true, y_pred=y_predicted)
        f1 = f1_score(y_true=y_true, y_pred=y_predicted)
        metrics = [accuracy, precision, recall, f1]
        metrics_names = ["Accuracy", "Precision", "Recall", "F1"]

    return np.array(metrics), metrics_names
