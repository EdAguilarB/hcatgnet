import os
import sys
from pathlib import Path

import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from hcatgnet.utils.plot_utils import parity_mean, plot_error_distribution
from hcatgnet.utils.utils_model import extract_metrics
from options.base_options import BaseOptions


def plot_results_GNN(
    experiments_gnn: Path = Path("results"),
    folds: int = 10,
    log_dir_results: Path = Path("results"),
):

    r2_gnn, mae_gnn, rmse_gnn = [], [], []
    accuracy_gnn, precision_gnn, recall_gnn = [], [], []

    results_all = pd.DataFrame(
        columns=[
            "index",
            "Test_Fold",
            "Val_Fold",
            "Method",
            "real_ddG",
            "predicted_ddG",
        ]
    )

    for outer in range(1, folds + 1):

        outer_gnn = os.path.join(experiments_gnn, f"Fold_{outer}_test_set")

        metrics_gnn = extract_metrics(
            file=outer_gnn + f"/performance_outer_test_fold{outer}.txt"
        )

        r2_gnn.append(metrics_gnn["R2"])
        mae_gnn.append(metrics_gnn["MAE"])
        rmse_gnn.append(metrics_gnn["RMSE"])
        accuracy_gnn.append(metrics_gnn["Accuracy"])
        precision_gnn.append(metrics_gnn["Precision"])
        recall_gnn.append(metrics_gnn["Recall"])

        for inner in range(1, folds):

            real_inner = inner + 1 if outer <= inner else inner

            gnn_dir = os.path.join(
                experiments_gnn, f"Fold_{outer}_test_set", f"Fold_{real_inner}_val_set"
            )

            df_gnn = pd.read_csv(gnn_dir + "/predictions_test_set.csv", index_col=0)

            df_gnn["Test_Fold"] = outer
            df_gnn["Val_Fold"] = real_inner
            df_gnn["Method"] = "GNN"

            results_all = pd.concat([results_all, df_gnn], axis=0)

    os.makedirs(log_dir_results, exist_ok=True)

    results_all["Error"] = results_all["real_ddG"] - results_all["predicted_ddG"]

    results_all = results_all.reset_index(drop=True)
    results_all.to_csv(f"{log_dir_results}/predictions_all.csv", index=False)

    gnn_predictions = results_all[results_all["Method"] == "GNN"]

    print("\n")

    gnn_predictions = (
        gnn_predictions.groupby(["index", "Method"])
        .agg(
            real_ddG=("real_ddG", "first"),
            mean_predicted_ddG=("predicted_ddG", "mean"),
            std_predicted_ddG=("predicted_ddG", "std"),
        )
        .reset_index()
    )

    parity_mean(df=gnn_predictions, save_path=log_dir_results)
    plot_error_distribution(df=gnn_predictions, save_path=log_dir_results)

    print("All plots have been saved in the directory {}".format(log_dir_results))


if __name__ == "__main__":
    opt = BaseOptions().parse()
    exp_dir = os.path.join(opt.log_dir_results, opt.filename[:-4], "test")
    plot_results_GNN(exp_dir, opt)
