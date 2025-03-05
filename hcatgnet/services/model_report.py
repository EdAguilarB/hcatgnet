import csv
import os
import re
from datetime import date, datetime

import numpy as np
import pandas as pd
import torch

from hcatgnet.options.enums import ProblemTypes
from hcatgnet.services.metrics import calculate_metrics
from hcatgnet.services.model_training import predict_network
from hcatgnet.services.plotting import (
    create_st_parity_plot,
    create_training_plot,
    plot_tsne_with_subsets,
)


def network_report(
    log_dir: str,
    loaders: tuple,
    outer: int,
    inner: int,
    loss_lists: tuple,
    save_all: bool,
    model: torch.nn.Module,
    model_params: dict,
    best_epoch: int,
):
    """
    Generates a report for the model performance

    Args:
        log_dir (str): directory to save the report
        loaders (tuple): training, validation and test loaders
        outer (int): outer fold number
        inner (int): inner fold number
        loss_lists (tuple): training, validation and test loss lists
        save_all (bool): whether to save all the loaders and model
        model (torch.nn.Module): trained model
        model_params (dict): model parameters
        best_epoch (int): best epoch

    Returns:
        str: path to the report
    """

    # 1) Create a directory to store the results
    log_dir = "{}/Fold_{}_test_set/Fold_{}_val_set".format(log_dir, outer, inner)
    os.makedirs(log_dir, exist_ok=True)

    # 2) Time of the run
    today = date.today()
    today_str = str(today.strftime("%d-%b-%Y"))
    time = str(datetime.now())[11:]
    time = time[:8]
    run_period = "{}, {}\n".format(today_str, time)

    # 3) Unfold loaders and save loaders and model
    train_loader, val_loader, test_loader = loaders[0], loaders[1], loaders[2]
    N_train, N_val, N_test = (
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset),
    )
    N_tot = N_train + N_val + N_test
    if save_all == True:
        torch.save(train_loader, "{}/train_loader.pth".format(log_dir))
        torch.save(val_loader, "{}/val_loader.pth".format(log_dir))
        torch.save(model, "{}/model.pth".format(log_dir))
        torch.save(model_params, "{}/model_params.pth".format(log_dir))
    torch.save(test_loader, "{}/test_loader.pth".format(log_dir))
    loss_function = "RMSE_%"

    # 4) loss trend during training
    train_list = loss_lists[0]
    val_list = loss_lists[1]
    test_list = loss_lists[2]
    if train_list is not None and val_list is not None and test_list is not None:
        with open(
            "{}/{}.csv".format(log_dir, "learning_process"), "w", newline=""
        ) as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "Epoch",
                    "Train_{}".format(loss_function),
                    "Val_{}".format(loss_function),
                    "Test_{}".format(loss_function),
                ]
            )
            for i in range(len(train_list)):
                writer.writerow([(i + 1) * 5, train_list[i], val_list[i], test_list[i]])
        create_training_plot(
            df="{}/{}.csv".format(log_dir, "learning_process"),
            save_path="{}".format(log_dir),
        )

    # 5) Start writting report
    file1 = open("{}/performance.txt".format(log_dir), "w")
    file1.write(run_period)
    file1.write("---------------------------------------------------------\n")
    file1.write("GNN TRAINING AND PERFORMANCE\n")
    file1.write("Best epoch: {}\n".format(best_epoch))
    file1.write("Dataset Size = {}\n".format(N_tot))
    file1.write("***************\n")

    model.load_state_dict(model_params)

    y_pred, y_true, idx, emb_train = predict_network(model, train_loader, True)
    emb_train["set"] = "training"
    metrics, metrics_names = calculate_metrics(y_true, y_pred, task=model.problem_type)

    file1.write("Training set\n")
    file1.write("Set size = {}\n".format(N_train))

    for name, value in zip(metrics_names, metrics):
        file1.write("{} = {:.3f}\n".format(name, value))

    file1.write("***************\n")
    y_pred, y_true, idx, emb_val = predict_network(model, val_loader, True)
    emb_val["set"] = "val"
    metrics, metrics_names = calculate_metrics(y_true, y_pred, task=model.problem_type)

    file1.write("Validation set\n")
    file1.write("Set size = {}\n".format(N_val))

    for name, value in zip(metrics_names, metrics):
        file1.write("{} = {:.3f}\n".format(name, value))

    file1.write("***************\n")

    y_pred, y_true, idx, emb_test = predict_network(model, test_loader, True)
    emb_test["set"] = "test"

    emb_all = pd.concat([emb_train, emb_val, emb_test], axis=0)

    plot_tsne_with_subsets(
        data_df=emb_all,
        feature_columns=[i for i in range(128)],
        color_column="ddG_exp",
        set_column="set",
        fig_name="tsne_emb_exp",
        save_path=log_dir,
    )
    # plot_tsne_with_subsets(data_df=emb_all, feature_columns=[i for i in range(128)], color_column='ddG_pred', set_column='set', fig_name='tsne_emb_pred', save_path=log_dir)
    emb_all.to_csv("{}/embeddings.csv".format(log_dir))

    pd.DataFrame({"real_ddG": y_true, "predicted_ddG": y_pred, "index": idx}).to_csv(
        "{}/predictions_test_set.csv".format(log_dir)
    )

    face_pred = np.where(y_pred > 0, 1, 0)
    face_true = np.where(y_true > 0, 1, 0)
    metrics, metrics_names = calculate_metrics(
        face_true, face_pred, task=ProblemTypes.Classification
    )

    correct_side_add = face_pred == face_true

    file1.write("Test set\n")
    file1.write("Set size = {}\n".format(N_test))

    for name, value in zip(metrics_names, metrics):
        file1.write("{} = {:.3f}\n".format(name, value))

    # error = abs(y_pred-y_true)
    # y_true = y_true[correct_side_add]
    # y_pred = y_pred[correct_side_add]
    # idx = idx[correct_side_add]

    file1.write(
        "Test Set Total Correct Face of Addition Predictions = {}\n".format(
            np.sum(correct_side_add)
        )
    )

    metrics, metrics_names = calculate_metrics(y_true, y_pred, task=model.problem_type)

    for name, value in zip(metrics_names, metrics):
        file1.write("{} = {:.3f}\n".format(name, value))

    file1.write("---------------------------------------------------------\n")

    create_st_parity_plot(
        real=y_true,
        predicted=y_pred,
        figure_name="outer_{}_inner_{}".format(outer, inner),
        save_path="{}".format(log_dir),
    )
    # create_it_parity_plot(real = y_true, predicted = y_pred, index = idx, figure_name='outer_{}_inner_{}.html'.format(outer, inner), save_path="{}".format(log_dir))

    file1.write("OUTLIERS (TEST SET)\n")
    error_test = [(y_pred[i] - y_true[i]) for i in range(len(y_pred))]
    abs_error_test = [abs(error_test[i]) for i in range(len(y_pred))]
    std_error_test = np.std(error_test)

    outliers_list, outliers_error_list, index_list = [], [], []

    counter = 0

    for sample in range(len(y_pred)):
        if abs_error_test[sample] >= 3 * std_error_test:
            counter += 1
            outliers_list.append(idx[sample])
            outliers_error_list.append(error_test[sample])
            index_list.append(sample)
            if counter < 10:
                file1.write(
                    "0{}) {}    Error: {:.2f} kJ/mol    (index={})\n".format(
                        counter, idx[sample], error_test[sample], sample
                    )
                )
            else:
                file1.write(
                    "{}) {}    Error: {:.2f} kJ/mol    (index={})\n".format(
                        counter, idx[sample], error_test[sample], sample
                    )
                )

    file1.close()

    return "Report saved in {}".format(log_dir)


def network_outer_report(
    log_dir: str,
    outer: int,
    folds: int,
):
    """
    Generates a report for the model performance

    Args:
        log_dir (str): directory to save the report
        outer (int): outer fold number
        folds (int): number of folds

    Returns:
        str: path to the report
    """

    accuracy, precision, recall, r2, mae, rmse = [], [], [], [], [], []

    files = [
        log_dir / f"Fold_{i}_val_set/performance.txt"
        for i in range(1, folds + 1)
        if i != outer
    ]

    # Define regular expressions to match metric lines
    accuracy_pattern = re.compile(r"Accuracy = (\d+\.\d+)")
    precision_pattern = re.compile(r"Precision = (\d+\.\d+)")
    recall_pattern = re.compile(r"Recall = (\d+\.\d+)")
    r2_pattern = re.compile(r"R2 = (\d+\.\d+)")
    mae_pattern = re.compile(r"MAE = (\d+\.\d+)")
    rmse_pattern = re.compile(r"RMSE = (\d+\.\d+)")

    for file in files:
        with open(os.path.join(file), "r") as f:
            content = f.read()

        # Split the content by '*' to separate different sets
        sets = content.split("*")

        for set_content in sets:
            # Check if "Test set" is in the set content
            if "Test set" in set_content:
                # Extract metric values using regular expressions
                accuracy_match = accuracy_pattern.search(set_content)
                accuracy.append(float(accuracy_match.group(1)))
                precision_match = precision_pattern.search(set_content)
                precision.append(float(precision_match.group(1)))
                recall_match = recall_pattern.search(set_content)
                recall.append(float(recall_match.group(1)))
                r2_match = r2_pattern.search(set_content)
                try:
                    r2.append(float(r2_match.group(1)))
                except:
                    r2.append(0)
                mae_match = mae_pattern.search(set_content)
                mae.append(float(mae_match.group(1)))
                rmse_match = rmse_pattern.search(set_content)
                rmse.append(float(rmse_match.group(1)))

    # Calculate mean and standard deviation for each metric
    accuracy_mean = np.mean(accuracy)
    accuracy_std = np.std(accuracy)
    precision_mean = np.mean(precision)
    precision_std = np.std(precision)
    recall_mean = np.mean(recall)
    recall_std = np.std(recall)
    r2_mean = np.mean(r2)
    r2_std = np.std(r2)
    mae_mean = np.mean(mae)
    mae_std = np.std(mae)
    rmse_mean = np.mean(rmse)
    rmse_std = np.std(rmse)

    # Write the results to the file
    file1 = open("{}/performance_outer_test_fold{}.txt".format(log_dir, outer), "w")
    file1.write("---------------------------------------------------------\n")
    file1.write("Test Set Metrics (mean ± std)\n")
    file1.write("Accuracy: {:.3f} ± {:.3f}\n".format(accuracy_mean, accuracy_std))
    file1.write("Precision: {:.3f} ± {:.3f}\n".format(precision_mean, precision_std))
    file1.write("Recall: {:.3f} ± {:.3f}\n".format(recall_mean, recall_std))
    file1.write("R2: {:.3f} ± {:.3f}\n".format(r2_mean, r2_std))
    file1.write("MAE: {:.3f} ± {:.3f}\n".format(mae_mean, mae_std))
    file1.write("RMSE: {:.3f} ± {:.3f}\n".format(rmse_mean, rmse_std))
    file1.write("---------------------------------------------------------\n")

    return "Report saved in {}".format(log_dir)
