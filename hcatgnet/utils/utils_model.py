import os
import pickle
import re
from copy import copy, deepcopy
from datetime import date, datetime

import numpy as np
import pandas as pd
from icecream import ic
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler

from hcatgnet.services.metrics import calculate_metrics
from hcatgnet.services.plotting import create_st_parity_plot

######################################
######################################
######################################
######  traditional ML functions #####
######################################
######################################
######################################


def extract_metrics(file):

    metrics = {
        "Accuracy": None,
        "Precision": None,
        "Recall": None,
        "R2": None,
        "MAE": None,
        "RMSE": None,
    }

    with open(file, "r") as file:
        content = file.read()

    # Define regular expressions to match metric lines
    accuracy_pattern = re.compile(r"Accuracy: (\d+\.\d+) ± (\d+\.\d+)")
    precision_pattern = re.compile(r"Precision: (\d+\.\d+) ± (\d+\.\d+)")
    recall_pattern = re.compile(r"Recall: (\d+\.\d+) ± (\d+\.\d+)")
    r2_pattern = re.compile(r"R2: (\d+\.\d+) ± (\d+\.\d+)")
    mae_pattern = re.compile(r"MAE: (\d+\.\d+) ± (\d+\.\d+)")
    rmse_pattern = re.compile(r"RMSE: (\d+\.\d+) ± (\d+\.\d+)")

    accuracy_match = accuracy_pattern.search(content)
    precision_match = precision_pattern.search(content)
    recall_match = recall_pattern.search(content)
    r2_match = r2_pattern.search(content)
    mae_match = mae_pattern.search(content)
    rmse_match = rmse_pattern.search(content)

    # Update the metrics dictionary with extracted values
    if accuracy_match:
        metrics["Accuracy"] = {
            "mean": float(accuracy_match.group(1)),
            "std": float(accuracy_match.group(2)),
        }
    if precision_match:
        metrics["Precision"] = {
            "mean": float(precision_match.group(1)),
            "std": float(precision_match.group(2)),
        }
    if recall_match:
        metrics["Recall"] = {
            "mean": float(recall_match.group(1)),
            "std": float(recall_match.group(2)),
        }
    if r2_match:
        metrics["R2"] = {
            "mean": float(r2_match.group(1)),
            "std": float(r2_match.group(2)),
        }
    if mae_match:
        metrics["MAE"] = {
            "mean": float(mae_match.group(1)),
            "std": float(mae_match.group(2)),
        }
    if rmse_match:
        metrics["RMSE"] = {
            "mean": float(rmse_match.group(1)),
            "std": float(rmse_match.group(2)),
        }

    return metrics


def load_variables(data, descriptors: list, target_variable: str):

    data = data.filter(descriptors)

    # remove erroneous data
    data = data.dropna(axis=0)

    X = data.drop([target_variable], axis=1)
    X = RobustScaler().fit_transform(np.array(X))
    y = data[target_variable]
    print("Features shape: ", X.shape)
    print("Y target variable shape: ", y.shape)

    return X, y, descriptors


def choose_model(best_params, algorithm):

    if best_params == None:
        if algorithm == "rf":
            return RandomForestRegressor()
        if algorithm == "lr":
            return LinearRegression()
        if algorithm == "gb":
            return GradientBoostingRegressor()

    else:
        if algorithm == "rf":
            return RandomForestRegressor(
                n_estimators=best_params["n_estimators"],
                max_depth=best_params["max_depth"],
                min_samples_leaf=best_params["min_samples_leaf"],
                min_samples_split=best_params["min_samples_split"],
                random_state=best_params["random_state"],
            )
        if algorithm == "lr":
            return LinearRegression()
        if algorithm == "gb":
            return GradientBoostingRegressor(
                loss=best_params["loss"],
                learning_rate=best_params["learning_rate"],
                n_estimators=best_params["n_estimators"],
                max_depth=best_params["max_depth"],
                min_samples_leaf=best_params["min_samples_leaf"],
                min_samples_split=best_params["min_samples_split"],
                random_state=best_params["random_state"],
            )


def hyperparam_tune(X, y, model, seed):

    np.random.seed(seed)

    print("ML algorithm to be tunned:", str(model))

    if str(model) == "LinearRegression()":
        return None

    else:
        if str(model) == "RandomForestRegressor()":
            hyperP = dict(
                n_estimators=[100, 300, 500, 800],
                max_depth=[None, 5, 8, 15, 25, 30],
                min_samples_split=[2, 5, 10, 15, 100],
                min_samples_leaf=[1, 2, 5, 10],
                random_state=[seed],
            )
        elif str(model) == "GradientBoostingRegressor()":
            hyperP = dict(
                loss=["squared_error"],
                learning_rate=[0.1, 0.2, 0.3],
                n_estimators=[100, 300, 500, 800],
                max_depth=[None, 5, 8, 15, 25, 30],
                min_samples_split=[2],
                min_samples_leaf=[1, 2],
                random_state=[seed],
            )

        gridF = GridSearchCV(model, hyperP, cv=3, verbose=1, n_jobs=-1)
        bestP = gridF.fit(X, y)
        params = bestP.best_params_
        print("Best hyperparameters:", params, "\n")

        return params


def split_data(df: pd.DataFrame):
    """
    splits a dataset in a given quantity of folds
    """

    for outer in np.unique(df["fold"]):
        proxy = copy(df)
        test = proxy[proxy["fold"] == outer]

        for inner in np.unique(df.loc[df["fold"] != outer, "fold"]):

            val = proxy.loc[proxy["fold"] == inner]
            train = proxy.loc[(proxy["fold"] != outer) & (proxy["fold"] != inner)]
            yield deepcopy((train, val, test))


def predict_tml(model, data: pd.DataFrame, descriptors: list):

    y_pred = model.predict(data[descriptors])
    y_true = list(data["ddG"])
    idx = list(data.index)

    return np.array(y_pred), np.array(y_true), np.array(idx)


def calculate_morgan_fingerprints(
    df, smiles_cols, radius=2, n_bits=2048, variance_threshold=0.01
):
    """
    Featurize molecules in a pandas DataFrame using Morgan fingerprints, and remove columns with low variance.

    Parameters:
    - df: pandas DataFrame, containing SMILES strings.
    - smiles_cols: list of column names that contain SMILES strings.
    - radius: int, optional, the radius for the Morgan fingerprint (default is 2).
    - n_bits: int, optional, the length of the fingerprint vector (default is 2048).
    - variance_threshold: float, optional, the threshold below which fingerprint bits will be removed for low variance (default is 0.01).

    Returns:
    - pd.DataFrame: a DataFrame with the Morgan fingerprints as features, filtered for low variance bits.
    """

    def smiles_to_fingerprint(smiles):
        """Convert a SMILES string to a Morgan fingerprint."""
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius, nBits=n_bits, useChirality=True
            )
            return np.array(fp)
        else:
            return np.zeros(n_bits)  # Return a zero array if the SMILES is invalid

    # Initialize an empty DataFrame to store the fingerprints
    fingerprint_df = pd.DataFrame()

    for col in smiles_cols:
        # Apply fingerprint calculation for each SMILES column
        fp_array = df[col].apply(smiles_to_fingerprint)
        fp_df = pd.DataFrame(
            fp_array.tolist(),
            index=df.index,
            columns=[f"{col}_fp_{i}" for i in range(n_bits)],
        )

        # Concatenate the new fingerprints with the main dataframe
        fingerprint_df = pd.concat([fingerprint_df, fp_df], axis=1)

    # Calculate the variance of each fingerprint column
    variances = fingerprint_df.var()

    # Filter out columns with low variance
    selected_columns = variances[variances > variance_threshold].index
    filtered_fingerprint_df = fingerprint_df[selected_columns]

    return filtered_fingerprint_df


def calculate_circus_fingerprints(df, opt):

    if os.path.exists(
        "data",
        "datasets",
        "circus_descriptors",
        f"{opt.filename[:-4]}_circus_descriptors.csv",
    ):
        descriptors = pd.read_csv(
            "data",
            "datasets",
            "circus_descriptors",
            f"{opt.filename[:-4]}_circus_descriptors.csv",
        )
        return descriptors

    else:
        from CGRtools import smiles
        from doptools.chem.chem_features import ChythonCircus

        unique_smiles = []

        for mol_col in opt.smiles_cols:
            unique_smiles.append(df[mol_col].unique().tolist())

        clean_smiles = [[] for _ in range(len(opt.smiles_cols))]

        for i, mol_col in enumerate(opt.smiles_cols):

            for mol in unique_smiles[i]:
                clean_smiles[i].append(smiles(mol))
                clean_smiles[i][-1].clean2d()

        df_list = []

        for i, mol_col in enumerate(opt.smiles_cols):
            circus = ChythonCircus(lower=0, upper=2)
            circus.fit(clean_smiles[i])
            fp = circus.transform(clean_smiles[i])
            fp.rename(columns={col: f"{mol_col}_{col}" for col in fp.columns})
            fp.index = unique_smiles[i]
            df_list.append(fp)

        cols = [df.columns for df in df_list]

        fps_all = pd.DataFrame(index=df.index, columns=cols)

        for i, row in df.iterrows():
            fps = []
            for j, mol_col in enumerate(opt.smiles_cols):
                mol_fp = df_list.loc[row[mol_col]]
                fps.append(mol_fp)

            fps_all.loc[i] = pd.concat(fps, axis=0)

        fps_all.to_csv(
            "data",
            "datasets",
            "circus_descriptors",
            f"{opt.filename[:-4]}_circus_descriptors.csv",
        )

        return fps_all


def tml_report(log_dir, outer, inner, model, data, descriptors, save_all=True):

    # 1) create a directory to store the results
    log_dir = "{}/Fold_{}_test_set/Fold_{}_val_set".format(log_dir, outer, inner)
    os.makedirs(log_dir, exist_ok=True)

    # 2) Get time of the run
    today = date.today()
    today_str = str(today.strftime("%d-%b-%Y"))
    time = str(datetime.now())[11:]
    time = time[:8]
    run_period = "{}, {}\n".format(today_str, time)

    # 3) Unfold  train/val/test dataloaders
    train_data, val_data, test_data = data[0], data[1], data[2]
    N_train, N_val, N_test = len(train_data), len(val_data), len(test_data)
    N_tot = N_train + N_val + N_test

    # 4) Save dataframes for future use
    if save_all:
        train_data.to_csv("{}/train.csv".format(log_dir))
        val_data.to_csv("{}/val.csv".format(log_dir))
        pickle.dump(model, open("{}/model.sav".format(log_dir), "wb"))

    test_data.to_csv("{}/test.csv".format(log_dir))

    # 5) Performance Report
    file1 = open("{}/performance.txt".format(log_dir), "w")
    file1.write(run_period)
    file1.write("---------------------------------------------------------\n")
    file1.write("Traditional ML algorithm Performance\n")
    file1.write("Dataset Size = {}\n".format(N_tot))
    file1.write("***************\n")

    y_pred, y_true, idx = predict_tml(model, train_data, descriptors)
    metrics, metrics_names = calculate_metrics(y_true, y_pred, task="regression")

    file1.write("Training set\n")
    file1.write("Set size = {}\n".format(N_train))

    for name, value in zip(metrics_names, metrics):
        file1.write("{} = {:.3f}\n".format(name, value))

    file1.write("***************\n")
    y_pred, y_true, idx = predict_tml(model, val_data, descriptors)
    metrics, metrics_names = calculate_metrics(y_true, y_pred, task="regression")

    file1.write("Validation set\n")
    file1.write("Set size = {}\n".format(N_val))

    for name, value in zip(metrics_names, metrics):
        file1.write("{} = {:.3f}\n".format(name, value))

    file1.write("***************\n")

    y_pred, y_true, idx = predict_tml(
        model=model, data=test_data, descriptors=descriptors
    )

    pd.DataFrame({"real_ddG": y_true, "predicted_ddG": y_pred, "index": idx}).to_csv(
        "{}/predictions_test_set.csv".format(log_dir)
    )

    face_pred = np.where(y_pred > 0, 1, 0)
    face_true = np.where(y_true > 0, 1, 0)

    metrics, metrics_names = calculate_metrics(
        face_true, face_pred, task="classification"
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

    metrics, metrics_names = calculate_metrics(y_true, y_pred, task="regression")

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
