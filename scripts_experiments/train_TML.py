import os
import sys
from math import sqrt
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_squared_error

from hcatgnet.services.model_report import network_outer_report

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from hcatgnet.utils.utils_model import (
    calculate_morgan_fingerprints,
    choose_model,
    hyperparam_tune,
    load_variables,
    split_data,
    tml_report,
)
from options.base_options import BaseOptions


def train_tml_model_nested_cv(
    data_csv: pd.DataFrame,
    mol_cols: list,
    descriptors: list,
    target_variable: str,
    tml_algorithm: str = "rf",
    representation: str = "bespoke",
    folds: int = 10,
    log_results_dir: Path = Path("results"),
    global_seed: int = 20232023,
) -> None:

    print(
        "Initialising chiral ligands selectivity prediction using a traditional ML approach."
    )

    if representation == "morgan":
        data_csv = data_csv[mol_cols + ["temp", "ddG", "fold", "index"]]
        fingerprints = calculate_morgan_fingerprints(
            df=data_csv, smiles_cols=mol_cols, variance_threshold=0.01
        )
        data_csv = data_csv.drop(mol_cols, axis=1)
        descriptors = ["temp"] + fingerprints.columns.tolist()
        print(f"Using {len(descriptors)} fingerprints")
        data_csv = pd.concat([data_csv, fingerprints], axis=1)

    elif representation == "circus_fp":
        data_csv = data_csv[mol_cols + ["temp", "ddG", "fold", "index"]]
        if opt.filename == "biaryl.csv":
            fingerprints = pd.read_csv(
                "data/datasets/circus_descriptors/biaryl_circus_descriptors.csv"
            )
        else:
            fingerprints = pd.read_csv(
                "data/datasets/circus_descriptors/diene_circus_descriptors.csv"
            )
        data_csv = data_csv.drop(mol_cols, axis=1)
        descriptors = ["temp"] + fingerprints.columns.tolist()
        print(f"Using {len(descriptors)} fingerprints")
        data_csv = pd.merge(data_csv, fingerprints, left_index=True, right_index=True)

    else:
        pass

    # Nested cross validation
    ncv_iterator = split_data(data_csv)

    # Initiate the counter of the total runs and the total number of runs
    counter = 0
    TOT_RUNS = folds * (folds - 1)
    print("Number of splits: {}".format(folds))
    print("Total number of runs: {}".format(TOT_RUNS))

    # Hyperparameter optimisation
    print("Hyperparameter optimisation starting...")
    X, y, _ = load_variables(
        data=data_csv,
        descriptors=descriptors + [target_variable],
        target_variable=target_variable,
    )
    best_params = hyperparam_tune(
        X,
        y,
        choose_model(best_params=None, algorithm=tml_algorithm),
        global_seed,
    )
    print("Hyperparameter optimisation has finalised")
    print("Training starting...")
    print("********************************")

    # Loop through the nested cross validation iterators
    # The outer loop is for the outer fold or test fold
    for outer in range(1, folds + 1):
        # The inner loop is for the inner fold or validation fold
        for inner in range(1, folds):

            # Inner fold is incremented by 1 to avoid having same inner and outer fold number for logging purposes
            real_inner = inner + 1 if outer <= inner else inner
            # Increment the counter
            counter += 1

            # Get the train, validation and test sets
            train_set, val_set, test_set = next(ncv_iterator)
            # Choose the model
            model = choose_model(best_params, tml_algorithm)
            # Fit the model
            model.fit(train_set[descriptors], train_set["ddG"])
            # Predict the train set
            preds = model.predict(train_set[descriptors])
            train_rmse = sqrt(mean_squared_error(train_set["ddG"], preds))
            # Predict the validation set
            preds = model.predict(val_set[descriptors])
            val_rmse = sqrt(mean_squared_error(val_set["ddG"], preds))
            # Predict the test set
            preds = model.predict(test_set[descriptors])
            test_rmse = sqrt(mean_squared_error(test_set["ddG"], preds))

            print(
                "Outer: {} | Inner: {} | Run {}/{} | Train RMSE {:.3f} kJ/mol | Val RMSE {:.3f} kJ/mol | Test RMSE {:.3f} kJ/mol".format(
                    outer,
                    real_inner,
                    counter,
                    TOT_RUNS,
                    train_rmse,
                    val_rmse,
                    test_rmse,
                )
            )

            # Generate a report of the model performance
            tml_report(
                log_dir=log_results_dir,
                data=(train_set, val_set, test_set),
                outer=outer,
                inner=real_inner,
                model=model,
                descriptors=descriptors,
            )

            # Reset the variables of the training
            del model, train_set, val_set, test_set

        print("All runs for outer test fold {} completed".format(outer))
        print("Generating outer report")

        # Generate a report of the model performance for the outer/test fold
        network_outer_report(
            log_dir=log_results_dir / f"Fold_{outer}_test_set/",
            outer=outer,
            folds=folds,
        )

        print("---------------------------------")

    print("All runs completed")


if __name__ == "__main__":
    opt = BaseOptions().parse()
    train_tml_model_nested_cv(opt)
