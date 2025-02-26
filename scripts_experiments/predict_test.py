import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from hcatgnet.services.model_report import network_outer_report, network_report
from hcatgnet.utils.utils_model import tml_report
from options.base_options import BaseOptions


def predict_final_test(
    graph_dataset,
    handcrafted_descriptors_dataset,
    descriptors,
    folds: int = 10,
    GNN_experiment_path: Path = Path("results"),
    TML_experiment_path: Path = Path("results"),
    GNN_log_dir: Path = Path("results"),
    TML_log_dir: Path = Path("results"),
) -> None:

    test_loader = DataLoader(graph_dataset, shuffle=False)

    # elif representation == "morgan":
    #     fingerprints = calculate_morgan_fingerprints(
    #         df=test_set, smiles_cols=opt.mol_cols, variance_threshold=0
    #     )
    #     test_set = pd.concat([test_set, fingerprints], axis=1)
    # elif representation == "circus_fp":
    #     test_set = test_set[opt.mol_cols + ["temp", "ddG", "index"]]
    #     if opt.filename == "biaryl.csv":
    #         fingerprints = pd.read_csv(
    #             "data/datasets/circus_descriptors/biaryl_circus_descriptors.csv"
    #         )
    #     else:
    #         fingerprints = pd.read_csv(
    #             "data/datasets/circus_descriptors/diene_circus_descriptors.csv"
    #         )
    #     test_set = test_set.drop(opt.mol_cols, axis=1)
    #     descriptors = ["temp"] + fingerprints.columns.tolist()
    #     test_set = pd.merge(test_set, fingerprints, left_index=True, right_index=True)

    for outer in range(1, folds + 1):
        print("Analysing models trained using as test set {}".format(outer))
        for inner in range(1, folds):

            real_inner = inner + 1 if outer <= inner else inner

            print(
                "Analysing models trained using as validation set {}".format(real_inner)
            )

            model_dir = (
                GNN_experiment_path
                / f"Fold_{outer}_test_set"
                / f"Fold_{real_inner}_val_set"
            )

            model = torch.load(model_dir / "model.pth", weights_only=False)
            model_params = torch.load(model_dir / "model_params.pth", weights_only=True)
            train_loader = torch.load(
                model_dir / "train_loader.pth", weights_only=False
            )
            val_loader = torch.load(model_dir / "val_loader.pth", weights_only=False)

            network_report(
                log_dir=GNN_log_dir,
                loaders=(train_loader, val_loader, test_loader),
                outer=outer,
                inner=real_inner,
                loss_lists=[None, None, None],
                model=model,
                model_params=model_params,
                best_epoch=None,
                save_all=False,
            )

            tml_dir = (
                TML_experiment_path
                / f"Fold_{outer}_test_set"
                / f"Fold_{real_inner}_val_set"
            )

            model = joblib.load(tml_dir / "model.sav")
            train_data = pd.read_csv(tml_dir / "train.csv")
            val_data = pd.read_csv(tml_dir / "val.csv")

            tml_report(
                log_dir=TML_log_dir,
                outer=outer,
                inner=real_inner,
                model=model,
                data=(train_data, val_data, handcrafted_descriptors_dataset),
                save_all=False,
                descriptors=descriptors,
            )

        network_outer_report(
            log_dir=GNN_log_dir / f"Fold_{outer}_test_set",
            outer=outer,
            folds=folds,
        )

        network_outer_report(
            log_dir=TML_log_dir / f"Fold_{outer}_test_set/",
            outer=outer,
            folds=folds,
        )


if __name__ == "__main__":
    opt = BaseOptions().parse()
    predict_final_test(opt)
