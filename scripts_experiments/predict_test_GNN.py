import os
import sys
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from icecream import ic

from hcatgnet.services.model_report import network_outer_report, network_report
from options.base_options import BaseOptions


def predict_final_test_GNN(
    graph_dataset,
    folds: int = 10,
    GNN_experiment_path: Path = Path("results"),
    log_results_dir: Path = Path("results"),
) -> None:

    test_loader = DataLoader(graph_dataset, shuffle=False)

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
                log_dir=log_results_dir,
                loaders=(train_loader, val_loader, test_loader),
                outer=outer,
                inner=real_inner,
                loss_lists=[None, None, None],
                model=model,
                model_params=model_params,
                best_epoch=None,
                save_all=False,
            )

        network_outer_report(
            log_dir=log_results_dir / f"Fold_{outer}_test_set",
            outer=outer,
            folds=folds,
        )


if __name__ == "__main__":
    opt = BaseOptions().parse()
    predict_final_test_GNN(opt)
