import os
from pathlib import Path

import pandas as pd

from data.rhcaa import rhcaa_diene
from options.base_options import BaseOptions
from scripts_experiments.compare_gnn_tml import plot_results
from scripts_experiments.explain_gnn import (GNNExplainer_node_feats,
                                             denoise_graphs, shapley_analysis)
from scripts_experiments.predict_test import predict_final_test
from scripts_experiments.train_GNN import train_network_nested_cv
from scripts_experiments.train_TML import train_tml_model_nested_cv


def run_all_exp():

    opt = BaseOptions().parse()

    filename = opt.filename
    root = Path(opt.root)
    mol_cols = opt.mol_cols
    target_variable = opt.target_col
    include_Hs = opt.Add_Hs
    folds = opt.folds
    global_seed = opt.global_seed

    if opt.train_GNN:

        dataset = rhcaa_diene(
            filename=filename,
            root=root,
            molcols=mol_cols,
            target_variable=target_variable,
            include_Hs=include_Hs,
            num_folds=folds,
            random_seed=global_seed,
        )

        dir_results = Path("results") / dataset._name / "learning"
        dir_results_GNN = dir_results / "results_GNN"

        if not dir_results_GNN.exists():
            train_network_nested_cv(
                graph_dataset=dataset,
                log_results_dir=dir_results_GNN,
                folds=folds,
                global_seed=global_seed,
            )
        else:
            print("GNN model has already been trained")

    if opt.train_tml:

        data = pd.read_csv(root / "raw" / dataset.filename)

        if opt.descriptors == "bespoke":
            descriptors = [
                "LVR1",
                "LVR2",
                "LVR3",
                "LVR4",
                "LVR5",
                "LVR6",
                "LVR7",
                "VB",
                "ER1",
                "ER2",
                "ER3",
                "ER4",
                "ER5",
                "ER6",
                "ER7",
                "SStoutR1",
                "SStoutR2",
                "SStoutR3",
                "SStoutR4",
                "temp",
            ]
            data = data[descriptors + [target_variable, "fold", "index"]]

        else:
            raise ValueError("Descriptors not recognised")

        def dir_results_TML(algorithm: str, representation: str) -> Path:
            return dir_results / "results_TML" / algorithm / representation

        algorithm = opt.tml_algorithm
        representation = opt.descriptors

        dir_results_tml = dir_results_TML(algorithm, representation)

        if not dir_results_tml.exists():
            train_tml_model_nested_cv(
                data_csv=data,
                mol_cols=mol_cols,
                descriptors=descriptors,
                target_variable=target_variable,
                log_results_dir=dir_results_TML("rf", "bespoke"),
                tml_algorithm="rf",
                global_seed=global_seed,
            )
        else:
            print("TML model has already been trained")

    if opt.predict_unseen:
        if not os.path.exists(
            os.path.join(
                opt.log_dir_results,
                opt.filename_final_test[:-4],
                "results_TML",
                opt.tml_algorithm,
                opt.descriptors,
            )
        ):
            predict_final_test(opt)
        else:
            print("Prediction of unseen data has already been done")

    if opt.compare_models:
        if not os.path.exists(
            os.path.join(
                opt.log_dir_results,
                opt.filename[:-4],
                "comparison",
                f"GNN_vs_{opt.tml_algorithm}",
                opt.descriptors,
            )
        ):
            plot_results(os.path.join(opt.log_dir_results, opt.filename[:-4]), opt)
        else:
            print(
                f"GNN and TML ({opt.tml_algorithm}) Models have already been compared for {opt.filename[:-4]} dataset."
            )

        if not os.path.exists(
            os.path.join(
                opt.log_dir_results,
                opt.filename_final_test[:-4],
                "comparison",
                f"GNN_vs_{opt.tml_algorithm}",
                opt.descriptors,
            )
        ):
            plot_results(
                exp_dir=os.path.join(opt.log_dir_results, opt.filename_final_test[:-4]),
                opt=opt,
            )
        else:
            print(
                f"GNN and TML ({opt.tml_algorithm}) Models have already been compared for {opt.filename_final_test[:-4]} dataset."
            )

    if opt.denoise_graph:
        denoise_graphs(
            opt,
            exp_path=os.path.join(
                os.getcwd(),
                opt.log_dir_results,
                opt.filename[:-4],
                "learning",
                "results_GNN",
            ),
        )

    if opt.GNNExplainer:
        GNNExplainer_node_feats(
            opt,
            exp_path=os.path.join(
                os.getcwd(),
                opt.log_dir_results,
                opt.filename[:-4],
                "learning",
                "results_GNN",
            ),
        )

    if opt.shapley_analysis:
        shapley_analysis(
            opt,
            exp_path=os.path.join(
                os.getcwd(),
                opt.log_dir_results,
                opt.filename[:-4],
                "learning",
                "results_GNN",
            ),
        )


if __name__ == "__main__":
    run_all_exp()
