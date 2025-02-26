import re

import pandas as pd
from torch.utils.data import Subset
from torch_geometric.explain import CaptumExplainer, Explainer, GNNExplainer

from hcatgnet.gnn_interpretation.utils_explain import (
    explain_dataset, plot_denoised_mols, plot_importances,
    plot_molecule_importance, visualize_score_features)


def denoise_mol(
    model,
    mol_dataset,
    mol_index,
    denoise_mol,
    include_Hs=False,
    analyse_feature=None,
    norm_denoise=True,
):

    denoise_mol = re.sub(r"\s+", "", denoise_mol).lower()

    mol_graph = Subset(mol_dataset, [mol_index])

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="regression",
            task_level="graph",
            return_type="raw",
        ),
    )

    masks = explain_dataset(
        dataset=mol_dataset,
        mol_graph=mol_graph,
        explainer=explainer,
        include_Hs=include_Hs,
    )

    mask = masks[denoise_mol]

    plot_denoised_mols(
        mask=mask,
        mol_dataset=mol_dataset,
        mol_graph=mol_graph[0],
        mol=denoise_mol,
        analysis=analyse_feature,
        norm=norm_denoise,
    )


def explain_node_feats(model, dataset, include_Hs, feature_sizes) -> None:

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="regression",
            task_level="graph",
            return_type="raw",
        ),
    )

    masks = explain_dataset(dataset, dataset, explainer, include_Hs)

    results_all = pd.DataFrame()

    for mol, mask in masks.items():

        mol_feats_score = visualize_score_features(
            score=mask, feature_sizes=feature_sizes
        )
        mol_feats_score = mol_feats_score.loc[mol_feats_score["score"] != 0]
        mol_feats_score["labels"] = mol_feats_score["labels"].apply(
            lambda m: f"{mol[0].upper()}. " + m
        )
        print(f"{mol} node features score: \n", mol_feats_score)

        results_all = pd.concat([results_all, mol_feats_score])

    results_all = results_all.sort_values("score", ascending=False)
    results_all["score"] = results_all["score"].astype(int)

    plot_importances(
        df=results_all,
    )


def shapley_analysis(model, dataset, explain_mol, plot_mol) -> None:

    plot_mol = re.sub(r"\s+", "", plot_mol).lower()

    mol = Subset(dataset, [explain_mol])
    mol = mol[0]

    explainer = Explainer(
        model=model,
        algorithm=CaptumExplainer("ShapleyValueSampling"),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="regression",
            task_level="graph",
            return_type="raw",
        ),
    )

    print("Reaction ddG: {:.2f}".format(mol.y.item()))
    print(
        "Reaction predicted ddG: {:.2f}".format(
            explainer.get_prediction(
                x=mol.x, edge_index=mol.edge_index, batch_index=mol.batch
            ).item()
        )
    )
    explanation = explainer(x=mol.x, edge_index=mol.edge_index, batch_index=mol.batch)
    plot_molecule_importance(
        mol_graph=mol, mol=plot_mol, explanation=explanation, palette="normal"
    )
