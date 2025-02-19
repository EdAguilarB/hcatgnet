import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import AddHs
from seaborn import barplot
from torch.utils.data import Subset
from torch_geometric.data import Dataset
from torch_geometric.explain import Explainer
from torch_geometric.loader import DataLoader

from hcatgnet.options.plot_mol import colors_n, sizes


def explain_dataset(
    dataset: Dataset,
    mol_graph: Subset,
    explainer: Explainer,
    include_Hs: bool,
):
    """ """

    # Creates a loader object from the dataset
    loader = DataLoader(dataset=mol_graph)

    # Iterate over the graphs in the loader
    mol_masks = {}

    for mol_name in dataset[0].mols.keys():
        mol_masks[mol_name] = []

    na_atoms = {}

    for graph in loader:

        ia = 0

        for mol_name, smiles in graph.mols.items():
            mol = Chem.MolFromSmiles(smiles[0])
            if include_Hs:
                mol = AddHs(mol)
            na_atoms[mol_name] = (ia, ia + mol.GetNumAtoms())
            ia += mol.GetNumAtoms()
        # Run the explanation function over the reaction graph
        explanation = explainer(
            x=graph.x, edge_index=graph.edge_index, batch_index=graph.batch
        )

        # Get the masks for each node within the molecule
        masks = explanation.node_mask

        # masks = masks / torch.max(masks.sum(dim=1))

        for mol_name, (ia, fa) in na_atoms.items():
            mol_masks[mol_name].append(masks[ia:fa])

    for mol_name in mol_masks.keys():
        mol_masks[mol_name] = torch.cat(mol_masks[mol_name], dim=0)

    return mol_masks


def plot_denoised_mols(
    mask: torch.Tensor,
    mol_dataset: Dataset,
    mol_graph: Subset,
    mol: str,
    analysis: str = None,
    norm: bool = True,
):
    """
    Plots denoised molecules by analyzing feature importance.

    Args:
        mask (torch.Tensor): The importance mask tensor.
        mol_dataset (Dataset): The dataset containing molecular features.
        mol_graph (Subset): The molecular graph subset.
        mol (str): The molecule identifier.
        analysis (str, optional): Specific feature to analyze. Defaults to None.
        norm (bool, optional): Whether to normalize importance scores. Defaults to True.

    Returns:
        np.ndarray: The computed importance scores.
    """

    # Compute feature importances
    start = 0
    importances = {}
    for name, size in mol_dataset.atom_feats_length.items():
        importances[name] = mask[:, start : start + size].sum(dim=1).cpu().numpy()
        start += size

    # Select importance based on analysis type
    if analysis in importances:
        importance = importances[analysis]
    else:
        importance = mask.sum(dim=1).cpu().numpy()  # Default to full mask importance

    for mol_name, smiles in mol_graph.mols.items():
        if mol_name == mol:
            plot_smiles = smiles

    addHs = mol_dataset._include_Hs

    plot_weighted_mol(importance, plot_smiles, norm, addHs)


def plot_weighted_mol(mask, smiles: str, norm=False, addHs=False):

    mol = Chem.MolFromSmiles(smiles)

    if addHs:
        mol = AddHs(mol)

    AllChem.EmbedMolecule(mol)
    AllChem.UFFOptimizeMolecule(mol)

    atoms = mol.GetNumAtoms()
    coords = mol.GetConformer().GetPositions()
    atom_symbol = [atom.GetSymbol() for atom in mol.GetAtoms()]

    edge_idx = []

    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_idx += [[u, v], [v, u]]

    edge_idx = np.array(edge_idx).T
    edge_coords = dict(zip(range(atoms), coords))

    coords_edges = [
        (
            np.concatenate(
                [
                    np.expand_dims(edge_coords[u], axis=1),
                    np.expand_dims(edge_coords[v], axis=1),
                ],
                axis=1,
            )
        )
        for u, v in zip(edge_idx[0], edge_idx[1])
    ]

    if norm == True:
        mask = mask / np.max(mask)

    mask = np.where(mask < 0.6, np.power(mask, 2), np.sqrt(mask))

    atoms_trace = trace_atoms(
        atom_symbol=atom_symbol,
        coords=coords,
        sizes=sizes,
        colors=colors_n,
        transparencies=mask,
    )

    edges_trace = trace_bonds(coords_edges=coords_edges, edge_mask_dict=edge_idx[0])

    traces = atoms_trace + edges_trace

    fig = go.Figure(data=traces)
    fig.update_layout(template="plotly_white")
    fig.update_layout(
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            xaxis_title="",
            yaxis_title="",
            zaxis_title="",
        )
    )
    fig.show()


def trace_atoms(atom_symbol, coords, sizes, colors, transparencies=None):
    trace_atoms = [None] * len(atom_symbol)
    for i in range(len(atom_symbol)):
        marker_dict = {
            "symbol": "circle",
            "size": sizes[atom_symbol[i]],
            "color": colors[atom_symbol[i]],
        }

        if transparencies is not None:
            marker_dict["opacity"] = transparencies[i]

        trace_atoms[i] = go.Scatter3d(
            x=[coords[i][0]],
            y=[coords[i][1]],
            z=[coords[i][2]],
            mode="markers",
            text=f"atom {atom_symbol[i]}",
            legendgroup="Atoms",
            showlegend=False,
            marker=marker_dict,
        )
    return trace_atoms


def trace_atom_imp(coords, opacity, atom_symbol, sizes, color):
    trace_atoms_imp = [None] * len(atom_symbol)
    for i in range(len(atom_symbol)):

        trace_atoms_imp[i] = go.Scatter3d(
            x=[coords[i][0]],
            y=[coords[i][1]],
            z=[coords[i][2]],
            mode="markers",
            showlegend=False,
            opacity=opacity[i],
            text=f"atom {atom_symbol[i]}",
            legendgroup="Atom importance",
            marker=dict(
                symbol="circle", size=sizes[atom_symbol[i]] * 1.7, color=color[i]
            ),
        )
    return trace_atoms_imp


def trace_bonds(coords_edges, edge_mask_dict):
    trace_edges = [None] * len(edge_mask_dict)

    for i in range(len(edge_mask_dict)):
        trace_edges[i] = go.Scatter3d(
            x=coords_edges[i][0],
            y=coords_edges[i][1],
            z=coords_edges[i][2],
            mode="lines",
            showlegend=False,
            legendgroup="Bonds",
            line=dict(color="black", width=2),
            hoverinfo="none",
        )

    return trace_edges


def trace_bond_imp(coords_edges, edge_mask_dict, opacity, color_edges):
    trace_edge_imp = [None] * len(edge_mask_dict)
    for i in range(len(edge_mask_dict)):
        trace_edge_imp[i] = go.Scatter3d(
            x=coords_edges[i][0],
            y=coords_edges[i][1],
            z=coords_edges[i][2],
            mode="lines",
            showlegend=False,
            legendgroup="Bond importance",
            opacity=opacity[i],
            line=dict(color=color_edges[i], width=opacity[i] * 15),
            hoverinfo="none",
        )

    return trace_edge_imp


def all_traces(atoms, atoms_imp, bonds, bonds_imp):
    traces = atoms + atoms_imp + bonds + bonds_imp
    fig = go.Figure(data=traces)
    fig.add_trace(
        go.Scatter3d(
            x=[None],
            y=[None],
            z=[None],
            mode="markers",
            legendgroup="Atoms",
            name="Atoms",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[None],
            y=[None],
            z=[None],
            mode="markers",
            legendgroup="Atom importance",
            name="Atom importance",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[None],
            y=[None],
            z=[None],
            mode="markers",
            legendgroup="Bonds",
            name="Bonds",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[None],
            y=[None],
            z=[None],
            mode="markers",
            legendgroup="Bond importance",
            name="Bond importance",
            showlegend=True,
        )
    )

    fig.update_layout(template="plotly_white")

    fig.update_layout(
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            xaxis_title="",
            yaxis_title="",
            zaxis_title="",
        )
    )

    fig.show()


def visualize_score_features(score: torch.Tensor, feature_sizes: dict):
    """
    Visualizes the importance of node features in the graph,
    showing the contributions of different atomic properties.

    Returns:
        A DataFrame with sorted feature importances.
    """

    # Compute feature importances
    start = 0
    importances = []
    for size in feature_sizes.values():
        importances.append(score[:, start : start + size].sum().cpu().item())
        start += size

    # Create DataFrame
    df = pd.DataFrame(
        {"score": importances, "labels": feature_sizes.keys()},
        index=feature_sizes.keys(),
    )
    df = df.sort_values("score", ascending=False).round(3)

    return df


def plot_importances(
    df,
):
    plt.figure(figsize=(10, 6))

    ax = barplot(df, x="score", y="labels", estimator="sum", errorbar=None)
    ax.bar_label(ax.containers[0], fontsize=10)
    # ax.set_yticklabels(ax.get_yticklabels(), verticalalignment='center', horizontalalignment='right')

    plt.xlabel("Feature Importance Score", fontsize=16)
    plt.ylabel("Feature", fontsize=16)

    # Display the plot
    plt.show()

    plt.close()


def plot_molecule_importance(mol_graph, mol, explanation, palette):

    edge_idx = mol_graph.edge_index
    fa, la, coords, atom_symbol = mol_prep(mol_graph=mol_graph, plot_mol=mol)
    edge_coords = dict(zip(range(fa, la), coords))
    edge_mask_dict, node_mask = get_masks(
        explanation=explanation, fa=fa, la=la, edge_idx=edge_idx
    )

    edge_mask_dict, node_mask = normalise_masks(
        edge_mask_dict=edge_mask_dict, node_mask=node_mask
    )

    coords_edges = [
        (
            np.concatenate(
                [
                    np.expand_dims(edge_coords[u], axis=1),
                    np.expand_dims(edge_coords[v], axis=1),
                ],
                axis=1,
            )
        )
        for u, v in edge_mask_dict.keys()
    ]

    edge_weights = list(edge_mask_dict.values())
    opacity_edges = [(x + 1) / 2 for x in edge_weights]
    opacity_nodes = [(x + 1) / 2 for x in node_mask]

    neg_edges = [True if num < 0 else False for num in list(edge_mask_dict.values())]
    neg_nodes = [True if num < 0 else False for num in node_mask]

    colors_atoms = colors_n
    color_nodes_imp = ["red" if boolean else "blue" for boolean in neg_nodes]
    color_edges_imp = ["red" if boolean else "blue" for boolean in neg_edges]

    atoms = trace_atoms(
        atom_symbol=atom_symbol, coords=coords, sizes=sizes, colors=colors_atoms
    )
    atoms_imp = trace_atom_imp(
        coords=coords,
        opacity=opacity_nodes,
        atom_symbol=atom_symbol,
        sizes=sizes,
        color=color_nodes_imp,
    )
    bonds = trace_bonds(coords_edges=coords_edges, edge_mask_dict=edge_mask_dict)
    bond_imp = trace_bond_imp(
        coords_edges=coords_edges,
        edge_mask_dict=edge_mask_dict,
        opacity=opacity_edges,
        color_edges=color_edges_imp,
    )

    all_traces(atoms=atoms, atoms_imp=atoms_imp, bonds=bonds, bonds_imp=bond_imp)


def normalise_masks(edge_mask_dict, node_mask):
    neg_edge = [True if num < 0 else False for num in list(edge_mask_dict.values())]
    min_value_edge = abs(min(edge_mask_dict.values(), key=abs))
    max_value_edge = abs(max(edge_mask_dict.values(), key=abs))

    abs_dict = {key: abs(value) for key, value in edge_mask_dict.items()}
    abs_dict = {
        key: (value - min_value_edge) / (max_value_edge - min_value_edge)
        for key, value in abs_dict.items()
    }

    edge_mask_dict_norm = {
        key: -value if convert else value
        for (key, value), convert in zip(abs_dict.items(), neg_edge)
    }

    node_mask = node_mask.sum(axis=1)
    node_mask = [val.item() for val in node_mask]
    neg_nodes = [True if num < 0 else False for num in node_mask]
    max_node = abs(max(node_mask, key=abs))
    min_node = abs(min(node_mask, key=abs))
    abs_node = [abs(w) for w in node_mask]
    abs_node = [(w - min_node) / (max_node - min_node) for w in abs_node]
    node_mask_norm = [
        -w if neg_nodes else w for w, neg_nodes in zip(abs_node, neg_nodes)
    ]

    return edge_mask_dict_norm, node_mask_norm


def get_masks(explanation, fa, la, edge_idx):
    edge_mask = explanation.edge_mask
    node_mask = explanation.node_mask

    edge_mask_dict = defaultdict(float)
    for val, u, v in list(zip(edge_mask, *edge_idx)):
        u, v = u.item(), v.item()
        if u in range(fa, la):
            if u > v:
                u, v = v, u
            edge_mask_dict[(u, v)] += val.item()

    node_mask = node_mask[fa:la]

    return edge_mask_dict, node_mask


def mol_prep(mol_graph, plot_mol: str, include_Hs=True):

    fa = 0

    for mol_name, smiles in mol_graph.mols.items():
        print(mol_name)
        print(smiles)
        mol = Chem.MolFromSmiles(smiles)
        if include_Hs:
            mol = AddHs(mol)
        la = fa + mol.GetNumAtoms()

        if mol_name == plot_mol:
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            coords = mol.GetConformer().GetPositions()
            atom_symbol = [atom.GetSymbol() for atom in mol.GetAtoms()]
            break
        else:
            fa += mol.GetNumAtoms()

    return fa, la, coords, atom_symbol
