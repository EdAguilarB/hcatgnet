import argparse
from copy import copy, deepcopy

from torch_geometric.loader import DataLoader

from hcatgnet.options.enums import Networks


def make_network(
    network_name: str, n_node_features: int, n_edge_features: int, seed: int = 20232023
):
    if network_name == Networks.GCN:
        from hcatgnet.gnn_training.models.gcn import GCN

        return GCN(
            n_node_features=n_node_features, n_edge_features=n_edge_features, seed=seed
        )
    else:
        raise ValueError(f"Network {network_name} not implemented")


def create_loaders(dataset, folds, batch_size):
    """
    Creates training, validation and testing loaders for cross validation and
    inner cross validation training-evaluation processes.
    Args:
    dataset: pytorch geometric datset
    batch_size (int): size of batches
    val (bool): whether or not to create a validation set
    folds (int): number of folds to be used
    num points (int): number of points to use for the training and evaluation process

    Returns:
    (tuple): DataLoaders for training, validation and test set

    """

    folds = [[] for _ in range(folds)]
    for data in dataset:
        folds[data.fold].append(data)

    for outer in range(len(folds)):
        proxy = copy(folds)
        test_loader = DataLoader(proxy.pop(outer), batch_size=batch_size, shuffle=False)
        for inner in range(len(proxy)):  # length is reduced by 1 here
            proxy2 = copy(proxy)
            val_loader = DataLoader(
                proxy2.pop(inner), batch_size=batch_size, shuffle=False
            )
            flatten_training = [
                item for sublist in proxy2 for item in sublist
            ]  # flatten list of lists
            train_loader = DataLoader(
                flatten_training, batch_size=batch_size, shuffle=True
            )
            yield deepcopy((train_loader, val_loader, test_loader))
