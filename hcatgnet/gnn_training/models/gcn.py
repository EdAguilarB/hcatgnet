import argparse

import torch
import torch.nn as nn
from icecream import ic
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap

from hcatgnet.gnn_training.models.networks import BaseNetwork
from hcatgnet.options.enums import (Optimizers, Pooling, ProblemTypes,
                                    Schedulers)


class GCN(BaseNetwork):

    def __init__(
        self,
        n_node_features: int,
        n_edge_features: int,
        improved: bool = True,
        pooling: str = Pooling.GlobalMeanMaxPool,
        n_convolutions: int = 2,
        embedding_dim: int = 64,
        readout_layers: int = 2,
        problem_type: str = ProblemTypes.Regression,
        n_classes: int = 1,
        seed=20232023,
        optimizer=Optimizers.Adam,
        lr=0.01,
        scheduler=Schedulers.ReduceLROnPlateau,
        step_size=7,
        gamma=0.7,
        min_lr=1e-08,
    ):
        super().__init__(
            n_node_features=n_node_features,
            n_edge_features=n_edge_features,
            pooling=pooling,
            n_convolutions=n_convolutions,
            embedding_dim=embedding_dim,
            readout_layers=readout_layers,
            problem_type=problem_type,
            n_classes=n_classes,
            seed=seed,
            optimizer=optimizer,
            lr=lr,
            scheduler=scheduler,
            step_size=step_size,
            gamma=gamma,
            min_lr=min_lr,
        )

        self._name = "GCN"
        self.improved = improved

        # First convolution and activation function
        self.conv1 = GCNConv(
            self.n_node_features, self.embedding_dim, improved=self.improved
        )
        self.relu1 = nn.LeakyReLU()

        # Convolutions
        self.conv_layers = nn.ModuleList([])
        for _ in range(self.n_convolutions - 1):
            self.conv_layers.append(
                GCNConv(self.embedding_dim, self.embedding_dim, self.improved)
            )

        # graph embedding is the concatenation of the global mean and max pooling, thus 2*embedding_dim
        graph_embedding = self.embedding_dim * 2

        # Readout layers
        self.readout = nn.ModuleList([])

        for _ in range(self.readout_layers - 1):
            reduced_dim = int(graph_embedding / 2)
            self.readout.append(
                nn.Sequential(nn.Linear(graph_embedding, reduced_dim), nn.LeakyReLU())
            )
            graph_embedding = reduced_dim

        # Final readout layer
        self.readout.append(nn.Linear(graph_embedding, self.n_classes))

        self._make_loss()
        self._make_optimizer(optimizer, lr)
        self._make_scheduler(scheduler, step_size=step_size, gamma=gamma, min_lr=min_lr)

    def forward(
        self,
        x=None,
        edge_index=None,
        batch_index=None,
        edge_weight=None,
        return_graph_embedding=False,
    ):

        x = self.conv1(x, edge_index, edge_weight)
        x = self.relu1(x)

        for i in range(self.n_convolutions - 1):
            x = self.conv_layers[i](x, edge_index)
            x = nn.LeakyReLU()(x)

        x = torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1)

        graph_emb = x

        for i in range(self.readout_layers):
            x = self.readout[i](x)

        if return_graph_embedding == True:
            return x, graph_emb
        else:
            return x
