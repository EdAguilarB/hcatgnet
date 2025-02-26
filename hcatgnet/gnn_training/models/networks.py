import torch
import torch.nn as nn
from torch_geometric.seed import seed_everything

from hcatgnet.options.enums import (Optimizers, Pooling, ProblemTypes,
                                    Schedulers)


class BaseNetwork(nn.Module):

    def __init__(
        self,
        n_node_features: int,
        n_edge_features: int,
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

        super().__init__()
        self._name = "BaseNetwork"
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features
        self.pooling = pooling
        self.n_convolutions = n_convolutions
        self.embedding_dim = embedding_dim
        self.readout_layers = readout_layers
        self.problem_type = problem_type
        self.n_classes = n_classes
        self._seed_everything(seed)

        if self.pooling == Pooling.GlobalMeanMaxPool:
            self.graph_embedding_dim = 2 * self.embedding_dim
        else:
            self.graph_embedding_dim = self.embedding_dim

    def forward(self):
        raise NotImplementedError

    @property
    def name(self):
        return self._name

    def _make_loss(self):
        if self.problem_type == ProblemTypes.Classification:
            self.loss = nn.CrossEntropyLoss()
        elif self.problem_type == ProblemTypes.Regression:
            self.loss = nn.MSELoss()
        else:
            raise ValueError(f"Problem type {self.problem_type} not supported")

    def _make_optimizer(self, optimizer, lr):
        if optimizer == Optimizers.Adam:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-9)
        elif optimizer == Optimizers.SGD:
            self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        elif optimizer == Optimizers.RMSprop:
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=lr)
        else:
            raise NotImplementedError(f"Optimizer type {optimizer} not implemented")

    def _make_scheduler(self, scheduler, step_size, gamma, min_lr):
        if scheduler == Schedulers.StepLR:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )
        elif scheduler == Schedulers.MultiStepLR:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=step_size, gamma=gamma
            )
        elif scheduler == Schedulers.ExponentialLR:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=gamma
            )
        elif scheduler == Schedulers.ReduceLROnPlateau:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=gamma,
                patience=step_size,
                min_lr=min_lr,
            )
        else:
            raise NotImplementedError(f"Scheduler type {scheduler} not implemented")

    def _seed_everything(self, seed):
        seed_everything(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.enabled = False
        # torch.use_deterministic_algorithms(True)
