import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList
from torch_geometric.nn import GCNConv, GATConv, NNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.seed import seed_everything
import numpy as np




class GCN_loop(torch.nn.Module):

    def __init__(self, num_features, embedding_size, gnn_layers, improved, SEED = 123456789, task ='r'):
        super(GCN_loop, self).__init__()

        seed_everything(SEED)
        torch.cuda.manual_seed(SEED)
        np.random.RandomState(SEED)
        torch.backends.cudnn.deterministic = True

        self.gnn_layers = gnn_layers
        self.embedding_size = embedding_size
        self.task = task

        # GCN layers
        self.initial_conv = GCNConv(num_features, 
                                    embedding_size, 
                                    improved=improved)
        
        self.conv_layers = ModuleList([])
        for _ in range(self.gnn_layers - 1):
            self.conv_layers.append(GCNConv(embedding_size,
                                            embedding_size,
                                            improved=improved))
            
        # Output layer
        self.readout1 = Linear(2*embedding_size, embedding_size)
        self.readout2 = Linear(embedding_size, 1)
        
    def forward(self, x, edge_index, batch_index, edge_weight = None):

        hidden = self.initial_conv(x, edge_index, edge_weight)
        hidden = F.leaky_relu(hidden)

        for i in range(self.gnn_layers-1):
            hidden = self.conv_layers[i](hidden, edge_index, edge_weight)
            hidden = F.leaky_relu(hidden)

        hidden = torch.cat([gmp(hidden, batch_index), 
                            gap(hidden, batch_index)], dim=1)
        
        hidden = self.readout1(hidden)
        hidden = F.leaky_relu(hidden)

        out = self.readout2(hidden)

        out = F.sigmoid(out)*100

        return out
        

