import torch.nn as nn
from torch_geometric.nn import global_add_pool
from pyRates.Model.MessagePassingModel.MPNN import MPNNLayer
from pyRates.Model.FeedForwardModel.MixtureDensityNetwork import MDN

class MPNN_MDN(nn.Module):

    def __init__(self, node_input_dim,
                 edge_input_dim,
                 node_hidden_dim=128,
                 mpnn_layers=3,
                 num_tasks=1,
                 n_components=3):

        super().__init__()

        self.node_embedding = nn.Sequential(
            nn.Linear(node_input_dim, node_hidden_dim),
            nn.ReLU()
        )

        self.mpnn_layers = nn.ModuleList([
            MPNNLayer(node_hidden_dim, edge_input_dim, node_hidden_dim)
            for _ in range(mpnn_layers)
        ])

        self.mdn = MDN(
            input_dim=node_hidden_dim,
            output_dim=num_tasks,
            n_components=n_components
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch

        x = self.node_embedding(x)

        for layer in self.mpnn_layers:
            x = layer(x, edge_index, edge_attr)

        graph_embedding = global_add_pool(x, batch)

        return self.mdn(graph_embedding)