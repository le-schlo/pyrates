import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class MPNNLayer(MessagePassing):
    """
    Message Passing Neural Network Layer
    """
    def __init__(self, node_dim, edge_dim, hidden_dim, dropout=0.0):
        super(MPNNLayer, self).__init__(aggr='add')

        # Message function
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Update function (GRU-style)
        self.gru = nn.GRUCell(hidden_dim, node_dim)

    def forward(self, x, edge_index, edge_attr):
        """
        x: node features [num_nodes, node_dim]
        edge_index: [2, num_edges]
        edge_attr: edge features [num_edges, edge_dim]
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        """
        x_i: target node features
        x_j: source node features
        edge_attr: edge features
        """
        # Concatenate source node, target node, and edge features
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        message = self.message_mlp(msg_input)
        return message

    def update(self, aggr_out, x):
        """
        Update node features using GRU
        """
        new_x = self.gru(aggr_out, x)
        return new_x