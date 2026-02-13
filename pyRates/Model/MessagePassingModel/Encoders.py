import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_add_pool, MessagePassing, GINEConv, TransformerConv


class MPNNLayer(MessagePassing):
    """
    Standard Message Passing Layer (Message -> Aggregate -> Update)
    Using GRU for update.
    """
    def __init__(self, node_dim, edge_dim, hidden_dim, dropout=0.0):
        super(MPNNLayer, self).__init__(aggr='add')  # Sum aggregation

        # Message function: MLP(cat(x_i, x_j, e_ij))
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Update function: GRU
        self.gru = nn.GRUCell(hidden_dim, node_dim)

    def forward(self, x, edge_index, edge_attr):
        """
        x: node features [num_nodes, node_dim]
        edge_index: [2, num_edges]
        edge_attr: edge features [num_edges, edge_dim]
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # x_i: target, x_j: source
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_mlp(msg_input)

    def update(self, aggr_out, x):
        # x is the previous node state
        new_x = self.gru(aggr_out, x)
        return new_x


class MPNN(nn.Module):
    """Standard MPNN Encoder (undirected message passing)"""
    def __init__(self, node_features, edge_features, hidden_size, depth=3, dropout=0.1, **kwargs):
        super(MPNN, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth

        # Initial embedding
        self.node_embedding = nn.Sequential(
            nn.Linear(node_features, hidden_size),
            nn.ReLU()
        )

        self.mpnn_layers = nn.ModuleList([
            MPNNLayer(
                node_dim=hidden_size,
                edge_dim=edge_features,
                hidden_dim=hidden_size,
                dropout=dropout
            )
            for _ in range(depth)
        ])

    def forward(self, data, return_node_embeddings=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.node_embedding(x)

        for layer in self.mpnn_layers:
            x = layer(x, edge_index, edge_attr)

        if return_node_embeddings:
            return x

        graph_out = global_add_pool(x, data.batch)
        return graph_out


class DMPNN(nn.Module):
    """Directed Message Passing Neural Network (D-MPNN)"""
    def __init__(self, node_features, edge_features, hidden_size, depth=3, dropout=0.1, **kwargs):
        super(DMPNN, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.dropout = nn.Dropout(dropout)

        self.W_i = nn.Linear(node_features + edge_features, hidden_size, bias=False)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_o = nn.Linear(node_features + hidden_size, hidden_size)

    def forward(self, data, return_node_embeddings=False):
        x, edge_index, edge_attr, rev_edge_index = data.x, data.edge_index, data.edge_attr, data.rev_edge_index
        num_nodes = x.size(0)
        row, col = edge_index

        input_ = torch.cat([x[row], edge_attr], dim=1)
        message = torch.relu(self.W_i(input_))
        message = self.dropout(message)

        for _ in range(self.depth - 1):
            rev_messages = message[rev_edge_index]
            aggr_message = torch.zeros(num_nodes, self.hidden_size, device=x.device)
            aggr_message.index_add_(0, col, message)
            message = torch.relu(self.W_h(aggr_message[row] - rev_messages))
            message = self.dropout(message)

        aggr_message = torch.zeros(num_nodes, self.hidden_size, device=x.device)
        aggr_message.index_add_(0, col, message)

        node_h = torch.cat([x, aggr_message], dim=1)
        node_h = torch.relu(self.W_o(node_h))
        node_h = self.dropout(node_h)

        if return_node_embeddings:
            return node_h

        graph_out = global_add_pool(node_h, data.batch)
        return graph_out


class GAT(nn.Module):
    """Graph Attention Network (GAT) with Edge Feature Support"""
    def __init__(self, node_features, edge_features, hidden_size, num_heads=8, depth=3, dropout=0.1, **kwargs):
        super(GAT, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.depth = depth
        self.dropout = dropout

        self.initial_embedding = nn.Linear(node_features, hidden_size)
        self.gat_layers = nn.ModuleList()

        # Input Layer
        # edge_dim tells GATConv to expect edge features of this size
        self.gat_layers.append(GATConv(
            hidden_size,
            hidden_size,
            heads=num_heads,
            dropout=dropout,
            edge_dim=edge_features
        ))

        # Hidden Layers
        for _ in range(depth - 2):
            self.gat_layers.append(GATConv(
                hidden_size * num_heads,
                hidden_size,
                heads=num_heads,
                dropout=dropout,
                edge_dim=edge_features
            ))

        # Output Layer (heads=1, concat=False)
        self.gat_layers.append(GATConv(
            hidden_size * num_heads,
            hidden_size,
            heads=1,
            concat=False,
            dropout=dropout,
            edge_dim=edge_features
        ))

        self.readout = nn.Linear(hidden_size, hidden_size)

    def forward(self, data, return_node_embeddings=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.initial_embedding(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for i, layer in enumerate(self.gat_layers):
            # Pass edge_attr to the GAT layer
            x = layer(x, edge_index, edge_attr=edge_attr)

            if i < self.depth - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.relu(self.readout(x))

        if return_node_embeddings:
            return x

        graph_out = global_add_pool(x, data.batch)
        return graph_out


class GIN(nn.Module):
    """
    Graph Isomorphism Network (GIN) with Edge Features (GINEConv).
    Known for high expressiveness in graph regression tasks.
    """
    def __init__(self, node_features, edge_features, hidden_size, depth=3, dropout=0.1, **kwargs):
        super(GIN, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.dropout = dropout

        # Initial embedding
        self.node_embedding = nn.Linear(node_features, hidden_size)

        # Edge embedding needed because GINEConv expects edge_attr to match node dim
        self.edge_embedding = nn.Linear(edge_features, hidden_size)

        self.layers = nn.ModuleList()
        for _ in range(depth):
            # GIN requires an MLP for the aggregation step
            mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size)
            )
            # GINEConv expects edge features
            self.layers.append(GINEConv(mlp, train_eps=True))

    def forward(self, data, return_node_embeddings=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = F.relu(self.node_embedding(x))
        edge_attr = F.relu(self.edge_embedding(edge_attr)) # Project edges to hidden_dim

        for layer in self.layers:
            # GINEConv needs edge_attr to have same dim as x, hence the embedding above
            x = layer(x, edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if return_node_embeddings:
            return x

        graph_out = global_add_pool(x, data.batch)
        return graph_out


class GraphTransformer(nn.Module):
    """
    Graph Transformer Network.
    More powerful attention mechanism than GAT, natively supporting edge features.
    """
    def __init__(self, node_features, edge_features, hidden_size, num_heads=4, depth=3, dropout=0.1, **kwargs):
        super(GraphTransformer, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.dropout = dropout

        self.node_embedding = nn.Linear(node_features, hidden_size)

        self.layers = nn.ModuleList()
        for _ in range(depth):
            # TransformerConv supports edge_dim directly
            self.layers.append(TransformerConv(
                in_channels=hidden_size,
                out_channels=hidden_size // num_heads, # PyG concatenates heads by default
                heads=num_heads,
                edge_dim=edge_features,
                dropout=dropout
            ))

    def forward(self, data, return_node_embeddings=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = F.relu(self.node_embedding(x))

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            # TransformerConv applies dropout internally, but we can add layer dropout
            x = F.dropout(x, p=self.dropout, training=self.training)

        if return_node_embeddings:
            return x

        graph_out = global_add_pool(x, data.batch)
        return graph_out

