import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool
from pyRates.Model.MessagePassingModel.MPNN import MPNNLayer
from pyRates.Model.FeedForwardModel.MixtureDensityNetwork import MDN


class MPNN_MDN(nn.Module):
    """
    Message Passing Neural Network with Mixture Density Network head.

    Combines graph-based message passing for molecular structure learning
    with probabilistic prediction via Mixture Density Networks.
    """

    def __init__(self,
                 node_hidden_dim=128,
                 mpnn_layers=3,
                 mpnn_dropout=0.1,
                 num_tasks=1,
                 n_components=3,
                 mdn_hidden_dims=[128, 256, 128],
                 mdn_dropout=0.1,
                 mdn_activation='relu',
                 use_batch_norm=True,
                 epsilon=1e-6,
                 device='auto'):
        """
        Initialize MPNN-MDN model.

        Args:
            node_hidden_dim: Hidden dimension for node embeddings
            mpnn_layers: Number of message passing layers
            mpnn_dropout: Dropout rate for MPNN layers
            num_tasks: Number of output tasks/targets
            n_components: Number of Gaussian components in MDN
            mdn_hidden_dims: List of hidden layer dimensions for MDN
            mdn_dropout: Dropout rate for MDN
            mdn_activation: Activation function for MDN ('relu', 'gelu', etc.)
            use_batch_norm: Whether to use batch normalization in MDN
            epsilon: Small constant for numerical stability
            device: Device to run model on ('auto', 'cuda', 'cpu', 'cuda:0', etc.)
        """
        super().__init__()

        # Device handling
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif 'cuda' in device:
            if not torch.cuda.is_available():
                raise ValueError(
                    "CUDA requested but not available. "
                    "Please check your PyTorch installation and GPU setup."
                )
            # Validate specific GPU if specified
            if ':' in device:
                gpu_id = int(device.split(':')[1])
                if gpu_id >= torch.cuda.device_count():
                    raise ValueError(
                        f"GPU {gpu_id} requested but only "
                        f"{torch.cuda.device_count()} GPU(s) available."
                    )

        self.device = torch.device(device)

        # Store hyperparameters
        self.node_hidden_dim = node_hidden_dim
        self.mpnn_layers_count = mpnn_layers
        self.mpnn_dropout = mpnn_dropout
        self.num_tasks = num_tasks
        self.n_components = n_components
        self.mdn_hidden_dims = mdn_hidden_dims
        self.mdn_dropout = mdn_dropout
        self.mdn_activation = mdn_activation
        self.use_batch_norm = use_batch_norm
        self.epsilon = epsilon

        # Layers (initialized lazily on first forward pass)
        self.node_embedding = None
        self.mpnn_layers = None
        self.mdn = None
        self._initialized = False

    def _initialize(self, node_input_dim, edge_input_dim):
        """
        Initialize layers with actual input dimensions.

        Called automatically on first forward pass.

        Args:
            node_input_dim: Dimension of node features
            edge_input_dim: Dimension of edge features
        """
        self.node_embedding = nn.Sequential(
            nn.Linear(node_input_dim, self.node_hidden_dim),
            nn.ReLU()
        ).to(self.device)

        self.mpnn_layers = nn.ModuleList([
            MPNNLayer(
                node_dim=self.node_hidden_dim,
                edge_dim=edge_input_dim,
                hidden_dim=self.node_hidden_dim,
                dropout=self.mpnn_dropout
            )
            for _ in range(self.mpnn_layers_count)
        ]).to(self.device)

        self.mdn = MDN(
            input_dim=self.node_hidden_dim,
            output_dim=self.num_tasks,
            n_components=self.n_components,
            hidden_dims=self.mdn_hidden_dims,
            dropout=self.mdn_dropout,
            activation=self.mdn_activation,
            use_batch_norm=self.use_batch_norm,
            epsilon=self.epsilon
        ).to(self.device)

        self._initialized = True

    def forward(self, data):
        """
        Forward pass through the network.

        Args:
            data: PyTorch Geometric Data object with attributes:
                - x: Node features [num_nodes, node_dim]
                - edge_index: Edge connectivity [2, num_edges]
                - edge_attr: Edge features [num_edges, edge_dim]
                - batch: Batch assignment [num_nodes]

        Returns:
            Tuple of (mu, sigma, pi) from MDN:
                - mu: Means of Gaussian components [batch_size, output_dim, n_components]
                - sigma: Std devs of components [batch_size, output_dim, n_components]
                - pi: Mixture weights [batch_size, output_dim, n_components]
        """
        # Move data to device
        data = data.to(self.device)
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch

        # Initialize layers on first forward pass
        if not self._initialized:
            node_input_dim = x.shape[1]
            edge_input_dim = edge_attr.shape[1]
            self._initialize(node_input_dim, edge_input_dim)

        # Node embedding
        x = self.node_embedding(x)

        # Message passing layers
        for layer in self.mpnn_layers:
            x = layer(x, edge_index, edge_attr)

        # Graph-level pooling
        graph_embedding = global_add_pool(x, batch)

        # MDN prediction
        return self.mdn(graph_embedding)

    def get_config(self):
        """
        Get model configuration as dictionary.

        Returns:
            Dictionary of hyperparameters
        """
        return {
            'node_hidden_dim': self.node_hidden_dim,
            'mpnn_layers': self.mpnn_layers_count,
            'mpnn_dropout': self.mpnn_dropout,
            'num_tasks': self.num_tasks,
            'n_components': self.n_components,
            'mdn_hidden_dims': self.mdn_hidden_dims,
            'mdn_dropout': self.mdn_dropout,
            'mdn_activation': self.mdn_activation,
            'use_batch_norm': self.use_batch_norm,
            'epsilon': self.epsilon,
            'device': str(self.device)
        }

    def __repr__(self):
        return (
            f"MPNN_MDN(\n"
            f"  node_hidden_dim={self.node_hidden_dim},\n"
            f"  mpnn_layers={self.mpnn_layers_count},\n"
            f"  num_tasks={self.num_tasks},\n"
            f"  n_components={self.n_components},\n"
            f"  device={self.device},\n"
            f"  initialized={self._initialized}\n"
            f")"
        )