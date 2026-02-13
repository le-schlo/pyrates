import torch
import torch.nn as nn
from pyRates.Model.FeedForwardModel.MixtureDensityNetwork import MDN
from pyRates.Model.MessagePassingModel.Encoders import DMPNN, GAT, MPNN, GIN, GraphTransformer
from torch_geometric.nn import global_add_pool


class MPNN_MDN(nn.Module):
    """
    A modular graph neural network model with a Mixture Density Network head.
    It can use different graph encoders like D-MPNN, GAT or MPNN.
    """

    def __init__(self,
                 encoder_type='dmpnn',
                 node_hidden_dim=128,
                 encoder_layers=3,
                 encoder_dropout=0.1,
                 num_tasks=None, # Changed default to None to detect usage
                 num_graph_tasks=0,
                 num_node_tasks=0,
                 n_components=3,
                 mdn_hidden_dims=[128, 256, 128],
                 mdn_dropout=0.1,
                 mdn_activation='relu',
                 use_batch_norm=True,
                 epsilon=1e-6,
                 device='auto',
                 atom_wise=False,
                 **encoder_args):
        """
        Initialize the model.

        Args:
            encoder_type (str): Type of encoder to use ('dmpnn', 'gat', or 'mpnn').
            node_hidden_dim (int): Hidden dimension for node embeddings.
            encoder_layers (int): Number of layers in the encoder.
            encoder_dropout (float): Dropout rate for the encoder.
            num_tasks (int): Legacy argument. If used, maps to graph or node tasks based on atom_wise.
            num_graph_tasks (int): Number of graph-level output tasks.
            num_node_tasks (int): Number of atom-level output tasks.
            n_components (int): Number of Gaussian components in MDN.
            mdn_hidden_dims (list): Hidden layer dimensions for MDN.
            mdn_dropout (float): Dropout rate for MDN.
            mdn_activation (str): Activation function for MDN.
            use_batch_norm (bool): Whether to use batch normalization in MDN.
            epsilon (float): Small constant for numerical stability.
            device (str): Device to run the model on.
            atom_wise (bool): Legacy flag. Whether to predict atom-wise properties (no pooling).
            **encoder_args: Additional arguments for the encoder (e.g., num_heads for GAT).
        """
        super().__init__()

        # Device handling
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif 'cuda' in device and not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available.")
        self.device = torch.device(device)

        # Handle Legacy Arguments
        if num_tasks is not None:
            if atom_wise:
                num_node_tasks = num_tasks
            else:
                num_graph_tasks = num_tasks
        else:
             # If num_tasks not provided, rely on explicit args.
             # If defaults (0,0) used, this might be an issue if user didn't provide num_tasks.
             # Assume num_tasks=1 default if nothing provided?
             if num_graph_tasks == 0 and num_node_tasks == 0:
                 if atom_wise:
                     num_node_tasks = 1
                 else:
                     num_graph_tasks = 1

        # Store hyperparameters
        self.encoder_type = encoder_type
        self.node_hidden_dim = node_hidden_dim
        self.encoder_layers = encoder_layers
        self.encoder_dropout = encoder_dropout
        self.encoder_args = encoder_args
        self.num_graph_tasks = num_graph_tasks
        self.num_node_tasks = num_node_tasks
        self.n_components = n_components
        self.mdn_hidden_dims = mdn_hidden_dims
        self.mdn_dropout = mdn_dropout
        self.mdn_activation = mdn_activation
        self.use_batch_norm = use_batch_norm
        self.epsilon = epsilon
        # atom_wise is stored effectively by checking if num_node_tasks > 0

        # Layers (initialized lazily)
        self.encoder = None
        self.mdn_graph = None
        self.mdn_node = None
        self._initialized = False

    def _initialize(self, node_input_dim, edge_input_dim):
        """Initialize layers with actual input dimensions."""
        encoder_classes = {
            'dmpnn': DMPNN,
            'gat': GAT,
            'mpnn': MPNN,
            'gin': GIN,
            'transformer': GraphTransformer
        }
        encoder_class = encoder_classes.get(self.encoder_type.lower())
        if not encoder_class:
            raise ValueError(f"Unknown encoder_type: {self.encoder_type}")

        self.encoder = encoder_class(
            node_features=node_input_dim,
            edge_features=edge_input_dim,
            hidden_size=self.node_hidden_dim,
            depth=self.encoder_layers,
            dropout=self.encoder_dropout,
            **self.encoder_args
        ).to(self.device)

        if self.num_graph_tasks > 0:
            self.mdn_graph = MDN(
                input_dim=self.node_hidden_dim,
                output_dim=self.num_graph_tasks,
                hidden_dims=self.mdn_hidden_dims,
                n_components=self.n_components,
                dropout=self.mdn_dropout,
                activation=self.mdn_activation,
                use_batch_norm=self.use_batch_norm,
                epsilon=self.epsilon
            ).to(self.device)

        if self.num_node_tasks > 0:
            self.mdn_node = MDN(
                input_dim=self.node_hidden_dim,
                output_dim=self.num_node_tasks,
                hidden_dims=self.mdn_hidden_dims,
                n_components=self.n_components,
                dropout=self.mdn_dropout,
                activation=self.mdn_activation,
                use_batch_norm=self.use_batch_norm,
                epsilon=self.epsilon
            ).to(self.device)

        self._initialized = True

    def forward(self, data):
        """Forward pass through the model."""
        data = data.to(self.device)
        if not self._initialized:
            node_input_dim = data.x.size(1)
            edge_input_dim = data.edge_attr.size(1) if data.edge_attr is not None else 0
            self._initialize(node_input_dim, edge_input_dim)

        # Always get node embeddings first
        # Exception: if ONLY graph tasks and we trust encoder pooling.
        # But we unified encoders to support return_node_embeddings.
        # To reuse code, let's just get node embeddings and pool ourselves.

        node_embeddings = self.encoder(data, return_node_embeddings=True)

        out_graph = None
        out_node = None

        if self.num_graph_tasks > 0:
            # Pool
            graph_embedding = global_add_pool(node_embeddings, data.batch)
            out_graph = self.mdn_graph(graph_embedding)

        if self.num_node_tasks > 0:
            out_node = self.mdn_node(node_embeddings)

        # Return logic
        if self.num_graph_tasks > 0 and self.num_node_tasks > 0:
            return out_graph, out_node
        elif self.num_graph_tasks > 0:
            return out_graph
        elif self.num_node_tasks > 0:
            return out_node
        else:
             raise RuntimeError("No tasks configured.")

    def get_config(self):
        """Get model configuration as a dictionary."""
        config = {
            'encoder_type': self.encoder_type,
            'node_hidden_dim': self.node_hidden_dim,
            'encoder_layers': self.encoder_layers,
            'encoder_dropout': self.encoder_dropout,
            'num_graph_tasks': self.num_graph_tasks,
            'num_node_tasks': self.num_node_tasks,
            'n_components': self.n_components,
            'mdn_hidden_dims': self.mdn_hidden_dims,
            'mdn_dropout': self.mdn_dropout,
            'mdn_activation': self.mdn_activation,
            'use_batch_norm': self.use_batch_norm,
            'epsilon': self.epsilon,
            'device': str(self.device),
            'atom_wise': self.atom_wise
        }
        config.update(self.encoder_args)
        return config

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  encoder_type='{self.encoder_type}',\n"
            f"  node_hidden_dim={self.node_hidden_dim},\n"
            f"  encoder_layers={self.encoder_layers},\n"
            f"  num_graph_tasks={self.num_graph_tasks},\n"
            f"  num_node_tasks={self.num_node_tasks},\n"
            f"  n_components={self.n_components},\n"
            f"  device={self.device},\n"
            f"  initialized={self._initialized}\n"
            f")"
        )
