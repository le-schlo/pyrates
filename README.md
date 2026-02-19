# pyRates

Framework for predicting excited state properties.

## Installation

### Prerequisites

- Python >= 3.12
- Linux
- [uv](https://github.com/astral-sh/uv)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (optional, for GPU support)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/pyRates.git
   cd pyRates
   ```

2. **Setup environment and install:**
   Create a virtual environment and install dependencies defined in `pyproject.toml` using `uv`.
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -e .
   ```

## Structure

- `pyRates/`: Core library code.
  - `Model/`: Neural network architectures.
    - `MessagePassingModel/`: Graph encoders (D-MPNN, GAT, GIN, etc.).
    - `FeedForwardModel/`: Mixture Density Network (MDN) head.
    - `MPNN_MDN.py`: Main model class combining graph encoders with MDN.
  - `Utilities/`: Helper functionalities.
    - `Dataset/`: Data loading and graph construction (including one-hot encoding).
    - `Plotting/`: Visualization tools.
- `playground/`: Experimental scripts.
  - `benchmarking/`: Scripts for comparing different model architectures (`gat_dmpnn`).
  - `model_setup/`: Training and inference scripts.
- `data/`: Directory for datasets.

## Modules

### Graph Encoders (`pyRates.Model.MessagePassingModel`)
The framework supports multiple graph encoder architectures, selectable via the `encoder_type` parameter in `MPNN_MDN`:

- **D-MPNN (`dmpnn`)**: Directed Message Passing Neural Network. Passes messages along directed edges (bonds) to capture local chemical environments without immediate backtracking. Often the state-of-the-art for molecular property prediction.
- **GAT (`gat`)**: Graph Attention Network. Uses attention mechanisms to weight the importance of neighboring atoms. Our implementation integrates edge features directly into the attention mechanism.
- **MPNN (`mpnn`)**: Standard undirected Message Passing Neural Network using GRU updates. Robust baseline that effectively aggregates local information.
- **GIN (`gin`)**: Graph Isomorphism Network (specifically `GINEConv`). Designed to maximize the theoretical discriminative power of the GNN, natively supporting edge features.
- **Graph Transformer (`transformer`)**: Adapts the Transformer architecture for graphs, utilizing multi-head attention mechanisms that explicitly incorporate edge attributes.

### Mixture Density Network (`pyRates.Model.FeedForwardModel.MDN`)
Instead of predicting single scalar values, the model outputs parameters for a Gaussian Mixture Model (means, standard deviations, and weights). This enables:
- Prediction of full probability distributions for target properties.
- Capture of aleatoric uncertainty (data noise).
- Modeling of multi-modal outputs (e.g., if a property can have multiple valid states).

### Data Handling (`pyRates.Utilities.Dataset`)
- **Graph Construction**: Converts SMILES strings into graph objects (`torch_geometric.data.Data`).
- **Featurization**: Uses comprehensive one-hot encoding for atom (atomic number, degree, hybridization, etc.) and bond (type, stereo) properties.
- **Normalization**: Targets are standardized using `StandardScaler`.

