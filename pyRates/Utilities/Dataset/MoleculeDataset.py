import torch
from torch.utils.data import Dataset
from pyRates.Utilities.Dataset.GraphConstruction import smiles_to_graph
from pyRates.Model.MessagePassingModel.PositionalEncoding import *
from torch_geometric.data import Batch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import joblib
import numpy as np

class DualScaler:
    """Wrapper for handling separate scalers for graph and node targets."""
    def __init__(self, graph_scaler=None, node_scaler=None):
        self.graph_scaler = graph_scaler
        self.node_scaler = node_scaler

    def transform(self, targets):
        pass

    def inverse_transform(self, data):
        # Data is keys 'graph' and 'node' or tuple
        if isinstance(data, dict):
            res = {}
            if 'graph' in data and self.graph_scaler:
                res['graph'] = self.graph_scaler.inverse_transform(data['graph'])
            elif 'graph' in data:
                res['graph'] = data['graph']

            if 'node' in data and self.node_scaler:
                res['node'] = self.node_scaler.inverse_transform(data['node'])
            elif 'node' in data:
                res['node'] = data['node']
            return res
        return data

class MoleculeDataset(Dataset):

    def __init__(self, smiles_list, targets, positional_encoding_kwargs=None, **kwargs):
        self.graphs = []

        # Analyze targets to split into graph and node targets
        self.graph_targets = []
        self.node_targets = []

        # Heuristic: Check first sample
        self.has_graph_targets = False
        self.has_node_targets = False
        self.graph_indices = []
        self.node_indices = []

        if len(targets) > 0:
            sample = targets[0]
            # If standard numpy array (all graph tasks)
            if isinstance(sample, np.ndarray) and sample.ndim == 1 and not isinstance(sample[0], (list, np.ndarray)):
                 pass # Will handle later as pure graph

            # If list/array of mixed types (object array or list of lists)
            n_tasks = len(sample)

            for i in range(n_tasks):
                elem = sample[i]
                if isinstance(elem, (list, np.ndarray)):
                    self.node_indices.append(i)
                else:
                    self.graph_indices.append(i)

            self.has_node_targets = len(self.node_indices) > 0
            self.has_graph_targets = len(self.graph_indices) > 0 or (not self.has_node_targets)

        # Process Graphs
        for i, smiles in enumerate(smiles_list):
            graph = smiles_to_graph(smiles, use_explicit_hydrogens=False, use_stereochemistry=True)
            if graph is not None:
                #Add positional encoding for GraphGPS, stored in graph.rwse and graph.rrwp_edge
                if positional_encoding_kwargs is not None:
                    walk_length = positional_encoding_kwargs['walk_length']
                    graph = add_random_walk_positional_encoding(graph, walk_length=walk_length)
                    graph = add_relative_random_walk_positional_encoding(graph, walk_length=walk_length)

                self.graphs.append(graph)

                # Process Targets
                if len(targets) == 0:
                    continue
                t = targets[i]

                # --- VALIDATION START ---
                # Check atom consistency if node targets exist
                consistent = True
                if self.has_node_targets:
                    # Determine node target parts
                    if self.has_graph_targets:
                         n_t = [t[k] for k in self.node_indices]
                    else:
                         n_t = [t] if not isinstance(t, (list, np.ndarray)) or len(self.node_indices)<=1 else t
                         if isinstance(n_t, (list, np.ndarray)) and isinstance(n_t[0], (list, np.ndarray))==False:
                              n_t = [n_t] # Wrap single array

                    # Check first part (assuming all parts aligned)
                    # n_t is list of arrays/lists
                    for part in n_t:
                        if isinstance(part, (list, np.ndarray)):
                             if len(part) != graph.num_nodes:
                                 print(f"Warning: Discarding molecule idx {i} (SMILES: {smiles}). "
                                       f"Node count ({graph.num_nodes}) != Target length ({len(part)}).")
                                 self.graphs.pop() # Remove graph added above
                                 consistent = False
                                 break

                if not consistent:
                    continue
                # --- VALIDATION END ---

                if self.has_node_targets and self.has_graph_targets:
                    # Mixed case
                     g_t = [t[k] for k in self.graph_indices]
                     n_t = [t[k] for k in self.node_indices]

                     self.graph_targets.append(torch.tensor(g_t, dtype=torch.float))
                     # Node targets are list of arrays, stack them column-wise
                     # n_t is list of [array(atoms), array(atoms)]
                     # We need (atoms, tasks)
                     # Check alignment
                     stacked_n = np.column_stack(n_t)
                     self.node_targets.append(torch.tensor(stacked_n, dtype=torch.float))

                elif self.has_node_targets:
                    # Only node targets
                    # t is list of arrays, or just array
                    if isinstance(t, (list, np.ndarray)) and len(self.node_indices) > 1:
                        stacked_n = np.column_stack(t)
                        self.node_targets.append(torch.tensor(stacked_n, dtype=torch.float))
                    elif isinstance(t, (list, np.ndarray)) and len(self.node_indices) == 1:
                         # Single node task, might be wrapped or not
                         if isinstance(t[0], (list, np.ndarray)):
                             val = t[0]
                         else:
                             val = t # It is the array itself
                         self.node_targets.append(torch.tensor(val, dtype=torch.float).unsqueeze(1))
                    else:
                         # fallback
                         self.node_targets.append(torch.tensor(t, dtype=torch.float))

                else:
                    # Only graph targets
                     self.graph_targets.append(t)

        # Finalize storage
        if self.has_graph_targets:
            try:
                self.graph_targets = torch.tensor(self.graph_targets, dtype=torch.float)
            except:
                 self.graph_targets = torch.tensor(np.array(self.graph_targets), dtype=torch.float)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        data = self.graphs[idx].clone()

        # Determine what goes into data.y (primary target)
        if self.has_graph_targets and self.has_node_targets:
             # Mixed mode: y is graph, node_y is node
             y = self.graph_targets[idx]
             if y.ndim == 0: y = y.unsqueeze(0)
             if y.ndim == 1: y = y.unsqueeze(0)
             data.y = y
             data.node_y = self.node_targets[idx]

        elif self.has_graph_targets:
             # Graph only
             y = self.graph_targets[idx]
             if y.ndim == 0: y = y.unsqueeze(0)
             if y.ndim == 1: y = y.unsqueeze(0)
             data.y = y

        elif self.has_node_targets:
             # Node only -> treat as primary y
             data.y = self.node_targets[idx]

        return data

def collate_fn(batch):
    # Use PyG's own batching which handles attributes
    return Batch.from_data_list(batch)

def create_dataloaders(smiles_list,
                       targets,
                       train_idx,
                       val_idx,
                       test_idx,
                       batch_size=64,
                       num_workers=8,
                       scaler_path='checkpoints/scaler.pkl',
                       positional_encoding_kwargs=None, **kwargs):
    '''
    Create dataloaders for training, validation, and testing. and scale targets using StandardScaler.
    '''
    smiles_train = smiles_list[train_idx]
    smiles_val = smiles_list[val_idx]
    smiles_test = smiles_list[test_idx]

    y_train = targets[train_idx]
    y_val = targets[val_idx]
    y_test = targets[test_idx]

    # Analyze structure of targets from first sample
    sample = y_train[0]
    graph_indices = []
    node_indices = []

    # Check if this is a mixed/complex target list (object array or list of mixed types)
    is_mixed = False
    if isinstance(sample, (list, np.ndarray)) and (isinstance(sample[0], (list, np.ndarray)) or any(isinstance(x, (list, np.ndarray)) for x in sample)):
         is_mixed = True
         n_tasks = len(sample)
         for i in range(n_tasks):
             if isinstance(sample[i], (list, np.ndarray)):
                 node_indices.append(i)
             else:
                 graph_indices.append(i)

    # If not mixed, check if it is purely atom-wise (one task, but variable length)
    is_pure_atom_wise = False
    if not is_mixed:
         if isinstance(sample, (list, np.ndarray)):
             # Could be graph targets (if fixed length) or atom targets (if variable)
             # We assume if it's a list/array but not mixed, it's either all graph or all node
             # Use stack check
             try:
                 np.stack(y_train)
                 # If stackable, treated as graph targets unless strictly one column and variable len
             except ValueError:
                 is_pure_atom_wise = True

    scaler = None

    if is_mixed:
        graph_scaler = StandardScaler() if graph_indices else None
        node_scaler = StandardScaler() if node_indices else None

        # Process Graph Targets
        if graph_indices:
            # Extract
            g_train = np.array([[y[i] for i in graph_indices] for y in y_train])
            graph_scaler.fit(g_train)

        # Process Node Targets
        if node_indices:
            # Extract and Flatten
            n_train_flat = []
            for y in y_train:
                for i in node_indices:
                    n_train_flat.extend(y[i])
            n_train_flat = np.array(n_train_flat).reshape(-1, 1)
            node_scaler.fit(n_train_flat)

        scaler = DualScaler(graph_scaler, node_scaler)

        def transform_mixed(y_data, s):
            new_y = []
            for y in y_data:
                res = [None] * len(y)
                # Transform graph parts
                if s.graph_scaler:
                    g_part = np.array([y[i] for i in graph_indices]).reshape(1, -1)
                    g_trans = s.graph_scaler.transform(g_part)[0]
                    for idx, val in zip(graph_indices, g_trans):
                        res[idx] = val

                # Transform node parts
                if s.node_scaler:
                    for i in node_indices:
                         val = np.array(y[i]).reshape(-1, 1)
                         res[i] = s.node_scaler.transform(val).flatten()
                new_y.append(res)
            return new_y

        y_train = transform_mixed(y_train, scaler)
        y_val = transform_mixed(y_val, scaler)
        y_test = transform_mixed(y_test, scaler)

    elif is_pure_atom_wise:
        scaler = StandardScaler()
        y_train_flat = np.concatenate(y_train).reshape(-1, 1)
        scaler.fit(y_train_flat)

        y_train = [scaler.transform(np.array(y).reshape(-1, 1)).flatten() for y in y_train]
        y_val = [scaler.transform(np.array(y).reshape(-1, 1)).flatten() for y in y_val]
        y_test = [scaler.transform(np.array(y).reshape(-1, 1)).flatten() for y in y_test]

    else:
        # Standard Graph Targets
        scaler = StandardScaler()
        if y_train.ndim == 1:
             y_train = y_train.reshape(-1, 1)
             y_val = y_val.reshape(-1, 1)
             y_test = y_test.reshape(-1, 1)

        y_train = scaler.fit_transform(y_train)
        y_val = scaler.transform(y_val)
        y_test = scaler.transform(y_test)

    if scaler_path:
        joblib.dump(scaler, scaler_path)

    train_dataset = MoleculeDataset(smiles_train, y_train, positional_encoding_kwargs=positional_encoding_kwargs)
    val_dataset = MoleculeDataset(smiles_val, y_val, positional_encoding_kwargs=positional_encoding_kwargs)
    test_dataset = MoleculeDataset(smiles_test, y_test, positional_encoding_kwargs=positional_encoding_kwargs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            collate_fn=collate_fn, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             collate_fn=collate_fn, num_workers=num_workers)

    return train_loader, val_loader, test_loader, scaler
