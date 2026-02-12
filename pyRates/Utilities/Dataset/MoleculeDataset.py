import torch
from torch.utils.data import Dataset
from pyRates.Utilities.Dataset.GraphConstruction import smiles_to_graph
from torch_geometric.data import Batch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import joblib
import numpy as np

class MoleculeDataset(Dataset):

    def __init__(self, smiles_list, targets):
        self.graphs = []
        self.targets = []

        for i, smiles in enumerate(smiles_list):
            graph = smiles_to_graph(smiles)
            if graph is not None:
                self.graphs.append(graph)
                self.targets.append(targets[i])

        # Convert targets to tensor efficiently
        if isinstance(self.targets, list):
            # If list of numpy arrays or mixed types
            self.targets = torch.tensor(
                np.array(self.targets),
                dtype=torch.float32
            )
        elif isinstance(self.targets, np.ndarray):
            # If already numpy array
            self.targets = torch.from_numpy(self.targets).float()
        elif isinstance(self.targets, torch.Tensor):
            # If already tensor
            self.targets = self.targets.float()
        else:
            # Fallback
            self.targets = torch.tensor(self.targets, dtype=torch.float32)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.targets[idx]

def collate_fn(batch):
    graphs, targets = zip(*batch)
    batch_graph = Batch.from_data_list(graphs)
    batch_graph.y = torch.stack(targets)
    return batch_graph

def create_dataloaders(smiles_list, targets, train_idx, val_idx, test_idx, batch_size=64, num_workers=8, scaler_path='checkpoints/scaler.pkl'):
    '''
    Create dataloaders for training, validation, and testing. and scale targets using StandardScaler.
    :param smiles_list: List of SMILES strings representing the molecules.
    :param targets: List of target values corresponding to each molecule.
    :param train_idx: Indices for the training set.
    :param val_idx: Indices for the validation set.
    :param test_idx: Indices for the test set.
    :param batch_size: Batch size for the dataloaders.
    :param num_workers: Number of worker processes for data loading.
    :return: train_loader, val_loader, test_loader, scaler: DataLoaders for training, validation, and testing. And the fitted StandardScaler object.
    '''
    smiles_train = smiles_list[train_idx]
    smiles_val = smiles_list[val_idx]
    smiles_test = smiles_list[test_idx]

    y_train = targets[train_idx]
    y_val = targets[val_idx]
    y_test = targets[test_idx]

    scaler = StandardScaler()
    y_train = scaler.fit_transform(y_train)
    y_val = scaler.transform(y_val)
    y_test = scaler.transform(y_test)

    if scaler_path:
        joblib.dump(scaler, scaler_path)

    train_dataset = MoleculeDataset(smiles_train, y_train)
    val_dataset = MoleculeDataset(smiles_val, y_val)
    test_dataset = MoleculeDataset(smiles_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            collate_fn=collate_fn, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             collate_fn=collate_fn, num_workers=num_workers)

    return train_loader, val_loader, test_loader, scaler
