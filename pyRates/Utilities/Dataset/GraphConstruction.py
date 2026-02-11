import torch
from torch_geometric.data import Data
from rdkit import Chem

def mol_to_graph(smiles):

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    node_features = []
    for atom in mol.GetAtoms():
        node_features.append([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            float(atom.GetHybridization()),
            float(atom.GetIsAromatic()),
            atom.GetTotalNumHs(),
        ])

    node_features = torch.tensor(node_features, dtype=torch.float)

    edge_indices = []
    edge_features = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bond_feature = [
            bond.GetBondTypeAsDouble(),
            float(bond.GetIsConjugated()),
            float(bond.IsInRing()),
        ]

        edge_indices += [[i, j], [j, i]]
        edge_features += [bond_feature, bond_feature]

    if len(edge_indices) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 3), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)