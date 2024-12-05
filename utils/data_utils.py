# utils/data_utils.py

import torch
from torch_geometric.data import Data

def create_hypergraph(num_nodes, num_edges, num_features, num_classes):
    x = torch.rand((num_nodes, num_features), dtype=torch.float32)
    labels = torch.randint(0, num_classes, (num_nodes,))

    edge_index = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if labels[i] == labels[j]:
                edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    data = Data(x=x, edge_index=edge_index)
    return data, labels

def augment_data(data):
    data.x = data.x + torch.randn_like(data.x) * 0.01  # Add Gaussian noise
    return data
