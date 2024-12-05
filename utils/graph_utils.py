# utils/graph_utils.py

import torch

def graph_regularization_loss(data, x):
    edge_index = data.edge_index
    edge_diffs = x[edge_index[0]] - x[edge_index[1]]
    reg_loss = torch.mean(torch.norm(edge_diffs, p=2, dim=1))
    return reg_loss
