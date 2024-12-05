# Install required libraries
!pip install torch_geometric gudhi adabelief-pytorch geoopt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv
from torch_geometric.data import Data
from gudhi import RipsComplex
import numpy as np
from adabelief_pytorch import AdaBelief
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


class HOINetwork(nn.Module):
    def __init__(self, num_nodes, num_features, num_classes):
        super(HOINetwork, self).__init__()

        self.hypergraph_conv1 = HypergraphConv(num_features, 64)
        self.bn1 = nn.BatchNorm1d(64)

        self.hypergraph_conv2 = HypergraphConv(64, 128)
        self.bn2 = nn.BatchNorm1d(128)

        self.hypergraph_conv3 = HypergraphConv(128, 64)
        self.bn3 = nn.BatchNorm1d(64)

        self.topology_fc = nn.Linear(1, 64)  # Align topo feature size to 64

        self.fc1 = nn.Linear(64 + 64, 128)  # Adjust for concatenation
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.bn1(self.hypergraph_conv1(x, edge_index)))
        x = F.relu(self.bn2(self.hypergraph_conv2(x, edge_index)))
        x = self.bn3(self.hypergraph_conv3(x, edge_index))

        topo_features = self.compute_topological_features(data)
        topo_embedding = F.relu(self.topology_fc(topo_features))

        topo_embedding_weight = 2.0
        combined = torch.cat([x, topo_embedding.unsqueeze(0).repeat(x.shape[0], 1) * topo_embedding_weight], dim=1)

        out = F.relu(self.fc1(combined))
        out = self.fc2(out)
        return F.log_softmax(out, dim=1)

    def compute_topological_features(self, data):
        embeddings = data.x.detach().numpy()
        distances = np.linalg.norm(embeddings[:, None] - embeddings, axis=2)

        rips_complex = RipsComplex(distance_matrix=distances, max_edge_length=2.0)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
        persistence = simplex_tree.persistence()

        persistence_features = np.array([pair[1][1] - pair[1][0] for pair in persistence if pair[0] == 1])
        if len(persistence_features) == 0:
            persistence_features = np.array([0.0])
        persistence_features = torch.tensor([persistence_features.mean()], dtype=torch.float32)
        return persistence_features


def create_hypergraph(num_nodes, num_edges, num_features, num_classes):
    """
    Create a structured hypergraph dataset for learning.
    """
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
    """
    Augment data by injecting noise into node features.
    """
    data.x += torch.randn_like(data.x) * 0.01  # Add Gaussian noise
    return data


def graph_regularization_loss(data, x):
    """
    Compute a graph regularization loss to enforce smoothness in embeddings.
    """
    edge_index = data.edge_index
    edge_diffs = x[edge_index[0]] - x[edge_index[1]]
    reg_loss = torch.mean(torch.norm(edge_diffs, p=2, dim=1))
    return reg_loss


# Hyperparameters
num_nodes, num_features, num_classes, num_edges = 100, 10, 5, 200
data, labels = create_hypergraph(num_nodes, num_edges, num_features, num_classes)

model = HOINetwork(num_nodes, num_features, num_classes)

# Optimizer and Scheduler
optimizer = AdaBelief(model.parameters(), lr=0.001, eps=1e-16, weight_decay=1e-4, rectify=True)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

# Loss Function
class_weights = torch.tensor([1.0] * num_classes)  # Adjust if class imbalance exists
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Training Loop
num_epochs, lambda_reg = 50, 0.1
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Data augmentation
    augmented_data = augment_data(data)

    # Forward pass
    out = model(augmented_data)
    loss = criterion(out, labels)

    # Regularization loss
    reg_loss = graph_regularization_loss(data, out)
    total_loss = loss + lambda_reg * reg_loss

    total_loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    scheduler.step(epoch + (epoch % scheduler.T_0) / scheduler.T_0)  # Adjust for warm restarts

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss.item()}")
