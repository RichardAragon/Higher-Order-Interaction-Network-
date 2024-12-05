# models/hoin_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv
from gudhi import RipsComplex
import numpy as np

class HOINetwork(nn.Module):
    def __init__(self, num_features, num_classes, topo_embedding_weight=2.0):
        super(HOINetwork, self).__init__()
        self.topo_embedding_weight = topo_embedding_weight

        self.hypergraph_conv1 = HypergraphConv(num_features, 64)
        self.bn1 = nn.BatchNorm1d(64)

        self.hypergraph_conv2 = HypergraphConv(64, 128)
        self.bn2 = nn.BatchNorm1d(128)

        self.hypergraph_conv3 = HypergraphConv(128, 64)
        self.bn3 = nn.BatchNorm1d(64)

        self.topology_fc = nn.Linear(1, 64)

        self.fc1 = nn.Linear(64 + 64, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.bn1(self.hypergraph_conv1(x, edge_index)))
        x = F.relu(self.bn2(self.hypergraph_conv2(x, edge_index)))
        x = self.bn3(self.hypergraph_conv3(x, edge_index))

        topo_features = self.compute_topological_features(data)
        topo_embedding = F.relu(self.topology_fc(topo_features))

        combined = torch.cat(
            [x, topo_embedding.unsqueeze(0).repeat(x.shape[0], 1) * self.topo_embedding_weight],
            dim=1
        )

        out = F.relu(self.fc1(combined))
        out = self.fc2(out)
        return F.log_softmax(out, dim=1)

    def compute_topological_features(self, data):
        embeddings = data.x.detach().cpu().numpy()
        distances = np.linalg.norm(embeddings[:, None] - embeddings, axis=2)

        rips_complex = RipsComplex(distance_matrix=distances, max_edge_length=2.0)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
        persistence = simplex_tree.persistence()

        persistence_features = np.array(
            [pair[1][1] - pair[1][0] for pair in persistence if pair[0] == 1]
        )
        if len(persistence_features) == 0:
            persistence_features = np.array([0.0])
        persistence_features = torch.tensor([persistence_features.mean()], dtype=torch.float32)
        return persistence_features
