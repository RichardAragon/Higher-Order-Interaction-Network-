# train.py

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from adabelief_pytorch import AdaBelief
from sklearn.metrics import accuracy_score
import numpy as np

# Import custom modules (assuming they are in the 'models' and 'utils' directories)
from models.hoin_model import HOINetwork
from utils.data_utils import create_hypergraph, augment_data
from utils.graph_utils import graph_regularization_loss

def train():
    # Hyperparameters
    num_nodes = 100
    num_features = 10
    num_classes = 5
    num_edges = 200
    num_epochs = 50
    lambda_reg = 0.1
    topo_embedding_weight = 2.0
    learning_rate = 0.001

    # Prepare data
    data, labels = create_hypergraph(num_nodes, num_edges, num_features, num_classes)
    labels = labels.long()

    # Initialize model
    model = HOINetwork(num_features=num_features, num_classes=num_classes, topo_embedding_weight=topo_embedding_weight)

    # Optimizer and scheduler
    optimizer = AdaBelief(model.parameters(), lr=learning_rate, eps=1e-16, weight_decay=1e-4, rectify=True)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # Loss function
    class_weights = torch.tensor([1.0] * num_classes)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Data augmentation
        augmented_data = augment_data(data)

        # Forward pass
        out = model(augmented_data)
        loss = criterion(out, labels)

        # Regularization loss
        reg_loss = graph_regularization_loss(augmented_data, out)
        total_loss = loss + lambda_reg * reg_loss

        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step(epoch + (epoch % scheduler.T_0) / scheduler.T_0)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss.item():.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        out = model(data)
        predictions = torch.argmax(out, dim=1)
        accuracy = accuracy_score(labels.numpy(), predictions.numpy())
        print(f"\nTraining Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    train()
