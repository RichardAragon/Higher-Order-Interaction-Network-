# Higher-Order Interaction Network (HOIN)

![Higher-Order Interaction Network](https://img.shields.io/badge/license-MIT-blue.svg) ![Python](https://img.shields.io/badge/Python-3.7%2B-brightgreen.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange.svg)

Higher-Order Interaction Network (HOIN) is a novel neural network architecture designed to model complex **higher-order interactions** within graph-structured and topological data. HOIN integrates graph neural networks, persistent homology, and advanced optimization techniques to capture higher-order relationships and topological structures, enabling groundbreaking performance in tasks like brain signal analysis, social network modeling, and beyond.

## Features

- **Hypergraph Convolutions**: Capture higher-order relationships among nodes in hypergraphs.
- **Topology-Aware Embedding**: Utilize persistent homology to extract meaningful topological features from data.
- **Advanced Optimization**: Powered by the AdaBelief optimizer, gradient clipping, and cosine annealing with warm restarts for robust learning.
- **Batch Normalization**: Stabilize training through feature normalization across layers.
- **Graph Regularization**: Ensure smoothness and structural consistency in graph embeddings.

## License

This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute as per the license terms.

---

## Installation

To set up the environment and dependencies, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/RichardAragon/Higher-Order-Interaction-Network.git
   cd Higher-Order-Interaction-Network
   ```

2. Install required dependencies:
   ```bash
   pip install torch torch_geometric gudhi adabelief-pytorch geoopt
   ```

3. Verify installation:
   - Ensure your environment includes Python 3.7+ and PyTorch 1.10+.

---

## Usage

### Training the Model
Run the training script to train HOIN on a synthetic hypergraph dataset:
```bash
python train.py
```

### Key Components
- **Hypergraph Convolutions**: Efficiently model higher-order interactions.
- **Topological Features**: Automatically extracted using persistent homology (via GUDHI).
- **Regularization**: Enforces smoothness in learned embeddings.

### Example Output
Sample training logs:
```
Epoch 1/50, Loss: 1.5696
Epoch 10/50, Loss: 1.5419
Epoch 20/50, Loss: 1.3348
Epoch 50/50, Loss: 0.6102
```

---

## Project Structure

```plaintext
Higher-Order-Interaction-Network/
├── LICENSE               # MIT License file
├── README.md             # This ReadMe file
├── train.py              # Main training script
├── hoin_model.py         # HOIN model definition
├── requirements.txt      # Dependency list
├── utils/
│   ├── data_utils.py     # Synthetic dataset generation
│   ├── graph_utils.py    # Graph regularization and augmentation
│   ├── topology_utils.py # Topological feature computation
```

---

## Contributing

We welcome contributions from the community! Here's how you can get involved:

1. Fork the repository and create a feature branch.
2. Submit a pull request with your changes.
3. Ensure your code is well-documented and follows PEP8 guidelines.

---

## Citation

If you use `Higher-Order Interaction Network` in your research or project, please cite this repository:

```
@misc{hoin2024,
  author = {Richard Aragon},
  title = {Higher-Order Interaction Network},
  year = {2024},
  howpublished = {\url{https://github.com/yourusername/Higher-Order-Interaction-Network}},
  note = {MIT License}
}
```

---

## Acknowledgments

- **PyTorch Geometric**: For advanced graph operations.
- **GUDHI**: For persistent homology and topological computations.
- **AdaBelief**: For adaptive, robust optimization.

---
