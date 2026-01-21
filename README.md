# SCOUT

Structure-Aware Aspect and Anchor-Count Selection for Node Attribute Augmentation via Positional Information

**WWW 2026 (The Web Conference)**

SCOUT is a model-agnostic node attribute augmentation framework that improves
graph neural networks (GNNs) when node attributes are missing, sparse, or weak.


---

## 📌 Introduction

**SCOUT** is a model-agnostic node attribute augmentation method that enhances the performance of graph neural networks (GNNs) by learning graph-aware positional information. It intelligently selects positional aspects and anchor counts to generate augmented node attributes, especially when original attributes are absent.

SCOUT addresses two core challenges in positional information (PI)-based augmentation:
1. Selecting appropriate structural measures and distance metrics.
2. Automatically determining the optimal number of anchor nodes (K).

To solve this, SCOUT:
- Learns a graph-level attention over diverse centrality–similarity pairs (aspects).
- Uses an elbow detector over centrality rankings to determine anchor-count.
- Can be integrated into standard GNNs for tasks like node classification and link prediction.

📄 **Paper Title**: *SCOUT: Structure-Aware Aspect and Anchor-Count Selection for Node Attribute Augmentation via Positional Information*  
🔍 **Submission**: WWW 2026 (under review)  
📎 **Repository**: https://github.com/seinkim2001/SCOUT

---

## 📂 Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Features](#features)
- [Configuration](#configuration)
- [Dependencies](#dependencies)
- [Examples](#examples)
- [Contributors](#contributors)
- [License](#license)

---

## ⚙️ Installation

### Using Conda (recommended)
```bash
conda env create -f requirements_conda.yaml
conda activate scout
```

### Using pip
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

To run the core pipeline (e.g., link prediction), use the provided shell script:
```bash
bash scripts/run_linkpred.sh
```

Alternatively, you can run individual training scripts:
```bash
python src/core/train_linkpred.py
python src/core/train_nodeclf.py
```

---

## 🧱 Project Structure

```plaintext
SCOUT/
├── attrs/                      # Precomputed centrality and similarity attributes
│   └── Cora_concat_centrality/
├── datasets/                  # Graph datasets (e.g., Cora)
│   └── Cora/
├── logs/                      # Training logs
├── results/                   # Output results and evaluation metrics
├── scripts/                   # Shell scripts for experiment automation
│   └── run_linkpred.sh
├── src/
│   ├── core/                  # Core training and preprocessing logic
│   │   ├── train_linkpred.py
│   │   ├── train_nodeclf.py
│   │   ├── elbow_selector.py
│   │   └── generate_attributes.py
│   ├── models/                # GNN encoder, decoder, attribute gating module
│   │   ├── encoder.py
│   │   ├── decoder.py
│   │   └── attr_gate.py
│   └── utils/                 # Data loading and utility functions
│       └── data_loader.py
├── requirements.txt
├── requirements_conda.yaml
└── README.md
```

---

## ✨ Features

- Model-agnostic augmentation method.
- Learns graph-specific positional aspects (centrality–similarity pairs).
- Automatically detects anchor-count via elbow point.
- Compatible with GNNs for node classification & link prediction.
- Significant performance boost on standard benchmarks (e.g., ogbn-mag, ogbl-ddi, Cora).

---

## ⚙️ Configuration

- Place raw graph datasets in `datasets/` directory.
- Precomputed centrality & similarity features should be stored under `attrs/`.
- You may modify the anchor aspects and centrality settings inside `generate_attributes.py`.

---

## 📦 Dependencies

- Python ≥ 3.8
- PyTorch
- DGL or PyG
- NumPy
- SciPy
- tqdm

> See `requirements.txt` or `requirements_conda.yaml` for full environment setup.

---

## 🧪 Examples

Run link prediction on Cora without original node attributes:
```bash
bash scripts/run_linkpred.sh
```

Train node classification with SCOUT-augmented attributes:
```bash
python src/core/train_nodeclf.py
```

---

## 🧪 Experimental Environment

Experiments were run on the following machine:

```text
Machine: user@peace
GPU(s): 2x NVIDIA RTX A6000 (49GB each)
CUDA Version: 12.2
Driver Version: 535.247.01
```

Python environments were managed using both pip and conda:
- `pip install -r requirements.txt`
- `conda env create -f requirements_conda.yaml`

---

## 📄 License

This project is currently under review for WWW 2026. License details will be updated upon acceptance/publication.

---

