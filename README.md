# 🚀 SCOUT: Structure-Aware Aspect and Anchor-Count Selection for Node Attribute Augmentation via Positional Information

[![Paper - WWW 2026](https://img.shields.io/badge/WWW%202026-Accepted-blue.svg)](https://github.com/seinkim2001/SCOUT)  
[![License - TBD](https://img.shields.io/badge/license-TBD-lightgrey.svg)](./LICENSE)  
[![Python](https://img.shields.io/badge/python-≥3.8-blue.svg)]()  
[![Framework](https://img.shields.io/badge/framework-PyTorch%20%7C%20DGL%20%2F%20PyG-orange)]()

**SCOUT** is a **model-agnostic augmentation framework** that enhances graph neural networks (GNNs) when node attributes are **missing**, **sparse**, or **uninformative**, by leveraging **multi-aspect positional information** and a **graph-aware anchor selection mechanism**.

---

## 📖 Abstract

> When node attributes are absent or limited, GNNs often fail to distinguish structurally similar nodes, leading to degraded downstream performance.  
> **SCOUT** addresses this by:
> - Selecting **positional aspects** (centrality–similarity pairs) via a graph-level attention mechanism.
> - Determining the **anchor-count** per graph using a principled **elbow method** grounded in power-law centrality distributions.
> - Augmenting node features with positional information that complements original attributes (when present).

This results in significant gains across **link prediction** and **node classification** tasks, both with and without node attributes.

📄 **Paper**: _SCOUT: Structure-Aware Aspect and Anchor-Count Selection for Node Attribute Augmentation via Positional Information_  
🌐 **Conference**: [The Web Conference (WWW), 2026](https://www2026.thewebconf.org)  
📁 **Code**: [https://github.com/seinkim01/SCOUT](https://github.com/seinkim01/SCOUT)  

---

## 📚 Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Features](#features)
- [Configuration](#configuration)
- [Dependencies](#dependencies)
- [Examples](#examples)
- [Experimental Results](#experimental-results)
- [Troubleshooting](#troubleshooting)
- [Contributors](#contributors)
- [License](#license)

---

## ⚙️ Installation

We recommend using **Conda** to manage dependencies:

```bash
# Recommended
conda env create -f requirements_conda.yaml
conda activate scout
```

Or, use pip:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

To run **link prediction** on Cora with SCOUT:

```bash
bash scripts/run_linkpred.sh
```

For **node classification**:

```bash
python src/core/train_nodeclf.py
```

All major configurations can be modified via the config files or script arguments.

---

## 🧱 Project Structure

```
SCOUT/
├── attrs/                      # Precomputed centrality & similarity scores
├── datasets/                  # Graph datasets (e.g., Cora, Citeseer)
├── logs/                      # Training logs
├── results/                   # Output metrics and predictions
├── scripts/                   # Automation scripts for experiments
├── src/
│   ├── core/                  # Training pipelines & preprocessing
│   ├── models/                # GNN modules, decoder, attribute gates
│   └── utils/                 # Data loading, helper functions
├── requirements.txt
├── requirements_conda.yaml
└── README.md
```

---

## ✨ Key Features

- ✅ **Model-agnostic augmentation**: Integrates with any GNN backbone.
- 🧠 **Graph-level positional aspect selection**: Learns which structural features matter.
- 🎯 **Elbow-based anchor-count detection**: Automatically selects anchor nodes per graph.
- 📈 **Performance improvements** on OGB & citation benchmarks, both with/without features.
- 🔧 Supports downstream tasks: **link prediction** & **node classification**.

---

## ⚙️ Configuration

- Datasets: Place in `datasets/`
- Attributes: Precompute and store in `attrs/`
- Custom aspects: Modify `generate_attributes.py`
- Logging: Enabled via `logs/` directory
- Models: Can be swapped or extended under `src/models/`

---

## 📦 Dependencies

- Python ≥ 3.8
- PyTorch
- DGL or PyG
- NumPy, SciPy, tqdm
- cuGraph (optional for acceleration)

Check `requirements_conda.yaml` for exact versions.

---

## 🔍 Examples

```bash
# Link prediction with SCOUT augmentation
bash scripts/run_linkpred.sh

# Node classification on augmented features
python src/core/train_nodeclf.py
```

---

## 📊 Experimental Results

SCOUT achieves:

- **+26.88% Hits@20** on ogbl-ddi (w/o original attributes)
- **+11.69% accuracy** on ogbn-mag (w/ original attributes)
- Outperforms **HPLC, P-GNN, SEAL** and others across tasks

Refer to the paper or `results/` for detailed tables and plots.

---

## 🛠️ Troubleshooting

- **Issue: Attribute files not found?**  
  Ensure correct folder structure under `attrs/`.

- **Issue: CUDA memory overflow?**  
  Reduce batch size or number of anchors (`K`).

- **Using PyG or DGL?**  
  Modify model imports in `src/models/`.

---

## 👥 Contributors

- **Dong-Hyuk Seo** — Hanyang University  
- **Sein Kim** — Hanyang University
- **Taeri Kim** — Hanyang University
- **Won-Yong Shin** — Yonsei University  
- **Sang-Wook Kim** — Hanyang University

---

## 📄 License

📌 This code is currently under review for publication at **WWW 2026**.  
The license will be updated upon acceptance. For academic use only.

---

> 📣 For citation, please refer to the paper once it's published officially.
