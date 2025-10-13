# SCOUT: Structure-Aware Aspect and Anchor-Count Selection for Positional Attribute Augmentation

> 🚧 **Status:** *Submitted to The Web Conference (WWW) 2026 — Under Review*  
> This repository provides a clean, fully reproducible implementation of the **SCOUT** framework described in the submitted paper.

---

## 🧠 Overview

Graph Neural Networks (GNNs) often fail to distinguish **locally isomorphic nodes** when node attributes are absent.  
**SCOUT** (Structure-Aware Aspect and Anchor-Count Selection for Positional Attribute Augmentation) tackles this limitation by:

- Selecting *structural aspects* (centrality–similarity pairs) that best encode positional information per graph,  
- Learning an *attention-based gating mechanism* that adaptively fuses multiple aspects,  
- Automatically determining the *anchor count (K)* using an **elbow-based heuristic**.

SCOUT is **model-agnostic**, lightweight, and can be applied to **both link prediction and node classification** tasks.

---

## 🏗️ Repository Structure

SCOUT/
├── src/
│ ├── core/
│ │ └── train_linkpred.py # Main training script (Link Prediction)
│ ├── models/
│ │ ├── models.py # GCN / GraphSAGE encoders + MLP decoder
│ │ └── attr_gate.py # MeasureAttentionGateV3 (aspect gating module)
│ └── utils/
│ └── dataset.py # Dataset loader (Planetoid / OGB / Amazon / Coauthor)
│
├── attrs/ # Precomputed SCOUT node attributes
│ └── Cora_concat_centrality/
│ ├── concat_all_top10.684.npy
│ └── meta_concat_all_top10.684.json
│
├── scripts/
│ └── run_linkpred.sh # Unified bash script for both modes
│
├── results/
│ └── logs/
│ ├── cora_wofeat_gcn.log
│ └── cora_wfeat_gcn.log
│
├── requirements.txt # Pip environment (exact package versions)
├── requirements_conda.yaml # Conda environment (recommended)
└── README.md